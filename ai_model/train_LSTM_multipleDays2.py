import os
import joblib
import json
import argparse
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler # [추가] StandardScaler
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    """
    터미널에서 모델 학습에 필요한 인자들을 받아오는 함수
    """
    parser = argparse.ArgumentParser(description='Time Series Forecasting Model Training')

    parser.add_argument('--tag', type=str, default='base', help='실험을 식별하기 위한 태그')
    parser.add_argument('--use_all_features', action='store_true', help='True로 설정 시 모든 피처(요일, diff 포함)를 사용합니다.')
    parser.add_argument('--scaler', type=str, default='minmax', choices=['minmax', 'standard'], help='사용할 스케일러 선택')

    parser.add_argument('--sequence_length', type=int, default=120, help='입력 시퀀스 길이')
    parser.add_argument('--prediction_horizon', type=int, default=5, help='예측할 기간(일)')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM 은닉층의 크기')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM 레이어의 수')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='드롭아웃 확률')

    parser.add_argument('--num_epochs', type=int, default=200, help='학습 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='학습률')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    
    args = parser.parse_args()
    return args

def load_and_preprocess_data(args):
    """
    [변경] config 대신 args를 인자로 받도록 수정
    """
    if args.data_path.endswith('.csv'):
        df = pd.read_csv(args.data_path)
    else:
        df = pd.read_excel(args.data_path)
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    features_df = df[args.feature_columns]
    target_series = df[args.target_column]

    if args.scaler == 'standard':
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    else: # minmax
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(features_df)
    scaled_target = target_scaler.fit_transform(target_series.values.reshape(-1, 1))
    
    folder_path = model_folder_name(args)
    joblib.dump(feature_scaler, os.path.join(folder_path, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(folder_path, 'target_scaler.pkl'))
    
    X, y, dates = [], [], []
    for i in range(len(scaled_features) - args.sequence_length - args.prediction_horizon + 1):
        X.append(scaled_features[i : i + args.sequence_length])
        y.append(scaled_target[i + args.sequence_length : i + args.sequence_length + args.prediction_horizon].squeeze())
        dates.append(df['Date'].iloc[i + args.sequence_length])
    
    X, y = np.array(X), np.array(y)
    dates = pd.Series(dates).reset_index(drop=True)

    split_index = dates[dates < args.test_start_date].index[-1] + 1
    
    train_dates = dates[:split_index]
    test_dates = dates[split_index:]
    
    print("--- 데이터 기간 모니터링 ---")
    print(f"전체 데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    if not train_dates.empty:
        print(f"학습 데이터 기간: {train_dates.iloc[0].strftime('%Y-%m-%d')} ~ {train_dates.iloc[-1].strftime('%Y-%m-%d')}")
    if not test_dates.empty:
        print(f"테스트 데이터 기간: {test_dates.iloc[0].strftime('%Y-%m-%d')} ~ {test_dates.iloc[-1].strftime('%Y-%m-%d')}")
    print("--------------------------\n")

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(y_test) > 0:
        y_test = y_test[:len(test_dates)]
        X_test = X_test[:len(test_dates)]

    return X_train, y_train, X_test, y_test, target_scaler, test_dates

class ExchangeRateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, output_size)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden_state = out[:, -1, :]
        final_output = self.fc(last_hidden_state)
        return final_output

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    print("--- 모델 학습 시작 ---")
    model.train()
    for epoch in range(num_epochs):
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{num_epochs:03d}", leave=False)
        epoch_loss = 0
        for X_batch, y_batch in batch_iterator:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_iterator.set_postfix(loss=f"{loss.item():.6f}")
        avg_epoch_loss = epoch_loss / len(train_loader)
        if ((epoch + 1) % 10 == 0):
            print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | Average Loss: {avg_epoch_loss:.6f}")
    print("--- 모델 학습 완료 ---\n")

def evaluate_model(model, test_loader, target_scaler, device):
    print("--- 모델 평가 시작 ---")
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            outputs_rescaled = target_scaler.inverse_transform(outputs.cpu().numpy())
            y_batch_rescaled = target_scaler.inverse_transform(y_batch.numpy())
            predictions.extend(outputs_rescaled)
            actuals.extend(y_batch_rescaled)
            
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = np.mean(np.abs(predictions - actuals))
    print(f"테스트 데이터 전체 MAE (평균 절대 오차): {mae:.2f} 원")
    for i in range(predictions.shape[1]):
        mae_day = np.mean(np.abs(predictions[:, i] - actuals[:, i]))
        print(f"  - {i+1}일 후 예측 MAE: {mae_day:.2f} 원")
    print("--- 모델 평가 완료 ---\n")
    return predictions, actuals

def save_model(model, args):
    folder_path = model_folder_name(args)
    model_path = os.path.join(folder_path, 'best_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"✅ 학습된 모델이 '{model_path}' 경로에 저장되었습니다.")
    
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    print(f"✅ 모델 설정이 '{config_path}' 경로에 저장되었습니다.\n")

def model_folder_name(args):
    folder_name = (
        f"seq_{args.sequence_length}-pred_{args.prediction_horizon}"
        f"-hidden_{args.hidden_size}-layers_{args.num_layers}"
        f"-batch_{args.batch_size}-lr_{args.learning_rate}"
        f"-scaler_{args.scaler}-tag_{args.tag}"
    )
    folder_path = os.path.join('models', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"'{folder_path}' 폴더를 생성했습니다.")
    return folder_path

def plot_test_results(args, test_dates, predictions, actuals):
    plt.figure(figsize=(15, 8))
    plt.plot(test_dates, actuals[:, 0], label='Actual (ECOS Rate)', color='blue', marker='o', markersize=4, linestyle='-')
    plt.plot(test_dates, predictions[:, 0], label='Predicted Rate (1-day ahead)', color='red', linestyle='--', marker='x', markersize=4)
    if len(test_dates) > args.prediction_horizon:
        last_prediction_start_date = test_dates.iloc[-args.prediction_horizon]
        future_dates = pd.date_range(start=last_prediction_start_date, periods=args.prediction_horizon)
        plt.plot(future_dates, actuals[-args.prediction_horizon, :], 's', color='limegreen', markersize=6, label=f'Actuals for last {args.prediction_horizon} days')
        plt.plot(future_dates, predictions[-args.prediction_horizon, :], '^', color='orange', markersize=6, label=f'Prediction for last {args.prediction_horizon} days')
    plt.title(f'Exchange Rate Prediction vs Actual ({args.prediction_horizon}-day Horizon)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('KRW / USD', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    folder_path = model_folder_name(args)
    fig_path = os.path.join(folder_path, 'prediction_vs_actual.png')
    plt.savefig(fig_path)
    print(f"✅ 예측 결과 그래프가 '{fig_path}'에 저장되었습니다.")
    plt.show()

if __name__ == '__main__':
    args = get_args()
    
    # 1. 피처 컬럼 설정
    simple_features = ['Inv_Close', 'ECOS_Close', 'DXY_Close', 'US10Y_Close']
    all_features = [
        'Inv_Close', 'Inv_Open', 'Inv_High', 'Inv_Low', 'Inv_Change(%)', 
        'ECOS_Close', 'DXY_Close', 'US10Y_Close',
        'is_Mon', 'is_Tue', 'is_Wed', 'is_Thu', 'is_Fri', 'diff'
    ]
    if args.use_all_features:
        args.feature_columns = all_features
        args.data_path = 'data/train/train_final_with_onehot_20100104_20250811_with_diff.xlsx' # 전체 피처용 데이터
    else:
        args.feature_columns = simple_features
        args.data_path = 'data/train/train_final_with_onehot_20100104_20250811_simple.xlsx' # 기본 피처용 데이터

    # 2. 파생되는 설정값 추가
    args.input_size = len(args.feature_columns)
    args.output_size = args.prediction_horizon
    args.target_column = 'target'
    args.test_start_date = '2025-01-01'
    
    # 3. 시드 고정 및 장치 설정
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- 데이터 로딩 및 전처리 ---
    X_train, y_train, X_test, y_test, target_scaler, test_dates = load_and_preprocess_data(args)

    # --- 데이터셋 및 데이터로더 ---
    train_dataset = ExchangeRateDataset(X_train, y_train)
    test_dataset = ExchangeRateDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- 모델, 손실함수, 옵티마이저 ---
    model = LSTMModel(
        input_size=args.input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        output_size=args.output_size,
        dropout_prob=args.dropout_prob
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # --- 학습 및 평가 ---
    train_model(model, train_loader, criterion, optimizer, args.num_epochs, device)
    save_model(model, args)
    
    if len(X_test) > 0:
        predictions, actuals = evaluate_model(model, test_loader, target_scaler, device)
        plot_test_results(args, test_dates, predictions, actuals)
    else:
        print("테스트 데이터가 없어 평가 및 시각화를 건너뜁니다.")