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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from model import LSTMModel
from utils.early_stopping import EarlyStopping
from utils.exchange_dataset import ExchangeRateDataset

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
    # [추가] 조기 종료 인자
    parser.add_argument('--patience', type=int, default=10, help='Early stopping을 위한 patience')
    
    args = parser.parse_args()
    return args

def load_and_preprocess_data(args):
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
    else:
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

    # 1. 테스트 데이터 분리
    test_split_index = dates[dates < args.test_start_date].index[-1] + 1
    
    X_rem, X_test = X[:test_split_index], X[test_split_index:]
    y_rem, y_test = y[:test_split_index], y[test_split_index:]
    dates_rem, test_dates = dates[:test_split_index], dates[test_split_index:]

    # 2. 남은 데이터를 학습/검증 데이터로 분리
    val_split_index = int(len(X_rem) * 0.9)
    
    X_train, X_val = X_rem[:val_split_index], X_rem[val_split_index:]
    y_train, y_val = y_rem[:val_split_index], y_rem[val_split_index:]
    
    # [추가] dates도 학습/검증용으로 분리
    train_dates, val_dates = dates_rem[:val_split_index], dates_rem[val_split_index:]

    # [변경] 데이터 기간과 샘플 수를 함께 출력하도록 개선
    print("--- 데이터 분할 모니터링 ---")
    print(f"전체 원본 데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    if not train_dates.empty:
        print(f"학습 데이터 기간:   {train_dates.iloc[0].strftime('%Y-%m-%d')} ~ {train_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 수: {len(X_train)})")
    if not val_dates.empty:
        print(f"검증 데이터 기간:   {val_dates.iloc[0].strftime('%Y-%m-%d')} ~ {val_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 수: {len(X_val)})")
    if not test_dates.empty:
        print(f"테스트 데이터 기간: {test_dates.iloc[0].strftime('%Y-%m-%d')} ~ {test_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 수: {len(X_test)})")
    print("--------------------------\n")

    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, test_dates

def train_model(args, model, train_loader, val_loader, criterion, optimizer, device):
    """
    [변경] val_loader와 args를 인자로 받고, 조기 종료 로직을 포함
    """
    print("--- 모델 학습 시작 ---")
    
    # 조기 종료 설정
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(model_folder_name(args), 'best_model.pth'))

    for epoch in range(args.num_epochs):
        # --- 학습 단계 ---
        model.train()
        train_loss = 0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.num_epochs:03d} [Train]", leave=False)
        for X_batch, y_batch in train_iterator:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iterator.set_postfix(loss=f"{loss.item():.6f}")

        # --- 검증 단계 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{args.num_epochs:03d} [Valid]", leave=False)
            for X_batch, y_batch in val_iterator:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_iterator.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:03d}/{args.num_epochs:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # --- 조기 종료 호출 ---
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    print("--- 모델 학습 완료 ---\n")
    # 가장 좋았던 모델의 가중치를 다시 불러옴
    model.load_state_dict(torch.load(early_stopping.path))
    return model

# evaluate_model, save_model, model_folder_name, plot_test_results 함수는 이전과 동일
# ... (이전 코드 붙여넣기) ...
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

def save_model_config(args):
    """
    최종 모델이 아닌, 실행된 설정(config)만 저장하는 함수
    """
    folder_path = model_folder_name(args)
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
    # ... (이전 코드와 동일) ...
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
    
    # ... (피처, 설정 구성 부분은 이전과 동일) ...
    #simple_features = ['Inv_Close', 'ECOS_Close', 'DXY_Close', 'US10Y_Close']
    simple_features = ['Inv_Close', 'ECOS_Close', 'diff']
    all_features = [
        'Inv_Close', 'Inv_Open', 'Inv_High', 'Inv_Low', 'Inv_Change(%)', 
        'ECOS_Close', 'DXY_Close', 'US10Y_Close',
        'is_Mon', 'is_Tue', 'is_Wed', 'is_Thu', 'is_Fri', 'diff'
    ]
    if args.use_all_features:
        args.feature_columns = all_features
        args.data_path = 'data/train/train_final_with_onehot_20100104_20250812_all.xlsx'
    else:
        args.feature_columns = simple_features
        args.data_path = 'data/train/train_final_with_onehot_20100104_20250812_all.csv'

    args.input_size = len(args.feature_columns)
    args.output_size = args.prediction_horizon
    args.target_column = 'target'
    args.test_start_date = '2025-01-01'
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 반환받는 변수 수정
    X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, test_dates = load_and_preprocess_data(args)

    # 검증 데이터셋 및 로더 추가
    train_dataset = ExchangeRateDataset(X_train, y_train)
    val_dataset = ExchangeRateDataset(X_val, y_val)
    test_dataset = ExchangeRateDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = LSTMModel(
        input_size=args.input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        output_size=args.output_size,
        dropout_prob=args.dropout_prob
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # [변경] train_model 호출 방식 수정
    model = train_model(args, model, train_loader, val_loader, criterion, optimizer, device)
    
    # 설정 파일은 학습이 끝나면 항상 저장
    save_model_config(args)
    
    if len(X_test) > 0:
        predictions, actuals = evaluate_model(model, test_loader, target_scaler, device)
        plot_test_results(args, test_dates, predictions, actuals)
    else:
        print("테스트 데이터가 없어 평가 및 시각화를 건너뜁니다.")