import os
import joblib
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# -- 1. 모델 및 학습 파라미터 설정 --
class Config:
    data_path = 'data/train/train_final_with_onehot_20100104_20250811_with_diff_cutted.xlsx'
    
    target_column = 'target'
    # feature_columns = [
    #     'Inv_Close', 'Inv_Open', 'Inv_High', 'Inv_Low', 'Inv_Change(%)', 'ECOS_Close', 'DXY_Close', 'US10Y_Close',
    #     'is_Mon', 'is_Tue', 'is_Wed', 'is_Thu', 'is_Fri', 'diff'
    # ]
    feature_columns = [
        'Inv_Close', 'ECOS_Close', 'DXY_Close', 'US10Y_Close', 'diff',
        'is_Mon', 'is_Tue', 'is_Wed', 'is_Thu', 'is_Fri', 
    ]
    
    tag = 'shorten'
    train_start_date = '2010-01-04'
    test_start_date = '2025-01-01'
    
    sequence_length = 120      # 입력 시퀀스 길이
    prediction_horizon = 5     # [추가] 예측할 기간 (일)

    # 모델 하이퍼파라미터
    input_size = len(feature_columns)
    hidden_size = 128
    num_layers = 2
    output_size = prediction_horizon # [변경] 출력 크기를 예측 기간으로 설정
    
    # 학습 관련
    num_epochs = 200
    learning_rate = 0.001
    batch_size = 16


# -- 2. 데이터 전처리 및 시퀀스 생성 --
def load_and_preprocess_data(config):
    if config.data_path.endswith('.csv'):
        df = pd.read_csv(config.data_path)
    else:
        df = pd.read_excel(config.data_path)
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    features_df = df[config.feature_columns]
    target_series = df[config.target_column]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(features_df)
    scaled_target = target_scaler.fit_transform(target_series.values.reshape(-1, 1))
    
    # 폴더 생성 및 스케일러 저장
    folder_path = model_folder_name(config)
    joblib.dump(feature_scaler, os.path.join(folder_path, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(folder_path, 'target_scaler.pkl'))
    
    X, y, dates = [], [], []
    # [변경] 루프 범위를 prediction_horizon 만큼 추가로 확보
    for i in range(len(scaled_features) - config.sequence_length - config.prediction_horizon + 1):
        X.append(scaled_features[i : i + config.sequence_length])
        # [변경] y 값을 prediction_horizon 길이의 시퀀스로 저장
        y.append(scaled_target[i + config.sequence_length : i + config.sequence_length + config.prediction_horizon].squeeze())
        dates.append(df['Date'].iloc[i + config.sequence_length])
    
    X, y = np.array(X), np.array(y)
    dates = pd.Series(dates).reset_index(drop=True)

    # test_start_date 이전의 마지막 날짜를 기준으로 분할 인덱스 결정
    # 이렇게 해야 test_dates와 y_test의 길이가 정확히 일치함
    split_index = dates[dates < config.test_start_date].index[-1] + 1
    
    train_dates = dates[:split_index]
    test_dates = dates[split_index:]
    
    print("--- 데이터 기간 모니터링 ---")
    print(f"전체 데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"학습 데이터 기간: {train_dates.iloc[0].strftime('%Y-%m-%d')} ~ {train_dates.iloc[-1].strftime('%Y-%m-%d')}")
    # test_dates가 비어있지 않은 경우에만 출력
    if not test_dates.empty:
        print(f"테스트 데이터 기간: {test_dates.iloc[0].strftime('%Y-%m-%d')} ~ {test_dates.iloc[-1].strftime('%Y-%m-%d')}")
    else:
        print("테스트 데이터가 없습니다.")
    print("--------------------------\n")

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # [추가] X_test와 y_test의 길이를 일치시키기 위한 조정
    # y_test의 각 샘플은 미래 N일(prediction_horizon)을 예측해야 하므로,
    # X_test의 마지막 샘플은 y_test가 끝나는 지점보다 N-1일 앞에 있어야 함.
    if len(y_test) > 0:
        y_test = y_test[:len(test_dates)]
        X_test = X_test[:len(test_dates)]


    return X_train, y_train, X_test, y_test, target_scaler, test_dates


# -- 3. PyTorch Dataset 클래스 정의 --
class ExchangeRateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -- 4. LSTM 모델 구조 정의 (구조 변경 없음) --
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.2):
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


# -- 5. 모델 학습 함수 (변경 없음) --
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    print("--- 모델 학습 시작 ---")
    model.train()
    
    for epoch in range(num_epochs):
        batch_iterator = tqdm(train_loader, 
                              desc=f"Epoch {epoch+1:03d}/{num_epochs:03d}", 
                              leave=False)
        
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


# -- 6. 모델 평가 함수 --
def evaluate_model(model, test_loader, target_scaler, device):
    print("--- 모델 평가 시작 ---")
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            outputs = model(X_batch)
            
            # [변경] inverse_transform은 CPU에서 수행되어야 하며, numpy 배열이어야 함
            outputs_rescaled = target_scaler.inverse_transform(outputs.cpu().numpy())
            y_batch_rescaled = target_scaler.inverse_transform(y_batch.numpy())
            
            predictions.extend(outputs_rescaled)
            actuals.extend(y_batch_rescaled)
            
    # [변경] predictions와 actuals를 numpy 배열로 변환하여 계산
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # [변경] 모든 예측 지점에 대한 MAE 계산
    mae = np.mean(np.abs(predictions - actuals))
    print(f"테스트 데이터 전체 MAE (평균 절대 오차): {mae:.2f} 원")

    # [추가] 예측 기간별 MAE 계산
    for i in range(predictions.shape[1]):
        mae_day = np.mean(np.abs(predictions[:, i] - actuals[:, i]))
        print(f"  - {i+1}일 후 예측 MAE: {mae_day:.2f} 원")

    print("--- 모델 평가 완료 ---\n")
    return predictions, actuals


def save_model(model, config):
    folder_path = model_folder_name(config)
    model_path = os.path.join(folder_path, 'best_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"✅ 학습된 모델이 '{model_path}' 경로에 저장되었습니다.\n")

    config_path = os.path.join(folder_path, 'config.json')
    config_dict = {key: getattr(config, key) for key in dir(config) if not key.startswith('__') and not callable(getattr(config, key))}
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 모델 설정이 '{config_path}' 경로에 저장되었습니다.\n")


def model_folder_name(config):
    folder_name = (
        f"seq_{config.sequence_length}"
        f"-pred_{config.prediction_horizon}" # [추가] 폴더명에 예측 기간 추가
        f"-hidden_{config.hidden_size}"
        f"-layers_{config.num_layers}"
        f"-batch_{config.batch_size}"
        f"-date_{config.train_start_date.replace('-', '')}"
        f"-tag_{config.tag}"
    )
    folder_path = os.path.join('models', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"'{folder_path}' 폴더를 생성했습니다.")
    return folder_path

def plot_test_results(config, test_dates, predictions, actuals):
    """
    테스트 결과를 시각화합니다.
    """
    # [변경] 시각화 코드: 다중 예측 결과 중 첫 번째 날(t+1)을 기준으로 비교
    plt.figure(figsize=(15, 8))
    # 실제값은 모든 기간에 대해 플로팅 (첫번째 값 기준)
    plt.plot(test_dates, actuals[:, 0], label='Actual (ECOS Rate)', color='blue', marker='o', markersize=4, linestyle='-')
    # 예측값은 1일 후 예측치(t+1)를 사용
    plt.plot(test_dates, predictions[:, 0], label=f'Predicted Rate (1-day ahead)', color='red', linestyle='--', marker='x', markersize=4)

    # [추가] 마지막 예측 시퀀스를 함께 시각화하여 다중 예측을 보여줌
    if len(test_dates) > config.prediction_horizon:
        last_prediction_start_date = test_dates.iloc[-config.prediction_horizon]
        future_dates = pd.date_range(start=last_prediction_start_date, periods=config.prediction_horizon)
        
        plt.plot(future_dates, actuals[-config.prediction_horizon, :], 's', color='limegreen', markersize=6, label=f'Actuals for last {config.prediction_horizon} days')
        plt.plot(future_dates, predictions[-config.prediction_horizon, :], '^', color='orange', markersize=6, label=f'Prediction for last {config.prediction_horizon} days')

    plt.title(f'Exchange Rate Prediction vs Actual ({config.prediction_horizon}-day Horizon)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('KRW / USD', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    folder_path = model_folder_name(config)
    fig_path = os.path.join(folder_path, 'Figure1.png')
    plt.savefig(fig_path)

    plt.show()

# -- 7. 메인 실행 블록 --
if __name__ == '__main__':
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X_train, y_train, X_test, y_test, target_scaler, test_dates = load_and_preprocess_data(config)

    train_dataset = ExchangeRateDataset(X_train, y_train)
    test_dataset = ExchangeRateDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = LSTMModel(config.input_size, config.hidden_size, config.num_layers, config.output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_model(model, train_loader, criterion, optimizer, config.num_epochs, device)
    save_model(model, config)
    
    predictions, actuals = evaluate_model(model, test_loader, target_scaler, device)
    
    plot_test_results(config, test_dates, predictions, actuals)