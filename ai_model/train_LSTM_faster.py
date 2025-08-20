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
import time # 시간 측정을 위해 추가

# -- 1. 모델 및 학습 파라미터 설정 --
class Config:
    data_path = 'data/train/train_final_with_onehot_20100104_20250812_with_diff.xlsx'
    processed_data_path = 'data/processed' # 전처리된 데이터를 저장할 경로

    target_column = 'target'
    feature_columns = [
        'Inv_Close', 'Inv_Open', 'Inv_High', 'Inv_Low', 'Inv_Change(%)', 'ECOS_Close', 'DXY_Close', 'US10Y_Close',
        'is_Mon', 'is_Tue', 'is_Wed', 'is_Thu', 'is_Fri', 'diff'
    ]
    
    train_start_date = '2015-01-02'
    test_start_date = '2025-01-01'
    sequence_length = 120

    # 모델 하이퍼파라미터
    input_size = len(feature_columns)
    hidden_size = 128
    num_layers = 2
    output_size = 1
    
    # 학습 관련
    num_epochs = 200
    learning_rate = 0.001
    batch_size = 16
    # ⭐ 개선점: DataLoader 최적화를 위한 파라미터 추가
    num_workers = os.cpu_count() // 2 # 사용 가능한 CPU 코어의 절반을 사용


# -- 2. 데이터 전처리 및 시퀀스 생성 --
def load_and_preprocess_data(config):
    """
    매번 원본 데이터 파일을 읽어와 전처리 및 시퀀스 생성을 수행합니다.
    """
    print("--- 데이터 전처리 시작 ---")
    
    # 1. 데이터 로드
    if config.data_path.endswith('.csv'):
        df = pd.read_csv(config.data_path)
    else:
        df = pd.read_excel(config.data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 2. 피처와 타겟 분리 및 스케일링
    features_df = df[config.feature_columns]
    target_series = df[config.target_column]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(features_df)
    scaled_target = target_scaler.fit_transform(target_series.values.reshape(-1, 1))
    
    # 스케일러 저장 (예측 시 필요하므로 저장 로직 유지)
    joblib.dump(feature_scaler, os.path.join(model_folder_name(config), 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(model_folder_name(config), 'target_scaler.pkl'))
    
    # 3. 시퀀스 데이터 생성
    X, y, dates = [], [], []
    for i in range(len(scaled_features) - config.sequence_length):
        X.append(scaled_features[i:i + config.sequence_length])
        y.append(scaled_target[i + config.sequence_length])
        dates.append(df['Date'].iloc[i + config.sequence_length])
    
    X, y = np.array(X), np.array(y)
    dates = pd.Series(dates).reset_index(drop=True)

    # 4. 학습/테스트 데이터 분리
    try:
        split_index = dates[dates >= config.test_start_date].index[0]
    except IndexError:
        # 테스트 시작 날짜가 데이터에 없는 경우, 마지막 20%를 테스트 데이터로 사용
        split_index = int(len(X) * 0.8) 
        print(f"경고: 테스트 시작 날짜 '{config.test_start_date}'를 찾을 수 없습니다. 데이터의 80% 지점에서 분할합니다.")

    test_dates = dates[split_index:]
    
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 5. PyTorch 텐서로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    print("--- 데이터 전처리 완료 ---\n")

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, target_scaler, test_dates


# -- 3. PyTorch Dataset 클래스 정의 (입력 텐서를 그대로 사용) --
class ExchangeRateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X # 이미 텐서이므로 변환 필요 없음
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -- 4. LSTM 모델 구조 정의 (기존과 동일) --
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 초기 은닉 상태는 지정하지 않아도 자동으로 0으로 초기화됨 (더 간결한 코드)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# -- 5. 모델 학습 함수 (개선) --
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    print("--- 모델 학습 시작 ---")
    model.train()
    
    # ⭐ 개선점: AMP(Automatic Mixed Precision)를 위한 GradScaler 추가
    scaler = torch.amp.GradScaler(enabled=(device=='cuda'))

    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{num_epochs:03d}", leave=False)
        
        for X_batch, y_batch in batch_iterator:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # ⭐ 개선점: autocast 컨텍스트 내에서 순전파 실행
            with torch.amp.autocast(enabled=(device=='cuda')):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            # ⭐ 개선점: Scaler를 사용하여 역전파
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batch_iterator.set_postfix(loss=f"{loss.item():.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        if ((epoch + 1) % 10 == 0):
            print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | Average Loss: {avg_epoch_loss:.6f}")

    print("--- 모델 학습 완료 ---\n")


# -- 6. 모델 평가 함수 (기존과 거의 동일) --
def evaluate_model(model, test_loader, target_scaler, device):
    print("--- 모델 평가 시작 ---")
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            outputs = target_scaler.inverse_transform(outputs.cpu().numpy())
            y_batch = target_scaler.inverse_transform(y_batch.cpu().numpy())
            
            predictions.extend(outputs.flatten().tolist())
            actuals.extend(y_batch.flatten().tolist())
            
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    print(f"테스트 데이터 MAE (평균 절대 오차): {mae:.2f} 원")
    print("--- 모델 평가 완료 ---\n")
    return predictions, actuals

def save_model(model, config):
    """
    모델의 가중치(state_dict)와 설정(config)을 JSON 파일로 함께 저장합니다.
    """ 
    folder_path = model_folder_name(config)
    model_path = os.path.join(folder_path, 'best_model.pth')

    torch.save(model.state_dict(), model_path)
    print(f"✅ 학습된 모델이 '{model_path}' 경로에 저장되었습니다.")

    config_path = os.path.join(folder_path, 'config.json')
    config_dict = {key: getattr(config, key) for key in dir(config) if not key.startswith('__') and not callable(getattr(config, key))}
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 모델 설정이 '{config_path}' 경로에 저장되었습니다.\n")

def model_folder_name(config):
    folder_name = (
        f"seq_{config.sequence_length}-hidden_{config.hidden_size}-"
        f"layers_{config.num_layers}-batch_{config.batch_size}-"
        f"date_{config.train_start_date.replace('-', '')}"
    )
    folder_path = os.path.join('models', folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# -- 7. 메인 실행 블록 (개선) --
if __name__ == '__main__':
    start_time = time.time() # 전체 실행 시간 측정 시작
    
    config = Config()
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"사용 장치: {device.upper()}")

    X_train, y_train, X_test, y_test, target_scaler, test_dates = load_and_preprocess_data(config)

    train_dataset = ExchangeRateDataset(X_train, y_train)
    test_dataset = ExchangeRateDataset(X_test, y_test)
    
    # ⭐ 개선점: DataLoader 최적화 옵션 적용
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True # GPU 사용 시 데이터 전송 속도 향상
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    model = LSTMModel(config.input_size, config.hidden_size, config.num_layers, config.output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    train_model(model, train_loader, criterion, optimizer, config.num_epochs, device)
    save_model(model, config)
    
    predictions, actuals = evaluate_model(model, test_loader, target_scaler, device)
    
    end_time = time.time() # 실행 시간 측정 종료
    print(f"총 실행 시간: {end_time - start_time:.2f}초")

    # 시각화 (기존과 동일)
    plt.figure(figsize=(15, 8))
    plt.plot(test_dates, actuals, label='Actual', color='blue')
    plt.plot(test_dates, predictions, label='Predicted', color='red', linestyle='--')
    plt.title('Exchange Rate Prediction vs Actual', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('KRW / USD', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()