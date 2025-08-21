import os
import joblib
import json
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class Config:
    data_path = 'data/train/train_final_with_onehot_20100104_20250811_with_diff_cutted.xlsx' 
    
    target_column = 'target'
    feature_columns = [
        'Inv_Close', 'Inv_Open', 'Inv_High', 'Inv_Low', 'Inv_Change(%)', 'ECOS_Close', 'DXY_Close', 'US10Y_Close',
        'is_Mon', 'is_Tue', 'is_Wed', 'is_Thu', 'is_Fri', 'diff'
    ]
    
    tag = 'simple'
    train_start_date = '2010-01-04'  # 학습 데이터 시작 날짜
    test_start_date = '2025-01-01' # 테스트 데이터 시작 날짜
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

def set_seed(seed):
    """
    재현성을 위해 모든 랜덤 시드를 고정하는 함수
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    
    # cuDNN 설정
    # 이 설정은 연산 속도를 느리게 할 수 있지만, 재현성을 보장합니다.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def load_and_preprocess_data(config):
    if config.data_path.endswith('.csv'):
        df = pd.read_csv(config.data_path)
    else:
        df = pd.read_excel(config.data_path)
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 설정된 피처만 선택
    features_df = df[config.feature_columns]
    target_series = df[config.target_column]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(features_df)
    scaled_target = target_scaler.fit_transform(target_series.values.reshape(-1, 1))
    
    joblib.dump(feature_scaler, os.path.join(model_folder_name(config), 'feature_scaler.pkl') )
    joblib.dump(target_scaler, os.path.join(model_folder_name(config), 'target_scaler.pkl') )
    
    X, y, dates = [], [], []
    for i in range(len(scaled_features) - config.sequence_length):
        X.append(scaled_features[i:i+config.sequence_length])
        y.append(scaled_target[i+config.sequence_length])
        dates.append(df['Date'].iloc[i+config.sequence_length])
    
    X, y = np.array(X), np.array(y)
    dates = pd.Series(dates).reset_index(drop=True)

    split_index = dates[dates >= config.test_start_date].index[0]
    
    train_dates = dates[:split_index]
    test_dates = dates[split_index:]
    
    print("--- 데이터 기간 모니터링 ---")
    print(f"전체 데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"학습 데이터 기간: {train_dates.iloc[0].strftime('%Y-%m-%d')} ~ {train_dates.iloc[-1].strftime('%Y-%m-%d')}")
    print(f"테스트 데이터 기간: {test_dates.iloc[0].strftime('%Y-%m-%d')} ~ {test_dates.iloc[-1].strftime('%Y-%m-%d')}")
    print("--------------------------\n")

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

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


# -- 4. LSTM 모델 구조 정의 (출력층 개선) --
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.2):
        """
        모델 초기화 함수
        Args:
            input_size (int): 입력 피처의 수
            hidden_size (int): LSTM 은닉 상태의 크기
            num_layers (int): LSTM 레이어의 수
            output_size (int): 최종 출력값의 크기
            dropout_prob (float): 드롭아웃 확률
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 첫 번째 선형 레이어 (차원 축소)
            nn.ReLU(),                                 # 비선형성을 추가하는 활성화 함수
            nn.Dropout(dropout_prob),                  # 과적합 방지를 위한 드롭아웃
            nn.Linear(hidden_size // 2, output_size)   # 최종 출력을 위한 두 번째 선형 레이어
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden_state = out[:, -1, :]
        
        final_output = self.fc(last_hidden_state)
        
        return final_output


# -- 5. 모델 학습 함수 --
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    print("--- 모델 학습 시작 ---")
    model.train()
    
    # 전체 에포크에 대한 진행률 표시줄 (바깥쪽 루프)
    for epoch in range(num_epochs):
        
        # desc: 진행률 표시줄의 제목, leave=False: 내부 루프 완료 후 표시줄 삭제
        batch_iterator = tqdm(train_loader, 
                              desc=f"Epoch {epoch+1:03d}/{num_epochs:03d}", 
                              leave=False)
        
        epoch_loss = 0
        
        for X_batch, y_batch in batch_iterator:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 순전파
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            batch_iterator.set_postfix(loss=f"{loss.item():.6f}")

        # 한 에포크의 학습이 끝나면 평균 Loss를 출력
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
    모델의 주요 하이퍼파라미터를 포함하는 폴더명에
    학습된 모델의 가중치(state_dict)를 저장합니다.
    """ 
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
    """
    모델의 주요 하이퍼파라미터를 기반으로 폴더 이름을 생성합니다.
    예: seq_120-hidden_128-layers_2-batch_16-date_20250102
    """
    folder_name = (
        f"seq_{config.sequence_length}"
        f"-hidden_{config.hidden_size}"
        f"-layers_{config.num_layers}"
        f"-batch_{config.batch_size}"
        f"-date_{config.train_start_date.replace('-', '')}"
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
    plt.figure(figsize=(15, 8))
    plt.plot(test_dates, actuals, label='Actual (ECOS Rate)', color='blue', marker='.')
    plt.plot(test_dates, predictions, label='Predicted Rate', color='red', linestyle='--', marker='.')
    plt.title('Exchange Rate Prediction vs Actual', fontsize=16)
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
    set_seed(42)

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
