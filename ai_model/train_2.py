import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- 모델 및 학습 설정 ---
WINDOW_SIZE = 30          # 1. Window Size
PREDICTION_DAYS = 5       # 2. Prediction Days
MODEL_TYPE = 'LSTM'       # 3. 'LSTM', 'GRU', 'StackedLSTM'
EPOCHS = 100              # 4. 학습 관련 하이퍼파라미터
BATCH_SIZE = 32
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1
DATA_FILE_PATH = 'data/한국은행_20100101_20250812_자른거_filled.xlsx'
MODEL_DIR = 'models'
MODEL_EXT = '.pt'         # 파이토치 체크포인트 확장자

# --- 공통 유틸 ---

def create_dataset(data, window_size, prediction_days):
    """
    시계열 데이터를 학습용 (X, y)로 변환
    data: 스케일된 (N, 1)
    X: (samples, window_size, 1)
    y: (samples, prediction_days)  <- (중요) 학습 편의를 위해 2D로 정리
    """
    X, y = [], []
    for i in range(len(data) - window_size - prediction_days + 1):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size : i + window_size + prediction_days])
    X = np.array(X)                        # (S, W, 1)
    y = np.array(y)                        # (S, P, 1)
    y = y.reshape(y.shape[0], y.shape[1])  # (S, P)
    return X, y

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 모델 빌드 ---

class TimeSeriesModel(nn.Module):
    def __init__(self, model_type: str, input_size: int, hidden_sizes, prediction_days: int, dropout=0.2):
        super().__init__()
        self.model_type = model_type

        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
            last_hidden_size = 64
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=64, batch_first=True)
            last_hidden_size = 64
        elif model_type == 'StackedLSTM':
            # 2층 LSTM: 첫 층 64, 둘째 층 32
            self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
            self.rnn2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
            last_hidden_size = 32
        else:
            raise ValueError("지원하지 않는 모델 타입입니다. 'LSTM', 'GRU', 'StackedLSTM' 중에서 선택하세요.")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(last_hidden_size, prediction_days)

    def forward(self, x):
        # x: (B, T, F)
        if self.model_type in ['LSTM', 'GRU']:
            out, h = self.rnn(x)              # out: (B, T, H)
        else:
            out1, _ = self.rnn1(x)
            out, _  = self.rnn2(out1)

        # 마지막 타임스텝의 hidden
        last = out[:, -1, :]                  # (B, H)
        last = self.dropout(last)
        y = self.fc(last)                     # (B, PREDICTION_DAYS)
        return y

def build_model(model_type, input_shape, prediction_days):
    """
    지정된 타입의 PyTorch 모델을 생성
    input_shape: (WINDOW_SIZE, 1)
    """
    _, in_features = input_shape
    model = TimeSeriesModel(model_type=model_type,
                            input_size=in_features,
                            hidden_sizes=(64, 32),
                            prediction_days=prediction_days,
                            dropout=0.2)
    return model

# --- 학습/검증 루프 ---

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * Xb.size(0)
    return running / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            preds = model(Xb)
            loss = criterion(preds, yb)
            running += loss.item() * Xb.size(0)
    return running / len(loader.dataset)

# --- 메인 ---

if __name__ == "__main__":
    # 1. 데이터 로드 및 준비
    print("1. 데이터 로드를 시작합니다...")
    df = pd.read_excel(DATA_FILE_PATH, index_col=0, parse_dates=True)
    print(df)
    exchange_rate = df['DATA_VALUE'].values.reshape(-1, 1)

    # 2. 데이터 스케일링 (0~1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(exchange_rate)
    print("데이터 스케일링 완료.")

    # 3. 학습 데이터셋 생성 (Windowing)
    X, y = create_dataset(scaled_data, WINDOW_SIZE, PREDICTION_DAYS)  # X:(S,W,1), y:(S,P)
    print(f"학습 데이터셋 생성 완료. X shape: {X.shape}, y shape: {y.shape}")

    # 4. 훈련 / 테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, shuffle=False
    )
    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

    # 데이터로더
    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds  = TimeSeriesDataset(X_test,  y_test)

    # 검증 분할(훈련 내에서 validation split)
    n_train = len(train_ds)
    n_val = int(VALIDATION_SPLIT * n_train)
    n_subtrain = n_train - n_val
    subtrain_ds, val_ds = torch.utils.data.random_split(
        train_ds, [n_subtrain, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(subtrain_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,      batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,     batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 5. 모델 빌드
    input_shape = (WINDOW_SIZE, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(MODEL_TYPE, input_shape, PREDICTION_DAYS).to(device)
    print(model)

    # 6. 모델 저장 경로 & 콜백(에뮬레이션)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_filename = f"{MODEL_DIR}/{MODEL_TYPE}_W{WINDOW_SIZE}_P{PREDICTION_DAYS}_best_model{MODEL_EXT}"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float('inf')
    patience = 10
    bad_epochs = 0
    history = {'loss': [], 'val_loss': []}

    # 7. 모델 학습
    print("\n7. 모델 학습을 시작합니다...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch:03d} | loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        # 체크포인트 & 얼리스탑
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            bad_epochs = 0
            torch.save({'model_state_dict': model.state_dict()}, model_filename)
            print(f"  ↳ Best model updated and saved to {model_filename}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # 베스트 모델 로드
    ckpt = torch.load(model_filename, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\n✅ 모델 학습이 완료되었습니다.")
    print(f"최적의 모델이 '{model_filename}' 경로에 저장되었습니다.")

    # 8. 학습 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{MODEL_TYPE} Model Loss (W: {WINDOW_SIZE}, P: {PREDICTION_DAYS})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 9. 테스트 데이터로 예측 및 성능 평가
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\n테스트 데이터 손실(MSE): {test_loss:.6f}")

    # 예측
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        print("X_test_t: ", X_test_t)
        preds_scaled_t = model(X_test_t)             # (N_test, P)
        predictions_scaled = preds_scaled_t.cpu().numpy()

    # y_test는 이미 (N_test, P)
    y_test_scaled = y_test

    # 스케일 복원: (N, P) → (-1, 1) → inverse → (N, P)
    def inverse_2d(arr_2d, scaler):
        flat = arr_2d.reshape(-1, 1)
        inv  = scaler.inverse_transform(flat)
        return inv.reshape(arr_2d.shape)

    predictions = inverse_2d(predictions_scaled, scaler)
    y_test_actual = inverse_2d(y_test_scaled, scaler)

    print("\n--- 예측 결과 샘플 비교 ---")
    print(f"예측 {PREDICTION_DAYS}일 환율: {predictions[0].flatten()}")
    print(f"실제 {PREDICTION_DAYS}일 환율: {y_test_actual[0].flatten()}")
