import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from model import build_model

# --- 모델 및 학습 설정 ---
WINDOW_SIZE = 60
PREDICTION_DAYS = 5
MODEL_TYPE = 'LSTM'
SCALER_TYPE = 'MinMaxScaler'
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1
DATA_FILE_PATH = 'data/한국은행_20200101_20250812_자른거_filled.xlsx'
MODEL_DIR = 'models'
MODEL_EXT = '.ckpt'

# --- 공통 유틸 ---
def create_dataset(data, window_size, prediction_days):
    """기존과 동일"""
    X, y = [], []
    for i in range(len(data) - window_size - prediction_days + 1):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size : i + window_size + prediction_days])
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(y.shape[0], y.shape[1])
    return X, y

class TimeSeriesDataset(Dataset):
    """기존과 동일"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 학습/검증 루프 ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    """기존과 동일"""
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
    """기존과 동일"""
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

## [추가] 2D 배열 스케일 복원 함수
def inverse_2d(arr_2d, scaler):
    flat = arr_2d.reshape(-1, 1)
    inv  = scaler.inverse_transform(flat)
    return inv.reshape(arr_2d.shape)

## [추가] 변화량 예측 결과를 실제 가격으로 복원하는 함수
def reconstruct_from_changes(last_known_prices, predicted_changes):
    """
    last_known_prices: (N_samples, 1) - 각 예측 시점 직전의 실제 가격
    predicted_changes: (N_samples, P_days) - 모델이 예측한 변화량
    """
    # 누적합을 통해 변화량을 가격 변동으로 변환
    cumulative_changes = np.cumsum(predicted_changes, axis=1)
    # 마지막 실제 가격에 가격 변동을 더하여 최종 예측 가격 계산
    reconstructed_prices = last_known_prices + cumulative_changes
    return reconstructed_prices


# --- 메인 ---

if __name__ == "__main__":
    # 1. 데이터 로드 및 준비
    print("1. 데이터 로드를 시작합니다...")
    df = pd.read_excel(DATA_FILE_PATH, index_col=0, parse_dates=True)
    exchange_rate = df['DATA_VALUE'].values.reshape(-1, 1)

    ## [추가] 데이터 차분 (Differencing)
    exchange_rate_diff = np.diff(exchange_rate, axis=0)
    print(f"데이터 차분 완료. 원본: {exchange_rate.shape}, 차분 후: {exchange_rate_diff.shape}")

    # 2. 데이터 스케일링 (0~1)
    ## [변경] 차분된 데이터를 기준으로 스케일러를 fit합니다.
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(exchange_rate_diff)
    print("데이터 스케일링 완료.")

    # 3. 학습 데이터셋 생성 (Windowing)
    ## [변경] 스케일링된 '변화량' 데이터로 데이터셋을 만듭니다.
    X, y = create_dataset(scaled_data, WINDOW_SIZE, PREDICTION_DAYS)
    print(f"학습 데이터셋 생성 완료. X shape: {X.shape}, y shape: {y.shape}")

    # 4. 훈련 / 테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, shuffle=False
    )
    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    
    # 데이터로더 (기존과 동일)
    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds  = TimeSeriesDataset(X_test,  y_test)
    n_train = len(train_ds)
    n_val = int(VALIDATION_SPLIT * n_train)
    n_subtrain = n_train - n_val
    subtrain_ds, val_ds = torch.utils.data.random_split(
        train_ds, [n_subtrain, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(subtrain_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,      batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,     batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 5. 모델 빌드 (기존과 동일)
    input_shape = (WINDOW_SIZE, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(MODEL_TYPE, input_shape, PREDICTION_DAYS).to(device)
    print(model)

    # 6. 모델 저장 경로 및 설정 파일 준비
    model_basename = f"{MODEL_TYPE}_diff_W{WINDOW_SIZE}_P{PREDICTION_DAYS}" ## [변경] 파일명에 diff 추가
    model_dir_name = f"{MODEL_DIR}/{model_basename}"
    os.makedirs(f"{model_dir_name}", exist_ok=True)
    #model_basename = f"{MODEL_TYPE}_diff_W{WINDOW_SIZE}_P{PREDICTION_DAYS}" ## [변경] 파일명에 diff 추가
    model_filename = f"{model_dir_name}/{model_basename}_best_model{MODEL_EXT}"
    config_filename = f"{model_dir_name}/{model_basename}.config.json"
    scaler_filename = f"{model_dir_name}/{model_basename}_scaler.pkl" ## [변경] scaler.pkl 경로 명시적 정의

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val = float('inf')
    patience = 10
    bad_epochs = 0
    history = {'loss': [], 'val_loss': []}

    # 6.5 설정 스냅샷 (기존과 거의 동일, 일부 키 추가)
    CONFIG = {
        "run": {"framework": "pytorch", "seed": 42, "device": str(device)},
        "data": {
            "file_path": DATA_FILE_PATH,
            "feature_name": "DATA_VALUE",
            "preprocessing": "differencing", ## [추가] 전처리 방식 명시
            "scaler": SCALER_TYPE,
            "window_size": WINDOW_SIZE,
            "prediction_days": PREDICTION_DAYS,
            "test_split": TEST_SPLIT,
            "validation_split": VALIDATION_SPLIT,
        },
        "model": {
             "type": MODEL_TYPE,
             "input_dim": 1, ## [변경] 명확한 파라미터명 사용
             "hidden_dim": 64,
             "num_layers": 2 if MODEL_TYPE == 'StackedLSTM' else 1,
             "output_dim": PREDICTION_DAYS
        },
        "training": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam",
            "learning_rate": LEARNING_RATE, "criterion": "MSELoss", "early_stopping_patience": patience
        },
        "artifacts": {
            "model_dir": MODEL_DIR, "model_name": os.path.basename(model_filename),
            "config_path": config_filename, "scaler_path": scaler_filename
        }
    }

    # 7. 모델 학습
    print("\n7. 모델 학습을 시작합니다...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch:03d} | loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            bad_epochs = 0
            torch.save({'model_state_dict': model.state_dict(), 'config': CONFIG}, model_filename)
            with open(config_filename, "w", encoding="utf-8") as f:
                json.dump(CONFIG, f, ensure_ascii=False, indent=2)
            ## [추가] 최적 모델과 함께 스케일러도 저장합니다.
            joblib.dump(scaler, scaler_filename)
            print(f"   ↳ Best model, config, scaler saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    
    # (학습 루프 후 나머지 코드는 기존과 유사하나, 예측 결과 복원 로직이 중요합니다.)
    # ...
    
    # 9. 테스트 데이터로 예측 및 성능 평가
    # 베스트 모델 로드
    ckpt = torch.load(model_filename, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("\n✅ 최적 모델을 로드하여 테스트를 진행합니다.")

    test_loss_on_changes = evaluate(model, test_loader, criterion, device)
    print(f"\n테스트 데이터에 대한 손실(MSE on scaled changes): {test_loss_on_changes:.6f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        preds_scaled_t = model(X_test_t)
        predictions_scaled = preds_scaled_t.cpu().numpy()

    # 스케일된 '변화량'을 실제 '변화량'으로 복원
    predicted_changes = inverse_2d(predictions_scaled, scaler)
    y_test_actual_changes = inverse_2d(y_test, scaler)

    ## [변경] '변화량'을 실제 '가격'으로 복원하는 로직
    # 테스트셋의 각 샘플이 시작되는 시점의 실제 가격이 필요합니다.
    # train/test 분리 시 shuffle=False이므로 인덱스 계산이 가능합니다.
    test_start_index = len(X_train) + WINDOW_SIZE
    
    # 각 test 샘플 window의 마지막 시점의 '실제 가격'을 가져옵니다.
    # 원본 데이터는 차분으로 길이가 1 줄었으므로 인덱스 +1을 해줍니다.
    last_known_prices_for_test = exchange_rate[test_start_index-1 : test_start_index-1 + len(X_test)]
    
    # 가격 복원
    predictions = reconstruct_from_changes(last_known_prices_for_test, predicted_changes)
    y_test_actual = reconstruct_from_changes(last_known_prices_for_test, y_test_actual_changes)
    
    # 복원된 '가격' 기준으로 최종 RMSE 계산
    final_rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"\n복원된 실제 가격 기준 최종 RMSE: {final_rmse:.4f}")

    print("\n--- 예측 결과 샘플 비교 (복원된 가격 기준) ---")
    print(f"예측 {PREDICTION_DAYS}일 환율: {predictions[0].flatten()}")
    print(f"실제 {PREDICTION_DAYS}일 환율: {y_test_actual[0].flatten()}")