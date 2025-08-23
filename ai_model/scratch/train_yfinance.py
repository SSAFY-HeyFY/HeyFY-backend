import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# model.py가 필요합니다. build_model 함수는 입력/출력 차원을 인자로 받아야 합니다.
from model import build_model 

# --- 모델 및 학습 설정 ---
WINDOW_SIZE = 60
PREDICTION_DAYS = 5
MODEL_TYPE = 'LSTM'
EPOCHS = 100
BATCH_SIZE = 32
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1
## [변경] 새로운 데이터 파일 경로로 수정
DATA_FILE_PATH = 'data/yfinance_KRW_data_with_indicators_20100325_20250816.xlsx'
MODEL_DIR = 'models'
MODEL_EXT = '.ckpt'

## [추가] 입출력으로 사용할 Feature 목록 정의
INPUT_FEATURES = ['Close', 'High', 'Low', 'Open', 'SMA_5', 'SMA_20', 'SMA_60', 'RSI_14']
TARGET_FEATURES = ['Close', 'High', 'Low', 'Open']

# --- 공통 유틸 ---
## [변경] 다변량 데이터를 처리하도록 create_dataset 함수 수정
def create_dataset(data, target_data, window_size, prediction_days):
    """
    다변량 시계열 데이터를 학습용 (X, y)로 변환
    data: 입력 Feature로 스케일된 (N, num_input_features)
    target_data: 출력 Feature로 스케일된 (N, num_target_features)
    """
    X, y = [], []
    for i in range(len(data) - window_size - prediction_days + 1):
        X.append(data[i:(i + window_size)])
        y.append(target_data[i + window_size : i + window_size + prediction_days])
    
    X = np.array(X)  # (Samples, Window, InputFeatures)
    y = np.array(y)  # (Samples, PredDays, TargetFeatures)
    
    # Loss 계산을 위해 y를 (Samples, PredDays * TargetFeatures) 형태로 펼침
    y = y.reshape(y.shape[0], -1) 
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
# 기존 코드와 동일 (변경 없음)
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
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
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            loss = criterion(preds, yb)
            running += loss.item() * Xb.size(0)
    return running / len(loader.dataset)

# --- 메인 ---
if __name__ == "__main__":
    # 1. 데이터 로드 및 준비
    print("1. 데이터 로드를 시작합니다...")
    df = pd.read_excel(DATA_FILE_PATH, index_col=0, parse_dates=True)
    
    ## [변경] 정의된 입력 Feature만 선택
    df_features = df[INPUT_FEATURES]
    print("선택된 입력 Features:")
    print(df_features.head())

    # 2. 데이터 스케일링 (0~1)
    ## [변경] 다변량 데이터 전체에 대해 하나의 스케일러를 적용
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)
    print("데이터 스케일링 완료. Shape:", scaled_data.shape)
    
    # 스케일링된 데이터를 다시 DataFrame으로 만들어 컬럼 정보 유지
    scaled_df = pd.DataFrame(scaled_data, columns=INPUT_FEATURES, index=df_features.index)

    # 3. 학습 데이터셋 생성 (Windowing)
    ## [변경] 입력 데이터와 타겟 데이터를 분리하여 create_dataset에 전달
    input_data_for_windowing = scaled_df[INPUT_FEATURES].values
    target_data_for_windowing = scaled_df[TARGET_FEATURES].values
    
    X, y = create_dataset(input_data_for_windowing, target_data_for_windowing, WINDOW_SIZE, PREDICTION_DAYS)
    print(f"학습 데이터셋 생성 완료. X shape: {X.shape}, y shape: {y.shape}")

    # 4. 훈련 / 테스트 분리 (기존과 동일)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, shuffle=False
    )
    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    
    # 데이터로더 (기존과 동일)
    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds  = TimeSeriesDataset(X_test,  y_test)
    n_train = len(train_ds)
    n_val = int(VALIDATION_SPLIT * n_train)
    subtrain_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train - n_val, n_val])
    train_loader = DataLoader(subtrain_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 모델 빌드
    ## [변경] 모델의 입출력 차원을 다변량에 맞게 수정
    num_input_features = len(INPUT_FEATURES)
    num_target_features = len(TARGET_FEATURES)
    
    # 새로운 input_shape 정의: (윈도우 크기, 입력 Feature 개수)
    input_shape = (WINDOW_SIZE, num_input_features) 
    
    # 최종 출력 뉴런 수 계산: (예측할 기간 * 예측할 Feature 개수)
    output_size = PREDICTION_DAYS * num_target_features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build_model 함수를 그대로 사용하되, 인자를 새로운 shape에 맞게 전달
    # PREDICTION_DAYS 위치에 최종 출력 뉴런 수(output_size)를 전달합니다.
    model = build_model(MODEL_TYPE, input_shape, output_size).to(device)
    print("✅ 모델이 성공적으로 빌드되었습니다.")
    print(model)

    # 6. 모델 저장 경로 및 설정 파일 준비
    # os.makedirs(MODEL_DIR, exist_ok=True)
    # model_basename = f"{MODEL_TYPE}_multi_W{WINDOW_SIZE}_P{PREDICTION_DAYS}" ## [변경] 파일명에 multi 추가
    # model_filename = os.path.join(MODEL_DIR, f"{model_basename}_best_model{MODEL_EXT}")
    # config_filename = os.path.join(MODEL_DIR, f"{model_basename}.config.json")
    # scaler_filename = os.path.join(MODEL_DIR, f"{model_basename}_scaler.pkl")
    model_basename = f"{MODEL_TYPE}_diff_W{WINDOW_SIZE}_P{PREDICTION_DAYS}" ## [변경] 파일명에 diff 추가
    model_dir_name = f"{MODEL_DIR}/{model_basename}"
    os.makedirs(f"{model_dir_name}", exist_ok=True)
    #model_basename = f"{MODEL_TYPE}_diff_W{WINDOW_SIZE}_P{PREDICTION_DAYS}" ## [변경] 파일명에 diff 추가
    model_filename = f"{model_dir_name}/{model_basename}_best_model{MODEL_EXT}"
    config_filename = f"{model_dir_name}/{model_basename}.config.json"
    scaler_filename = f"{model_dir_name}/{model_basename}_scaler.pkl" ## [변경] scaler.pkl 경로 명시적 정의

    # (학습 루프 및 CONFIG 설정은 기존 코드와 거의 동일하므로 생략... 일부만 표시)
    # ...
    # (학습 루프 시작 전)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val = float('inf')
    patience = 10
    bad_epochs = 0
    history = {'loss': [], 'val_loss': []}
    
    # CONFIG 설정 (필요 시 수정)
    CONFIG = {
        "data": {
            "file_path": DATA_FILE_PATH,
            "input_features": INPUT_FEATURES,
            "target_features": TARGET_FEATURES,
            "window_size": WINDOW_SIZE,
            "prediction_days": PREDICTION_DAYS,
        },
        "model": {
            "type": MODEL_TYPE,
            "input_dim": num_input_features,
            "output_dim": output_size,
        },
        "training": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam",
            "learning_rate": 1e-3, "criterion": "MSELoss",
        },
        "artifacts": {
            "model_dir": MODEL_DIR,
            "model_name": os.path.basename(model_filename),
            "config_path": config_filename,
            "scaler_path": scaler_filename
        }
    }
    
    # =================================================================
    # ## 7. 모델 학습 (생략되었던 전체 학습 루프)
    # =================================================================
    print(f"\n7. 모델 학습을 시작합니다... (Device: {device})")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 체크포인트 & 얼리스탑
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            
            # 체크포인트 저장 (모델 + config)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': CONFIG
            }, model_filename)

            # 별도 JSON 설정 파일 저장
            with open(config_filename, "w", encoding="utf-8") as f:
                json.dump(CONFIG, f, ensure_ascii=False, indent=4)
            
            # 스케일러 저장
            joblib.dump(scaler, scaler_filename)
            
            print(f"   ↳ Best model, config, scaler updated and saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break
    
    print("\n✅ 모델 학습이 완료되었습니다.")

    # 9. 테스트 데이터로 예측 및 성능 평가 (루프 종료 후)
    # 베스트 모델 로드
    ckpt = torch.load(model_filename, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("\n✅ 최적 모델을 로드하여 테스트를 진행합니다.")
    
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        preds_scaled_flat = model(X_test_t).cpu().numpy()

    # 스케일 복원을 위한 준비
    # 1. 예측 결과를 (Samples, PredDays, TargetFeatures) 형태로 복원
    predictions_scaled = preds_scaled_flat.reshape(-1, PREDICTION_DAYS, num_target_features)
    # 2. 실제값도 동일한 형태로 복원
    y_test_scaled = y_test.reshape(-1, PREDICTION_DAYS, num_target_features)
    
    ## [변경] 다변량 스케일 복원 로직
    # Scaler는 8개 Feature에 대해 학습되었으므로, 복원 시에도 8개 Feature 구조가 필요합니다.
    # 예측된 4개 Feature를 원래 위치에 넣고 나머지는 0으로 채운 뒤 inverse_transform을 수행합니다.
    
    # 원본 Feature 순서에서 Target Feature들의 인덱스를 찾음
    target_indices = [INPUT_FEATURES.index(col) for col in TARGET_FEATURES]
    
    def inverse_transform_multivariate(scaled_values, scaler, target_indices, original_num_features):
        # scaled_values shape: (Samples, PredDays, TargetFeatures)
        num_samples = scaled_values.shape[0]
        num_pred_days = scaled_values.shape[1]
        
        # 전체 Feature 수만큼 0으로 채워진 임시 배열 생성
        temp_array_flat = np.zeros((num_samples * num_pred_days, original_num_features))
        
        # 예측된 값을 올바른 위치(target_indices)에 삽입
        scaled_values_flat = scaled_values.reshape(-1, len(target_indices))
        temp_array_flat[:, target_indices] = scaled_values_flat
        
        # 전체 스케일 복원
        inversed_flat = scaler.inverse_transform(temp_array_flat)
        
        # 복원된 값들 중에서 Target Feature에 해당하는 값만 추출
        inversed_targets_flat = inversed_flat[:, target_indices]
        
        # 최종 형태로 복원: (Samples, PredDays, TargetFeatures)
        return inversed_targets_flat.reshape(num_samples, num_pred_days, len(target_indices))

    predictions = inverse_transform_multivariate(predictions_scaled, scaler, target_indices, num_input_features)
    y_test_actual = inverse_transform_multivariate(y_test_scaled, scaler, target_indices, num_input_features)

    # 최종 RMSE 계산 (Close 가격 기준)
    rmse_close = np.sqrt(mean_squared_error(y_test_actual[:, :, 0], predictions[:, :, 0]))
    print(f"\n복원된 실제 가격 기준 최종 RMSE (Close): {rmse_close:.4f}")

    print("\n--- 예측 결과 샘플 비교 (첫 번째 테스트 데이터) ---")
    for i in range(PREDICTION_DAYS):
        print(f"\n+{i+1}일 후 예측:")
        pred_dict = {col: f"{predictions[0, i, j]:.2f}" for j, col in enumerate(TARGET_FEATURES)}
        actual_dict = {col: f"{y_test_actual[0, i, j]:.2f}" for j, col in enumerate(TARGET_FEATURES)}
        print(f"  예측값: {pred_dict}")
        print(f"  실제값: {actual_dict}")