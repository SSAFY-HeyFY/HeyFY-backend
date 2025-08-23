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
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from model import LSTMDirectMH
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
    parser.add_argument('--tag', type=str, default='return_predict', help='실험을 식별하기 위한 태그') # [변경] 기본 태그 수정
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
    parser.add_argument('--patience', type=int, default=10, help='Early stopping을 위한 patience')
    
    # jupyter notebook 환경에서는 아래 코드를 활성화하세요.
    # args = parser.parse_args([])
    # production 환경에서는 아래 코드를 활성화하세요.
    args = parser.parse_args()
    return args

def load_and_preprocess_data(args):
    """
    학습목표: gap-residual
      - gap_return_t     = Inv_Close_t / ECOS_Close_t - 1
      - next_return_t    = target_t   / ECOS_Close_t - 1   (target=다음날 ECOS)
      - y_t (residual)   = next_return_t - gap_return_t

    주의: gap 기반 목적은 horizon=1을 가정.
    """
    assert args.prediction_horizon == 1, "gap-residual 목적은 현재 horizon=1에서만 지원됩니다. --prediction_horizon 1 로 실행하세요."

    print(f"데이터셋 경로: {args.data_path}")

    # 1) 로드 & 정렬
    if args.data_path.endswith('.csv'):
        df = pd.read_csv(args.data_path)
    else:
        df = pd.read_excel(args.data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 2) 가격/갭/다음날 수익률 계산
    #    target 컬럼은 '다음날 ECOS_Close'로 들어있음 (파일 확인 완료)
    ecos = df['ECOS_Close'].astype(float).values
    inv  = df['Inv_Close'].astype(float).values
    nxt  = df[args.target_column].astype(float).values  # 다음날 ECOS

    gap_return   = (inv / ecos) - 1.0                  # 밤사이 미국 갭 (당일 새벽 5시 close 기준)
    next_return  = (nxt / ecos) - 1.0                  # 다음날 한국 매매기준율 수익률
    residual_ret = next_return - gap_return            # 우리가 학습할 타깃

    # 피처 구성 (네가 지정한 columns 그대로 사용)
    features_df = df[args.feature_columns].copy()

    # 3) 시퀀스 생성(비스케일) — horizon=1
    X, y, dates, base_prices, gap_arr = [], [], [], [], []
    N = len(features_df)
    T = args.sequence_length
    H = 1

    feat_np = features_df.values.astype(np.float32)

    for i in range(N - T - H + 1):
        X.append(feat_np[i:i+T])                   # (T, F)
        y.append(residual_ret[i+T])                # 스칼라(= 1-step residual return)
        dates.append(df['Date'].iloc[i+T])
        base_prices.append(ecos[i+T-1])            # 기준가: 당일 ECOS
        gap_arr.append(gap_return[i+T])            # 예측시 더해줄 밤사이 갭(다음날에 대응)

    X = np.array(X, dtype=np.float32)              # (M, T, F)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)   # (M, 1)
    dates = pd.Series(dates).reset_index(drop=True)
    base_prices = np.array(base_prices, dtype=np.float32)  # (M,)
    gap_arr = np.array(gap_arr, dtype=np.float32).reshape(-1, 1)  # (M,1)

    # 4) 날짜 split
    test_split_index = dates[dates < args.test_start_date].index[-1] + 1
    X_rem, X_test = X[:test_split_index], X[test_split_index:]
    y_rem, y_test = y[:test_split_index], y[test_split_index:]
    gap_rem, test_gap = gap_arr[:test_split_index], gap_arr[test_split_index:]
    dates_rem, test_dates = dates[:test_split_index], dates[test_split_index:]
    base_prices_rem, test_base_prices = base_prices[:test_split_index], base_prices[test_split_index:]

    val_split_index = int(len(X_rem) * 0.9)
    X_train, X_val = X_rem[:val_split_index], X_rem[val_split_index:]
    y_train, y_val = y_rem[:val_split_index], y_rem[val_split_index:]
    gap_train, gap_val = gap_rem[:val_split_index], gap_rem[val_split_index:]
    train_dates, val_dates = dates_rem[:val_split_index], dates_rem[val_split_index:]

    # 5) 스케일러 준비 → train으로만 fit
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    feature_scaler = StandardScaler() if args.scaler == 'standard' else MinMaxScaler()
    target_scaler  = StandardScaler()  # ★ 타깃은 항상 표준화 (수익률 분산 보존)

    feature_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    target_scaler.fit(y_train.reshape(-1, 1))

    # 6) 각 분할 transform (타깃은 residual만 스케일)
    def scale_X(X_):
        s = X_.reshape(-1, X_.shape[-1])
        s = feature_scaler.transform(s)
        return s.reshape(X_.shape)

    def scale_y(y_):
        s = target_scaler.transform(y_.reshape(-1, 1))
        return s.reshape(y_.shape)

    X_train, X_val, X_test = scale_X(X_train), scale_X(X_val), scale_X(X_test)
    y_train, y_val, y_test = scale_y(y_train), scale_y(y_val), scale_y(y_test)
    # gap은 "그 자체"가 물리량(이미 비스케일 수익률)이므로 스케일하지 않고 그대로 둔다.

    # 7) 스케일러 저장
    folder_path = model_folder_name(args)
    joblib.dump(feature_scaler, os.path.join(folder_path, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(folder_path, 'target_scaler.pkl'))

    # 8) 로그
    print("--- 데이터 분할 모니터링 ---")
    print(f"전체 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    if not train_dates.empty:
        print(f"학습: {train_dates.iloc[0].strftime('%Y-%m-%d')} ~ {train_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 {len(X_train)})")
    if not val_dates.empty:
        print(f"검증: {val_dates.iloc[0].strftime('%Y-%m-%d')} ~ {val_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 {len(X_val)})")
    if not test_dates.empty:
        print(f"테스트: {test_dates.iloc[0].strftime('%Y-%m-%d')} ~ {test_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 {len(X_test)})")
    print("--------------------------")
    print("y_train mean (scaled residual):", np.mean(y_train, axis=0).round(6))
    print("y_train std  (scaled residual):",  np.std(y_train,  axis=0).round(6))
    print()

    # 반환: test_gap(밤사이 갭)을 함께 넘겨서 평가시 복원에 사용
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, test_dates, test_base_prices, test_gap



# train_model 함수는 이전과 동일합니다.
def train_model(args, model, train_loader, val_loader, criterion, optimizer, device):
    print("--- 모델 학습 시작 ---")
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(model_folder_name(args), 'best_model.pth'))

    for epoch in range(args.num_epochs):
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
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    print("--- 모델 학습 완료 ---\n")
    model.load_state_dict(torch.load(early_stopping.path))
    return model

def evaluate_model(model, test_loader, target_scaler, device, test_base_prices, test_gap_returns):
    """
    model 출력: scaled residual return  (shape: (B, 1))
    평가 로직:
      residual_pred = inverse_transform(outputs_scaled)
      return_pred   = gap + residual_pred
      price_pred    = base * (1 + return_pred)
    """
    print("--- 모델 평가 시작 (gap-residual) ---")
    model.eval()
    predictions_price, actuals_price = [], []

    # 베이스라인도 함께 계산
    baseline_persist_prices, baseline_gapfill_prices = [], []

    batch_offset = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            # 1) 예측/실제 (scaled residual)
            outputs_scaled = model(X_batch).cpu().numpy()      # (B,1)
            actuals_scaled = y_batch.numpy()                   # (B,1)

            # 2) 역스케일링 (residual)
            B, H = outputs_scaled.shape[0], outputs_scaled.shape[1]
            residual_pred = target_scaler.inverse_transform(outputs_scaled.reshape(-1,1)).reshape(B,H)  # (B,1)
            residual_true = target_scaler.inverse_transform(actuals_scaled.reshape(-1,1)).reshape(B,H)

            # 3) 배치 기준가 & 갭
            base_batch = test_base_prices[batch_offset : batch_offset + B].reshape(-1,1)   # (B,1)
            gap_batch  = test_gap_returns[batch_offset : batch_offset + B].reshape(-1,1)   # (B,1)

            # 4) 수익률/가격 복원
            pred_return = gap_batch + residual_pred    # (B,1)
            true_return = gap_batch + residual_true    # (B,1)

            pred_price = base_batch * (1.0 + pred_return)
            true_price = base_batch * (1.0 + true_return)

            predictions_price.extend(pred_price.squeeze(1))
            actuals_price.extend(true_price.squeeze(1))

            # 베이스라인 1) 전날 그대로
            baseline_persist_prices.extend(base_batch.squeeze(1))
            # 베이스라인 2) 갭만 100% 반영
            baseline_gapfill_prices.extend((base_batch * (1.0 + gap_batch)).squeeze(1))

            batch_offset += B

    predictions = np.array(predictions_price)
    actuals     = np.array(actuals_price)
    baseline_persist = np.array(baseline_persist_prices)
    baseline_gapfill = np.array(baseline_gapfill_prices)

    # MAE
    mae_model    = np.mean(np.abs(predictions - actuals))
    mae_persist  = np.mean(np.abs(baseline_persist - actuals))
    mae_gapfill  = np.mean(np.abs(baseline_gapfill - actuals))

    print(f"모델 MAE: {mae_model:.2f} 원")
    print(f"베이스라인(전날 그대로) MAE: {mae_persist:.2f} 원")
    print(f"베이스라인(갭 100% 반영) MAE: {mae_gapfill:.2f} 원")
    print("--- 모델 평가 완료 ---\n")

    return predictions.reshape(-1,1), actuals.reshape(-1,1)


# [추가] 수익률을 가격으로 변환하는 헬퍼 함수
def returns_to_prices(base_prices, returns):
    """
    기준 가격과 수익률 시퀀스를 바탕으로 실제 가격 시퀀스를 계산합니다.
    Args:
        base_prices (np.array): (batch_size, ) 형태의 기준 가격 배열
        returns (np.array): (batch_size, prediction_horizon) 형태의 수익률 배열
    Returns:
        np.array: (batch_size, prediction_horizon) 형태의 실제 가격 배열
    """
    # (batch_size, 1) 형태로 만들어 브로드캐스팅 준비
    base_prices = base_prices.reshape(-1, 1)
    
    # (1 + 수익률) 누적곱 계산. cumprod는 누적곱을 계산하는 numpy 함수
    # 예: [r1, r2, r3] -> [1+r1, (1+r1)(1+r2), (1+r1)(1+r2)(1+r3)]
    cumulative_returns = np.cumprod(1 + returns, axis=1)
    
    # 기준 가격과 누적곱을 곱하여 최종 가격 계산
    prices = base_prices * cumulative_returns
    return prices


# save_model_config, model_folder_name, plot_test_results 함수는 이전과 동일합니다.
def save_model_config(args):
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
    plt.rc('font', family='Malgun Gothic') # 한글 폰트 설정
    plt.figure(figsize=(15, 8))
    plt.plot(test_dates, actuals[:, 0], label='실제 환율', color='blue', marker='o', markersize=4, linestyle='-')
    plt.plot(test_dates, predictions[:, 0], label='1일 후 예측 환율', color='red', linestyle='--', marker='x', markersize=4)
    
    # 마지막 예측 구간을 상세히 보여주기 위한 로직
    if len(test_dates) >= args.prediction_horizon:
        last_prediction_start_date = test_dates.iloc[-1]
        future_dates = pd.date_range(start=last_prediction_start_date, periods=args.prediction_horizon)
        
        # 마지막 예측에 사용된 실제값들을 표시
        plt.plot(future_dates, actuals[-1, :], 's', color='limegreen', markersize=6, label=f'마지막 {args.prediction_horizon}일 실제값')
        # 마지막 예측값들을 표시
        plt.plot(future_dates, predictions[-1, :], '^', color='orange', markersize=6, label=f'마지막 {args.prediction_horizon}일 예측값')

    plt.title(f'환율 예측 vs 실제값 ({args.prediction_horizon}일 예측)', fontsize=16)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('원/달러', fontsize=12)
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
    
    # 피처 및 기본 설정
    simple_features = ['Inv_Close', 'ECOS_Close', 'DXY_Close', 'US10Y_Close']
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
        args.data_path = 'data/train/train_final_with_onehot_20150104_20250812_simple.csv'

    args.input_size = len(args.feature_columns)
    args.output_size = args.prediction_horizon
    args.target_column = 'target' 
    args.test_start_date = '2025-01-01'
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, test_dates, test_base_prices, test_gap = load_and_preprocess_data(args)

    train_dataset = ExchangeRateDataset(X_train, y_train)
    val_dataset = ExchangeRateDataset(X_val, y_val)
    test_dataset = ExchangeRateDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = LSTMDirectMH(
        input_size=args.input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        output_size=args.output_size,
        dropout_prob=args.dropout_prob
    ).to(device)
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()  # = HuberLoss, 변동성 보존에 유리
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    model = train_model(args, model, train_loader, val_loader, criterion, optimizer, device)
    
    save_model_config(args)
    
    if len(X_test) > 0:
        # [변경] evaluate_model에 test_base_prices를 전달
        predictions, actuals = evaluate_model(model, test_loader, target_scaler, device, test_base_prices, test_gap)
        plot_test_results(args, test_dates, predictions, actuals)
    else:
        print("테스트 데이터가 없어 평가 및 시각화를 건너뜁니다.")