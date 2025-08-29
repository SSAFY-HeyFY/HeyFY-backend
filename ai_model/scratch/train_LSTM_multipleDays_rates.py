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

# [변경] 수익률 변환 로직이 추가된 데이터 로더
def load_and_preprocess_data(args):
    """
    [수정]
    - pct_change() 적용 후 dropna()를 먼저 수행하여 데이터 불일치 문제를 해결합니다.
    - 모든 데이터(피처, 타겟, 원본 가격)가 dropna() 이후의 정제된 DataFrame에서 생성되도록 수정합니다.
    """
    if args.data_path.endswith('.csv'):
        df = pd.read_csv(args.data_path)
    else:
        df = pd.read_excel(args.data_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # [핵심 수정] 원본 가격을 먼저 복사해두고, 수익률 계산과 dropna를 먼저 수행합니다.
    original_target_prices = df[args.target_column].copy()
    df['return'] = df[args.target_column].pct_change() # 수익률을 'return'이라는 새 컬럼에 저장
    df = df.dropna().reset_index(drop=True)
    
    # dropna로 인해 줄어든 original_target_prices도 동일하게 인덱스를 맞춰줍니다.
    original_target_prices = original_target_prices.iloc[1:].reset_index(drop=True)
    
    # [핵심 수정] 이제 모든 피처와 타겟은 정렬이 완료된 df에서 가져옵니다.
    features_df = df[args.feature_columns]
    target_series = df['return'] # 타겟은 'return' 컬럼을 사용

    # 스케일러 설정
    if args.scaler == 'standard':
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler() # 이 스케일러는 '수익률'에 대해 학습(fit)됩니다.
    else:
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(features_df)
    scaled_target = target_scaler.fit_transform(target_series.values.reshape(-1, 1))
    
    # 스케일러 저장
    folder_path = model_folder_name(args)
    joblib.dump(feature_scaler, os.path.join(folder_path, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(folder_path, 'target_scaler.pkl'))
    
    X, y, dates, base_prices = [], [], [], []
    for i in range(len(scaled_features) - args.sequence_length - args.prediction_horizon + 1):
        X.append(scaled_features[i : i + args.sequence_length])
        y.append(scaled_target[i + args.sequence_length : i + args.sequence_length + args.prediction_horizon].squeeze())
        dates.append(df['Date'].iloc[i + args.sequence_length])
        
        # 기준 가격은 original_target_prices (원본 가격)에서 가져옵니다.
        base_price_idx = i + args.sequence_length - 1
        base_prices.append(original_target_prices.iloc[base_price_idx])
    
    X, y = np.array(X), np.array(y)
    dates = pd.Series(dates).reset_index(drop=True)
    base_prices = np.array(base_prices)

    # --- 데이터 분할 (이하 로직은 동일) ---
    test_split_index = dates[dates < args.test_start_date].index[-1] + 1
    
    X_rem, X_test = X[:test_split_index], X[test_split_index:]
    y_rem, y_test = y[:test_split_index], y[test_split_index:]
    dates_rem, test_dates = dates[:test_split_index], dates[test_split_index:]
    base_prices_rem, test_base_prices = base_prices[:test_split_index], base_prices[test_split_index:]

    val_split_index = int(len(X_rem) * 0.9)
    
    X_train, X_val = X_rem[:val_split_index], X_rem[val_split_index:]
    y_train, y_val = y_rem[:val_split_index], y_rem[val_split_index:]
    train_dates, val_dates = dates_rem[:val_split_index], dates_rem[val_split_index:]

    print("--- 데이터 분할 모니터링 ---")
    print(f"전체 원본 데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    if not train_dates.empty:
        print(f"학습 데이터 기간:   {train_dates.iloc[0].strftime('%Y-%m-%d')} ~ {train_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 수: {len(X_train)})")
    if not val_dates.empty:
        print(f"검증 데이터 기간:   {val_dates.iloc[0].strftime('%Y-%m-%d')} ~ {val_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 수: {len(X_val)})")
    if not test_dates.empty:
        print(f"테스트 데이터 기간: {test_dates.iloc[0].strftime('%Y-%m-%d')} ~ {test_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 수: {len(X_test)})")
    print("--------------------------\n")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, test_dates, test_base_prices

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

# [변경] 수익률을 가격으로 복원하는 로직이 추가된 평가 함수
def evaluate_model(model, test_loader, target_scaler, device, test_base_prices):
    """
    [변경]
    - test_base_prices를 인자로 받아 수익률 예측값을 실제 가격 예측값으로 변환합니다.
    - 실제값(y_batch)도 수익률이므로, 동일하게 가격으로 변환하여 MAE를 계산합니다.
    """
    print("--- 모델 평가 시작 ---")
    model.eval()
    predictions_price, actuals_price = [], []
    
    batch_offset = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            # 1. 모델은 '스케일링된 수익률'을 예측
            outputs_scaled_return = model(X_batch).cpu().numpy()
            actuals_scaled_return = y_batch.numpy()

            # 2. '스케일링된 수익률'을 '실제 수익률'로 복원 (inverse_transform)
            outputs_return = target_scaler.inverse_transform(outputs_scaled_return)
            actuals_return = target_scaler.inverse_transform(actuals_scaled_return)

            # 3. '실제 수익률'을 '실제 가격'으로 변환
            # 현재 배치의 base price들을 가져옴
            current_base_prices = test_base_prices[batch_offset : batch_offset + len(X_batch)]
            
            # 예측값과 실제값 모두 가격으로 변환
            predicted_prices_batch = returns_to_prices(current_base_prices, outputs_return)
            actual_prices_batch = returns_to_prices(current_base_prices, actuals_return)
            
            predictions_price.extend(predicted_prices_batch)
            actuals_price.extend(actual_prices_batch)
            
            batch_offset += len(X_batch)
            
    predictions = np.array(predictions_price)
    actuals = np.array(actuals_price)
    
    mae = np.mean(np.abs(predictions - actuals))
    print(f"테스트 데이터 전체 MAE (평균 절대 오차): {mae:.2f} 원")
    for i in range(predictions.shape[1]):
        mae_day = np.mean(np.abs(predictions[:, i] - actuals[:, i]))
        print(f"  - {i+1}일 후 예측 MAE: {mae_day:.2f} 원")
    print("--- 모델 평가 완료 ---\n")
    return predictions, actuals

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
        args.data_path = 'data/train/train_final_with_onehot_20100104_20250812_all.csv'

    args.input_size = len(args.feature_columns)
    args.output_size = args.prediction_horizon
    args.target_column = 'target' 
    args.test_start_date = '2025-01-01'
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # [변경] load_and_preprocess_data로부터 test_base_prices를 추가로 받음
    X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, test_dates, test_base_prices = load_and_preprocess_data(args)

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
    
    model = train_model(args, model, train_loader, val_loader, criterion, optimizer, device)
    
    save_model_config(args)
    
    if len(X_test) > 0:
        # [변경] evaluate_model에 test_base_prices를 전달
        predictions, actuals = evaluate_model(model, test_loader, target_scaler, device, test_base_prices)
        plot_test_results(args, test_dates, predictions, actuals)
    else:
        print("테스트 데이터가 없어 평가 및 시각화를 건너뜁니다.")