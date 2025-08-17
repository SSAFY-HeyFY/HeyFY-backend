# inference.py (1:1 날짜 매칭 버전)
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===== 학습 시와 동일한 하이퍼파라미터 =====
WINDOW_SIZE = 30
PREDICTION_DAYS = 5
MODEL_TYPE = 'LSTM'   # 'LSTM' | 'GRU' | 'StackedLSTM'
DATA_FILE_PATH = 'data/한국은행_20100101_20250812_자른거_filled.xlsx'
MODEL_DIR = 'models'
MODEL_NAME = f'{MODEL_TYPE}_diff_W{WINDOW_SIZE}_P{PREDICTION_DAYS}'
CKPT_PATH = f"{MODEL_DIR}/{MODEL_NAME}/{MODEL_NAME}_best_model.ckpt"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"   # 있으면 사용

# ===== 학습 때 쓰던 모델 정의 그대로 =====
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
            self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
            self.rnn2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
            last_hidden_size = 32
        else:
            raise ValueError("지원하지 않는 모델 타입입니다.")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(last_hidden_size, prediction_days)

    def forward(self, x):
        if self.model_type in ['LSTM', 'GRU']:
            out, _ = self.rnn(x)
        else:
            out1, _ = self.rnn1(x)
            out, _  = self.rnn2(out1)
        last = out[:, -1, :]
        last = self.dropout(last)
        y = self.fc(last)
        return y

def load_scaler_or_fit(series_1d):
    if os.path.exists(SCALER_PATH):
        print(f"scaler 로드: {SCALER_PATH}")
        return joblib.load(SCALER_PATH)
    print("scaler.pkl이 없어 CSV 전체 열로 MinMaxScaler를 새로 fit 합니다.")
    scaler = MinMaxScaler()
    scaler.fit(series_1d.reshape(-1, 1))
    return scaler

def inverse_2d(arr_2d, scaler):
    flat = arr_2d.reshape(-1, 1)
    inv  = scaler.inverse_transform(flat)
    return inv.reshape(arr_2d.shape)

def main():
    # 1) 데이터 로드
    df = pd.read_excel(DATA_FILE_PATH, index_col=0, parse_dates=True)
    prices = df['DATA_VALUE'].values.astype(np.float32)
    dates = df.index

    # 2) 스케일러 준비
    scaler = load_scaler_or_fit(prices)
    scaled = scaler.transform(prices.reshape(-1, 1)).astype(np.float32)

    # 3) 입력 윈도우를 P만큼 앞쪽으로 당겨서, 마지막 P일을 "정답 비교 구간"으로 남긴다
    need = WINDOW_SIZE + PREDICTION_DAYS
    if len(scaled) < need:
        raise ValueError(f"데이터가 부족합니다. 최소 {need} 포인트 필요 (현재 {len(scaled)}).")

    start_idx = len(scaled) - need
    window_scaled = scaled[start_idx : start_idx + WINDOW_SIZE]        # (W, 1)
    x_infer = window_scaled.reshape(1, WINDOW_SIZE, 1)                 # (1, W, 1)

    # 실제 비교 구간(정답): 마지막 P일
    target_actual = prices[start_idx + WINDOW_SIZE : start_idx + WINDOW_SIZE + PREDICTION_DAYS]   # (P,)
    target_dates  = dates[start_idx + WINDOW_SIZE : start_idx + WINDOW_SIZE + PREDICTION_DAYS]

    # 시각화용 과거 구간(입력 윈도우)
    past_dates  = dates[start_idx : start_idx + WINDOW_SIZE]
    past_actual = prices[start_idx : start_idx + WINDOW_SIZE]

    # 4) 모델 로드 & 추론
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeSeriesModel(MODEL_TYPE, input_size=1, hidden_sizes=(64, 32),
                            prediction_days=PREDICTION_DAYS, dropout=0.2).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_infer).to(device)
        pred_scaled = model(x_t).cpu().numpy()            # (1, P)
    pred = inverse_2d(pred_scaled, scaler).flatten()      # (P,)

    # 5) 플롯: 입력 윈도우(실제) + 정답구간(실제) + 예측(같은 날짜에 1:1 매칭)
    fig, ax = plt.subplots(figsize=(12, 5))

    # 입력 윈도우 실제
    ax.plot(past_dates, past_actual, label='Actual (Input Window)', linewidth=2)

    # 정답 구간 실제
    ax.plot(target_dates, target_actual, label='Actual (Target P days)', linewidth=2)

    # 예측 (같은 target_dates에 1:1로 올림)
    ax.plot(target_dates, pred, label=f'Forecast (+{PREDICTION_DAYS}d)', linewidth=2)

    # 입력 마지막 날과 타깃 첫 날 사이 점선(경계 시각화)a
    ax.plot([past_dates[-1], target_dates[0]], [past_actual[-1], target_actual[0]],
            linestyle='--', linewidth=1)

    # 경계선
    ax.axvline(past_dates[-1], linestyle=':', linewidth=1, color='gray')

    # 날짜 포맷
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title(f'{MODEL_TYPE} Forecast (W={WINDOW_SIZE}, P={PREDICTION_DAYS}) — 1:1 date match')
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # 6) 콘솔: 날짜별 예측 vs 실제
    print(f"\n예측 {PREDICTION_DAYS}일 환율 (날짜 1:1 매칭):")
    for d, p_hat, p_true in zip(target_dates, pred, target_actual):
        print(f"  {d.date()} | 예측: {p_hat:.4f} | 실제: {p_true:.4f}")

if __name__ == "__main__":
    main()
