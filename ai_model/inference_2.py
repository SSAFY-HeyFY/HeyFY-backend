# inference.py
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
CKPT_PATH = f"{MODEL_DIR}/{MODEL_TYPE}_W{WINDOW_SIZE}_P{PREDICTION_DAYS}_best_model.pt"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"   # 있으면 사용

# ===== 학습 때 쓰던 모델 정의 그대로 복사 =====
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
    """
    1) scaler.pkl이 있으면 로드
    2) 없으면 series를 0~1로 fit 해서 반환
    """
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
    df = pd.read_excel(DATA_FILE_PATH, index_col=0, parse_dates=True)
    prices = df['DATA_VALUE'].values.astype(np.float32)
    dates = df.index

    # 스케일러 로드/적합
    scaler = load_scaler_or_fit(prices)
    scaled = scaler.transform(prices.reshape(-1, 1)).astype(np.float32)

    # 최근 윈도우 입력
    if len(scaled) < WINDOW_SIZE:
        raise ValueError("데이터 길이가 WINDOW_SIZE보다 짧아서 추론 불가합니다.")
    last_window = scaled[-WINDOW_SIZE:]                   # (W, 1)
    x_infer = last_window.reshape(1, WINDOW_SIZE, 1)      # (1, W, 1)

    # 모델 로드 및 예측
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

    # ==== 날짜 축 구성 ====
    # 1) 과거 구간: 최근 WINDOW_SIZE일 실제 데이터
    past_dates = dates[-WINDOW_SIZE:]
    past_actual = prices[-WINDOW_SIZE:]

    # 2) 미래 구간: 마지막 날짜 이후 PREDICTION_DAYS개의 날짜 생성
    #    데이터의 빈도 추정 (영업일 데이터라면 'B'로 추정될 수도 있음)
    inferred = pd.infer_freq(dates)
    if inferred is None:
        # 추정 실패 시 하루 단위로 증가
        inferred = 'D'
    future_dates = pd.date_range(start=past_dates[-1], periods=PREDICTION_DAYS + 1, freq=inferred)[1:]

    # ==== 시각화 ====
    fig, ax = plt.subplots(figsize=(12, 5))

    # 실제: 최근 WINDOW_SIZE일
    ax.plot(past_dates, past_actual, label='Actual (Past Window)', linewidth=2)

    # 예측: 미래 P일
    ax.plot(future_dates, pred, label=f'Forecast (+{PREDICTION_DAYS}d)', linewidth=2)

    # 마지막 실제 값과 예측의 연속성을 시각적으로 강조(점선으로 연결)
    ax.plot([past_dates[-1], future_dates[0]], [past_actual[-1], pred[0]],
            linestyle='--', linewidth=1)

    # 경계선 표시(세로선)
    ax.axvline(past_dates[-1], linestyle=':', linewidth=1, color='gray')

    # 날짜 포맷팅
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title(f'{MODEL_TYPE} Forecast (W={WINDOW_SIZE}, P={PREDICTION_DAYS})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()  # 날짜 라벨 기울임/정리
    plt.tight_layout()
    plt.show()

    # 콘솔 출력
    print(f"\n예측 {PREDICTION_DAYS}일 환율:")
    for i, (d, v) in enumerate(zip(future_dates, pred), 1):
        print(f"  {d.date()} ( +{i}일 ): {v:.4f}")

if __name__ == "__main__":
    main()