import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from model_before import TimeSeriesModel, build_model_from_config

WINDOW_SIZE = 30
PREDICTION_DAYS = 5
MODEL_TYPE = 'LSTM'   # 'LSTM' | 'GRU' | 'StackedLSTM'
DATA_FILE_PATH = 'data/한국은행_20100101_20250812_자른거_filled.xlsx'
MODEL_DIR = 'models'
CKPT_PATH = f"{MODEL_DIR}/{MODEL_TYPE}_W{WINDOW_SIZE}_P{PREDICTION_DAYS}_best_model.pt"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"


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

    scaler = load_scaler_or_fit(prices)
    scaled = scaler.transform(prices.reshape(-1, 1)).astype(np.float32)

    if len(scaled) < WINDOW_SIZE:
        raise ValueError("데이터 길이가 WINDOW_SIZE보다 짧아서 추론 불가합니다.")
    last_window = scaled[-WINDOW_SIZE:]                   # (W, 1)
    x_infer = last_window.reshape(1, WINDOW_SIZE, 1)      # (1, W, 1)

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

    past_dates = dates[-WINDOW_SIZE:]
    past_actual = prices[-WINDOW_SIZE:]
    inferred = pd.infer_freq(dates)
    if inferred is None:
        inferred = 'D'
    future_dates = pd.date_range(start=past_dates[-1], periods=PREDICTION_DAYS + 1, freq=inferred)[1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(past_dates, past_actual, label='Actual (Past Window)', linewidth=2)
    ax.plot(future_dates, pred, label=f'Forecast (+{PREDICTION_DAYS}d)', linewidth=2)
    ax.plot([past_dates[-1], future_dates[0]], [past_actual[-1], pred[0]],
            linestyle='--', linewidth=1)
    ax.axvline(past_dates[-1], linestyle=':', linewidth=1, color='gray')

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title(f'{MODEL_TYPE} Forecast (W={WINDOW_SIZE}, P={PREDICTION_DAYS})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    print(f"\n예측 {PREDICTION_DAYS}일 환율:")
    for i, (d, v) in enumerate(zip(future_dates, pred), 1):
        print(f"  {d.date()} ( +{i}일 ): {v:.4f}")

if __name__ == "__main__":
    main()