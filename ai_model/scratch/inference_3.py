import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from model_before import TimeSeriesModel, build_model_from_config

MODEL_DIR = 'models'
CONFIG_PATH = f"{MODEL_DIR}/LSTM_W30_P5.config.json"


# 유틸 함수

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    # 필수 키 점검 (간단 체크)
    for key in ['data', 'model', 'artifacts']:
        if key not in cfg:
            raise KeyError(f"config에 '{key}' 섹션이 없습니다: {config_path}")
    return cfg

def load_scaler_or_fit(series_1d: np.ndarray, scaler_path: str | None) -> MinMaxScaler:
    """
    1) scaler_path가 존재하면 로드
    2) 없으면 series를 0~1로 fit 해서 반환
    """
    if scaler_path and os.path.exists(scaler_path):
        print(f"[scaler] 로드: {scaler_path}")
        return joblib.load(scaler_path)
    print("[scaler] 파일이 없어 데이터 전체로 MinMaxScaler를 새로 fit 합니다.")
    scaler = MinMaxScaler()
    scaler.fit(series_1d.reshape(-1, 1))
    return scaler

def inverse_2d(arr_2d: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    flat = arr_2d.reshape(-1, 1)
    inv  = scaler.inverse_transform(flat)
    return inv.reshape(arr_2d.shape)

# =========================
# 메인 추론 로직
# =========================
def main():
    cfg_path = CONFIG_PATH
    print(f"[config] 사용 파일: {cfg_path}")
    cfg = load_config(cfg_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    data_path = cfg['data']['file_path']
    feature   = cfg['data'].get('feature_name', 'DATA_VALUE')
    window    = int(cfg['data']['window_size'])
    pred_days = int(cfg['data']['prediction_days'])

    df = pd.read_excel(data_path, index_col=0, parse_dates=True)
    prices = df[feature].values.astype(np.float32)
    dates = df.index

    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    scaler = load_scaler_or_fit(prices, scaler_path)
    scaled = scaler.transform(prices.reshape(-1, 1)).astype(np.float32)

    if len(scaled) < window:
        raise ValueError(f"데이터 길이({len(scaled)})가 WINDOW_SIZE({window})보다 짧아서 추론 불가합니다.")
    last_window = scaled[-window:]                   # (W, 1)
    x_infer = last_window.reshape(1, window, 1)      # (1, W, 1)

    model, ckpt_path = build_model_from_config(cfg, device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트 파일이 없습니다: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    ckpt_cfg = ckpt.get('config')
    if ckpt_cfg and ckpt_cfg.get('model', {}).get('type') != cfg.get('model', {}).get('type'):
        print("[경고] 체크포인트에 내장된 config와 JSON의 모델 타입이 다릅니다. JSON 기준으로 진행합니다.")

    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_infer).to(device)
        pred_scaled = model(x_t).cpu().numpy()            # (1, P)
    pred = inverse_2d(pred_scaled, scaler).flatten()      # (P,)

    past_dates = dates[-window:]
    past_actual = prices[-window:]

    inferred = pd.infer_freq(dates)
    if inferred is None:
        inferred = 'D'
    future_dates = pd.date_range(start=past_dates[-1], periods=pred_days + 1, freq=inferred)[1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(past_dates, past_actual, label='Actual (Past Window)', linewidth=2)
    ax.plot(future_dates, pred, label=f'Forecast (+{pred_days}d)', linewidth=2)
    ax.plot([past_dates[-1], future_dates[0]], [past_actual[-1], pred[0]],
            linestyle='--', linewidth=1)
    ax.axvline(past_dates[-1], linestyle=':', linewidth=1)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title(f"{cfg['model']['type']} Forecast (W={window}, P={pred_days})")
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    print(f"\n예측 {pred_days}일 환율:")
    for i, (d, v) in enumerate(zip(future_dates, pred), 1):
        print(f"  {d.date()} ( +{i}일 ): {v:.4f}")

if __name__ == "__main__":
    main()
