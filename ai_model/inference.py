import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime
import joblib  # 모델/스케일러 저장용

# ======================= 설정 =======================
FILE_PATH = "data/한국은행_20100101_20250812_자른거.xlsx"
DATE_COL  = "TIME"                   # 엑셀의 날짜열 이름
TARGET_COL = "DATA_VALUE"            # 이번엔 단일 열만 사용

SEQ_LEN   = 60
EPOCHS    = 50
BATCH     = 32
LR        = 1e-3
HIDDEN    = 128
LAYERS    = 2
DROPOUT   = 0.2
VAL_RATIO = 0.2                      # train/test = 0.8/0.2

def load_artifacts(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    scaler = joblib.load(run_dir / "scaler_returns.joblib")

    model = LSTMRegressor(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_size=config["output_size"],
    )
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, scaler, config

# =================== 데이터 로드/전처리 ===================
def load_from_excel(path: str, value_col: str, date_col: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"엑셀 파일을 찾을 수 없습니다: {p.resolve()}")
    df = pd.read_excel(p)

    if date_col not in df.columns:
        raise ValueError(f"엑셀에 날짜 컬럼 '{date_col}'이 없습니다. 실제 열 이름을 지정해 주세요.\n열 목록: {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"필수 열 '{value_col}'이 없습니다. 실제 열 이름을 지정해 주세요.\n열 목록: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).set_index(date_col)
    # 단일 열만 float으로 캐스팅
    out = df[[value_col]].astype(float).dropna()
    return out  # index=Datetime, column=[DATA_VALUE]


def make_sequences(arr: np.ndarray, seq_len: int):
    """arr: (N, 1) -> X:(N-seq_len, seq_len, 1), y:(N-seq_len, 1)  (다음 시점의 r 예측)"""
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len])
    return np.array(X), np.array(y)

class SeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ======================= 모델 =======================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)        # (B,T,H)
        out = out[:, -1, :]          # (B,H)
        out = self.fc(out)           # (B,1)  -> r̂_{t+1}
        return out


def predict(model, X_test: np.ndarray, device):
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_test).float().to(device)
        pred = model(xb).cpu().numpy()        # (N_test, 1)  -> r̂ (scaled)
    return pred

# -------- 아티팩트 저장/로드(원하면 사용) --------
def save_artifacts(run_dir, model, scaler, config, example_input=None, save_torchscript=False):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    joblib.dump(scaler, run_dir / "scaler_returns.joblib")

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if save_torchscript:
        model.eval()
        if example_input is None:
            example_input = torch.randn(1, config["seq_len"], config["input_size"])
        scripted = torch.jit.trace(model, example_input.to(next(model.parameters()).device))
        scripted.save(str(run_dir / "model_ts.pt"))



    torch.manual_seed(42); np.random.seed(42)

    # 1) 단일 열(DATA_VALUE) 로드
    df      = load_from_excel(FILE_PATH, TARGET_COL, DATE_COL)  # index=Datetime, col=[DATA_VALUE]
    dates   = df.index.values
    prices  = df.values                      # (N,1) — 절대값(가격/지표)

    # 2) 로그수익률 r_t = log(P_t / P_{t-1})  (시간축: dates[1:])
    rets    = np.log(prices[1:] / prices[:-1])   # (N-1,1)
    r_dates = df.index[1:]                       # returns의 날짜축(=다음 날)

    # 3) 시퀀스 생성(수익률 기준)
    X_all_raw, y_all_raw = make_sequences(rets, SEQ_LEN)  # (M,T,1), (M,1)
    M = len(X_all_raw)
    split = int(M * (1 - VAL_RATIO))  # 순서 유지 분할

    X_tr_raw, y_tr_raw = X_all_raw[:split], y_all_raw[:split]
    X_te_raw, y_te_raw = X_all_raw[split:], y_all_raw[split:]

    # 4) 스케일링(수익률에 대해 train만 fit)
    scaler = StandardScaler().fit(
        np.vstack([X_tr_raw.reshape(-1, X_tr_raw.shape[-1]), y_tr_raw])
    )
    X_train = scaler.transform(X_tr_raw.reshape(-1, X_tr_raw.shape[-1])).reshape(X_tr_raw.shape)
    y_train = scaler.transform(y_tr_raw)
    X_test  = scaler.transform(X_te_raw.reshape(-1, X_te_raw.shape[-1])).reshape(X_te_raw.shape)
    y_test  = scaler.transform(y_te_raw)




# 어딘가 다른 스크립트/세션에서
load_dir = "runs/usdkrw_lstm_C_20250813_171413"  # 실제 저장 폴더

device = "cuda" if torch.cuda.is_available() else "cpu"
model, scaler, cfg = load_artifacts(load_dir, device=device)

torch.manual_seed(42); np.random.seed(42)

# 1) 단일 열(DATA_VALUE) 로드
df      = load_from_excel(FILE_PATH, TARGET_COL, DATE_COL)  # index=Datetime, col=[DATA_VALUE]
dates   = df.index.values
prices  = df.values                      # (N,1) — 절대값(가격/지표)

# 2) 로그수익률 r_t = log(P_t / P_{t-1})  (시간축: dates[1:])
rets    = np.log(prices[1:] / prices[:-1])   # (N-1,1)
r_dates = df.index[1:]         

# === 5영업일 롤링 예측(미래 실제값 미사용) ===
N_DAYS = 5
BACKTEST_DAYS = 5

# 마지막 SEQ_LEN일 수익률 시퀀스(스케일)
last_seq_ret = rets[-SEQ_LEN:]                                  # (SEQ_LEN,1)
last_seq_s   = scaler.transform(last_seq_ret)                    # (SEQ_LEN,1)

# 시작 가격 = 가장 최근 실제 가격
start_idx = len(df) - BACKTEST_DAYS - 1
prev_price = float(df[TARGET_COL].iloc[start_idx])
prev_price = float(prices[-1, 0])
start_date = df.index[-1]

pred_prices_future = []
pred_returns_future = []

model.eval()
with torch.no_grad():
    cur_seq = last_seq_s.copy()
    for _ in range(N_DAYS):
        xb = torch.from_numpy(cur_seq.reshape(1, SEQ_LEN, 1)).float().to(device)
        pred_ret_s_step = model(xb).cpu().numpy()[0]             # (1,) scaled returns
        pred_ret_step   = scaler.inverse_transform(pred_ret_s_step.reshape(1, -1))[0, 0]

        # 가격 복원: P_{t+1} = P_t * exp(r̂_{t+1})
        next_price = prev_price * float(np.exp(pred_ret_step))
        pred_prices_future.append(next_price)
        pred_returns_future.append(pred_ret_step)

        # 창 업데이트(스케일 공간): 맨 앞 제거 + 새 예측 수익률 추가
        cur_seq = np.vstack([cur_seq[1:], pred_ret_s_step.reshape(1, 1)])
        prev_price = next_price

pred_prices_future = np.array(pred_prices_future).reshape(-1, 1)  # (5,1)
pred_returns_future = np.array(pred_returns_future).reshape(-1, 1)

# 예측 날짜(영업일)
future_dates = pd.bdate_range(start=start_date + pd.Timedelta(days=1), periods=N_DAYS)

# 보기 좋게 DataFrame
forecast_df = pd.DataFrame(pred_prices_future, index=future_dates, columns=[TARGET_COL])
print("\n=== 5-Day Forecast (rolling) ===")
print(forecast_df.round(4))

# 최근 히스토리 + 예측 라인 플롯
hist_days = 60
plt.figure(figsize=(10, 4))
plt.plot(df.index[-hist_days:], df[TARGET_COL].values[-hist_days:], label=f"Actual {TARGET_COL}")
plt.plot(future_dates, forecast_df[TARGET_COL].values, label=f"Forecast {TARGET_COL}")
plt.title(f"5-Day Rolling Forecast — {TARGET_COL}")
plt.xlabel("Date"); plt.ylabel(TARGET_COL)
plt.legend(); plt.tight_layout(); plt.show()


# === Backtest: 마지막 5영업일을 '미래'로 가정해 순차 예측하고 실제와 비교 ===
BACKTEST_DAYS = 5

# 1) 입력 창: 마지막 5일 직전까지의 SEQ_LEN 길이 수익률(정보 누수 방지)
# rets 길이는 (N-1), prices 길이는 (N)
# 예측 대상: r_{N-5}..r_{N-1} → 입력은 r_{N-6}까지의 최근 SEQ_LEN
bt_last_seq_ret = rets[-(SEQ_LEN + BACKTEST_DAYS) : -BACKTEST_DAYS]   # (SEQ_LEN, 1)
if bt_last_seq_ret.shape[0] != SEQ_LEN:
    raise ValueError("데이터가 부족합니다. SEQ_LEN + 5 보다 많은 데이터가 필요해요.")

bt_last_seq_s = scaler.transform(bt_last_seq_ret)                     # (SEQ_LEN, 1)

# 2) 시작 가격: 예측 구간 바로 '직전' 날짜의 실제 가격 P_{N-5의 직전}=P_{N-6}
start_idx = len(df) - BACKTEST_DAYS - 1
prev_price = float(df[TARGET_COL].iloc[start_idx])
prev_price = float(prices[-(BACKTEST_DAYS + 1), 0])

bt_pred_prices = []
bt_pred_returns = []

model.eval()
with torch.no_grad():
    cur_seq = bt_last_seq_s.copy()
    for _ in range(BACKTEST_DAYS):
        xb = torch.from_numpy(cur_seq.reshape(1, SEQ_LEN, 1)).float().to(device)
        pred_ret_s = model(xb).cpu().numpy()[0]                          # (1,) scaled
        pred_ret   = scaler.inverse_transform(pred_ret_s.reshape(1, -1))[0, 0]

        next_price = prev_price * float(np.exp(pred_ret))
        bt_pred_prices.append(next_price)
        bt_pred_returns.append(pred_ret)

        # 창 업데이트(스케일 공간)
        cur_seq = np.vstack([cur_seq[1:], pred_ret_s.reshape(1, 1)])
        prev_price = next_price

bt_pred_prices = np.array(bt_pred_prices)          # (5,)
bt_dates       = df.index[-BACKTEST_DAYS:]         # 마지막 5영업일 실제 날짜
bt_true_prices = df[TARGET_COL].values[-BACKTEST_DAYS:]  # (5,)

# 3) 간단 지표
bt_mae = mean_absolute_error(bt_true_prices, bt_pred_prices)
bt_mse = mean_squared_error(bt_true_prices, bt_pred_prices)
bt_r2  = r2_score(bt_true_prices, bt_pred_prices)
print("\n[Backtest: Last 5 Business Days]")
print(f"{TARGET_COL} | MAE: {bt_mae:.6f} | MSE: {bt_mse:.6f} | R²: {bt_r2:.4f}")

# 4) 플롯(마지막 60일 실제 + 마지막 5일 예측 오버레이)
plt.figure(figsize=(10,4))
hist_days = 60
plt.plot(df.index[-hist_days:], df[TARGET_COL].values[-hist_days:], label=f"Actual {TARGET_COL}")
plt.plot(bt_dates, bt_true_prices, label="Actual (Last 5d)")
plt.plot(bt_dates, bt_pred_prices, label="Predicted (Last 5d)")
plt.title(f"Backtest on Last 5 Business Days — {TARGET_COL}")
plt.xlabel("Date"); plt.ylabel(TARGET_COL); plt.legend(); plt.tight_layout(); plt.show()

