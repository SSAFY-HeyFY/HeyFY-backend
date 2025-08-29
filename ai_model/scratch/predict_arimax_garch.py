# predict_arimax_garch.py
# pip install pandas numpy pmdarima arch joblib

import os, json, argparse, warnings
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import joblib

warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message=".*force_all_finite.*renamed to 'ensure_all_finite'.")

def future_business_days(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = last_date + BDay(1)
    return pd.date_range(start=start, periods=horizon, freq=BDay())

def load_fx_from_csv(
    path: str,
    date_col: str = "Date",
    us_col="Inv_Close",
    kr_col="ECOS_Close",
) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'. got: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    def pick(cands):
        if isinstance(cands, str): cands = (cands,)
        for c in cands:
            if c in df.columns: return c
        raise ValueError(f"Column not found among candidates: {cands}")

    df = df[[date_col, us_col, kr_col]].dropna()
    df = df.drop_duplicates(subset=[date_col], keep="last").sort_values(date_col).set_index(date_col)

    us = pd.to_numeric(df[us_col], errors="coerce").astype(float).rename("US").sort_index()
    kr = pd.to_numeric(df[kr_col], errors="coerce").astype(float).rename("KR").sort_index()

    # ✅ freq를 강제하지 않는다 (경고는 나중에 억제하거나, 그냥 무시)
    return us, kr

def load_models(model_dir: str):
    arimax = joblib.load(os.path.join(model_dir, "arimax.joblib"))
    garch  = joblib.load(os.path.join(model_dir, "garch.joblib"))
    with open(os.path.join(model_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return arimax, garch, meta

def predict_simulate(
    csv_path: str,
    model_dir: str,
    horizon: int = 5,
    n_paths: int = 500,
    date_col: str = "Date",
    us_col: str = "Inv_Close",
    kr_col: str = "ECOS_Close",
    business_days: bool = True,
    rng_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    arimax_fit, garch_fit, meta = load_models(model_dir)

    us, kr = load_fx_from_csv(csv_path, date_col=date_col, us_col=us_col, kr_col=kr_col)
    # 최신 구간 use_days만 사용 (학습과 동일 창)
    use_days = int(meta.get("use_days", 120))
    df = pd.concat([us.rename("US"), kr.rename("KR")], axis=1).dropna().sort_index().iloc[-use_days:]

    # 로그/수익률
    log_kr = np.log(df["KR"])
    log_us = np.log(df["US"])
    r_us = log_us.diff().fillna(0.0).values

    # 표준화 파라미터(학습 시 저장) 적용
    exog_mean = float(meta["exog_mean"])
    exog_std  = float(meta["exog_std"])
    resid_std = float(meta["resid_std"])
    us_hist_z = (r_us - exog_mean) / (exog_std + 1e-12)
    if np.std(us_hist_z) < 1e-8:
        us_hist_z = us_hist_z + rng.normal(0.0, 1e-4, size=us_hist_z.shape)

    # 날짜 인덱스
    last_date = df.index[-1]
    future_idx = future_business_days(last_date, horizon) if business_days \
                 else pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # GARCH H-step 분산 예측
    garch_fc = garch_fit.forecast(horizon=horizon)
    step_var = np.asarray(garch_fc.variance.values[-1]).reshape(-1) / (100.0 ** 2)
    if step_var.shape[0] != horizon:
        step_var = np.repeat(step_var.mean(), horizon)
    step_std = np.sqrt(step_var)

    # 시뮬레이션
    last_log_price = log_kr.iloc[-1]
    paths = np.zeros((n_paths, horizon))
    clip_sigma = 3.0 * max(resid_std, 1e-6)

    for i in range(n_paths):
        us_future_z = rng.choice(us_hist_z, size=horizon, replace=True).reshape(-1, 1)
        mean_r = arimax_fit.predict(n_periods=horizon, X=us_future_z)
        mean_r = np.clip(mean_r, -clip_sigma, clip_sigma)  # 안정화
        noise = rng.normal(0.0, step_std, size=horizon)
        log_path = last_log_price + np.cumsum(mean_r + noise)
        paths[i, :] = np.exp(log_path)

    median_path = np.median(paths, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    out = pd.DataFrame({
        "date": future_idx,
        "yhat": median_path,
        "yhat_p10": p10,
        "yhat_p90": p90,
        "sample_path": paths[0],
    })
    return out

if __name__ == "__main__":
    import pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/train/only_US_KOR_20100104_20250812_simple.csv")
    p.add_argument("--model_dir", default="models/fx_arimax_garch")
    p.add_argument("--h", type=int, default=5)
    p.add_argument("--date_col", default="Date")
    p.add_argument("--us_col", default="Inv_Close")
    p.add_argument("--kr_col", default="ECOS_Close")
    args = p.parse_args()

    df_out = predict_simulate(
        csv_path=args.csv,
        model_dir=args.model_dir,
        horizon=args.h,
        date_col=args.date_col,
        us_col=args.us_col,
        kr_col=args.kr_col,
        business_days=True,
    )
    print(df_out)
