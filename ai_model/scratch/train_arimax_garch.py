# train_arimax_garch.py
# pip install pandas numpy pmdarima arch joblib

import os, json, argparse, warnings
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from arch import arch_model
from pandas.tseries.offsets import BDay
import joblib

warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message=".*force_all_finite.*renamed to 'ensure_all_finite'.")

# ---------------- Utils ----------------
def future_business_days(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = last_date + BDay(1)
    return pd.date_range(start=start, periods=horizon, freq=BDay())

def _pick_column(df: pd.DataFrame, candidates) -> str:
    if isinstance(candidates, str):
        candidates = (candidates,)
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Column not found among candidates: {candidates}")

def load_fx_from_csv(
    path: str,
    date_col: str = "Date",
    us_col_candidates=("Inv_Close", "US_Close", "US", "USMarket", "US_FX"),
    kr_col_candidates=("ECOS_Close", "KR_Close", "KR", "KRW"),
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

    us_col = pick(us_col_candidates)
    kr_col = pick(kr_col_candidates)

    df = df[[date_col, us_col, kr_col]].dropna()
    df = df.drop_duplicates(subset=[date_col], keep="last").sort_values(date_col).set_index(date_col)

    us = pd.to_numeric(df[us_col], errors="coerce").astype(float).rename("US").sort_index()
    kr = pd.to_numeric(df[kr_col], errors="coerce").astype(float).rename("KR").sort_index()

    # ✅ freq를 강제하지 않는다 (경고는 나중에 억제하거나, 그냥 무시)
    return us, kr


# ---------------- Train & Save ----------------
def train_and_save(
    csv_path: str,
    model_dir: str,
    date_col: str = "Date",
    us_col=("Inv_Close",),
    kr_col=("ECOS_Close",),
    use_days: int = 120,
    random_seed: int = 42,
):
    os.makedirs(model_dir, exist_ok=True)
    us, kr = load_fx_from_csv(csv_path, date_col=date_col, us_col_candidates=us_col, kr_col_candidates=kr_col)

    df = pd.concat([us.rename("US"), kr.rename("KR")], axis=1).dropna().sort_index()
    if len(df) < max(60, use_days):
        raise ValueError(f"Not enough data: {len(df)}. Need >= {max(60,use_days)} days.")
    df = df.iloc[-use_days:]  # recent window

    # Log & returns
    log_us = np.log(df["US"])
    log_kr = np.log(df["KR"])
    r_us = log_us.diff().fillna(0.0)
    r_kr = log_kr.diff().fillna(0.0)

    # Standardize exogenous (US returns)
    exog_mean = float(r_us.mean())
    exog_std  = float(r_us.std() + 1e-12)
    exog_z = ((r_us - exog_mean) / exog_std).values.reshape(-1, 1)

    # ARIMAX on returns (target=r_kr, d=0)
    arimax = auto_arima(
        r_kr, exogenous=exog_z,
        d=0, seasonal=False, stepwise=True, suppress_warnings=True,
        max_p=3, max_q=3, max_d=0, error_action="ignore",
    )
    arimax_fit = arimax.fit(r_kr, exogenous=exog_z)

    # Residuals (returns space)
    ins_pred = pd.Series(arimax_fit.predict_in_sample(exogenous=exog_z), index=r_kr.index)
    resid = (r_kr - ins_pred).dropna()
    resid_std = float(resid.std())

    # GARCH on residuals (×100 scale for stability)
    garch = arch_model(resid * 100.0, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    garch_fit = garch.fit(disp="off")

    # Save models
    joblib.dump(arimax_fit, os.path.join(model_dir, "arimax.joblib"))
    joblib.dump(garch_fit,  os.path.join(model_dir, "garch.joblib"))

    # Save metadata
    meta = {
        "date_col": date_col,
        "us_col": us.name or "US",
        "kr_col": kr.name or "KR",
        "use_days": use_days,
        "random_seed": random_seed,
        "exog_mean": exog_mean,
        "exog_std": exog_std,
        "resid_std": resid_std,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved ARIMAX+GARCH to: {model_dir}")
    print(f"  exog_mean={exog_mean:.6g}, exog_std={exog_std:.6g}, resid_std={resid_std:.6g}")
    print(f"  train_range: {df.index.min().date()} ~ {df.index.max().date()} (N={len(df)})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/train/only_US_KOR_20100104_20250812_simple.csv")
    p.add_argument("--model_dir", default="models/fx_arimax_garch")
    p.add_argument("--date_col", default="Date")
    p.add_argument("--us_col", default="Inv_Close")
    p.add_argument("--kr_col", default="ECOS_Close")
    p.add_argument("--use_days", type=int, default=120)
    args = p.parse_args()

    train_and_save(
        csv_path=args.csv,
        model_dir=args.model_dir,
        date_col=args.date_col,
        us_col=args.us_col,
        kr_col=args.kr_col,
        use_days=args.use_days,
    )
