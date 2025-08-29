# pip install pandas numpy pmdarima arch openpyxl

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from arch import arch_model
from pandas.tseries.offsets import BDay
from typing import Iterable, Tuple

# ---------- 유틸 ----------
def future_business_days(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = last_date + BDay(1)
    return pd.date_range(start=start, periods=horizon, freq=BDay())

def _as_candidates(x) -> Tuple[str, ...]:
    if isinstance(x, str):
        return (x,)
    if isinstance(x, Iterable):
        return tuple(x)
    raise ValueError("컬럼 후보는 str 또는 Iterable[str]")

def _pick_column(df: pd.DataFrame, candidates) -> str:
    for c in _as_candidates(candidates):
        if c in df.columns:
            return c
    raise ValueError(f"해당 후보 열을 찾을 수 없습니다: {candidates}. 실제 컬럼: {list(df.columns)}")

# ---------- 1) CSV 로더 ----------
def load_fx_from_csv(
    path: str,
    date_col: str = "Date",
    us_col_candidates=("Inv_Close", "US_Close", "US", "USMarket", "US_FX"),
    kr_col_candidates=("ECOS_Close", "KR_Close", "KR", "KRW"),
) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"CSV에 '{date_col}' 컬럼이 없습니다. 실제 컬럼: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    us_col = _pick_column(df, us_col_candidates)
    kr_col = _pick_column(df, kr_col_candidates)

    df = df[[date_col, us_col, kr_col]].dropna()
    df = df.drop_duplicates(subset=[date_col], keep="last").sort_values(date_col).set_index(date_col)

    us = pd.to_numeric(df[us_col], errors="coerce").astype(float).rename("US")
    kr = pd.to_numeric(df[kr_col], errors="coerce").astype(float).rename("KR")
    return us, kr

# ---------- 2) ARIMAX(수익률) + GARCH + 시뮬 ----------
def simulate_arimax_garch(
    us: pd.Series,
    kr: pd.Series,
    horizon: int = 5,
    n_paths: int = 500,
    use_days: int = 120,
    business_days: bool = True,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    평균: KR의 '로그수익률'(d=1) 을 ARIMAX로 (exog = US 로그수익률, z-score 표준화)
    변동성: 잔차에 GARCH(1,1)
    미래: US 로그수익률 히스토리 부트스트랩 → ARIMAX 평균경로 + GARCH 잡음 → 누적합(수익률) → 가격 복원
    """
    rng = np.random.default_rng(rng_seed)

    # 공통 구간 정합 + 최근 구간만 사용
    df = pd.concat([us.rename("US"), kr.rename("KR")], axis=1).dropna().sort_index().iloc[-use_days:]
    if len(df) < max(60, use_days):
        raise ValueError(f"학습 데이터 부족: {len(df)}일. 최소 60일, 권장 {use_days}일.")

    # 로그가격 & 수익률
    log_us = np.log(df["US"])
    log_kr = np.log(df["KR"])
    r_us = log_us.diff().fillna(0.0)  # US 로그수익률
    r_kr = log_kr.diff().fillna(0.0)  # KR 로그수익률 (평균모형의 타깃)

    # 외생변수 표준화(Z-score) → 회귀 안정화
    exog = (r_us - r_us.mean()) / (r_us.std() + 1e-12)
    exog = exog.values.reshape(-1, 1)

    # 평균모형: ARIMAX on r_kr (d=0, 이미 차분된 수익률을 직접 모델링)
    arimax = auto_arima(
        r_kr, exogenous=exog,
        d=0,           # 중요: 타깃이 이미 차분된 '수익률'이므로 d=0
        seasonal=False, stepwise=True, suppress_warnings=True,
        max_p=3, max_q=3, max_d=0, error_action="ignore",
    )
    arimax_fit = arimax.fit(r_kr, exogenous=exog)

    # 인샘플 예측 & 잔차 (수익률 단위)
    ins_pred = pd.Series(arimax_fit.predict_in_sample(exogenous=exog), index=r_kr.index)
    resid = (r_kr - ins_pred).dropna()

    # 변동성: GARCH(1,1) on resid (×100 스케일)
    garch = arch_model(resid * 100.0, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    garch_fit = garch.fit(disp="off")

    # 미래 날짜 인덱스
    last_date = df.index[-1]
    future_idx = future_business_days(last_date, horizon) if business_days \
                 else pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # 미국장 미래 수익률 경로 부트스트랩(표준화 후 역변환 불필요: exog는 표준화 값 사용)
    us_hist = r_us.values
    if np.std(us_hist) < 1e-8:
        us_hist = us_hist + rng.normal(0.0, 1e-4, size=us_hist.shape)
    us_hist_z = (us_hist - np.mean(us_hist)) / (np.std(us_hist) + 1e-12)

    # GARCH H-step 조건부분산
    garch_fc = garch_fit.forecast(horizon=horizon)
    step_var = np.asarray(garch_fc.variance.values[-1]).reshape(-1) / (100.0 ** 2)
    if step_var.shape[0] != horizon:
        step_var = np.repeat(step_var.mean(), horizon)
    step_std = np.sqrt(step_var)

    # 수익률 경로 시뮬 + 가격 복원
    paths = np.zeros((n_paths, horizon))
    last_log_price = log_kr.iloc[-1]
    r_kr_std_cap = max(resid.std(), 1e-6)   # 평균경로의 비정상적 튀김 방지용 클리핑 기준

    for i in range(n_paths):
        # 미래 exog(z-score) H개 샘플
        us_future_z = rng.choice(us_hist_z, size=horizon, replace=True).reshape(-1, 1)
        # 평균 경로: '수익률' 단위 (필요시 클리핑으로 폭주 방지)
        mean_r = arimax_fit.predict(n_periods=horizon, X=us_future_z)
        mean_r = np.clip(mean_r, -3*r_kr_std_cap, 3*r_kr_std_cap)
        # GARCH 잡음
        noise = rng.normal(0.0, step_std, size=horizon)
        # 로그수익률 누적 → 로그가격
        log_path = last_log_price + np.cumsum(mean_r + noise)
        paths[i, :] = np.exp(log_path)

    price_paths = paths
    median_path = np.median(price_paths, axis=0)
    p10 = np.percentile(price_paths, 10, axis=0)
    p90 = np.percentile(price_paths, 90, axis=0)

    out = pd.DataFrame({
        "date": future_idx,
        "yhat": median_path,
        "yhat_p10": p10,
        "yhat_p90": p90,
        "sample_path": price_paths[0],
    })
    return out

# ---------- 3) 실행 ----------
if __name__ == "__main__":
    us, kr = load_fx_from_csv(
        "data/train/only_US_KOR_20100104_20250812_simple.csv",
        date_col="Date",
        us_col_candidates="Inv_Close",     # 문자열/튜플/리스트 모두 OK
        kr_col_candidates="ECOS_Close",
    )
    out = simulate_arimax_garch(us, kr, horizon=7, n_paths=500, use_days=120, business_days=True)
    print(out.head())
