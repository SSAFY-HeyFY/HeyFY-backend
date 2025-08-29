# fx_runner.py
# pip install pandas numpy pmdarima arch joblib matplotlib tqdm

import os, json, argparse, warnings
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from arch import arch_model
from pandas.tseries.offsets import BDay

# ----------------------------- QoL: warnings & print -----------------------------
warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message=".*force_all_finite.*renamed to 'ensure_all_finite'.")

def log(msg: str): print(f"[★] {msg}")

# ----------------------------- Utils -----------------------------
def future_business_days(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = last_date + BDay(1)
    return pd.date_range(start=start, periods=horizon, freq=BDay())

def _as_tuple(x) -> Tuple[str, ...]:
    if isinstance(x, str): return (x,)
    if isinstance(x, Iterable): return tuple(x)
    raise ValueError("column candidates must be str or Iterable[str]")

def _pick_column(df: pd.DataFrame, candidates) -> str:
    for c in _as_tuple(candidates):
        if c in df.columns:
            return c
    raise ValueError(f"Column not found. candidates={candidates}, actual={list(df.columns)}")

# ----------------------------- DataModule -----------------------------
@dataclass
class DataConfig:
    csv_path: str
    date_col: str = "Date"
    us_col: str = "Inv_Close"
    kr_col: str = "ECOS_Close"

class DataModule:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.csv_path)
        if self.cfg.date_col not in df.columns:
            raise ValueError(f"Missing date column '{self.cfg.date_col}'. columns={list(df.columns)}")
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col], errors="coerce")
        us_col = _pick_column(df, self.cfg.us_col)
        kr_col = _pick_column(df, self.cfg.kr_col)
        df = df[[self.cfg.date_col, us_col, kr_col]].dropna()
        df = df.drop_duplicates(subset=[self.cfg.date_col], keep="last").sort_values(self.cfg.date_col)
        df = df.rename(columns={us_col: "US", kr_col: "KR"}).set_index(self.cfg.date_col)
        df = df[["US", "KR"]].astype(float)
        self.df = df
        return df

    def window(self, use_days: int) -> pd.DataFrame:
        if self.df is None: self.load()
        return self.df.iloc[-use_days:].copy()

# ----------------------------- Model -----------------------------
@dataclass
class ModelConfig:
    use_days: int = 120
    random_seed: int = 42
    save_dir: str = "models/fx_arimax_garch"

class ARIMAXGARCHModel:
    def __init__(self, mcfg: ModelConfig):
        self.cfg = mcfg
        self.arimax = None
        self.garch = None
        self.meta = {}

    # ---- core transforms ----
    @staticmethod
    def _make_returns(df: pd.DataFrame):
        log_us = np.log(df["US"])
        log_kr = np.log(df["KR"])
        r_us = log_us.diff().fillna(0.0)
        r_kr = log_kr.diff().fillna(0.0)
        return r_us, r_kr, log_kr.iloc[-1]

    def fit(self, df: pd.DataFrame):
        np.random.seed(self.cfg.random_seed)

        # returns
        r_us, r_kr, _ = self._make_returns(df)

        # standardize exog
        ex_mean = float(r_us.mean()); ex_std = float(r_us.std() + 1e-12)
        exog_z = ((r_us - ex_mean) / ex_std).values.reshape(-1, 1)

        # ARIMAX on returns
        self.arimax = auto_arima(
            r_kr, exogenous=exog_z,
            d=0, seasonal=False, stepwise=True, suppress_warnings=True,
            max_p=3, max_q=3, max_d=0, error_action="ignore",
        ).fit(r_kr, exogenous=exog_z)

        # residuals
        ins_pred = pd.Series(self.arimax.predict_in_sample(exogenous=exog_z), index=r_kr.index)
        resid = (r_kr - ins_pred).dropna()
        resid_std = float(resid.std())

        # GARCH on resid
        self.garch = arch_model(resid * 100.0, vol="Garch", p=1, q=1, mean="Zero", dist="normal").fit(disp="off")

        # meta for inference
        self.meta = dict(
            use_days=self.cfg.use_days,
            random_seed=self.cfg.random_seed,
            exog_mean=ex_mean, exog_std=ex_std,
            resid_std=resid_std
        )

    def save(self, save_dir: Optional[str] = None):
        d = save_dir or self.cfg.save_dir
        os.makedirs(d, exist_ok=True)
        joblib.dump(self.arimax, os.path.join(d, "arimax.joblib"))
        joblib.dump(self.garch,  os.path.join(d, "garch.joblib"))
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load(self, save_dir: Optional[str] = None):
        d = save_dir or self.cfg.save_dir
        self.arimax = joblib.load(os.path.join(d, "arimax.joblib"))
        self.garch  = joblib.load(os.path.join(d, "garch.joblib"))
        with open(os.path.join(d, "metadata.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def forecast(self, df: pd.DataFrame, horizon: int, business_days: bool = True, rng_seed: Optional[int] = None) -> pd.DataFrame:
        assert self.arimax is not None and self.garch is not None
        rng = np.random.default_rng(self.cfg.random_seed if rng_seed is None else rng_seed)

        r_us, r_kr, last_log_price = self._make_returns(df)

        # exog hist (z)
        ex_mean = float(self.meta["exog_mean"]); ex_std = float(self.meta["exog_std"])
        us_hist_z = (r_us.values - ex_mean) / (ex_std + 1e-12)
        if np.std(us_hist_z) < 1e-8:
            us_hist_z = us_hist_z + rng.normal(0.0, 1e-4, size=us_hist_z.shape)

        # future dates
        last_date = df.index[-1]
        future_idx = future_business_days(last_date, horizon) if business_days \
                     else pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        # GARCH step std
        garch_fc = self.garch.forecast(horizon=horizon)
        step_var = np.asarray(garch_fc.variance.values[-1]).reshape(-1) / (100.0 ** 2)
        if step_var.shape[0] != horizon:
            step_var = np.repeat(step_var.mean(), horizon)
        step_std = np.sqrt(step_var)

        # simulate paths
        n_paths = 500
        resid_std = max(float(self.meta["resid_std"]), 1e-6)
        clip_sigma = 3.0 * resid_std

        paths = np.zeros((n_paths, horizon))
        for i in range(n_paths):
            us_future_z = rng.choice(us_hist_z, size=horizon, replace=True).reshape(-1, 1)
            mean_r = self.arimax.predict(n_periods=horizon, X=us_future_z)
            mean_r = np.clip(mean_r, -clip_sigma, clip_sigma)
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

# ----------------------------- Trainer -----------------------------
@dataclass
class TrainConfig:
    horizon: int = 5
    use_days: int = 120
    business_days: bool = True

class Trainer:
    def __init__(self, data: DataModule, model: ARIMAXGARCHModel, tcfg: TrainConfig):
        self.data = data
        self.model = model
        self.cfg = tcfg

    def fit(self):
        df_win = self.data.window(self.model.cfg.use_days)
        log(f"Training on window: {df_win.index.min().date()} ~ {df_win.index.max().date()} (N={len(df_win)})")
        self.model.fit(df_win)
        self.model.save()
        log(f"Saved to: {self.model.cfg.save_dir}")

    def forecast(self, show_plot: bool = True, rng_seed: Optional[int] = 42) -> pd.DataFrame:
        df_win = self.data.window(self.model.cfg.use_days)
        out = self.model.forecast(df_win, horizon=self.cfg.horizon, business_days=self.cfg.business_days, rng_seed=rng_seed)
        log("Forecast head:")
        print(out.head())

        if show_plot:
            self._plot_forecast(df_win, out, title=f"FX Forecast (h={self.cfg.horizon})")
        return out

    def backtest(self, n_eval: int = 30, show_plot: bool = True) -> pd.DataFrame:
        """
        롤링 원스텝(1-day) 백테스트: 최근 n_eval 영업일 동안,
        각 시점에서 use_days 창으로 재학습 → 하루 예측 → MAE/암시적 커버리지 체크.
        """
        df = self.data.load()
        window = self.model.cfg.use_days
        true_vals, pred_vals, naive_vals, dates = [], [], [], []

        it = tqdm(range(n_eval, 0, -1), desc="Backtest (1-step rolling)")
        for i in it:
            df_win = df.iloc[-(window+i): -i]
            # 학습
            self.model.fit(df_win)
            # 1-step 예측
            out = self.model.forecast(df_win, horizon=1, business_days=True)
            t_next = out["date"].iloc[0]
            if t_next not in df.index:
                continue
            y_true = df.loc[t_next, "KR"]
            y_pred = float(out["yhat"].iloc[0])
            y_naiv = float(df_win["KR"].iloc[-1])

            true_vals.append(y_true); pred_vals.append(y_pred); naive_vals.append(y_naiv); dates.append(t_next)

            it.set_postfix(mae=np.mean(np.abs(np.array(true_vals) - np.array(pred_vals))).round(3))

        res = pd.DataFrame({"date": dates, "y_true": true_vals, "y_pred": pred_vals, "y_naive": naive_vals}).sort_values("date")
        mae_pred = np.mean(np.abs(res["y_true"] - res["y_pred"]))
        mae_naiv = np.mean(np.abs(res["y_true"] - res["y_naive"]))
        cov = (np.minimum(res["y_pred"], res["y_naive"]) <= res["y_true"]).mean()  # 그냥 재미로 보는 지표

        log(f"Backtest MAE - pred: {mae_pred:.3f}, naive: {mae_naiv:.3f}, N={len(res)}")
        if show_plot:
            self._plot_backtest(res, title=f"Backtest (1-step, window={window}, N={len(res)})")
        return res

    @staticmethod
    def _plot_forecast(df_hist: pd.DataFrame, df_fc: pd.DataFrame, title: str = "Forecast"):
        plt.figure(figsize=(10, 5))
        plt.plot(df_hist.index[-120:], df_hist["KR"].iloc[-120:], label="KR (hist)")
        plt.plot(df_fc["date"], df_fc["yhat"], label="yhat")
        plt.fill_between(df_fc["date"], df_fc["yhat_p10"], df_fc["yhat_p90"], alpha=0.2, label="p10~p90")
        plt.plot(df_fc["date"], df_fc["sample_path"], linestyle="--", label="sample path")
        plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

    @staticmethod
    def _plot_backtest(res: pd.DataFrame, title: str = "Backtest"):
        plt.figure(figsize=(10,5))
        plt.plot(res["date"], res["y_true"], label="true")
        plt.plot(res["date"], res["y_pred"], label="pred")
        plt.plot(res["date"], res["y_naive"], label="naive", alpha=0.6)
        plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train/only_US_KOR_20100104_20250812_simple.csv")
    ap.add_argument("--date_col", default="Date")
    ap.add_argument("--us_col", default="Inv_Close")
    ap.add_argument("--kr_col", default="ECOS_Close")

    ap.add_argument("--mode", choices=["fit", "forecast", "backtest"], default="forecast")
    ap.add_argument("--use_days", type=int, default=120)
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--model_dir", default="models/fx_arimax_garch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--n_eval", type=int, default=30)  # backtest steps

    args = ap.parse_args()

    data = DataModule(DataConfig(csv_path=args.csv, date_col=args.date_col, us_col=args.us_col, kr_col=args.kr_col))
    model = ARIMAXGARCHModel(ModelConfig(use_days=args.use_days, random_seed=args.seed, save_dir=args.model_dir))
    trainer = Trainer(data, model, TrainConfig(horizon=args.h, use_days=args.use_days, business_days=True))

    if args.mode == "fit":
        trainer.fit()

    elif args.mode == "forecast":
        trainer.fit()  # 최신 윈도우로 학습
        trainer.forecast(show_plot=not args.no_plot, rng_seed=args.seed)

    elif args.mode == "backtest":
        # 최근 n_eval일 구간 롤링 1-step 평가
        trainer.backtest(n_eval=args.n_eval, show_plot=not args.no_plot)

if __name__ == "__main__":
    main()
