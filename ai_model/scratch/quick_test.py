# quick_test.py
import os, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from pandas.tseries.offsets import BDay

warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message=".*force_all_finite.*renamed to 'ensure_all_finite'.")

CSV = "data/train/only_US_KOR_20100104_20250812_simple.csv"
MODEL_DIR = "models/fx_arimax_garch"
H = 5

# 1) 학습 저장
print("[1/4] training & saving ...")
import train_arimax_garch as T
T.train_and_save(CSV, MODEL_DIR, date_col="Date", us_col="Inv_Close", kr_col="ECOS_Close", use_days=120)

assert Path(MODEL_DIR, "arimax.joblib").exists()
assert Path(MODEL_DIR, "garch.joblib").exists()
assert Path(MODEL_DIR, "metadata.json").exists()

# 2) 추론 2회 (재현성 체크)
print("[2/4] predicting (two runs for determinism) ...")
import predict_arimax_garch as P
out1 = P.predict_simulate(CSV, MODEL_DIR, horizon=H, date_col="Date", us_col="Inv_Close", kr_col="ECOS_Close", business_days=True, rng_seed=42)
out2 = P.predict_simulate(CSV, MODEL_DIR, horizon=H, date_col="Date", us_col="Inv_Close", kr_col="ECOS_Close", business_days=True, rng_seed=42)

# 3) 구조/형태/단조 체크
print("[3/4] structural assertions ...")
for out in (out1, out2):
    assert list(out.columns) == ["date","yhat","yhat_p10","yhat_p90","sample_path"]
    assert len(out)==H
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out[["yhat","yhat_p10","yhat_p90","sample_path"]].notna().all().all()
    # 신뢰밴드 관계
    assert (out["yhat_p10"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["yhat_p90"]).all()
    # 값이 환율 범위(대략 800~2000원) 안팎인지 대략 점검 (유연하게 마진 둠)
    assert (out[["yhat","yhat_p10","yhat_p90","sample_path"]] > 500).all().all()
    assert (out[["yhat","yhat_p10","yhat_p90","sample_path"]] < 3000).all().all()
    # 날짜가 영업일 간격인지 대략 점검
    diffs = out["date"].diff().dropna()
    assert diffs.apply(lambda d: d in [BDay(1), pd.Timedelta(days=1), pd.Timedelta(days=3)]).all()

# 4) 재현성: 동일 시드 → 동일 결과
print("[4/4] determinism ...")
np.testing.assert_allclose(out1["yhat"].values, out2["yhat"].values, rtol=0, atol=1e-12)
np.testing.assert_allclose(out1["yhat_p10"].values, out2["yhat_p10"].values, rtol=0, atol=1e-12)
np.testing.assert_allclose(out1["yhat_p90"].values, out2["yhat_p90"].values, rtol=0, atol=1e-12)
np.testing.assert_allclose(out1["sample_path"].values, out2["sample_path"].values, rtol=0, atol=1e-12)

print("\n✅ QUICK TEST PASSED")
print(out1)
