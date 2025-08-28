import pandas as pd
from statsmodels.tsa.stattools import ccf, grangercausalitytests
import statsmodels.api as sm

# 1. 데이터 준비
# df["US"], df["KR"]는 각각 미국/한국 환율 종가라고 가정
file_path = "data/correlation/merged_US10Y_ECOS_CLOSE.xlsx"
df = pd.read_excel(file_path)

df["spread"] = df["US10Y_Close"] - df["ECOS_Close"]
df["kr_change"] = df["ECOS_Close"].shift(-1) - df["ECOS_Close"]
 #df["ECOS_next"] = df["ECOS_Close"].shift(-1)
df = df.dropna(subset=["spread", "kr_change"])

# 2. 단순 상관
corr = df["spread"].corr(df["kr_change"])
print("Corr(spread_t, KR_t+1):", corr)

cross_corr = ccf(df["spread"], df["kr_change"])
print("Cross_corr(spread_t, KR_t+1):", cross_corr)

# 4. 단순 회귀
X = sm.add_constant(df["spread"].dropna())
y = df["kr_change"].dropna()
model = sm.OLS(y, X).fit()
print(model.summary())

# 5. Granger 인과 검정 (최대 3일 lag)
grangercausalitytests(df[["kr_change", "spread"]].dropna(), maxlag=3)
