import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# 1. 산점도 + 회귀선
plt.figure(figsize=(8,6))
sns.regplot(x=df["spread"], y=df["kr_change"], line_kws={"color":"red"}, scatter_kws={"alpha":0.3})
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(0, color='black', linestyle='--', linewidth=1)

plt.title("미국-한국 환율 차익 vs 다음날 한국 환율 변화", fontsize=14)
plt.xlabel("Spread (US_t - KR_t)", fontsize=12)
plt.ylabel("ΔKR (KR_t+1 - KR_t)", fontsize=12)
plt.show()