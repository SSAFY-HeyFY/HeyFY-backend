import pandas as pd

inv_path = "data/Investing_미국10년물_국채_금리_채권수익률_20100104_20250816.xlsx"
ecos_path = "data/한국은행_20100101_20250812_자른거.xlsx"

key = "US10Y"

df_inv = pd.read_excel(inv_path).rename(columns={
    "Close": f"{key}_Close", "Open": f"{key}_Open", "High": f"{key}_High",
    "Low": f"{key}_Low", "Change": f"{key}_Change(%)"
})
df_ecos = pd.read_excel(ecos_path).rename(columns={"DATA_VALUE": "ECOS_Close"})

df_inv["Date"] = pd.to_datetime(df_inv["Date"], errors="coerce").dt.date
df_inv[f"{key}_Change(%)"] = df_inv[f"{key}_Change(%)"] * 100
df_ecos["Date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df_ecos["Date"], unit="D")
df_ecos["Date"] = df_ecos["Date"].dt.date

df_merged = (
    pd.merge(df_inv, df_ecos[["Date", "ECOS_Close"]], on="Date", how="inner")
      .sort_values("Date")
      .set_index("Date")
)

print("병합된 데이터 (최신 5일):")
print(df_merged[[f"{key}_Open", f"{key}_High", f"{key}_Low", f"{key}_Close", f"{key}_Change(%)", "ECOS_Close"]].tail())

out_path = f"data/merged_{key}_ECOS.xlsx"
df_merged.to_excel(out_path)
print("엑셀 저장:", out_path)
