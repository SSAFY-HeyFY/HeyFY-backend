import pandas as pd

inv_path = "data/Investig_USD_KRW_20100101_20250816.xlsx"
ecos_path = "data/한국은행_20100101_20250812_자른거.xlsx"

df_inv = pd.read_excel(inv_path).rename(columns={
    "Close": "Inv_Close", "Open": "Inv_Open", "High": "Inv_High",
    "Low": "Inv_Low", "Change": "Inv_Change(%)"
})
df_ecos = pd.read_excel(ecos_path).rename(columns={"DATA_VALUE": "ECOS_Close"})

df_inv["date"] = pd.to_datetime(df_inv["date"], errors="coerce").dt.date
df_inv["Inv_Change(%)"] = df_inv["Inv_Change(%)"] * 100
df_ecos["date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df_ecos["date"], unit="D")
df_ecos["date"] = df_ecos["date"].dt.date

df_merged = (
    pd.merge(df_inv, df_ecos[["date", "ECOS_Close"]], on="date", how="inner")
      .sort_values("date")
      .set_index("date")
)

print("병합된 데이터 (최신 5일):")
print(df_merged[["Inv_Open", "Inv_High", "Inv_Low", "Inv_Close", "Inv_Change(%)", "ECOS_Close"]].tail())

out_path = "data/merged_USDKRW_Inv_ECOS.xlsx"
df_merged.to_excel(out_path)
print("엑셀 저장:", out_path)
