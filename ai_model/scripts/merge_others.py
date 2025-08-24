import pandas as pd

file1 = "data/merged_USDKRW_달러_지수.xlsx"
file2 = "data/Investing_미국10년물_국채_금리_채권수익률_20100104_20250816.xlsx"
save_name = "data/merged_USDKRW_달러_지수_US10Y_final.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

def ensure_datetime(col):
    out = pd.to_datetime(col, errors="coerce")
    mask_numlike = out.isna() & col.astype(str).str.fullmatch(r"\d+(\.\d+)?")
    if mask_numlike.any():
        out.loc[mask_numlike] = pd.to_datetime(col[mask_numlike].astype(float), unit="D", origin="1899-12-30")
    return out

#df1['Date'] = pd.to_datetime(df1['Date'], unit='D', origin='1899-12-30')
df2['Date'] = ensure_datetime(df2.iloc[:,0]).dt.normalize()
df2 = df2[['Date', 'Close']]
df2.rename(columns={'Close': 'US10Y_Close'}, inplace=True)

merged = pd.merge(df1, df2, on="Date", how="inner").set_index("Date")
print(merged.head())

merged.to_excel(save_name)
