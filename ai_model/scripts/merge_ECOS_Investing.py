import pandas as pd

df_bok = pd.read_excel('data/한국은행_20100101_20250812_자른거.xlsx')
df_investing = pd.read_csv('data/Investig_USD_KRW_20100101_20250816.csv', encoding='CP949')

df_bok['date'] = pd.to_datetime(df_bok['date'])
df_bok.rename(columns={'DATA_VALUE': 'BOK_Close'}, inplace=True)

df_investing.rename(columns={'날짜': 'date', '종가': 'Inv_Close', '시가': 'Inv_Open', '고가': 'Inv_High', '저가': 'Inv_Low', '변동 %': 'Inv_Change'}, inplace=True)
df_investing['date'] = pd.to_datetime(df_investing['date'])

df_merged = pd.merge(df_investing, df_bok, on='date', how='inner')
df_merged.set_index('date', inplace=True)
df_merged.sort_index(inplace=True)

print("병합 및 날짜 정제 후 데이터 (최신 5일):")
print(df_merged[['Inv_Open', 'Inv_High', 'Inv_Low', 'Inv_Close', 'BOK_Close']].tail())