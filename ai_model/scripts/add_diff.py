import pandas as pd

# 1. 데이터를 로드합니다.
excel_filename = 'data/train/train_final_with_onehot_20100104_20250812'

df = pd.read_excel(excel_filename + '.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df = df.reset_index(drop=True)

# 2. (핵심) 미국 시장 종가와 한국은행 종가의 차이를 계산하여 'diff'라는 새 열을 만듭니다.
# 이 계산은 target을 만들기 전에, 원본 데이터 상태에서 수행합니다.
df['diff'] = df['Inv_Close'] - df['ECOS_Close']

# 3. 'ECOS_Close' 열을 하루 앞으로 당겨서 'target' 열을 만듭니다.
df['target'] = df['ECOS_Close'].shift(-1)

# 4. 마지막 행은 target 값이 없으므로 제거합니다.
df = df.dropna()

# 5. 이제 학습에 사용할 feature 목록에 'diff'를 추가하면 됩니다.
# feature_columns = ['Inv_Close', 'DXY_Close', ..., 'diff']  <-- 여기에 추가!
# target_column = 'target'

# 변경된 데이터 확인 (Date, 미국 종가, 차이, 그리고 하루 뒤의 한국은행 종가)
print("새로운 'diff' 피처가 추가된 데이터:")
print(df[['Date', 'Inv_Close', 'ECOS_Close', 'diff']].head())

df.to_excel(excel_filename + '_with_diff.xlsx', index=False)