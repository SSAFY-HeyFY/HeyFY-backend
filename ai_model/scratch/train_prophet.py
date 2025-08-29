import os
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import matplotlib.pyplot as plt

# --- 설정 ---
FILE_PATH = 'data/train/only_US_KOR_20100104_20250812_simple.csv'
MODEL_DIR = 'models/prophet/'
MODEL_PATH = 'prophet_model.json'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_PATH)
TEST_DAYS = 30  # 마지막 30일을 테스트 데이터로 사용

# 1. 데이터 로드 및 전처리
print(f"'{FILE_PATH}'에서 데이터를 로드합니다...")
df = pd.read_csv(FILE_PATH)
df['ds'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'target': 'y'})

# 결측치가 있는 행 제거
df_prophet = df[['ds', 'y', 'Inv_Close', 'ECOS_Close']].dropna()
print("데이터 전처리 완료.")

# 2. 학습/테스트 기간 분리
print(f"데이터를 학습 기간과 테스트 기간(마지막 {TEST_DAYS}일)으로 분리합니다.")
train_df = df_prophet.iloc[:-TEST_DAYS]

print(f"학습 기간: {train_df['ds'].min().strftime('%Y-%m-%d')} ~ {train_df['ds'].max().strftime('%Y-%m-%d')}")
print(f"총 {len(train_df)}개의 데이터로 모델을 학습합니다.")


# 3. Prophet 모델 생성, 보조 지표 추가 및 학습
print("\nProphet 모델을 생성하고 학습을 시작합니다...")
model = Prophet(daily_seasonality=True)

# 보조 지표 (Regressors) 추가
model.add_regressor('Inv_Close')
model.add_regressor('ECOS_Close')

# 학습 데이터로 모델 학습
model.fit(train_df)
print("모델 학습 완료.")

# 4. 학습된 모델 저장
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"'{MODEL_DIR}' 폴더를 생성했습니다.")
print(f"\n학습된 모델을 '{MODEL_SAVE_PATH}' 파일로 저장합니다...")
with open(MODEL_SAVE_PATH, 'w') as fout:
    fout.write(model_to_json(model))
print("모델 저장 완료.")

# (선택) 학습 결과 시각화
# print("\n학습 데이터에 대한 예측 결과를 시각화합니다.")
# fig = model.plot(model.predict(train_df), xlabel='날짜', ylabel='Target 값')
# plt.title('학습 데이터에 대한 예측 결과')
# plt.show()