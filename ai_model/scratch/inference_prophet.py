import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error # MAE 계산을 위해 추가

# --- 설정 ---
FILE_PATH = 'data/train/only_US_KOR_20100104_20250812_simple.csv'
MODEL_LOAD_PATH = 'models/prophet/prophet_model.json'
TEST_DAYS = 365 # 학습 코드와 동일하게 설정

# 1. 저장된 모델 로드
print(f"'{MODEL_LOAD_PATH}'에서 학습된 모델을 로드합니다...")
with open(MODEL_LOAD_PATH, 'r') as fin:
    model = model_from_json(fin.read())
print("모델 로드 완료.")

# 2. 전체 데이터 로드 및 추론(테스트)용 데이터 준비
print(f"\n'{FILE_PATH}'에서 전체 데이터를 다시 로드하여 추론에 사용할 데이터를 준비합니다.")
df = pd.read_csv(FILE_PATH)
df['ds'] = pd.to_datetime(df['Date'])
df_prophet = df[['ds', 'Inv_Close', 'ECOS_Close']].dropna()

# 마지막 30일을 추론(테스트)할 기간으로 설정
future_df = df_prophet.tail(TEST_DAYS)

print(f"추론(테스트) 기간: {future_df['ds'].min().strftime('%Y-%m-%d')} ~ {future_df['ds'].max().strftime('%Y-%m-%d')}")
print(f"총 {len(future_df)}일의 데이터를 예측합니다.")

# 3. 미래 예측 실행
print("\n추론을 시작합니다...")
forecast = model.predict(future_df)
print(forecast.tail(7))
print("추론 완료.")

# 4. 결과 DataFrame 생성 및 MAE 계산
print("\n추론 결과와 실제 값을 비교하여 성능을 분석합니다.")
# 시각화 및 분석을 위해 실제 값(y) 불러오기
actuals = df.rename(columns={'target': 'y'})[['ds', 'y']].tail(TEST_DAYS)
# 예측 결과와 실제 값을 날짜(ds) 기준으로 합치기
result_df = pd.merge(forecast[['ds', 'yhat']], actuals, on='ds')

# MAE 계산
mae = mean_absolute_error(result_df['y'], result_df['yhat'])
print(f"✅ 평균 절대 오차 (MAE): {mae:.2f}")
print(f"-> 모델의 예측값은 실제값과 평균적으로 {mae:.2f} 정도의 차이를 보입니다.")


# 5. 결과 시각화
print("\n추론 결과를 실제 값과 비교하여 시각화합니다.")
plt.figure(figsize=(12, 8))
# 예측값 (파란 선) 및 불확실성 구간 (연한 파란 영역)
plt.plot(result_df['ds'], result_df['yhat'], 'b-', label=f'예측값 (Predicted)')
plt.fill_between(result_df['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='b', alpha=0.2)
# 실제값 (빨간 점)
plt.plot(result_df['ds'], result_df['y'], 'ro', label='실제값 (Actual)')
# 그래프 제목에 MAE 값 추가
plt.title(f'Prophet 모델 추론 결과 (MAE: {mae:.2f})')
plt.xlabel('날짜')
plt.ylabel('Target 값')
plt.legend()
plt.grid(True)
plt.show()

print(f"\n## 최근 {TEST_DAYS}일간의 예측 결과 ##")
print(result_df.tail(TEST_DAYS))