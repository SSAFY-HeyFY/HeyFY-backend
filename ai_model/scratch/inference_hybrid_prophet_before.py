import os
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc # 폰트 세팅을 위한 모듈 추가
font_path = "C:/Windows/Fonts/malgun.ttf" # 사용할 폰트명 경로 삽입
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

# --- 설정 ---
FILE_PATH = 'data/train/only_US_KOR_20100104_20250812_simple.csv'
MODEL_A_DIR = 'models/prophet_1day_model/'
MODEL_B_DIR = 'models/prophet_multi_day_model/'
MODEL_A_PATH = 'prophet_1day_model.json'    # 1일 예측용 (다변량)
MODEL_B_PATH = 'prophet_multi_day_model.json' # 2일 이상 예측용 (단변량)
MODEL_A_SAVE_PATH = os.path.join(MODEL_A_DIR, MODEL_A_PATH)
MODEL_B_SAVE_PATH = os.path.join(MODEL_B_DIR, MODEL_B_PATH)
FUTURE_DAYS = 5 # 총 예측할 미래 일수

print("--- 하이브리드 모델 추론 시작 ---")

# 1. 데이터 및 모델 로드
df = pd.read_csv(FILE_PATH)
df['ds'] = pd.to_datetime(df['Date'])
df_clean = df.dropna()
last_day_data = df_clean.tail(1)
last_date = df_clean['ds'].max()

print("저장된 모델들을 로드합니다...")
with open(MODEL_A_SAVE_PATH, 'r') as fin:
    model_A = model_from_json(fin.read()) # 1일 예측용
with open(MODEL_B_SAVE_PATH, 'r') as fin:
    model_B = model_from_json(fin.read()) # 2일+ 예측용
print("모델 로드 완료.")

# ======================================================================
# ***수정된 추론 및 보정 로직***
# ======================================================================
# 2. 두 모델로 각각 1일 후를 예측하여 차이(Offset) 계산
print("\n[모델 A, B]로 각각 1일 후를 예측하여 차이를 계산합니다...")
# 모델 A의 1일 후 예측
future_1day_A = pd.DataFrame({
    'ds': [last_date + pd.Timedelta(days=1)],
    'Inv_Close': [last_day_data['Inv_Close'].iloc[0]],
    'ECOS_Close': [last_day_data['ECOS_Close'].iloc[0]]
})
forecast_1day_A = model_A.predict(future_1day_A)
yhat_1day_A = forecast_1day_A['yhat'].iloc[0]

# 모델 B의 1일 후 예측
future_1day_B = pd.DataFrame({'ds': [last_date + pd.Timedelta(days=1)]})
forecast_1day_B = model_B.predict(future_1day_B)
yhat_1day_B = forecast_1day_B['yhat'].iloc[0]

# 두 모델 간의 차이(Offset) 계산
offset = yhat_1day_A - yhat_1day_B
print("yhat_1day_A: " + str(yhat_1day_A) + " yhat_1day_B: " + str(yhat_1day_B)) 
 
print(f"-> 1일차 예측 오프셋(차이) 계산 완료: {offset:.2f}")

# 3. 모델 B로 2~7일 후를 예측하고, 계산된 Offset으로 보정
print(f"\n[모델 B]로 2~{FUTURE_DAYS}일 후를 예측하고 오프셋을 적용합니다...")
future_weekdays = pd.bdate_range(start=last_date + pd.Timedelta(days=2), periods=FUTURE_DAYS - 1)
future_df_B = pd.DataFrame({'ds': future_weekdays})
forecast_multi_day_B = model_B.predict(future_df_B)

# 예측값(yhat)과 신뢰구간(lower, upper) 모두에 오프셋 적용
forecast_multi_day_B['yhat'] = forecast_multi_day_B['yhat'] + offset
forecast_multi_day_B['yhat_lower'] = forecast_multi_day_B['yhat_lower'] + offset
forecast_multi_day_B['yhat_upper'] = forecast_multi_day_B['yhat_upper'] + offset
print("-> 예측치 보정 완료.")

# 4. 최종 예측 결과 조합
print("\n두 모델의 예측 결과를 하나로 조합합니다...")
final_forecast_1day = forecast_1day_A[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
final_forecast_multi_day = forecast_multi_day_B[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
final_forecast = pd.concat([final_forecast_1day, final_forecast_multi_day], ignore_index=True)

# ======================================================================
# 결과 시각화
# ======================================================================
print("\n최종 추론 결과를 시각화합니다.")
plt.figure(figsize=(12, 8))
actuals_past = df.rename(columns={'target': 'y'})[['ds', 'y']].dropna().tail(30)
plt.plot(actuals_past['ds'], actuals_past['y'], 'k-', label='과거 실제값')
plt.plot(final_forecast['ds'], final_forecast['yhat'], 'r-o', label=f'미래 예측값 (보정됨)')
plt.fill_between(final_forecast['ds'], final_forecast['yhat_lower'], final_forecast['yhat_upper'], color='r', alpha=0.2)
plt.title(f'하이브리드 전략 예측 (Jump 현상 보정)')
plt.xlabel('날짜')
plt.ylabel('Target 값')
plt.legend()
plt.grid(True)
plt.show()

print("\n## 최종 예측 결과 (보정됨) ##")
print(final_forecast)
