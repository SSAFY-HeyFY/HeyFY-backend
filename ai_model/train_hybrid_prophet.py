import os
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

# --- 설정 ---
FILE_PATH = 'data/train/prophet_train_2020102_20250812.csv'
MODEL_A_DIR = 'models/prophet_1day_model/'
MODEL_B_DIR = 'models/prophet_multi_day_model/'
MODEL_A_PATH = 'prophet_1day_model.json'      # 1일 예측용 (다변량)
MODEL_B_PATH = 'prophet_multi_day_model.json' # 2일 이상 예측용 (단변량)
MODEL_A_SAVE_PATH = os.path.join(MODEL_A_DIR, MODEL_A_PATH)
MODEL_B_SAVE_PATH = os.path.join(MODEL_B_DIR, MODEL_B_PATH)

print("--- 하이브리드 모델 학습 시작 ---")

# 1. 데이터 로드 및 기본 전처리
df = pd.read_csv(FILE_PATH)
df['ds'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'target': 'y'})
df_clean = df.dropna()

# ======================================================================
# 모델 A 학습 (1일 예측 전문가 - 다변량)
# ======================================================================
print(f"\n[모델 A] 1일 예측용 다변량 모델을 학습합니다...")
df_model_A = df_clean[['ds', 'y', 'Inv_Close', 'ECOS_Close']]

model_A = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.1)
model_A.add_regressor('Inv_Close')
model_A.add_regressor('ECOS_Close')
model_A.fit(df_model_A)

# 모델 A 저장
if not os.path.exists(MODEL_A_DIR):
    os.makedirs(MODEL_A_DIR)
    print(f"'{MODEL_A_DIR}' 폴더를 생성했습니다.")
with open(MODEL_A_SAVE_PATH, 'w') as fout:
    fout.write(model_to_json(model_A))
print(f"-> 모델 A를 '{MODEL_A_SAVE_PATH}'에 저장했습니다.")


# ======================================================================
# ✨ 모델 B 학습 (2일 이상 예측 전문가 - 단변량, 성능 개선 버전) ✨
# ======================================================================
print(f"\n[모델 B] 2일 이상 예측용 단변량 모델을 학습합니다...")
# 모델 B는 외부 변수 없이 오직 시간(ds)과 환율(y)만으로 학습합니다.
df_model_B = df_clean[['ds', 'ECOS_Close']]
df_model_B = df_model_B.rename(columns={'ECOS_Close': 'y'})

# 데이터 분석을 통해 도출한 최적의 파라미터로 모델을 설정합니다.
model_B = Prophet(
    growth='linear',
    changepoint_prior_scale=0.1,
    seasonality_mode='additive'
)
# 2일 주기의 단기 패턴과 20일 주기의 월별 패턴을 모델에 추가합니다.
model_B.add_seasonality(name='daily_short_term', period=2, fourier_order=5)
model_B.add_seasonality(name='monthly', period=20, fourier_order=5)

model_B.fit(df_model_B)

# 모델 B 저장
if not os.path.exists(MODEL_B_DIR):
    os.makedirs(MODEL_B_DIR)
    print(f"'{MODEL_B_DIR}' 폴더를 생성했습니다.")
with open(MODEL_B_SAVE_PATH, 'w') as fout:
    fout.write(model_to_json(model_B))
print(f"-> 모델 B를 '{MODEL_B_SAVE_PATH}'에 저장했습니다.")

print("\n--- 모든 모델의 학습 및 저장이 완료되었습니다. ---")


import holidays # 대한민국 공휴일 정보를 위한 라이브러리
PREDICT_DAYS = 20
# 3-1. 학습 데이터의 마지막 날짜 확인
last_date = df_model_B['ds'].max()

# 3-2. 예측 시작일로부터 주말을 제외한 영업일(Business day)을 넉넉하게 생성
# 공휴일 때문에 예측일수가 줄어들 것을 대비해 PREDICT_DAYS + 10일 만큼 생성
future_business_days = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=PREDICT_DAYS + 10, # 버퍼를 줌
    freq='B'  # 'B'는 Business day frequency (주말 제외)
)

# 3-3. 해당 연도의 대한민국 공휴일 목록 가져오기
sk_holidays = holidays.KR(years=future_business_days.year.unique())

# 3-4. 생성된 영업일 중 공휴일에 해당하는 날짜 제외
actual_future_dates = []
for date in future_business_days:
    if date not in sk_holidays:
        actual_future_dates.append(date)
    # 원하는 예측일수(PREDICT_DAYS) 만큼 채워지면 중단
    if len(actual_future_dates) == PREDICT_DAYS:
        break

# 3-5. 최종적으로 예측에 사용할 날짜로 DataFrame 생성
future = pd.DataFrame({'ds': actual_future_dates})


# 4. 예측 수행 및 결과 확인
forecast = model_B.predict(future)

# 예측 결과 확인 (주말/공휴일이 제외된 7일)
print(f"--- 주말/공휴일 제외된 미래 {PREDICT_DAYS}일 예측 결과 ---")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])