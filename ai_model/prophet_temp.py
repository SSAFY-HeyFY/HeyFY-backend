from prophet import Prophet
import pandas as pd
import holidays # 대한민국 공휴일 정보를 위한 라이브러리

# --- 기존 코드와 동일 ---
#FILE_PATH = 'data/train/prophet_train.csv'
FILE_PATH = 'data/train/temp/prophet_train_5years.csv'
#FILE_PATH = 'data/train/prophet_train_3years.csv'
PREDICT_DAYS = 5

# 1. 데이터 로드 및 기본 전처리
df = pd.read_csv(FILE_PATH)
df['ds'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'ECOS_Close': 'y'})
df_clean = df.dropna()

# 2. Prophet 모델 생성 및 학습 (기존 코드와 동일)
model = Prophet(
    growth='linear',
    changepoint_prior_scale=0.1,
    seasonality_mode='additive'
)
model.add_seasonality(name='daily', period=2, fourier_order=5)
model.add_seasonality(name='monthly', period=20, fourier_order=5)
model.fit(df_clean)


# --- ✨ 주말/공휴일 제외를 위해 수정된 부분 ✨ ---

# 3. 예측할 미래 날짜 생성 (주말/공휴일 제외)

# 3-1. 학습 데이터의 마지막 날짜 확인
last_date = df['ds'].max()

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
forecast = model.predict(future)

# 예측 결과 확인 (주말/공휴일이 제외된 7일)
print(f"--- 주말/공휴일 제외된 미래 {PREDICT_DAYS}일 예측 결과 ---")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])