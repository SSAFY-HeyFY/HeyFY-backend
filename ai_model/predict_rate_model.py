import os
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
import holidays

# --- 설정 ---
MODEL_A_PATH = 'models/prophet_1day_model/prophet_1day_model.json'
MODEL_B_PATH = 'models/prophet_multi_day_model/prophet_multi_day_model.json'

def get_hybrid_prophet_forecast(historical_df: pd.DataFrame, predict_days: int = 5) -> pd.DataFrame:
    """
    하이브리드 Prophet 모델을 사용해, 주어진 과거 데이터프레임을 기반으로 미래 N일(영업일 기준)의 환율을 예측합니다.
    실시간 추론 상황에 맞게 마지막 행의 target(y) 값이 NaN이어도 정상 동작합니다.

    Args:
        historical_df (pd.DataFrame): 'Date', 'target', 'Inv_Close', 'ECOS_Close' 컬럼을 포함하는 과거 데이터.
                                     마지막 행의 'target'은 비어있을 수 있습니다.
        predict_days (int): 예측할 미래 일수 (영업일 기준).

    Returns:
        pd.DataFrame: 최종 예측 결과. 'ds', 'yhat', 'yhat_lower', 'yhat_upper' 컬럼 포함.
    """
    # 1. 모델 파일 존재 여부 확인
    if not os.path.exists(MODEL_A_PATH) or not os.path.exists(MODEL_B_PATH):
        raise FileNotFoundError("학습된 Prophet 모델 파일(.json)을 찾을 수 없습니다. 먼저 학습을 실행해주세요.")

    # 2. 입력받은 데이터프레임 처리 (dropna 제거)
    df = historical_df.copy()
    df['ds'] = pd.to_datetime(df['Date'])
    if 'target' in df.columns:
        df = df.rename(columns={'target': 'y'})

    # ✨ 핵심 수정: dropna()를 사용하지 않고 마지막 행의 데이터를 그대로 사용
    last_day_data = df.tail(1)
    if last_day_data.empty:
        raise ValueError("입력 데이터가 비어있습니다.")
    
    last_date = last_day_data['ds'].iloc[0] # 예측 시작 기준 날짜
    
    # 3. Prophet 모델 로드
    with open(MODEL_A_PATH, 'r') as fin:
        model_A = model_from_json(fin.read())
    with open(MODEL_B_PATH, 'r') as fin:
        model_B = model_from_json(fin.read())

    # 4. 예측 대상 날짜 생성 (주말/공휴일 자동 제외)
    future_business_days = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=predict_days + 15,
        freq='B'
    )
    sk_holidays = holidays.KR(years=future_business_days.year.unique())
    final_future_dates = [date for date in future_business_days if date not in sk_holidays][:predict_days]
            
    if not final_future_dates:
        raise ValueError("예측할 미래 영업일이 없습니다. 날짜를 확인해주세요.")

    # 5. 하이브리드 추론 및 보정 (y값 없이 외부 변수만 사용)
    # 5-1. Offset 계산
    next_business_day = final_future_dates[0]
    future_1day_A = pd.DataFrame({
        'ds': [next_business_day],
        'Inv_Close': [last_day_data['Inv_Close'].iloc[0]], # 마지막 날의 외부 변수 사용
        'ECOS_Close': [last_day_data['ECOS_Close'].iloc[0]] # 마지막 날의 외부 변수 사용
    })
    forecast_1day_A = model_A.predict(future_1day_A)
    
    future_1day_B = pd.DataFrame({'ds': [next_business_day]})
    forecast_1day_B = model_B.predict(future_1day_B)
    
    offset = forecast_1day_A['yhat'].iloc[0] - forecast_1day_B['yhat'].iloc[0]

    # 5-2. 장기 예측 및 결과 조합
    final_forecast_1day = forecast_1day_A[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    if len(final_future_dates) > 1:
        multi_day_dates = final_future_dates[1:]
        future_df_B = pd.DataFrame({'ds': multi_day_dates})
        forecast_multi_day_B = model_B.predict(future_df_B)
        
        forecast_multi_day_B['yhat'] += offset
        forecast_multi_day_B['yhat_lower'] += offset
        forecast_multi_day_B['yhat_upper'] += offset
        
        final_forecast_multi_day = forecast_multi_day_B[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        final_forecast = pd.concat([final_forecast_1day, final_forecast_multi_day], ignore_index=True)
    else:
        final_forecast = final_forecast_1day

    return final_forecast

# --- 스크립트 직접 실행 시 테스트를 위한 부분 ---
if __name__ == '__main__':
    print("--- Prophet 하이브리드 모델 예측 테스트 (실시간 추론 상황 가정) ---")
    try:
        # 테스트를 위해 CSV 파일에서 데이터를 직접 로드
        DATA_FILE_PATH = 'data/train/only_US_KOR_20100104_20250812_simple.csv'
        historical_data = pd.read_csv(DATA_FILE_PATH)
        
        # 실시간 상황 시뮬레이션: 마지막 행의 'target' 값을 강제로 NaN으로 만듦
        historical_data.loc[historical_data.index[-1], 'target'] = None
        
        print("\n테스트 데이터 마지막 행 (target=NaN):")
        print(historical_data.tail(1))
        
        # 수정된 함수에 데이터 전달
        prediction_df = get_hybrid_prophet_forecast(
            historical_df=historical_data, 
            predict_days=5
        )
        print("\n✅ 예측 성공!")
        print(prediction_df)

    except Exception as e:
        print(f"\n❌ 예측 중 오류 발생: {e}")