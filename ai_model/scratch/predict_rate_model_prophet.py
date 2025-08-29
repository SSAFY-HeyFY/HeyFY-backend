import os
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json

class Predictor:
    """
    Prophet 하이브리드 모델을 사용하여 환율을 예측하는 Predictor 클래스입니다.
    학습된 두 종류의 Prophet 모델을 로드하고, 새로운 데이터에 대한 미래 예측을 수행합니다.
    - 모델 A: 단기(1일) 예측을 위한 다변량 모델
    - 모델 B: 장기(2일 이상) 예측을 위한 단변량 모델
    """
    def __init__(self, model_dir):
        """
        Args:
            model_dir (str): 학습된 Prophet 모델(.json) 파일들이 저장된 디렉토리 경로
        """
        print(f"'{model_dir}' 경로에서 Prophet 모델들을 로드합니다.")
        
        model_a_path = os.path.join(model_dir, 'prophet_1day_model.json')
        model_b_path = os.path.join(model_dir, 'prophet_multi_day_model.json')

        if not os.path.exists(model_a_path) or not os.path.exists(model_b_path):
            raise FileNotFoundError("필수 Prophet 모델 파일(.json)을 찾을 수 없습니다.")

        # 1. Prophet 모델 로드
        with open(model_a_path, 'r') as fin:
            self.model_A = model_from_json(fin.read())
        with open(model_b_path, 'r') as fin:
            self.model_B = model_from_json(fin.read())
            
        self.future_days = 7  # 예측할 미래 일수
        print("✅ Prophet 하이브리드 모델 로딩 완료.")

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        최신 환율 데이터를 입력받아, 향후 7일간의 환율을 예측합니다.
        '앵커링 후 조정' 및 '그래프 연결' 기법이 적용된 최종 예측 결과를 반환합니다.

        Args:
            input_df (pd.DataFrame): 최신 데이터를 포함하는 데이터프레임.
                                     'Date', 'target', 'Inv_Close', 'ECOS_Close' 컬럼이 필요합니다.

        Returns:
            pd.DataFrame: 예측 결과. 컬럼: ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                          (첫 행은 그래프 연결을 위한 실제 마지막 값)
        """
        if input_df.empty:
            raise ValueError("입력 데이터프레임이 비어있습니다.")
            
        # 1. 입력 데이터 전처리
        df = input_df.copy()
        df['ds'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'target': 'y'})
        df_clean = df.dropna(subset=['ds', 'y', 'Inv_Close', 'ECOS_Close'])
        
        if df_clean.empty:
            raise ValueError("필수 컬럼에 NaN 값이 많아 예측을 진행할 수 없습니다.")
            
        last_day_data = df_clean.tail(1)
        last_date = df_clean['ds'].max()

        # 2. '앵커링 후 조정' 로직
        # 모델 A의 1일 후 예측
        future_1day_A = pd.DataFrame({
            'ds': [last_date + pd.Timedelta(days=1)],
            'Inv_Close': [last_day_data['Inv_Close'].iloc[0]],
            'ECOS_Close': [last_day_data['ECOS_Close'].iloc[0]]
        })
        forecast_1day_A = self.model_A.predict(future_1day_A)
        yhat_1day_A = forecast_1day_A['yhat'].iloc[0]

        # 모델 B의 1일 후 예측
        future_1day_B = pd.DataFrame({'ds': [last_date + pd.Timedelta(days=1)]})
        forecast_1day_B = self.model_B.predict(future_1day_B)
        yhat_1day_B = forecast_1day_B['yhat'].iloc[0]

        # 두 모델 간의 차이(Offset) 계산
        offset = yhat_1day_A - yhat_1day_B
        
        # 모델 B로 2~7일 후를 예측하고, 계산된 Offset으로 보정
        future_weekdays = pd.bdate_range(start=last_date + pd.Timedelta(days=2), periods=self.future_days - 1)
        future_df_B = pd.DataFrame({'ds': future_weekdays})
        forecast_multi_day_B = self.model_B.predict(future_df_B)

        forecast_multi_day_B['yhat'] += offset
        forecast_multi_day_B['yhat_lower'] += offset
        forecast_multi_day_B['yhat_upper'] += offset

        # 3. 최종 예측 결과 조합
        final_forecast_1day = forecast_1day_A[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        final_forecast_multi_day = forecast_multi_day_B[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast_df = pd.concat([final_forecast_1day, final_forecast_multi_day], ignore_index=True)

        # 4. 그래프 연결을 위해 실제값 마지막 행 추가
        last_actual_row_data = last_day_data[['ds', 'y']].rename(columns={'y':'yhat'})
        last_actual_row_data['yhat_lower'] = last_actual_row_data['yhat']
        last_actual_row_data['yhat_upper'] = last_actual_row_data['yhat']
        
        final_forecast_linked = pd.concat([last_actual_row_data, forecast_df], ignore_index=True)
        
        return final_forecast_linked


if __name__ == '__main__':
    # --- Prophet Predictor 클래스 사용 예시 ---
    
    # 1. 학습된 Prophet 모델들이 저장된 디렉토리 경로
    MODEL_DIRECTORY = "models"  # 예: 'models/prophet_1day_model.json' 이 있는 폴더
    DATA_PATH = "data/only_US_KOR_20100104_20250812_simple.csv" # 예측에 사용할 전체 데이터

    if not os.path.exists(MODEL_DIRECTORY):
        print(f"오류: 모델 디렉토리 '{MODEL_DIRECTORY}'를 찾을 수 없습니다.")
    else:
        try:
            # 2. 추론기 인스턴스 생성
            predictor = Predictor(model_dir=MODEL_DIRECTORY)

            # 3. 예측에 사용할 최신 데이터 로드 (실제로는 API가 DB 등에서 가져옴)
            # 여기서는 CSV 파일 전체를 사용하지만, 실제로는 마지막 1일치 데이터만 있어도 예측 가능
            input_data = pd.read_csv(DATA_PATH)
            
            # 4. 예측 수행
            predictions = predictor.predict(input_data)
            
            print("\n--- 예측 결과 ---")
            print(f"입력 데이터 마지막 날짜: {pd.to_datetime(input_data['Date'].iloc[-1]).strftime('%Y-%m-%d')}")
            print("다음 7일 후 환율 예측 (첫 행은 현재 값):")
            print(predictions)

        except Exception as e:
            print(f"\n--- 예측 테스트 중 오류 발생 ---")
            print(e)