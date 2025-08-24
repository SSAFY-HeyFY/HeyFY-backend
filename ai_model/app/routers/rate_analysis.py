import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

# --- Pydantic 모델 정의 ---
# API가 최종적으로 반환할 데이터 구조를 정의합니다.
class AnalysisResponse(BaseModel):
    api_called_at: str          # API가 호출된 시각
    today_rate: float           # 분석 기준이 되는 '오늘'의 환율
    ai_predicted_rate: float    # AI가 예측한 실제 환율 값
    historical_analysis: Optional[str] # 과거 데이터 기반 분석 문구
    ai_prediction: Optional[str]    # AI 예측 기반 문구

# --- 라우터 생성 ---
router = APIRouter()
PREDICTION_CACHE_FILE = "prediction_cache.json"

@router.get(
    "/rate-analysis",
    response_model=AnalysisResponse,
    summary="AI 분석 및 예측 문구 조회 (캐시 기반)",
    description="캐시 파일을 기반으로 과거 데이터 분석 및 AI 예측 조언 문구를 제공합니다."
)
def get_rate_analysis_from_cache():
    """
    [로직 요약]
    1. 'prediction_cache.json' 파일을 읽습니다.
    2. 데이터를 과거/예측으로 분리하고 '오늘의 환율'을 정의합니다.
    3. 과거 데이터를 기반으로 최고/최저점 분석 문구를 생성합니다.
    4. AI 예측 데이터를 기반으로 다음 날 예측 문구를 생성합니다.
    5. 모든 정보를 Wrapper 모델에 담아 반환합니다.
    """
    if not os.path.exists(PREDICTION_CACHE_FILE):
        raise HTTPException(
            status_code=404, 
            detail=f"'{PREDICTION_CACHE_FILE}'을 찾을 수 없습니다. AI 예측 스케줄러가 아직 실행되지 않았을 수 있습니다."
        )

    try:
        with open(PREDICTION_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data_list = json.load(f).get('predictions', [])
        
        if not cache_data_list:
            raise HTTPException(status_code=404, detail="캐시 파일에 데이터가 없습니다.")

        # 1. 데이터 분리 및 기준 환율 설정
        historical_points = [p for p in cache_data_list if not p['is_prediction']]
        predicted_points = [p for p in cache_data_list if p['is_prediction']]

        if not historical_points:
            raise HTTPException(status_code=404, detail="캐시에서 과거 데이터를 찾을 수 없습니다.")
        
        today_rate = historical_points[-1]['rate']

        # 2. 과거 데이터 기반 분석 (Historical Analysis)
        df_historical = pd.DataFrame(historical_points)
        min_rate_hist = df_historical['rate'].min()
        max_rate_hist = df_historical['rate'].max()
        
        historical_analysis_msg = None
        if today_rate >= max_rate_hist:
            historical_analysis_msg = "Over the past 30 days, today shows the highest exchange rate."
        elif today_rate <= min_rate_hist:
            historical_analysis_msg = "Today's rate is the lowest in the last 30 days. You might want to wait."

        # 3. AI 예측 기반 분석 (AI Prediction)
        ai_prediction_msg = None
        # predicted_points 리스트에는 '브릿지' 포인트와 '실제 예측' 포인트가 포함됩니다.
        # 실제 예측은 두 번째 요소이므로, 리스트 길이가 2 이상인지 확인합니다.
        if len(predicted_points) > 1:
            actual_prediction = predicted_points[1] # 0번은 브릿지, 1번이 실제 예측
            pred_rate = actual_prediction['rate']
            ai_predicted_rate_value = pred_rate
            diff = round(pred_rate - today_rate, 2)
            
            # [수정됨] 예측 날짜를 파싱하여 요일 정보를 추출합니다.
            prediction_date = datetime.strptime(actual_prediction['date'], '%Y-%m-%d')
            day_name = prediction_date.strftime('%A') # e.g., Monday

            # [수정됨] AI 예측 문구에 요일 정보를 포함하여 더 구체적으로 변경합니다.
            if diff > 0:
                ai_prediction_msg = (
                    f"Our AI model forecasts an increase of about {diff:.2f}₩ by this coming {day_name}, "
                    f"suggesting a favorable time for exchange. "
                    f"Consider exchanging your money then for better value."
                )
            else:
                ai_prediction_msg = (
                    f"Our AI model projects a potential decrease of about {-diff:.2f}₩ by this coming {day_name}. "
                    f"It might be better to wait for a more favorable rate. "
                    f"You might get more value by waiting."
                )

        # 4. 최종 응답 생성
        return AnalysisResponse(
            api_called_at=datetime.now().isoformat(),
            today_rate=today_rate,
            ai_predicted_rate=ai_predicted_rate_value,
            historical_analysis=historical_analysis_msg,
            ai_prediction=ai_prediction_msg
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="캐시 파일을 파싱하는 데 실패했습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 처리 중 오류 발생: {e}")
