import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# --- Pydantic 모델 정의 ---
# API가 최종적으로 반환할 데이터 구조를 정의합니다.
class AnalysisResponse(BaseModel):
    api_called_at: str            # API가 호출된 시각
    today_rate: float             # 분석 기준이 되는 '오늘'의 환율
    ai_predicted_rate: float      # AI가 예측한 실제 환율 값
    historical_analysis: Optional[str] # 과거 데이터 기반 분석 문구
    ai_prediction: Optional[str]       # AI 예측 기반 문구

# --- 라우터 생성 ---
router = APIRouter()
CACHE_BASE_PATH = os.getenv('CACHE_DIR', './logs')
PREDICTION_CACHE_FILE = os.path.join(CACHE_BASE_PATH, 'prediction_cache.json')

# --- 헬퍼 함수 ---
# 금융 스타일의 분석 문구를 생성하는 헬퍼 함수입니다.
def describe_finance_style(current: float, past30: pd.Series) -> str:
    """
    현재 환율이 과거 30일 데이터 범위 내 어느 지점에 위치하는지 분석하여
    금융 리포트 스타일의 문구를 반환합니다.
    """
    low = past30.min()
    high = past30.max()
    
    # 최고점/최저점 특별 케이스를 먼저 처리합니다.
    if current >= high:
        return "Over the past 30 days, today shows the highest exchange rate."
    if current <= low:
        return "Today's rate is the lowest in the last 30 days. You might want to wait."
        
    # 변동 없는 경우를 방지합니다.
    if high == low:
        return "The exchange rate has shown no fluctuation over the past 30 days."
        
    pos = (current - low) / (high - low)

    if pos <= 0.2:
        return "The current exchange rate is positioned near the bottom of the 30-day range, reflecting a relatively weak dollar level."
    elif pos <= 0.4:
        return "The exchange rate is trading in the lower segment of the past 30 days, slightly below the monthly average."
    elif pos <= 0.6:
        return "The current rate is around the midpoint of the 30-day range, indicating a neutral positioning within recent trends."
    elif pos <= 0.8:
        return "The exchange rate is situated in the upper segment of the monthly range, showing moderate upward pressure."
    else: # pos > 0.8
        return "The current rate is close to the 30-day peak, signaling strong dollar momentum against the won."

# --- API 엔드포인트 ---
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
        historical_points = [p for p in cache_data_list if not p.get('is_prediction')]
        predicted_points = [p for p in cache_data_list if p.get('is_prediction')]

        if not historical_points:
            raise HTTPException(status_code=404, detail="캐시에서 과거 데이터를 찾을 수 없습니다.")
        
        today_rate = historical_points[-1]['rate']

        # 2. 과거 데이터 기반 분석
        df_historical = pd.DataFrame(historical_points)
        historical_analysis_msg = describe_finance_style(
            current=today_rate,
            past30=df_historical['rate']
        )

        # 3. AI 예측 기반 분석
        ai_prediction_msg = None
        ai_predicted_rate_value = 0.0
        if len(predicted_points) > 1:
            actual_prediction = predicted_points[1] # 0번은 브릿지, 1번이 실제 예측
            pred_rate = actual_prediction['rate']
            ai_predicted_rate_value = pred_rate
            diff = pred_rate - today_rate
            
            prediction_date = datetime.strptime(actual_prediction['date'], '%Y-%m-%d')
            day_name = prediction_date.strftime('%A')

            if diff > 0.01:
                ai_prediction_msg = f"Our AI model forecasts a potential increase to around {pred_rate:,.2f}₩ by this coming {day_name}, suggesting a more favorable time for selling dollars."
            elif diff < -0.01:
                ai_prediction_msg = f"Our AI model projects a potential decrease to around {pred_rate:,.2f}₩ by this coming {day_name}, indicating a better opportunity for buying dollars might be ahead."
            else:
                ai_prediction_msg = f"Our AI model suggests the rate will remain stable around {pred_rate:,.2f}₩ through this coming {day_name}, indicating no significant short-term fluctuation."

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
