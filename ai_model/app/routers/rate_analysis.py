import os
import json
from datetime import datetime, date
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic 모델 정의 ---

# Endpoint 1: /rate-analysis 용 모델
class AnalysisResponse(BaseModel):
    api_called_at: str
    today_rate: float
    # ai_predicted_rate 필드는 5일치 중 마지막 날 예측으로 대체
    final_predicted_rate: float
    historical_analysis: Optional[str]
    ai_prediction: Optional[str]

# Endpoint 2: /rate-prediction-summary 용 모델
class PredictionHighlight(BaseModel):
    trend_type: str  # "Bullish" 또는 "Bearish"
    header: str
    change_label: str  # "+1.17% in 5 days" 형태의 문자열
    highlight_date: str
    highlight_rate: float

class PredictionSummaryResponse(BaseModel):
    api_called_at: str
    today_rate: float
    prediction: PredictionHighlight

# --- 라우터 생성 및 설정 ---
router = APIRouter()
CACHE_BASE_PATH = os.getenv('CACHE_DIR', './logs')
PREDICTION_CACHE_FILE = os.path.join(CACHE_BASE_PATH, 'prediction_cache.json')

# --- 헬퍼 함수 ---
def describe_finance_style(current: float, past30: pd.Series) -> str:
    # 추천 문구 길이 줄임
    low, high = past30.min(), past30.max()
    if current >= high: return "Today's rate is the highest in the last 30 days. You may consider exchanging now."
    if current <= low: return "Today's rate is the lowest in the last 30 days. You might want to wait."
    if high == low: return "The exchange rate has remained unchanged for the past 30 days."
    pos = (current - low) / (high - low)
    if pos <= 0.2: return "Today's rate is near the bottom of the 30-day range."
    elif pos <= 0.4: return "Today's rate is below the monthly average level."
    elif pos <= 0.6: return "Today's rate is around the middle of the 30-day range."
    elif pos <= 0.8: return "Today's rate is in the upper part of this month's range."
    else: return "Today's rate is close to the 30-day peak level. It's worth considering for exchange."

def load_and_prepare_data():
    """캐시 파일을 로드하고 데이터를 분리하는 공통 함수"""
    if not os.path.exists(PREDICTION_CACHE_FILE):
        raise HTTPException(status_code=404, detail="AI 예측 캐시 파일을 찾을 수 없습니다.")
    try:
        with open(PREDICTION_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data_list = json.load(f).get('predictions', [])
        
        if not cache_data_list:
            raise HTTPException(status_code=404, detail="캐시 파일에 데이터가 없습니다.")

        historical_points = [p for p in cache_data_list if not p.get('is_prediction')]
        # 실제 예측값만 필터링 ('브릿지' 데이터 제외 및 모델별로 가져옴)
        predicted_points = [p for p in cache_data_list if p.get('is_prediction') and 'rate' in p]
        
        if not historical_points or not predicted_points:
            raise HTTPException(status_code=404, detail="과거 또는 예측 데이터가 부족합니다.")
        
        today_rate = historical_points[-1]['rate']
        
        return historical_points, predicted_points, today_rate
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 처리 중 오류 발생: {e}")

# --- API 엔드포인트 ---
@router.get(
    "/rate-analysis",
    response_model=AnalysisResponse,
    summary="AI 종합 분석 문구 조회 (5일 추세 반영)",
    description="5일치 예측 데이터를 종합하여 전반적인 추세에 대한 분석 문구를 제공합니다.\n" +
    "지난 30일 환율 분석은 최고점/최저점, 0%~20%, 20%~40%, 40%~60%, 60%~80%, 80~100% 구간별로 다르게 제공됩니다.\n" +
    "AI 예측 결과는 다음 영업일만 분석해줌.(/rate-prediction-summary API가 5일치 분석용)\n" +
    "AI 예측 결과 문구는 다음 영업일의 환율이 1원 이상 오르거나, 1원 이하로 내리거나, 1원 미만으로 움직일 때 총 3가지입니다."
)
def get_rate_analysis_from_cache():
    historical_points, predicted_points, today_rate = load_and_prepare_data()

    # 1. 과거 데이터 기반 분석
    df_historical = pd.DataFrame(historical_points)
    historical_analysis_msg = describe_finance_style(current=today_rate, past30=df_historical['rate'])

    # 2. AI 예측 기반 분석 (5일 전체 추세 반영 및 문구 간소화)
    actual_predictions = [p for p in predicted_points if p.get('model_name') in ['1D Est.', '5D Est.']]
    
    ai_prediction_msg = "AI is currently analyzing market trends."
    ai_predicted_rate_value = today_rate

    # 핵심 수정: 오늘 날짜 이후의 예측만 필터링하여 사용
    today_str = datetime.now().strftime('%Y-%m-%d')
    future_predictions = [p for p in actual_predictions if p['date'] > today_str]
    
    if actual_predictions:
        # 필터링된 미래 예측 리스트의 첫 번째 항목이 '진짜' 다음 날 예측
        next_day_pred = future_predictions[0]
        ai_predicted_rate_value = next_day_pred['rate']
        
        diff = ai_predicted_rate_value - today_rate
        
        prediction_date = datetime.strptime(next_day_pred['date'], '%Y-%m-%d')
        date_formatted = prediction_date.strftime('%B %d (%A)')

        if diff > 1.0:
            ai_prediction_msg = f"The rate may increase by ₩{diff:,.2f} by this coming {date_formatted}."
        elif diff < -1.0:
            ai_prediction_msg = f"The rate may decrease by ₩{abs(diff):,.2f} by this coming {date_formatted}."
        else:
            ai_prediction_msg = f"The rate is expected to remain stable until this coming {date_formatted}."

    # 3. 최종 응답 생성
    return AnalysisResponse(
        api_called_at=datetime.now().isoformat(),
        today_rate=today_rate,
        final_predicted_rate=ai_predicted_rate_value,
        historical_analysis=historical_analysis_msg,
        ai_prediction=ai_prediction_msg
    )


@router.get(
    "/rate-prediction-summary",
    response_model=PredictionSummaryResponse,
    summary="AI 환율 예측 강세/약세 정보 조회 (Bullish/Bearish 자동 선택)",
    description="5일 AI 예측 중 최고점(Bullish)과 최저점(Bearish)을 분석하여 변동폭이 더 큰 하나를 선택하여 제공합니다."
)
def get_rate_prediction_summary():
    historical_points, predicted_points, today_rate = load_and_prepare_data()
    
    actual_predictions = [p for p in predicted_points if p.get('model_name') in ['1D Est.', '5D Est.']]
    if not actual_predictions:
        raise HTTPException(status_code=404, detail="분석할 AI 예측 데이터가 없습니다.")

    df_preds = pd.DataFrame(actual_predictions)
    df_preds['date_obj'] = pd.to_datetime(df_preds['date'])
    today_date = date.today()

    # 1. 최고점(Bullish) 분석
    highest_point = df_preds.loc[df_preds['rate'].idxmax()]
    high_rate = highest_point['rate']
    high_date_obj = highest_point['date_obj'].date()
    high_days_span = (high_date_obj - today_date).days if (high_date_obj - today_date).days > 0 else 1
    high_percent_change = ((high_rate - today_rate) / today_rate) * 100 if today_rate != 0 else 0

    # change_label 문자열 생성
    bullish_change_label = f"+{high_percent_change:.2f}% in {high_days_span} days"

    bullish_pred = PredictionHighlight(
        trend_type="Bullish Prediction",
        header=f"The rate is expected to rise over the next {high_days_span} days",
        change_label=bullish_change_label, # <-- 수정된 필드 적용
        highlight_date=high_date_obj.strftime('%B %d, %Y'),
        highlight_rate=high_rate
    )

    # 2. 최저점(Bearish) 분석
    lowest_point = df_preds.loc[df_preds['rate'].idxmin()]
    low_rate = lowest_point['rate']
    low_date_obj = lowest_point['date_obj'].date()
    low_days_span = (low_date_obj - today_date).days if (low_date_obj - today_date).days > 0 else 1
    low_percent_change = ((low_rate - today_rate) / today_rate) * 100 if today_rate != 0 else 0

    # change_label 문자열 생성
    bearish_change_label = f"{low_percent_change:.2f}% in {low_days_span} days"

    bearish_pred = PredictionHighlight(
        trend_type="Bearish Prediction",
        header=f"The rate might decline over the next {low_days_span} days",
        change_label=bearish_change_label, # <-- 수정된 필드 적용
        highlight_date=low_date_obj.strftime('%B %d, %Y'),
        highlight_rate=low_rate
    )

    # --- Bullish/Bearish 중 변동폭이 더 큰 예측을 선택 ---
    abs_change_high = abs(high_rate - today_rate)
    abs_change_low = abs(low_rate - today_rate)

    final_prediction = bullish_pred if abs_change_high >= abs_change_low else bearish_pred
    
    return PredictionSummaryResponse(
        api_called_at=datetime.now().isoformat(),
        today_rate=today_rate,
        prediction=final_prediction
    )