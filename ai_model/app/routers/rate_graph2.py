import json
import os
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import FinanceDataReader as fdr
import pandas as pd

from app.services.exchange_rate_crawler import get_detailed_exchange_rates

# yfinance는 AI 모델 입력에 대한 설명으로 이해하고, API 출력에서는 사용하지 않습니다.
# import yfinance as yf

# --- 서비스 임포트 (실제 환경에 맞게 수정 필요) ---
# from app.services.exchange_rate_crawler import get_detailed_exchange_rates
# 아래는 get_detailed_exchange_rates 함수의 모의(mock) 구현입니다.
# 실제 프로젝트에서는 주석을 해제하고 실제 크롤러 함수를 사용하세요.
class MockRateDetail(BaseModel):
    currency_code: str
    rate: float

async def get_detailed_exchange_rates() -> List[MockRateDetail]:
    # 실제 크롤링 로직 대신 임시 데이터를 반환합니다.
    # 테스트를 위해 실행 시점마다 약간의 변동을 줍니다.
    base_rate = 1380.0
    fluctuation = (datetime.now().second / 60.0) * 2 - 1 # -1 to +1
    return [MockRateDetail(currency_code='USDKRW', rate=base_rate + fluctuation)]
# --- 모의 구현 끝 ---


# --- Pydantic 모델 정의 ---
class ChartDataPoint(BaseModel):
    date: str
    rate: float
    is_prediction: bool = False
    difference_from_today: Optional[float] = None # 오늘 환율과의 차이

class RateStatistics(BaseModel):
    updated_at: datetime
    average: float
    min_rate: float
    max_rate: float
    change: float
    change_percent: float

class ExchangeRateResponse(BaseModel):
    source: str
    currency_pair: str = "USDKRW"
    today_rate: float # 오늘자 환율
    statistics: RateStatistics
    ai_prediction_summary: Optional[str] = None # AI 예측 요약 메시지 (영문)
    historical_summary: Optional[str] = None # 과거 데이터 기반 요약 메시지 (영문)
    data: List[ChartDataPoint]

# --- 라우터 생성 ---
router = APIRouter()
PREDICTION_CACHE_FILE = "prediction_cache.json"

def create_mock_prediction_cache():
    """AI 예측 캐시 파일이 없을 경우, 테스트용 모의 파일을 생성합니다."""
    if not os.path.exists(PREDICTION_CACHE_FILE):
        print(f"'{PREDICTION_CACHE_FILE}'을 찾을 수 없어 테스트용 모의 파일을 생성합니다.")
        today = datetime.now()
        mock_data = {
            "last_updated": (today - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
            "predictions": [
                {
                    "date": (today + timedelta(days=1)).strftime('%Y-%m-%d'),
                    "rate": 1395.50,
                    "is_prediction": True
                },
                {
                    "date": (today + timedelta(days=2)).strftime('%Y-%m-%d'),
                    "rate": 1402.10,
                    "is_prediction": True
                },
                {
                    "date": (today + timedelta(days=3)).strftime('%Y-%m-%d'),
                    "rate": 1398.80,
                    "is_prediction": True
                },
                {
                    "date": (today + timedelta(days=4)).strftime('%Y-%m-%d'),
                    "rate": 1405.20,
                    "is_prediction": True
                },
                {
                    "date": (today + timedelta(days=5)).strftime('%Y-%m-%d'),
                    "rate": 1410.00,
                    "is_prediction": True
                }
            ]
        }
        with open(PREDICTION_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, indent=4)

@router.get(
    "/rate-graph-ai",
    response_model=ExchangeRateResponse,
    summary="AI 예측을 포함한 환율 그래프 데이터 조회",
    description="과거(ECOS), 현재(실시간 크롤링), 미래(AI 예측)의 USD/KRW 환율 데이터를 통합하고, AI 분석 메시지를 함께 제공합니다."
)
async def get_rate_graph_data_with_ai(days: int = 30):
    """
    요청된 기능 요구사항을 모두 포함하는 API 함수입니다.
    1. 과거 30일 + 오늘 실시간 환율 데이터 제공
    2. 최대 5일치 AI 예측 환율 데이터 추가
    3. AI 예측 기반 영문 요약 메시지 생성
    4. 과거 30일 최고/최저점 비교 영문 메시지 생성
    5. 예측 데이터에 오늘 환율과의 차액 정보 포함
    """
    # 테스트용 모의 예측 파일 생성 함수 호출
    create_mock_prediction_cache()

    # --- 1. 과거 및 오늘 데이터 가져오기 ---
    try:
        # ECOS 데이터는 주말/공휴일 데이터가 없으므로 넉넉하게 조회
        start_date_fdr = datetime.now() - timedelta(days=days + 20)
        end_date_fdr = datetime.now() - timedelta(days=1)
        
        # [수정됨] AI 모델 입력과 일관성을 맞추기 위해 ECOS 매매기준율 데이터로 변경
        df_past = fdr.DataReader('ECOS-KEYSTAT:K152', start_date_fdr, end_date_fdr)
        df_past.reset_index(inplace=True)
        # 컬럼명을 'date', 'rate'로 표준화
        df_past.columns = ['date', 'rate']
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"과거 환율 데이터(ECOS) 조회 실패: {e}")

    try:
        realtime_rates = await get_detailed_exchange_rates()
        today_rate_detail = next((r for r in realtime_rates if r.currency_code == 'USDKRW'), None)
        if not today_rate_detail:
            raise ValueError("실시간 USD/KRW 환율 정보를 찾을 수 없습니다.")
        
        today_rate = round(today_rate_detail.rate, 2)
        today_data = {'date': datetime.now(), 'rate': today_rate}
        df_today = pd.DataFrame([today_data])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"실시간 환율 데이터 조회 실패: {e}")

    df_historical = pd.concat([df_past, df_today], ignore_index=True)
    df_historical['date'] = pd.to_datetime(df_historical['date']).dt.strftime('%Y-%m-%d')
    df_historical.dropna(inplace=True)
    df_historical.drop_duplicates(subset=['date'], keep='last', inplace=True)
    
    # 그래프에 표시할 최종 과거 데이터는 요청된 'days' 만큼만 사용
    df_display = df_historical.tail(days).copy()
    if df_display.empty:
        raise HTTPException(status_code=404, detail=f"요청하신 기간({days}일)의 유효한 환율 데이터가 없습니다.")

    historical_points = [ChartDataPoint(**row) for row in df_display.to_dict('records')]

    # --- 2. AI 예측 데이터 처리 ---
    predicted_points = []
    ai_prediction_summary = None
    if os.path.exists(PREDICTION_CACHE_FILE):
        with open(PREDICTION_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            predictions = cache.get('predictions', [])
            
            for pred in predictions:
                pred_rate = round(pred.get('rate', 0.0), 2)
                # 요구사항 6: AI 예측 데이터와 오늘 환율의 차이 계산
                difference = round(pred_rate - today_rate, 2)
                predicted_points.append(ChartDataPoint(
                    date=pred['date'],
                    rate=pred_rate,
                    is_prediction=True,
                    difference_from_today=difference
                ))

            # 요구사항 5: AI 예측 요약 메시지 생성 (영문)
            if predicted_points:
                first_prediction = predicted_points[0]
                diff = first_prediction.difference_from_today
                change_percent = round((diff / today_rate) * 100, 2) if today_rate != 0 else 0
                
                prediction_date = datetime.strptime(first_prediction.date, '%Y-%m-%d')
                day_name = prediction_date.strftime('%A') # e.g., Monday

                if diff > 0:
                    # 외국인 유학생 기준: 환율이 오르면(더 많은 원화를 받음) 긍정적
                    summary = (
                        f"Based on recent market shifts, the AI predicts the rate will increase "
                        f"by approximately {diff:.2f} KRW (+{change_percent}%) by this coming {day_name}. "
                        f"This could be a great time to exchange your currency."
                    )
                else:
                    summary = (
                        f"The AI forecasts a slight decrease in the exchange rate "
                        f"by about {-diff:.2f} KRW ({change_percent}%) by this {day_name}. "
                        f"You might get more value by waiting."
                    )
                ai_prediction_summary = summary


    # --- 3. 과거 30일 데이터 기반 메시지 생성 ---
    historical_summary = None
    min_rate_30d = df_display['rate'].min()
    max_rate_30d = df_display['rate'].max()

    # 요구사항 7: 지난 30일 최고/최저가 비교 메시지 (영문)
    if today_rate >= max_rate_30d:
        historical_summary = "Today's rate is the highest it's been in the last 30 days. We recommend exchanging your money today for the best value."
    elif today_rate <= min_rate_30d:
        historical_summary = "Today's rate is the lowest in the last 30 days. You might get a better rate by waiting for it to rise."


    # --- 4. 통계 계산 및 최종 데이터 결합 ---
    all_data = historical_points + predicted_points
    
    start_rate = df_display['rate'].iloc[0]
    end_rate = df_display['rate'].iloc[-1] # 통계는 예측 제외, 실제 데이터의 마지막 날 기준
    change = end_rate - start_rate
    stats = RateStatistics(
        updated_at=datetime.now(),
        average=round(df_display['rate'].mean(), 2),
        min_rate=min_rate_30d,
        max_rate=max_rate_30d,
        change=round(change, 2),
        change_percent=round((change / start_rate) * 100, 2) if start_rate != 0 else 0
    )

    return ExchangeRateResponse(
        source="ECOS, yfinance, AI Model",
        today_rate=today_rate,
        statistics=stats,
        ai_prediction_summary=ai_prediction_summary,
        historical_summary=historical_summary,
        data=all_data
    )

# FastAPI 앱에 라우터를 포함시키기 위한 코드 (예시)
# from fastapi import FastAPI
# app = FastAPI()
# app.include_router(router, prefix="/api/exchange", tags=["exchange-rate"])
