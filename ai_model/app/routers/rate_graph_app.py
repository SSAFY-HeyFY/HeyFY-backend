import json
import os
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import FinanceDataReader as fdr
import pandas as pd

# --- 서비스 임포트 ---
from app.services.exchange_rate_crawler import get_detailed_exchange_rates

# --- Pydantic 모델 정의 ---
class ChartDataPoint(BaseModel):
    date: str
    rate: float
    is_prediction: bool = False

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
    statistics: RateStatistics
    data: List[ChartDataPoint]

# --- 라우터 생성 ---
router = APIRouter()
PREDICTION_CACHE_FILE = "prediction_cache.json"

@router.get(
    "/rate-graph",
    response_model=ExchangeRateResponse,
    summary="AI 예측을 포함한 환율 그래프 데이터 조회",
    description="과거(ECOS), 현재(실시간 크롤링), 미래(AI 예측)의 USD/KRW 환율 데이터를 통합하여 제공합니다."
)
async def get_rate_graph_data(days: int = 30):
    """
    - 과거 (days-1일): ECOS 한국은행 기준환율을 사용합니다.
    - 오늘 (1일): 실시간 크롤러로 가져옵니다.
    - 미래 (3일): 미리 계산된 AI 예측 캐시를 사용합니다.
    """
    today_str = datetime.now().strftime('%Y-%m-%d')
    yesterday = datetime.now() - timedelta(days=1)
    
    # 1. 과거 데이터 소스 변경: ECOS 기준환율 사용
    try:
        start_date_fdr = datetime.now() - timedelta(days=days + 100) # 넉넉하게 조회
        df_past = fdr.DataReader('ECOS-KEYSTAT:K152', start_date_fdr, yesterday)
        df_past.reset_index(inplace=True)
        df_past.columns = ['date', 'rate']
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"과거 환율 데이터(ECOS) 조회 실패: {e}")

    # 2. 오늘의 실시간 데이터 가져오기
    try:
        realtime_rates = await get_detailed_exchange_rates()
        today_rate_detail = next((r for r in realtime_rates if r.currency_code == 'USDKRW'), None)
        if not today_rate_detail:
            raise ValueError("실시간 USD/KRW 환율 정보를 찾을 수 없습니다.")
        
        today_data = {'date': datetime.now(), 'rate': today_rate_detail.rate}
        df_today = pd.DataFrame([today_data])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"실시간 환율 데이터 조회 실패: {e}")

    # 3. 과거 + 오늘 데이터 합치기
    df_historical = pd.concat([df_past, df_today], ignore_index=True)
    df_historical['date'] = pd.to_datetime(df_historical['date']).dt.strftime('%Y-%m-%d')
    df_historical.dropna(inplace=True)
    
    # 그래프에 표시할 최종 과거 데이터는 요청된 'days' 만큼만 사용
    df_display = df_historical.tail(days).copy()
    if df_display.empty:
        raise HTTPException(status_code=404, detail=f"요청하신 기간({days}일)의 유효한 환율 데이터가 없습니다.")

    # 4. AI 예측 데이터 읽기
    predicted_points = []
    if os.path.exists(PREDICTION_CACHE_FILE):
        with open(PREDICTION_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            predicted_points = [ChartDataPoint(**item) for item in cache.get('predictions', [])]

    # 5. 모든 데이터 결합
    historical_points = [ChartDataPoint(**row, is_prediction=False) for row in df_display.to_dict('records')]
    all_data = historical_points + predicted_points

    # 6. 통계 계산 (과거+오늘 데이터 기준)
    start_rate = df_display['rate'].iloc[0]
    end_rate = df_display['rate'].iloc[-1]
    change = end_rate - start_rate
    stats = RateStatistics(
        updated_at=datetime.now(),
        average=round(df_display['rate'].mean(), 2),
        min_rate=df_display['rate'].min(),
        max_rate=df_display['rate'].max(),
        change=round(change, 2),
        change_percent=round((change / start_rate) * 100, 2) if start_rate != 0 else 0
    )

    return ExchangeRateResponse(
        source="ECOS, Naver Finance, 자체 AI 모델",
        statistics=stats,
        data=all_data
    )
