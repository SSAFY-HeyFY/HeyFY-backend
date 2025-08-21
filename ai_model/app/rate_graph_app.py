import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import FinanceDataReader as fdr

# 개별 데이터 포인트 모델
class ChartDataPoint(BaseModel):
    date: str
    rate: float

# 통계 정보 모델
class RateStatistics(BaseModel):
    average: Optional[float] = None     # 평균 환율
    min_rate: Optional[float] = None    # 최저 환율
    max_rate: Optional[float] = None    # 최고 환율
    change: Optional[float] = None      # 기간 내 변동액
    change_percent: Optional[float] = None # 기간 내 변동률 (%)

# 최종 응답 모델
class ExchangeRateResponse(BaseModel):
    source: str = "한국은행 경제통계시스템 (ECOS)"
    currency_pair: str = "USD/KRW"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    count: int = 0
    statistics: RateStatistics
    data: List[ChartDataPoint]

# --- FastAPI 앱 생성 ---
app = FastAPI(
    title="📈 환율 정보 API",
    description="FinanceDataReader를 사용하여 최근 환율 데이터를 Vico 차트 라이브러리에 최적화된 형식으로 제공합니다.",
    version="1.0.0"
)

# --- API 엔드포인트 정의 ---
@app.get("/rate-graph", response_model=ExchangeRateResponse)
async def get_exchange_rate_data(days: int = 30):
    # 1. 조회 시작일 계산
    today = datetime.now()
    start_date = today - timedelta(days=days)
    start_date_str = start_date.strftime('%Y-%m-%d')
    print(f"📅 환율 데이터 조회 기간: {start_date_str} ~ {today.strftime('%Y-%m-%d')}")

    # 2. 데이터 가져오기
    try:
        df = fdr.DataReader('ECOS-KEYSTAT:K152', start_date_str)
    except Exception as e:
        raise HTTPException(
            status_code=503, # 503 Service Unavailable: 외부 서비스(fdr) 문제로 서비스 불가
            detail=f"FinanceReader에서 환율 정보를 가져오는데 실패했습니다. 원인: {e}"
        )

    # 데이터가 비어있을 경우 빈 응답 반환
    if df.empty:
        raise HTTPException(
            status_code=404, # 404 Not Found: 요청한 기간에 데이터가 없음
            detail=f"{start_date_str} 이후의 환율 데이터가 존재하지 않습니다."
        )

    # 3. 데이터 가공
    df.reset_index(inplace=True)
    df.columns = ['date', 'rate']
    df.dropna(subset=['rate'], inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # 데이터가 없는 경우를 다시 한번 확인
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"{start_date_str} 이후 유효한 환율 데이터가 존재하지 않습니다."
        )

    # 4. 메타데이터 및 통계 정보 계산
    start_rate = df['rate'].iloc[0]
    end_rate = df['rate'].iloc[-1]
    change = end_rate - start_rate
    
    stats = RateStatistics(
        average=round(df['rate'].mean(), 2),
        min_rate=df['rate'].min(),
        max_rate=df['rate'].max(),
        change=round(change, 2),
        change_percent=round((change / start_rate) * 100, 2) if start_rate != 0 else 0
    )

    # 5. 최종 응답 데이터 구성
    chart_data = df.to_dict('records')
    
    return ExchangeRateResponse(
        start_date=df['date'].iloc[0],
        end_date=df['date'].iloc[-1],
        count=len(df),
        statistics=stats,
        data=chart_data
    )

# --- 6. 서버 실행 (터미널에서 직접 실행) ---
# 예: uvicorn app.rate_graph_app:app --reload