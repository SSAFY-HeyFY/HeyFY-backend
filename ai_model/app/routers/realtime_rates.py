import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# --- Pydantic 모델 정의 ---
class ExchangeRateDetail(BaseModel):
    updated_at: str         # 데이터 업데이트 시각 (API 호출 시점으로 덮어쓰기 됨)
    currency: str           # 통화 코드 (예: "USD")
    rate: str               # 현재 환율 (매매기준율)
    change_direction: str
    change_abs: str         # 전일 대비 변동액 (예: "10.50")
    change_pct: str         # 전일 대비 등락률 (예: "2.30")
    cash_buy: Optional[str] = None
    cash_sell: Optional[str] = None
    wire_send: Optional[str] = None
    wire_receive: Optional[str] = None
    provider: str           # 정보 제공처 (예: "Naver-Shinhan")

# [추가됨] API 최종 응답을 위한 Wrapper 모델
class RealtimeRatesResponse(BaseModel):
    api_called_at: str      # API가 호출된 시각
    data: List[ExchangeRateDetail]

# --- 라우터 생성 ---
router = APIRouter()
REALTIME_CACHE_FILE = "logs/realtime_cache.json"

@router.get(
    "/realtime-rates",
    # [수정됨] 응답 모델을 새로운 Wrapper 모델로 변경
    response_model=RealtimeRatesResponse,
    summary="주요 통화 실시간 시세 조회 (캐시 기반)",
    description="스케줄러가 주기적으로 크롤링하여 저장한 JSON 캐시 파일에서 환율 정보를 신속하게 조회합니다."
)
def get_current_detailed_rates_from_cache():
    """
    [수정된 로직]
    1. 미리 생성된 'realtime_cache.json' 파일을 읽습니다.
    2. API가 호출된 현재 시각을 기록합니다.
    3. 캐시 데이터의 updated_at을 현재 시각으로 덮어쓴 후, 전체를 Wrapper 모델에 담아 반환합니다.
    """
    if not os.path.exists(REALTIME_CACHE_FILE):
        raise HTTPException(
            status_code=404, 
            detail=f"'{REALTIME_CACHE_FILE}'을 찾을 수 없습니다. 실시간 환율 스케줄러가 아직 실행되지 않았을 수 있습니다."
        )

    try:
        with open(REALTIME_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data_list = json.load(f).get('data', [])
        
        # [추가됨] API 호출 시점의 타임스탬프 생성
        api_call_time = datetime.now()
        api_call_time_iso = api_call_time.isoformat()

        processed_data = []
        for item in cache_data_list:
            # 캐시의 타임스탬프 대신 API 호출 시점으로 덮어쓰기
            item['updated_at'] = api_call_time_iso
            processed_data.append(ExchangeRateDetail(**item))
        
        # 최종 응답 모델에 담아 반환
        return RealtimeRatesResponse(
            api_called_at=api_call_time_iso,
            data=processed_data
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="캐시 파일을 파싱하는 데 실패했습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 처리 중 오류 발생: {e}")
