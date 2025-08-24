import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# --- Pydantic 모델 정의 ---
# [수정됨] exchange_rate_crawler.py의 출력과 UI에 필요한 정보를 바탕으로 모델을 재정의합니다.
# 이 모델은 스케줄러가 생성하고 API가 반환할 최종 데이터 형식을 정의합니다.
class ExchangeRateDetail(BaseModel):
    updated_at: str         # 데이터 업데이트 시각 (ISO 8601 형식)
    currency: str      # 통화 코드 (예: "USD")
    rate: str             # 현재 환율 (매매기준율)
    change_direction: str
    change_abs: str       # 전일 대비 변동액 (예: 10.50)
    change_pct: str       # 전일 대비 등락률 (예: 2.3)
    cash_buy: Optional[str] = None
    cash_sell: Optional[str] = None
    wire_send: Optional[str] = None
    wire_receive: Optional[str] = None
    provider: str           # 정보 제공처 (예: "Naver-Shinhan")

# --- 라우터 생성 ---
router = APIRouter()
REALTIME_CACHE_FILE = "realtime_cache.json"

@router.get(
    "/realtime-rates",
    response_model=List[ExchangeRateDetail],
    summary="주요 통화 실시간 시세 조회 (캐시 기반)",
    description="스케줄러가 주기적으로 크롤링하여 저장한 JSON 캐시 파일에서 환율 정보를 신속하게 조회합니다."
)
def get_current_detailed_rates_from_cache():
    """
    [수정된 로직]
    실시간으로 크롤링하는 대신, 미리 생성된 'realtime_cache.json' 파일을 읽어 반환합니다.
    이를 통해 API 응답 속도를 크게 향상시킵니다.
    """
    if not os.path.exists(REALTIME_CACHE_FILE):
        raise HTTPException(
            status_code=404, 
            detail=f"'{REALTIME_CACHE_FILE}'을 찾을 수 없습니다. 실시간 환율 스케줄러가 아직 실행되지 않았을 수 있습니다."
        )

    try:
        with open(REALTIME_CACHE_FILE, 'r', encoding='utf-8') as f:
            # 캐시 파일의 최상위 키가 'data'일 것을 가정합니다.
            cache_data = json.load(f).get('data', [])
        
        # Pydantic 모델을 사용하여 데이터 유효성 검증 후 반환
        return [ExchangeRateDetail(**item) for item in cache_data]
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="캐시 파일을 파싱하는 데 실패했습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 처리 중 오류 발생: {e}")

