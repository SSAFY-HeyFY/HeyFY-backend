from fastapi import APIRouter
from typing import List
# 1단계에서 리팩토링한 크롤링 서비스와 데이터 모델을 가져옵니다.
from app.services.exchange_rate_crawler import get_detailed_exchange_rates, ExchangeRateDetail

# --- 라우터 생성 ---
# 이 파일의 API 엔드포인트들을 그룹화합니다.
router = APIRouter()

# --- API 엔드포인트 정의 ---
@router.get(
    "/realtime-rates",
    response_model=List[ExchangeRateDetail],
    summary="실시간 상세 환율 조회",
    description="네이버 금융과 구글 금융에서 USD, CNY, VND의 현재 원화 환율 및 상세 고시 정보를 실시간으로 크롤링하여 제공합니다."
)
async def get_current_detailed_rates():
    """
    정의된 모든 통화(USD, CNY, VND)의 상세 환율 정보를 비동기적으로 가져옵니다.
    - USD, CNY: 네이버 금융 (신한은행 고시 기준)
    - VND: 구글 금융
    """
    detailed_rates = await get_detailed_exchange_rates()
    return detailed_rates
