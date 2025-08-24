import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# --- Pydantic 모델 정의 ---
class ChartDataPoint(BaseModel):
    date: str
    rate: float
    is_prediction: bool

# [추가됨] API 최종 응답을 위한 Wrapper 모델
class RateGraphResponse(BaseModel):
    api_called_at: str      # API가 호출된 시각
    data: List[ChartDataPoint]

# --- 라우터 생성 ---
router = APIRouter()
PREDICTION_CACHE_FILE = "prediction_cache.json"

@router.get(
    "/rate-graph",
    # [수정됨] 응답 모델을 새로운 Wrapper 모델로 변경
    response_model=RateGraphResponse,
    summary="환율 그래프 데이터 조회 (캐시 기반)",
    description="스케줄러가 생성한 캐시 파일에서 과거 및 예측 데이터를 읽어 그래프용으로 제공합니다."
)
def get_rate_graph_data_from_cache():
    """
    [수정된 로직]
    1. 미리 생성된 'prediction_cache.json' 파일을 읽습니다.
    2. API가 호출된 현재 시각을 기록합니다.
    3. 전체 데이터를 Wrapper 모델에 담아 반환합니다.
    """
    if not os.path.exists(PREDICTION_CACHE_FILE):
        raise HTTPException(
            status_code=404, 
            detail=f"'{PREDICTION_CACHE_FILE}'을 찾을 수 없습니다. AI 예측 스케줄러가 아직 실행되지 않았을 수 있습니다."
        )

    try:
        with open(PREDICTION_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data_list = json.load(f).get('predictions', [])
        
        # API 호출 시점의 타임스탬프 생성
        api_call_time_iso = datetime.now().isoformat()

        # Pydantic 모델로 데이터 유효성 검증
        validated_data = [ChartDataPoint(**item) for item in cache_data_list]
        
        # 최종 응답 모델에 담아 반환
        return RateGraphResponse(
            api_called_at=api_call_time_iso,
            data=validated_data
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="캐시 파일을 파싱하는 데 실패했습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 처리 중 오류 발생: {e}")
