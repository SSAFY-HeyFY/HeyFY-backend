import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional  # Optional을 임포트합니다.
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic 모델 정의 ---
class ChartDataPoint(BaseModel):
    date: str
    rate: float
    is_prediction: bool    
    model_name: Optional[str] = None # model_name 추가: 예측 데이터가 아닐 수도 있으므로 Optional로 설정

class RateGraphResponse(BaseModel):
    api_called_at: str
    data: List[ChartDataPoint]

# --- 라우터 생성 ---
router = APIRouter()
CACHE_BASE_PATH = os.getenv('CACHE_DIR', './logs')
PREDICTION_CACHE_FILE = os.path.join(CACHE_BASE_PATH, 'prediction_cache.json')

@router.get(
    "/rate-graph",
    response_model=RateGraphResponse,
    summary="환율 그래프 데이터 조회 (캐시 기반, 모델 정보 포함)",
    description="스케줄러가 생성한 캐시 파일에서 과거 및 모델별 예측 데이터를 읽어 그래프용으로 제공합니다."
)
def get_rate_graph_data_from_cache():
    """
    캐시 파일('prediction_cache.json')을 읽어, 모델 정보가 포함된
    환율 그래프 데이터를 반환합니다.
    """
    if not os.path.exists(PREDICTION_CACHE_FILE):
        raise HTTPException(
            status_code=404, 
            detail=f"'{PREDICTION_CACHE_FILE}'을 찾을 수 없습니다. AI 예측 스케줄러가 아직 실행되지 않았을 수 있습니다."
        )

    try:
        with open(PREDICTION_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data_list = json.load(f).get('predictions', [])
        
        api_call_time_iso = datetime.now().isoformat()

        # Pydantic 모델이 `model_name`을 포함하여 데이터 유효성을 검증합니다.
        validated_data = [ChartDataPoint(**item) for item in cache_data_list]
        
        return RateGraphResponse(
            api_called_at=api_call_time_iso,
            data=validated_data
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="캐시 파일을 파싱하는 데 실패했습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 처리 중 오류 발생: {e}")