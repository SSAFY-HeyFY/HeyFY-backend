from fastapi import FastAPI
# app/routers 폴더에서 각 기능별로 만든 라우터 파일을 가져옵니다.
from app.routers import realtime_rates, rate_graph

# --- 1. FastAPI 앱 생성 ---
# API의 이름, 설명, 버전 등 기본 정보를 설정합니다.
app = FastAPI(
    title="📈 환율 정보 및 예측 API",
    description="실시간 환율 정보와 과거 데이터 기반 AI 예측을 제공하는 API입니다.",
    version="1.2.0"
)

# --- 2. 라우터 포함 ---
# 기능별로 분리된 라우터를 메인 앱(app)에 연결합니다.
# 이렇게 하면 코드를 기능 단위로 깔끔하게 관리할 수 있습니다.

# (A) 실시간 환율 API 라우터 포함
app.include_router(
    realtime_rates.router,
    prefix="/api/v1",  # 이 라우터의 모든 API는 /api/v1/ 로 시작됩니다. (예: /api/v1/realtime-rates)
    tags=["실시간 환율 (Real-time Rates)"] # API 문서에서 보여줄 그룹 이름입니다.
)

# (B) 그래프용 데이터 API 라우터 포함
app.include_router(
    rate_graph.router,
    prefix="/api/v1", # 이 라우터의 모든 API도 /api/v1/ 로 시작됩니다. (예: /api/v1/rate-graph)
    tags=["환율 그래프 (Rate Graph with AI Prediction)"]
)

# --- 3. 루트 엔드포인트 ---
# API 서버가 살아있는지 간단히 확인할 수 있는 경로입니다.
# 브라우저에서 http://127.0.0.1:8000/ 로 접속하면 이 메시지가 보입니다.
@app.get("/", summary="API 상태 확인")
def read_root():
    """
    API 서버의 상태를 확인하는 기본 엔드포인트입니다.
    """
    return {"status": "ok", "message": "환율 정보 API 서버가 정상적으로 동작하고 있습니다."}

# --- 4. 서버 실행 방법 ---
# 프로젝트 최상위 폴더의 터미널에서 아래 명령어로 서버를 실행합니다.
# uvicorn app.main:app --reload --port 8000
