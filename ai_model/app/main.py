from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.routers import realtime_rates, rate_graph_ai, rate_analysis

# --- 1. FastAPI 앱 생성 ---
# API의 이름, 설명, 버전 등 기본 정보를 설정합니다.
app = FastAPI(
    title="📈 환율 정보 및 예측 API",
    description="실시간 환율 정보와 과거 데이터 기반 AI 예측을 제공하는 API입니다.",
    version="2.0.0"
)

# --- 2. 라우터 포함 ---
# 기능별로 분리된 라우터를 메인 앱(app)에 연결합니다.
# 이렇게 하면 코드를 기능 단위로 깔끔하게 관리할 수 있습니다.

# (A) 실시간 주요 환율 API (화면 ①번) (예: /api/realtime-rates)
app.include_router(
    realtime_rates.router,
    prefix="/api",
    tags=["① 실시간 주요 환율 (Real-time Rates)"]
)

# (B) 환율 그래프 데이터 API (화면 ②번) (예: /api/rate-graph)
app.include_router(
    rate_graph_ai.router,
    prefix="/api",
    tags=["② 환율 그래프 데이터 (Graph Data)"]
)
# (C) AI 분석 및 예측 문구 API (화면 ③번) (예: /api/rate-analysis)
app.include_router(
    rate_analysis.router,
    prefix="/api",
    tags=["③ AI 분석 및 예측 문구 (Analysis)"]
)

# --- 3. 루트 엔드포인트 ---
# API 서버가 살아있는지 간단히 확인할 수 있는 경로입니다.
# 브라우저에서 http://127.0.0.1:8000/ 로 접속하면 이 메시지가 보입니다.
# --- 3. 루트 엔드포인트 ---
@app.get("/", summary="API 상태 확인", include_in_schema=False)
def read_root():
    return {"status": "ok", "message": "환율 정보 API 서버가 정상적으로 동작하고 있습니다."}

# [추가됨] /api 루트 엔드포인트
@app.get("/api", summary="사용 가능한 API 목록 안내")
def read_api_root():
    """
    /api 경로로 접속 시, 사용 가능한 API의 주요 기능 그룹(태그) 목록을 안내합니다.
    """
    available_apis = [
        {"tag": "실시간 주요 환율 (Real-time Rates)", "path": "/api/realtime-rates"},
        {"tag": "환율 그래프 데이터 (Graph Data)", "path": "/api/rate-graph"},
        {"tag": "AI 분석 및 예측 문구 (Analysis)", "path": "/api/rate-analysis"},
        {"tag": "API Documentation", "path": "/docs"}
    ]
    
    return JSONResponse(content={
        "message": "환율 정보 API 입니다. 가능한 엔드포인트 목록을 확인하세요.",
        "available_endpoints": available_apis
    })

# --- 4. 서버 실행 방법 ---
# 프로젝트 최상위 폴더의 터미널에서 아래 명령어로 서버를 실행합니다.
# uvicorn app.main:app --reload --port 8888
