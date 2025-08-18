import requests
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------
# 기본 설정
# ---------------------------
APP_NAME = "Naver Finance Shinhan FX API"
HEADERS = {"User-Agent": "Mozilla/5.0"}

URLS = {
    "USDKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW_SHB",
    "EURKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_EURKRW_SHB",
    "CNYKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_CNYKRW_SHB",
}

# ---------------------------
# 아이콘 숫자 복원 매핑
# ---------------------------
ICON_MAP = {
    "no0": "0", "no1": "1", "no2": "2", "no3": "3", "no4": "4",
    "no5": "5", "no6": "6", "no7": "7", "no8": "8", "no9": "9",
    "jum": ".", "dash": "-", "minus": "-", "cm": ",", "comma": ",", "per": "%",
}

def _icon_text_to_str(container) -> str:
    """<span class="no7"> 같은 아이콘 스팬들을 문자열로 재조립."""
    out = []
    for sp in container.find_all("span", recursive=True):
        for cls in sp.get("class", []) or []:
            if cls in ICON_MAP:
                out.append(ICON_MAP[cls])
                break
    return "".join(out)

def _to_float(s: str) -> float:
    """'1,234.56%' → 1234.56 (float), 실패 시 0.0"""
    if not s:
        return 0.0
    s = s.replace(",", "").replace("%", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return 0.0

def fetch_parsed(url: str) -> dict:
    """네이버 환율 상세 페이지 파싱 → float 값으로 반환."""
    try:
        html = requests.get(url, headers=HEADERS, timeout=10).text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {e}")

    try:
        tables = pd.read_html(html)
    except ValueError:
        tables = []

    # (A) 단위별 금액: 보통 tables[0]에 '1' 컬럼이 있음
    unit_price_1 = 0.0
    if tables:
        df0 = tables[0].copy()
        if isinstance(df0.columns, pd.MultiIndex):
            df0.columns = [" ".join(map(str, c)).strip() for c in df0.columns]
        if "1" in df0.columns and not df0.empty:
            try:
                unit_price_1 = round(float(str(df0.iloc[0]["1"]).replace(",", "")), 2)
            except Exception:
                unit_price_1 = 0.0

    # (B) '구분/환율' 표에서 주요 요율
    cash_buy = cash_sell = wire_send = wire_receive = 0.0
    for df in tables:
        cols = list(map(str, df.columns))
        if "구분" in cols and "환율" in cols:
            for _, row in df.iterrows():
                g = str(row.get("구분", "")).strip()
                r = str(row.get("환율", "")).replace(",", "")
                try:
                    val = round(float(r), 2) if r and r != "-" else 0.0
                except ValueError:
                    val = 0.0
                if "현찰 사실때" in g:
                    cash_buy = val
                elif "현찰 파실때" in g:
                    cash_sell = val
                elif "송금 보내실때" in g:
                    wire_send = val
                elif "송금 받으실때" in g:
                    wire_receive = val
            break

    # (C) 전일대비(아이콘/글리프)
    soup = BeautifulSoup(html, "lxml")
    change_dir = ""
    change_abs = 0.0
    change_pct = 0.0

    box = soup.select_one("p.no_exday")
    if box:
        # ▲/▼ 아이콘
        ico = box.select_one("span.ico")
        if ico:
            cls = ico.get("class", [])
            if any(c == "up" for c in cls):
                change_dir = "▲"
            elif any(c == "down" for c in cls):
                change_dir = "▼"

        # 숫자 블록(<em>) 2개: 절대값, 퍼센트
        ems = box.find_all("em")
        if ems:
            change_abs = _to_float(_icon_text_to_str(ems[0]))
        if len(ems) >= 2:
            change_pct = _to_float(_icon_text_to_str(ems[1]))

    return {
        "unit_price_1": unit_price_1,      # 단위 1 (예: 1달러) 기준 원화 환율
        "cash_buy": cash_buy,              # 현찰 사실 때 (고객이 은행에서 살 때)
        "cash_sell": cash_sell,            # 현찰 파실 때 (고객이 은행에 팔 때)
        "wire_send": wire_send,            # 송금 보내실 때 (은행이 해외로 송금 보낼 때)
        "wire_receive": wire_receive,      # 송금 받으실 때 (해외에서 송금 받을 때)
        "change_direction": change_dir,    # 전일대비 방향 (▲ 상승 / ▼ 하락 / "" 변동 없음)
        "change_abs": change_abs,          # 전일대비 절대 금액 변화 (원)
        "change_pct": change_pct,          # 전일대비 퍼센트 변화 (%)
    }

# ---------------------------
# FastAPI 앱
# ---------------------------
app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 안드로이드/웹 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": APP_NAME}

@app.get("/rate/{pair}")
def get_rate(
    pair: str,
    # 강제 리프레시가 필요할 때(향후 캐시 적용 대비): /rate/USDKRW?refresh=1
    refresh: int = Query(0, ge=0, le=1)
):
    key = pair.upper()
    if key not in URLS:
        raise HTTPException(status_code=404, detail=f"Supported pairs: {', '.join(URLS.keys())}")
    url = URLS[key]
    data = fetch_parsed(url)
    return {"pair": key, "source": url, "data": data}

@app.get("/rates")
def get_all_rates():
    out = {}
    for pair, url in URLS.items():
        out[pair] = fetch_parsed(url)
    return {"data": out}

# 실행: uvicorn fastapi_app:app --reload --port 8000
# 전체: http://localhost:8000/rates
# 개별: http://localhost:8000/rate/USDKRW
