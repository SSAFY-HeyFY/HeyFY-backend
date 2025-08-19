import re
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
    "CNYKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_CNYKRW_SHB",
    "EURKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_EURKRW_SHB",    
}

GOOGLE_URLS = {
    "VNDKRW": "https://www.google.com/finance/quote/VND-KRW"
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
    
def _extract_number(text: str) -> float:
    if not text:
        return 0.0
    m = re.search(r"[+\-]?\d[\d,]*\.?\d*", text)
    if not m: return 0.0
    try:
        return round(float(m.group().replace(",", "")), 6)
    except ValueError:
        return 0.0

def fetch_parsed_naver(url: str) -> dict:
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

def parse_google_vnd(soup: BeautifulSoup) -> dict:
    """
    Google Finance VND/KRW 메인 카드를 대상으로 파싱 (수정된 버전)
    """
    # 메인 환율 카드 루트
    root = soup.select_one('div[jsname="OYCkv"]') or soup

    # 현재가 (기존 코드와 동일)
    price = 0.0
    price_node = root.select_one('div[jsname="ip75Cb"] .YMlKec')
    print("price_node", price_node)
    if price_node:
        price = _extract_number(price_node.get_text(strip=True))

    # --- 변동률 및 절대변동값 파싱 (개선된 부분) ---
    change_pct = 0.0
    change_abs = 0.0
    change_dir = ""

    # 1. 변동률(%)을 포함하는 기준 요소를 선택합니다.
    pct_wrap = root.select_one('span[jsname="Fe7oBc"]')
    print("pct_wrap", pct_wrap)
    if pct_wrap:
        # 2. aria-label을 이용해 상승/하락 방향을 결정합니다.
        aria = (pct_wrap.get("aria-label") or "").lower()
        if "상승" in aria or "up" in aria:
            change_dir = "▲"
        elif "하락" in aria or "down" in aria:
            change_dir = "▼"

        # 3. 변동률(%) 텍스트를 추출하고 숫자로 변환합니다.
        pct_text = pct_wrap.get_text(" ", strip=True)
        change_pct = _extract_number(pct_text)

        # 4. 변동률 요소의 '바로 다음 형제' span 태그를 찾아 절대변동값을 추출합니다.
        #    이 방법이 클래스 이름으로 찾는 것보다 구조적으로 더 안정적입니다.
        abs_span = pct_wrap.find_next_sibling('span')
        if abs_span:
            abs_text = abs_span.get_text(" ", strip=True)
            change_abs = _extract_number(abs_text)

        # 5. 방향(▼)에 따라 부호를 맞춰줍니다.
        if change_dir == "▼":
            if change_pct > 0: change_pct *= -1
            if change_abs > 0: change_abs *= -1

    return {
        "unit_price_1": round(price, 6),      # 1 VND당 원화
        "change_direction": change_dir,       # ▲ / ▼ / ""
        "change_abs": round(change_abs, 6),   # 절대변동 (원)
        "change_pct": round(change_pct, 2),   # 변동률 (%)
    }

def fetch_google_vnd(url: str) -> dict:
    """요청 → soup 생성 → 순수 파서 호출"""
    try:
        html = requests.get(url, headers=HEADERS, timeout=10).text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Google upstream fetch failed: {e}")
    soup = BeautifulSoup(html, "lxml")
    return parse_google_vnd(soup)

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
def get_rate(pair: str, refresh: int = Query(0, ge=0, le=1)):
    key = pair.upper()
    if key in URLS:
        url = URLS[key]
        data = fetch_parsed_naver(url)
        return {"pair": key, "source": url, "provider": "Naver-Shinhan", "data": data}
    if key in GOOGLE_URLS:
        url = GOOGLE_URLS[key]
        data = fetch_google_vnd(url)
        return {"pair": key, "source": url, "provider": "Google Finance", "data": data}
    raise HTTPException(status_code=404, detail=f"Supported pairs: {', '.join(list(URLS.keys()) + list(GOOGLE_URLS.keys()))}")

@app.get("/rates")
def get_all_rates():
    out = {}
    for pair, url in URLS.items():
        out[pair] = {"provider": "Naver-Shinhan", "source": url, "data": fetch_parsed_naver(url)}
    for pair, url in GOOGLE_URLS.items():
        out[pair] = {"provider": "Google Finance", "source": url, "data": fetch_google_vnd(url)}
    return {"data": out}

# 실행: uvicorn fastapi_app:app --reload --port 8000
# 전체: http://localhost:8000/rates
# 개별: http://localhost:8000/rate/USDKRW
