import asyncio
import re
import warnings
from datetime import datetime
from typing import List, Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# --- Pydantic 모델 정의 ---
# 크롤링 결과의 데이터 구조를 명확하게 정의합니다.
class ExchangeRateDetail(BaseModel):
    updated_at: datetime
    currency_code: str
    provider: str
    source_url: str    
    rate: float  # 매매 기준율 또는 현재가
    cash_buy: Optional[float] = None
    cash_sell: Optional[float] = None
    wire_send: Optional[float] = None
    wire_receive: Optional[float] = None
    change_direction: Optional[str] = None
    change_abs: Optional[float] = None
    change_pct: Optional[float] = None

# --- 설정 ---
HEADERS = {"User-Agent": "Mozilla/5.0"}
NAVER_URLS = {
    "USDKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW_SHB",
    "CNYKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_CNYKRW_SHB",
}
GOOGLE_URLS = {
    "VNDKRW": "https://www.google.com/finance/quote/VND-KRW"
}

# --- 유틸리티 함수 (기존 코드 재사용) ---
ICON_MAP = {
    "no0": "0", "no1": "1", "no2": "2", "no3": "3", "no4": "4",
    "no5": "5", "no6": "6", "no7": "7", "no8": "8", "no9": "9",
    "jum": ".", "dash": "-", "minus": "-", "cm": ",", "comma": ",", "per": "%",
}

def _icon_text_to_str(container) -> str:
    out = [ICON_MAP[cls] for sp in container.find_all("span", recursive=True) for cls in sp.get("class", []) if cls in ICON_MAP]
    return "".join(out)

def _to_float(s: str) -> float:
    if not s: return 0.0
    try:
        return round(float(s.replace(",", "").replace("%", "")), 6)
    except (ValueError, TypeError):
        return 0.0

def _extract_number(text: str) -> float:
    if not text: return 0.0
    m = re.search(r"[+\-]?\d[\d,]*\.?\d*", text)
    if not m: return 0.0
    return _to_float(m.group())

# --- 네이버 금융 파서 (비동기) ---
async def parse_naver_finance(client: httpx.AsyncClient, code: str) -> Optional[ExchangeRateDetail]:
    url = NAVER_URLS[code]
    try:
        response = await client.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "lxml")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            tables = pd.read_html(html)

        # 매매기준율 (unit_price_1)
        rate = _to_float(str(tables[0].iloc[0]["1"]))

        # 현찰/송금 정보
        cash_buy, cash_sell, wire_send, wire_receive = 0.0, 0.0, 0.0, 0.0
        df_rates = next((df for df in tables if "구분" in df.columns and "환율" in df.columns), None)
        if df_rates is not None:
            for _, row in df_rates.iterrows():
                g, r = str(row.get("구분", "")), str(row.get("환율", ""))
                if "현찰 사실때" in g: cash_buy = _to_float(r)
                elif "현찰 파실때" in g: cash_sell = _to_float(r)
                elif "송금 보내실때" in g: wire_send = _to_float(r)
                elif "송금 받으실때" in g: wire_receive = _to_float(r)

        # 전일 대비 정보
        change_dir, change_abs, change_pct = "", 0.0, 0.0
        box = soup.select_one("p.no_exday")
        if box:
            if "up" in (box.select_one("span.ico") or {}).get("class", []): change_dir = "▲"
            elif "down" in (box.select_one("span.ico") or {}).get("class", []): change_dir = "▼"
            ems = box.find_all("em")
            if ems: change_abs = _to_float(_icon_text_to_str(ems[0]))
            if len(ems) >= 2: change_pct = _to_float(_icon_text_to_str(ems[1]))

        return ExchangeRateDetail(
            updated_at=datetime.now(), currency_code=code, provider="Naver-Shinhan", source_url=url,
            rate=f"{rate:.2f}", cash_buy=f"{cash_buy:.2f}", cash_sell=f"{cash_sell:.2f}",
            wire_send=f"{wire_send:.2f}", wire_receive=f"{wire_receive:.2f}",
            change_direction=change_dir, change_abs=f"{change_abs:.2f}", change_pct=f"{change_pct:.2f}"
            )
    except Exception as e:
        print(f"[{code}] 네이버 파싱 실패: {e}")
        return None

# --- 구글 금융 파서 (동기, Selenium 사용) ---
def parse_google_finance_vnd() -> Optional[ExchangeRateDetail]:
    url = GOOGLE_URLS["VNDKRW"]
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.YMlKec.fxKbKc")))
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        rate = _extract_number(soup.select_one('div.YMlKec.fxKbKc').get_text(strip=True))
        
        change_dir, change_abs, change_pct = "", 0.0, 0.0
        change_container = soup.select_one('div[jsname="CGyduf"]')
        if change_container:
            pct_span = change_container.select_one('span[jsname="Fe7oBc"]')
            if pct_span:
                aria = (pct_span.get("aria-label") or "").lower()
                if "상승" in aria or "up" in aria: change_dir = "▲"
                elif "하락" in aria or "down" in aria: change_dir = "▼"
                else: change_dir = "-"
                change_pct = _extract_number(pct_span.get_text(strip=True))
            
            abs_span = change_container.select_one('span.P2Luy')
            if abs_span:
                change_abs = _extract_number(abs_span.get_text(strip=True))
            
            if change_dir == "▼":
                if change_pct > 0: change_pct *= -1
                if change_abs > 0: change_abs *= -1

        return ExchangeRateDetail(
            updated_at=datetime.now(), currency_code="VNDKRW", provider="Google Finance", source_url=url, 
            rate=f"{rate:g}", change_direction=change_dir, change_abs=f"{change_abs:.6f}", change_pct=f"{change_pct:g}"
        )
    except Exception as e:
        print(f"[VNDKRW] 구글 파싱 실패: {e}")
        return None
    finally:
        driver.quit()

# --- 메인 크롤링 함수 (비동기 오케스트레이터) ---
async def get_detailed_exchange_rates() -> List[ExchangeRateDetail]:
    """정의된 모든 통화에 대해 상세 환율 정보를 비동기적으로 가져옵니다."""
    async with httpx.AsyncClient() as client:
        # 비동기 작업(네이버)과 동기 작업(구글/Selenium)을 분리하여 태스크 생성
        naver_tasks = [parse_naver_finance(client, code) for code in NAVER_URLS]
        
        # 동기 함수인 Selenium 파서를 비동기 이벤트 루프에서 블로킹 없이 실행
        loop = asyncio.get_running_loop()
        google_task = loop.run_in_executor(None, parse_google_finance_vnd)

        # 모든 작업을 동시에 실행하고 결과 수집
        all_tasks = naver_tasks + [google_task]
        results = await asyncio.gather(*all_tasks)
        
        return [res for res in results if res is not None]

# --- 직접 실행 테스트 ---
if __name__ == "__main__":
    async def main():
        rates = await get_detailed_exchange_rates()
        for rate_info in rates:
            print(f"통화: {rate_info.currency_code}, 기준가: {rate_info.rate:,.4f}원, 전일대비: {rate_info.change_direction}{rate_info.change_abs} 등락률: {rate_info.change_pct}")

    asyncio.run(main())
