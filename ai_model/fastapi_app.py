import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Shinhan FX Scraper")

# 안드로이드 앱 호출을 위해 CORS 허용(필요시 도메인 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# 네이버 환율(신한) 상세 페이지
URLS = {
    "USDKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW_SHB",
    "CNYKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_CNYKRW_SHB",
    "EURKRW": "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_EURKRW_SHB",    
}

HEADERS = {
    # 간단한 UA 지정(차단 방지용)
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

def fetch_tables(url: str) -> list[dict]:
    try:
        # 인코딩 옵션은 requests에서 처리하고, read_html은 문자열을 파싱
        html = requests.get(url, headers=HEADERS, timeout=10).text
        # 테이블 전부 파싱
        tables = pd.read_html(html)
        # 각 테이블을 dict 목록(레코드)으로 변환
        out = []
        for i, df in enumerate(tables):
            # 컬럼이 멀티인덱스면 간단히 문자열로 합침
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join(map(str, col)).strip() for col in df.columns.values]
            out.append({
                "index": i,
                "columns": list(map(str, df.columns)),
                "records": df.fillna("").to_dict(orient="records"),
            })
        return out
    except ValueError:
        # read_html이 테이블을 못 찾을 때
        return []
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fetch error: {e}")

@app.get("/rates")
def get_all_rates():
    """
    USDKRW, EURKRW, JPYKRW 모두 반환
    """
    data = {}
    for pair, url in URLS.items():
        tables = fetch_tables(url)
        data[pair] = {
            "pair": pair,
            "source": url,
            "tables": tables,
        }
    return {"bank": "Shinhan", "provider": "Naver Finance", "data": data}

@app.get("/rate/{pair}")
def get_rate(pair: str):
    """
    개별 통화쌍만 반환: USDKRW, EURKRW, JPYKRW
    """
    pair = pair.upper()
    if pair not in URLS:
        raise HTTPException(status_code=404, detail="Supported pairs: USDKRW, EURKRW, JPYKRW")
    url = URLS[pair]
    tables = fetch_tables(url)
    return {"bank": "Shinhan", "provider": "Naver Finance", "pair": pair, "source": url, "tables": tables}

# uvicorn app:app --reload
