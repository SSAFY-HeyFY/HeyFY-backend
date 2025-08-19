import pandas as pd

shinhan_USDKRW_url = 'https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW_SHB'
shinhan_EURKRW_url = 'https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_EURKRW_SHB'
shinhan_JPYKRW_url = 'https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_JPYKRW'

exchange_lists = pd.read_html(shinhan_USDKRW_url, encoding='cp949')
#print(exchange_lists)

out = []
for i, df in enumerate(exchange_lists):
    # 컬럼이 멀티인덱스면 간단히 문자열로 합침
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(map(str, col)).strip() for col in df.columns.values]
    out.append({
        "index": i,
        "columns": list(map(str, df.columns)),
        "records": df.fillna("").to_dict(orient="records"),
    })

print(out)
