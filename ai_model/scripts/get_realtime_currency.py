import pandas as pd

shinhan_USDKRW_url = 'https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW_SHB'
shinhan_EURKRW_url = 'https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_EURKRW_SHB'
shinhan_JPYKRW_url = 'https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_JPYKRW'

exchange_lists = pd.read_html(shinhan_USDKRW_url, encoding='cp949')
print(exchange_lists)
