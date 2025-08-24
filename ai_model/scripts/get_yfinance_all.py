import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

ticker_symbol = 'KRW=X'
start_date = (datetime.now() - timedelta(30 + 150)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# --- 데이터 다운로드 ---
print(f"'{ticker_symbol}' 데이터를 다운로드합니다. 기간: {start_date} ~ {end_date}")
df = yf.download(ticker_symbol, start=start_date, end=end_date)

if df.empty:
    print("데이터를 다운로드하지 못했습니다. Ticker나 기간을 확인해주세요.")
else:
    print("✅ 데이터 다운로드 완료!")
    print("--- 원본 데이터 샘플 (최근 5일) ---")
    print(df.tail())

    print(df.columns)
    df.columns = df.columns.droplevel('Ticker')
    df_close = df[['Close']]
    df_close.columns.name = None
    print("--- 파싱 데이터 샘플 (최근 5일) ---")
    print(df_close.tail())

    # --- 파일로 저장 ---
    # data 폴더가 없으면 생성
    # import os
    # if not os.path.exists('data'):
    #     os.makedirs('data')

    # output_filename = f'data/{ticker_symbol}_data_with_indicators.xlsx'
    # final_df.to_excel(output_filename)
    # print(f"\n✅ 최종 데이터가 '{output_filename}' 파일로 저장되었습니다.")