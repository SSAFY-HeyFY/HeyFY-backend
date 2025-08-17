import yfinance as yf
import pandas as pd

def calculate_rsi(data: pd.Series, length: int = 14) -> pd.Series:
    """
    pandas_ta 라이브러리 없이 상대강도지수(RSI)를 계산합니다.
    
    :param data: 종가(Close) 등이 포함된 pandas Series
    :param length: RSI 계산 기간 (일반적으로 14)
    :return: RSI 값이 포함된 pandas Series
    """
    # 1. 가격 변화량 계산
    delta = data.diff()

    # 2. 상승분과 하락분 분리
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # 3. 지수이동평균(EMA)을 사용하여 평균 상승분과 하락분 계산
    # Wilder's Smoothing 방식(com=length-1)은 RSI 표준 계산법입니다.
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()

    # 4. 상대강도(RS) 및 RSI 계산
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# --- 데이터 설정 ---
# 야후 파이낸스에서 사용하는 Ticker를 입력합니다.
# - 원/달러 환율: 'KRW=X'
# - 삼성전자: '005930.KS'
# - S&P 500 지수: '^GSPC'
ticker_symbol = 'KRW=X'

# 데이터를 가져올 기간 설정 (10년 이상)
start_date = '2010-01-01'
# yfinance는 end_date를 포함하지 않으므로 하루 뒤 날짜로 설정
end_date = '2025-08-16' 

# --- 데이터 다운로드 ---
print(f"'{ticker_symbol}' 데이터를 다운로드합니다. 기간: {start_date} ~ {end_date}")
df = yf.download(ticker_symbol, start=start_date, end=end_date)

if df.empty:
    print("데이터를 다운로드하지 못했습니다. Ticker나 기간을 확인해주세요.")
else:
    print("✅ 데이터 다운로드 완료!")
    print("--- 원본 데이터 샘플 (최근 5일) ---")
    print(df.tail())

    # --- 기술적 지표 추가 (pandas_ta 없이) ---
    print("\n기술적 지표를 추가합니다...")
    
    # 1. 이동평균 (Simple Moving Average) - pandas 내장 함수 사용
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    # 2. 상대강도지수 (Relative Strength Index) - 직접 구현한 함수 사용
    df['RSI_14'] = calculate_rsi(df['Close'], length=14)
    
    # 3. 거래량 이동평균
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()

    # 결측치(NaN)가 있는 초기 데이터 제거
    df.dropna(inplace=True)
    
    print("✅ 기술적 지표 추가 완료!")
    print("\n--- 최종 데이터 샘플 (지표 추가 후) ---")
    
    # 'Adj Close'는 보통 수정 종가로, Close와 같을 경우 중복되므로 삭제
    if 'Adj Close' in df.columns:
        final_df = df.drop(columns=['Adj Close'])
    else:
        final_df = df.copy()

    print(final_df.tail())

    # --- 파일로 저장 ---
    # data 폴더가 없으면 생성
    import os
    if not os.path.exists('data'):
        os.makedirs('data')

    output_filename = f'data/{ticker_symbol}_data_with_indicators.xlsx'
    final_df.to_excel(output_filename)
    print(f"\n✅ 최종 데이터가 '{output_filename}' 파일로 저장되었습니다.")