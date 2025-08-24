import pandas as pd
import io

file_name = "한국은행_환율_데이터_2025-07-01.xlsx"
date_column_name = 'Date'
rate_column_name = 'DATA_VALUE'

try:
    df = pd.read_excel(file_name)
    print("--- 원본 데이터 (일부) ---")
    print(df.head())
    print("\n원본 데이터 정보:")
    df.info()

    df[date_column_name] = pd.to_datetime(df[date_column_name], unit='D', origin='1899-12-30')
    df.set_index(date_column_name, inplace=True)

    start_date = df.index.min()
    end_date = df.index.max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    df_resampled = df.reindex(full_date_range)
    df_resampled.index.name = date_column_name
    df_filled = df_resampled.ffill()
    df_filled.info()

    processed_file_name = "data/" + file_name.split('/')[-1].split('.')[0] + "_filled.xlsx"
    df_filled.to_excel(processed_file_name)
    
    print(f"\n'{processed_file_name}' 파일로 저장되었습니다.")
except FileNotFoundError:
    print(f"오류: '{file_name}'을 찾을 수 없습니다. 파일명과 경로를 확인해주세요.")
except KeyError as e:
    print(f"오류: 열 이름 '{e}'을(를) 찾을 수 없습니다. date_column_name 또는 rate_column_name 변수를 확인해주세요.")