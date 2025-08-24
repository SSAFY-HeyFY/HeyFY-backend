import pandas as pd

INPUT_FILE = 'data/merged_USDKRW_달러_지수_US10Y_final.xlsx'
OUTPUT_FILE = 'data/merged_USDKRW_달러_지수_US10Y_final_with_onehot.xlsx'

DATE_COLUMN = 'Date'

def add_day_of_week_onehot(input_filepath, output_filepath, date_col):
    """
    데이터 파일을 읽어 날짜(Date) 컬럼을 기반으로 
    요일 원-핫 인코딩 컬럼을 추가하고 새 파일로 저장합니다.
    """
    try:
        print(f"'{input_filepath}' 파일을 로드합니다...")
        df = pd.read_excel(input_filepath)
        print("파일 로드 완료.")

        if date_col not in df.columns:
            print(f"[오류] 데이터에서 '{date_col}' 컬럼을 찾을 수 없습니다.")
            print(f"사용 가능한 컬럼: {df.columns.tolist()}")
            return None
            
        # 1. 날짜 컬럼을 datetime 형식으로 변환 (에러 발생 시 해당 행은 NaT로)
        df[date_col] = pd.to_datetime(df[date_col], unit='D', origin='1899-12-30')
        
        # 날짜 변환에 실패한 행이 있다면 제거
        original_rows = len(df)
        df.dropna(subset=[date_col], inplace=True)
        if len(df) < original_rows:
            print(f"{original_rows - len(df)}개의 행에서 날짜 변환에 실패하여 제거했습니다.")

        # 2. 요일 이름 구하기 (월요일=Monday, 화요일=Tuesday...)
        day_names = df[date_col].dt.day_name()
        print("요일 정보 추출 완료.")

        # 3. 요일 이름으로 원-핫 인코딩 컬럼 생성 (get_dummies)
        day_dummies = pd.get_dummies(day_names)
        
        # 4. 모델 학습에 필요한 컬럼명 형식으로 변경 ('is_Mon'...)
        required_days = {
            'Monday': 'is_Mon', 
            'Tuesday': 'is_Tue', 
            'Wednesday': 'is_Wed', 
            'Thursday': 'is_Thu', 
            'Friday': 'is_Fri'
        }
        
        # 주말(Saturday, Sunday)은 필요 없으므로 제외하고, 필요한 요일만 추가
        for day_eng, col_name in required_days.items():
            if day_eng in day_dummies.columns:
                df[col_name] = day_dummies[day_eng]
            else:
                # 데이터에 해당 요일이 없는 경우 0으로 채움
                df[col_name] = 0
        print("원-핫 인코딩 컬럼 추가 완료.")
        
        # 5. 새로운 CSV 파일로 저장
        df.to_excel(output_filepath, index=False, float_format='%.4f')
        print(f"\n✅ 작업 완료! 요일 정보가 추가된 파일이 '{output_filepath}'로 저장되었습니다.")
        
        return df

    except FileNotFoundError:
        print(f"[오류] 입력 파일 '{input_filepath}'을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"처리 중 오류가 발생했습니다: {e}")
        return None


if __name__ == "__main__":
    final_df = add_day_of_week_onehot(INPUT_FILE, OUTPUT_FILE, DATE_COLUMN)
    
    if final_df is not None:
        print("\n--- 최종 데이터 미리보기 (상위 5개 행) ---")
        # 추가된 요일 컬럼들을 잘 보이도록 뒤쪽 컬럼 위주로 출력
        print(final_df.tail().iloc[:, -7:])