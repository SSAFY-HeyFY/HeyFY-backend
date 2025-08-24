import numpy as np
import pandas as pd
import FinanceDataReader as fdr

start_date = "2025-07-01"

# 기간을 지정하지 않으면 1999-05-06 ~ 현재
df = fdr.DataReader('ECOS-KEYSTAT:K152', start_date)
# df = fdr.SnapDataReader("ECOS/SNAP/529")
print(df)

excel_filename = "한국은행_환율_데이터_" + start_date +".xlsx"
df.to_excel(excel_filename)
print(f"   - ✅ 불러온 데이터를 '{excel_filename}' 파일로 저장했습니다.")