import os
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

# --- 설정 ---
FILE_PATH = 'data/train/only_US_KOR_20100104_20250812_simple.csv'
MODEL_A_DIR = 'models/prophet_1day_model/'
MODEL_B_DIR = 'models/prophet_multi_day_model/'
MODEL_A_PATH = 'prophet_1day_model.json'    # 1일 예측용 (다변량)
MODEL_B_PATH = 'prophet_multi_day_model.json' # 2일 이상 예측용 (단변량)
MODEL_A_SAVE_PATH = os.path.join(MODEL_A_DIR, MODEL_A_PATH)
MODEL_B_SAVE_PATH = os.path.join(MODEL_B_DIR, MODEL_B_PATH)

print("--- 하이브리드 모델 학습 시작 ---")

# 1. 데이터 로드 및 기본 전처리
df = pd.read_csv(FILE_PATH)
df['ds'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'target': 'y'})
df_clean = df.dropna()

# ======================================================================
# 모델 A 학습 (1일 예측 전문가 - 다변량)
# ======================================================================
print(f"\n[모델 A] 1일 예측용 다변량 모델을 학습합니다...")
df_model_A = df_clean[['ds', 'y', 'Inv_Close', 'ECOS_Close']]

model_A = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.1)
model_A.add_regressor('Inv_Close')
model_A.add_regressor('ECOS_Close')
model_A.fit(df_model_A)

# 모델 A 저장
if not os.path.exists(MODEL_A_DIR):
    os.makedirs(MODEL_A_DIR)
    print(f"'{MODEL_A_DIR}' 폴더를 생성했습니다.")
with open(MODEL_A_SAVE_PATH, 'w') as fout:
    fout.write(model_to_json(model_A))
print(f"-> 모델 A를 '{MODEL_A_SAVE_PATH}'에 저장했습니다.")


# ======================================================================
# 모델 B 학습 (2일 이상 예측 전문가 - 단변량)
# ======================================================================
print(f"\n[모델 B] 2일 이상 예측용 단변량 모델을 학습합니다...")
df_model_B = df_clean[['ds', 'y']]

model_B = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.1)
model_B.fit(df_model_B)

# 모델 B 저장
if not os.path.exists(MODEL_B_DIR):
    os.makedirs(MODEL_B_DIR)
    print(f"'{MODEL_B_DIR}' 폴더를 생성했습니다.")
with open(MODEL_B_SAVE_PATH, 'w') as fout:
    fout.write(model_to_json(model_B))
print(f"-> 모델 B를 '{MODEL_B_SAVE_PATH}'에 저장했습니다.")

print("\n--- 모든 모델의 학습 및 저장이 완료되었습니다. ---")