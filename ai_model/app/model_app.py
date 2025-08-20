import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta

# --- 1. 설정 (Configuration) ---
class Config:
    # 학습 시 사용했던 설정과 동일하게 맞춰야 합니다.
    SEQUENCE_LENGTH = 90
    INPUT_SIZE = 12  # feature_columns의 개수
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1
    
    # 모델 및 스케일러 경로
    MODEL_PATH = 'models/seq_90-hidden_128-layers_2-batch_16/best_model.pth'
    FEATURE_SCALER_PATH = 'scalers/feature_scaler.pkl'
    TARGET_SCALER_PATH = 'scalers/target_scaler.pkl'

    # 추천 로직 설정
    RECOMMENDATION_DAYS = 5  # 며칠 앞까지 예측할지
    MIN_RISE_THRESHOLD = 0.007 # 추천을 위한 최소 상승률 (0.7%)
    BASE_EXCHANGE_AMOUNT_USD = 1000 # 추천 문구의 기준 환전 금액

# --- 2. PyTorch 모델 클래스 정의 ---
# 저장된 모델을 로드하려면, 모델의 구조(클래스)를 알고 있어야 합니다.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 3. FastAPI 앱 및 모델/스케일러 로딩 ---
app = FastAPI(title="환율 예측 및 환전 타이밍 추천 API")

# 앱 시작 시 모델과 스케일러를 한번만 로드합니다.
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(Config.INPUT_SIZE, Config.HIDDEN_SIZE, Config.NUM_LAYERS, Config.OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    feature_scaler = joblib.load(Config.FEATURE_SCALER_PATH)
    target_scaler = joblib.load(Config.TARGET_SCALER_PATH)
    print("✅ 모델과 스케일러 로딩 완료.")
except Exception as e:
    print(f"🚨 모델 또는 스케일러 로딩 실패: {e}")
    model = None # 로딩 실패 시 None으로 설정

# --- 4. 시뮬레이션용 데이터 수집 함수 (실제로는 API 호출로 대체) ---
def get_live_market_data():
    """실시간 시장 데이터를 가져오는 것을 시뮬레이션합니다."""
    # 실제 서비스에서는 이 부분을 금융 데이터 API 호출 코드로 바꿔야 합니다.
    print("(시뮬레이션) 실시간 시장 데이터 수집 중...")
    return {
        "USD_KRW": 1385.50,  # 현재 실시간 NDF 환율
        "달러지수": 105.8,
        "달러지수선물": 105.9,
        "미_10년물": 4.25
    }

def get_recent_history_data():
    """모델 입력에 필요한 최근 과거 데이터를 가져오는 것을 시뮬레이션합니다."""
    # 실제로는 DB나 파일에서 최근 (SEQUENCE_LENGTH)일치 데이터를 읽어와야 합니다.
    print("(시뮬레이션) 최근 과거 데이터 수집 중...")
    # 아래는 예시 데이터이며, 실제로는 9개의 피처를 모두 포함해야 합니다.
    return pd.DataFrame({
        'USD_KRW': np.linspace(1350, 1380, Config.SEQUENCE_LENGTH),
        '달러지수': np.linspace(104, 105.5, Config.SEQUENCE_LENGTH),
        '달러지수선물': np.linspace(104.1, 105.6, Config.SEQUENCE_LENGTH),
        '미_10년물': np.linspace(4.1, 4.22, Config.SEQUENCE_LENGTH),
        'is_Mon': [0]*Config.SEQUENCE_LENGTH,
        'is_Tue': [0]*Config.SEQUENCE_LENGTH,
        'is_Wed': [1]*Config.SEQUENCE_LENGTH, # 예시로 수요일로 가정
        'is_Thu': [0]*Config.SEQUENCE_LENGTH,
        'is_Fri': [0]*Config.SEQUENCE_LENGTH,
    })

# --- 5. AI 모델 추론 파이프라인 ---
def predict_multi_day(live_data, history_df):
    """실시간 데이터를 바탕으로 N일 미래의 환율을 반복적으로 예측합니다."""
    if not model: return None
    
    print(f"{Config.RECOMMENDATION_DAYS}일 미래 예측 시작...")
    
    # 최근 과거 데이터와 실시간 데이터를 합쳐 초기 입력 시퀀스 생성
    live_df_row = pd.DataFrame([live_data])
    # 요일 정보 추가 (내일 기준)
    tomorrow = datetime.now() + timedelta(days=1)
    days = ['is_Mon', 'is_Tue', 'is_Wed', 'is_Thu', 'is_Fri']
    for i, day_col in enumerate(days):
        live_df_row[day_col] = 1 if i == tomorrow.weekday() else 0
        
    current_sequence_df = pd.concat([history_df.iloc[1:], live_df_row], ignore_index=True)
    
    future_predictions = []
    
    with torch.no_grad():
        for i in range(Config.RECOMMENDATION_DAYS):
            # 1. 입력 데이터 스케일링 및 텐서 변환
            scaled_sequence = feature_scaler.transform(current_sequence_df)
            input_tensor = torch.tensor([scaled_sequence], dtype=torch.float32).to(device)

            # 2. 모델 예측
            prediction_scaled = model(input_tensor)
            
            # 3. 예측 결과 스케일 복원
            prediction = target_scaler.inverse_transform(prediction_scaled.cpu().numpy())[0][0]
            future_predictions.append(prediction)
            
            # 4. 다음날 예측을 위한 입력 시퀀스 업데이트 (가상 데이터 생성)
            next_day_features = current_sequence_df.iloc[-1].copy()
            next_day_features['USD_KRW'] = prediction # 예측값을 다음날 NDF 값으로 가정
            
            # 다음날 요일 업데이트
            next_day = tomorrow + timedelta(days=i+1)
            for j, day_col in enumerate(days):
                next_day_features[day_col] = 1 if j == next_day.weekday() else 0

            # 가장 오래된 데이터를 버리고 새로운 예측 데이터를 추가
            current_sequence_df = pd.concat([current_sequence_df.iloc[1:], pd.DataFrame([next_day_features])], ignore_index=True)
            
    return future_predictions

# --- 6. 추천 로직 및 문구 생성 ---
def generate_recommendation(current_rate, predictions):
    """예측 결과를 바탕으로 조심스럽지만 유용한 추천 문구를 생성합니다."""
    max_rate = max(predictions)
    best_day = predictions.index(max_rate) + 1  # 1일 뒤, 2일 뒤...
    rise_percent = ((max_rate / current_rate) - 1)
    
    disclaimer = "\n\n(AI 예측은 참고 자료이며, 실제와 다를 수 있습니다.)"
    
    # 조건 1: 최소 상승률 이상으로 오를 것으로 예측될 때
    if rise_percent > Config.MIN_RISE_THRESHOLD:
        benefit = (max_rate - current_rate) * Config.BASE_EXCHANGE_AMOUNT_USD
        text = (f"{best_day}일 뒤 환율이 현재보다 약 {rise_percent:.2%} 상승할 것으로 AI가 예측했어요. "
                f"지금보다 약 {benefit:,.0f}원 더 이득을 볼 수 있어요. (1,000달러 환전 기준)")
    # 조건 2: 계속 하락할 것으로 예측될 때
    elif max_rate < current_rate:
        text = "AI가 향후 환율을 예측한 결과, 점진적인 하락이 예상돼요. 현재가 가장 유리한 환전 시점일 수 있어요."
    # 조건 3: 큰 변동이 없을 때
    else:
        text = "AI 예측 결과, 향후 며칠간 환율에 큰 변동은 없을 것 같아요. 서두르지 않고 편할 때 환전해도 괜찮아요."
        
    return {
        "recommendation_text": text + disclaimer,
        "current_rate": round(current_rate, 2),
        "predicted_peak_rate": round(max_rate, 2),
        "best_day_after": best_day,
        "forecast_rates": [round(p, 2) for p in predictions]
    }

# --- 7. FastAPI 엔드포인트 정의 ---
class RecommendationResponse(BaseModel):
    recommendation_text: str
    current_rate: float
    predicted_peak_rate: float
    best_day_after: int
    forecast_rates: list[float]

@app.get("/recommend/timing", response_model=RecommendationResponse)
async def get_timing_recommendation():
    if not model:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다. 서버 로그를 확인하세요.")
    
    # 1. 실시간 및 과거 데이터 수집
    live_data = get_live_market_data()
    history_df = get_recent_history_data()
    
    # 2. 다일 예측 실행
    predictions = predict_multi_day(live_data, history_df)
    if predictions is None:
        raise HTTPException(status_code=500, detail="모델 추론 중 오류가 발생했습니다.")
        
    # 3. 추천 문구 생성
    current_ndf_rate = live_data["USD_KRW"]
    recommendation = generate_recommendation(current_ndf_rate, predictions)
    
    return recommendation

# --- 8. 서버 실행 (터미널에서 직접 실행) ---
# 예: uvicorn app.model_app:app --reload