import os
import joblib
import json
import numpy as np
import pandas as pd
import torch

# 위에서 정의한 모델 구조를 가져옴
from model import LSTMModel

def returns_to_prices(base_prices, returns):
    """기준 가격과 수익률 시퀀스를 바탕으로 실제 가격 시퀀스를 계산합니다."""
    base_prices = base_prices.reshape(-1, 1)
    cumulative_returns = np.cumprod(1 + returns, axis=1)
    prices = base_prices * cumulative_returns
    return prices

class Predictor:
    """
    학습된 모델을 로드하고 새로운 데이터에 대한 예측을 수행하는 클래스.
    FastAPI 등에서 이 클래스를 가져와 사용합니다.
    """
    def __init__(self, model_dir):
        """
        Args:
            model_dir (str): 학습된 모델, 스케일러, 설정 파일이 저장된 디렉토리 경로
        """
        print(f"'{model_dir}' 경로에서 모델과 관련 파일을 로드합니다.")
        
        # 설정 파일 로드
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # 장치 설정
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 스케일러 로드
        self.feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
        self.target_scaler = joblib.load(os.path.join(model_dir, 'target_scaler.pkl'))

        # 모델 구조 초기화 및 가중치 로드
        self.model = LSTMModel(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout_prob=self.config['dropout_prob']
        ).to(self.device)
        
        model_path = os.path.join(model_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # 모델을 추론 모드로 설정
        print("✅ 모델 로딩 완료.")

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        새로운 입력 데이터프레임에 대해 미래 환율을 예측합니다.

        Args:
            input_df (pd.DataFrame): 최소 'sequence_length' 만큼의 최신 데이터를 포함하는 데이터프레임.
                                     학습에 사용된 모든 'feature_columns'를 포함해야 합니다.

        Returns:
            np.ndarray: 예측된 'prediction_horizon' 기간 동안의 환율 가격 배열.
        """
        seq_len = self.config['sequence_length']
        
        if len(input_df) < seq_len:
            raise ValueError(f"입력 데이터의 길이는 최소 {seq_len} 이상이어야 합니다. 현재 길이: {len(input_df)}")
            
        # 1. 전처리: 최신 시퀀스 데이터 추출 및 스케일링
        last_sequence = input_df.tail(seq_len)
        features = last_sequence[self.config['feature_columns']]
        scaled_features = self.feature_scaler.transform(features)

        # 2. 텐서 변환 및 배치 차원 추가
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. 모델 추론 (결과는 스케일링된 수익률)
        with torch.no_grad():
            predicted_scaled_return = self.model(input_tensor).cpu().numpy()

        # 4. 후처리: 수익률로 복원 -> 가격으로 복원
        predicted_return = self.target_scaler.inverse_transform(predicted_scaled_return)
        
        # 수익률을 가격으로 변환하기 위한 기준 가격 (시퀀스의 마지막 날 종가)
        base_price = last_sequence[self.config['target_column']].iloc[-1]
        
        # 예측된 가격 계산
        predicted_prices = returns_to_prices(np.array([base_price]), predicted_return)
        
        return predicted_prices.flatten()


if __name__ == '__main__':
    # --- Predictor 클래스 사용 예시 ---
    
    # 1. 예측을 수행할 학습된 모델의 경로 지정
    # 이 경로는 train.py 실행 후 생성된 폴더 중 하나를 선택해야 합니다.
    MODEL_DIRECTORY = "models/seq_90-pred_3-hidden_128-layers_2-batch_16-lr_0.0005-scaler_standard-tag_supersimpleAdamWRates2layer"

    if not os.path.exists(MODEL_DIRECTORY):
        print(f"오류: 모델 디렉토리 '{MODEL_DIRECTORY}'를 찾을 수 없습니다.")
        print("먼저 train.py를 실행하여 모델을 학습시켜 주세요.")
    else:
        # 2. 추론기 인스턴스 생성 (이 시점에 모델, 스케일러 등 로딩)
        predictor = Predictor(model_dir=MODEL_DIRECTORY)

        # 3. 예측에 사용할 최신 데이터 로드 (실제로는 API 요청 등으로 데이터를 받아옴)
        # 예시로 학습에 사용한 전체 데이터 로드 후 마지막 120일치 사용
        config = predictor.config
        if config['data_path'].endswith('.csv'):
            sample_df = pd.read_csv(config['data_path'])
        else:
            sample_df = pd.read_excel(config['data_path'])
        
        # 가장 마지막 120일 데이터를 예측 입력으로 사용한다고 가정
        input_data = sample_df.tail(config['sequence_length']).copy()
        
        # 4. 예측 수행
        predictions = predictor.predict(input_data)
        
        print("\n--- 예측 결과 ---")
        print(f"입력 데이터 마지막 날짜: {pd.to_datetime(sample_df['Date'].iloc[-1]).strftime('%Y-%m-%d')}")
        print(f"{config['prediction_horizon']}일 후 환율 예측:")
        for i, price in enumerate(predictions):
            print(f"  - {i+1}일 후: {price:,.2f} 원")