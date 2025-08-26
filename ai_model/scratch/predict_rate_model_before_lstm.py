import os
import joblib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import LSTMDirectMH

class Predictor:
    """
    새로운 'gap-residual' 학습 방식에 맞춰 수정된 Predictor 클래스입니다.
    학습된 모델을 로드하고 새로운 데이터에 대한 예측을 수행합니다.
    """
    def __init__(self, model_dir):
        """
        Args:
            model_dir (str): 학습된 모델, 스케일러, 설정 파일이 저장된 디렉토리 경로
        """
        print(f"'{model_dir}' 경로에서 모델과 관련 파일을 로드합니다.")
        
        # 1. 설정 파일 로드
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # 2. 장치 설정
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 3. 스케일러 로드
        self.feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
        self.target_scaler = joblib.load(os.path.join(model_dir, 'target_scaler.pkl'))

        # 4. 모델 구조 초기화 및 가중치 로드
        self.model = LSTMDirectMH(
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
        새로운 입력 데이터프레임에 대해 다음 날의 환율을 예측합니다.
        학습 로직(gap-residual)에 맞춰 가격을 복원합니다.

        Args:
            input_df (pd.DataFrame): 최소 'sequence_length' 만큼의 최신 데이터를 포함하는 데이터프레임.
                                     'Inv_Close', 'ECOS_Close' 및 학습에 사용된 모든 피처를 포함해야 합니다.

        Returns:
            np.ndarray: 예측된 1일 후의 환율 가격 배열 (e.g., array([1350.25]))
        """
        seq_len = self.config['sequence_length']
        
        if len(input_df) < seq_len:
            raise ValueError(f"입력 데이터의 길이는 최소 {seq_len} 이상이어야 합니다. 현재 길이: {len(input_df)}")
            
        # 1. 전처리: 최신 시퀀스 데이터 추출 및 스케일링
        last_sequence = input_df.tail(seq_len)
        features = last_sequence[self.config['feature_columns']]
        scaled_features = self.feature_scaler.transform(features)

        # 2. 가격 복원을 위한 주요 값 추출
        # 기준 가격: 시퀀스 마지막 날의 한국 매매기준율
        base_price = last_sequence['ECOS_Close'].iloc[-1]
        # 밤사이 갭(gap) 계산: 시퀀스 마지막 날의 미국 시장 종가와 한국 매매기준율의 차이
        inv_close_last = last_sequence['Inv_Close'].iloc[-1]
        gap_return = (inv_close_last / base_price) - 1.0 if base_price != 0 else 0.0

        # 3. 텐서 변환 및 모델 추론 (결과는 스케일링된 '잔차 수익률')
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # 모델 출력: scaled residual return, shape: (1, 1)
            scaled_pred_residual = self.model(input_tensor).cpu().numpy()

        # 4. 후처리: 잔차 수익률 복원 -> 전체 수익률 복원 -> 가격 복원
        # 4.1. 스케일링된 잔차 수익률 -> 실제 잔차 수익률
        pred_residual = self.target_scaler.inverse_transform(scaled_pred_residual)
        
        # 4.2. 전체 수익률 = 밤사이 갭 수익률 + 예측된 잔차 수익률
        pred_next_day_return = gap_return + pred_residual
        
        # 4.3. 최종 가격 = 기준 가격 * (1 + 전체 수익률)
        predicted_price = base_price * (1 + pred_next_day_return)
        
        return predicted_price.flatten()


if __name__ == '__main__':
    # --- Predictor 클래스 사용 예시 ---
    
    # 1. 예측을 수행할, 새로 학습된 모델의 경로 지정
    # 예시 경로이며, 실제 학습 후 생성된 폴더 경로로 변경해야 합니다.
    MODEL_DIRECTORY = "models/seq_120-pred_1-hidden_128-layers_2-batch_16-lr_0.001-scaler_standard-tag_return_predict"

    if not os.path.exists(MODEL_DIRECTORY):
        print(f"오류: 모델 디렉토리 '{MODEL_DIRECTORY}'를 찾을 수 없습니다.")
        print("먼저 새로운 train.py를 실행하여 모델을 학습시켜 주세요.")
    else:
        # 2. 추론기 인스턴스 생성
        try:
            predictor = Predictor(model_dir=MODEL_DIRECTORY)

            # 3. 예측에 사용할 최신 데이터 로드 (예시)
            # 실제 스케줄러에서는 FinanceDataReader를 통해 이 데이터를 동적으로 생성합니다.
            config = predictor.config
            if config['data_path'].endswith('.csv'):
                sample_df_full = pd.read_csv(config['data_path'])
            else:
                sample_df_full = pd.read_excel(config['data_path'])
            
            # 마지막 120일 데이터를 예측 입력으로 사용한다고 가정
            input_data = sample_df_full.tail(config['sequence_length']).copy()
            
            # 4. 예측 수행
            prediction = predictor.predict(input_data)
            
            print("\n--- 예측 결과 ---")
            print(f"입력 데이터 마지막 날짜: {pd.to_datetime(input_data['Date'].iloc[-1]).strftime('%Y-%m-%d')}")
            print(f"기준 가격 (ECOS_Close): {input_data['ECOS_Close'].iloc[-1]:,.2f} 원")
            print(f"다음 1일 후 환율 예측: {prediction[0]:,.2f} 원")

        except Exception as e:
            print(f"\n--- 예측 테스트 중 오류 발생 ---")
            print(e)

