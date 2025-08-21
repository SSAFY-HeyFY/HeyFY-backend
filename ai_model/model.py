import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM 모델의 구조를 정의하는 클래스.
    이 파일은 모델의 '설계도' 역할만 담당합니다.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0 # 마지막 레이어 제외 드롭아웃
        )        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM의 출력은 (출력 시퀀스, (마지막 은닉 상태, 마지막 셀 상태))
        # out: (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x)
        
        # 우리는 마지막 타임스텝의 은닉 상태만 필요로 함
        # out[:, -1, :]: (batch_size, hidden_size)
        last_hidden_state = out[:, -1, :]
        
        # Fully Connected Layer를 통과시켜 최종 예측값 출력
        # final_output: (batch_size, output_size)
        final_output = self.fc(last_hidden_state)
        
        return final_output