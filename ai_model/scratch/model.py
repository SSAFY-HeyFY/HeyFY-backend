import torch
import torch.nn as nn

class TimeSeriesModel(nn.Module):
    """
    LSTM/GRU 기반 시계열 예측 모델
    - 입력: (batch, seq_len, input_size)
    - 출력: (batch, pred_horizon)
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, pred_horizon: int = 5, model_type: str = "LSTM"):
        super().__init__()
        self.model_type = model_type.upper()
        self.pred_horizon = pred_horizon

        if self.model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'LSTM' or 'GRU'.")

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.pred_horizon)
        )

    def forward(self, x):
        # x: (B, T, F)
        if self.model_type == "LSTM":
            out, (hn, cn) = self.rnn(x)  # out: (B, T, H), hn: (num_layers, B, H)
        else:
            out, hn = self.rnn(x)
        last = out[:, -1, :]  # 마지막 타임스텝의 hidden
        y = self.fc(last)     # (B, pred_horizon)
        return y
