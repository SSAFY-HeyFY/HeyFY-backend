from __future__ import annotations
import torch
import torch.nn as nn

class TimeSeriesModel(nn.Module):
    """
    model_type: 'LSTM' | 'GRU' | 'StackedLSTM'
    hidden_sizes:
      - LSTM/GRU       -> (h,) 또는 h(int)
      - StackedLSTM    -> (h1, h2)
    prediction_days: 출력 차원(미래 예측 길이)
    """
    def __init__(
        self,
        model_type: str,
        input_size: int,
        hidden_sizes=(64, 32),
        prediction_days: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.model_type = model_type

        # hidden size 정규화
        if isinstance(hidden_sizes, (list, tuple)):
            hs = tuple(int(h) for h in hidden_sizes)
        else:
            hs = (int(hidden_sizes),)

        if model_type == 'LSTM':
            h = hs[0] if len(hs) >= 1 else 64
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=h, batch_first=True)
            last_hidden_size = h

        elif model_type == 'GRU':
            h = hs[0] if len(hs) >= 1 else 64
            self.rnn = nn.GRU(input_size=input_size, hidden_size=h, batch_first=True)
            last_hidden_size = h

        elif model_type == 'StackedLSTM':
            h1 = hs[0] if len(hs) >= 1 else 64
            h2 = hs[1] if len(hs) >= 2 else 32
            self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=h1, batch_first=True)
            self.rnn2 = nn.LSTM(input_size=h1,      hidden_size=h2, batch_first=True)
            last_hidden_size = h2

        else:
            raise ValueError("지원하지 않는 모델 타입입니다. 'LSTM', 'GRU', 'StackedLSTM' 중에서 선택하세요.")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(last_hidden_size, prediction_days)

    def forward(self, x):
        # x: (B, T, F)
        if self.model_type in ['LSTM', 'GRU']:
            out, _ = self.rnn(x)              # (B, T, H)
        else:
            out1, _ = self.rnn1(x)
            out, _  = self.rnn2(out1)
        last = out[:, -1, :]                  # (B, H)
        last = self.dropout(last)
        y = self.fc(last)                     # (B, prediction_days)
        return y


def build_model(
    model_type: str,
    input_shape: tuple[int, int],
    prediction_days: int,
    dropout: float = 0.2,
    hidden_sizes=(64, 32),
) -> TimeSeriesModel:
    """
    학습 스크립트에서 사용하던 형태를 그대로 유지:
      - input_shape: (WINDOW_SIZE, in_features)  ← 보통 (W, 1)
    """
    _, in_features = input_shape

    # 모델 타입별 hidden_sizes 정리
    if model_type == 'StackedLSTM':
        if not isinstance(hidden_sizes, (list, tuple)) or len(hidden_sizes) < 2:
            hidden_sizes = (64, 32)
    else:
        if isinstance(hidden_sizes, (list, tuple)):
            hidden_sizes = (int(hidden_sizes[0]),)
        else:
            hidden_sizes = (int(hidden_sizes),)

    model = TimeSeriesModel(
        model_type=model_type,
        input_size=in_features,
        hidden_sizes=hidden_sizes,
        prediction_days=prediction_days,
        dropout=dropout,
    )
    return model


def build_model_from_config(cfg: dict, device: torch.device) -> tuple[TimeSeriesModel, str]:
    """
    inference에서 config.json을 받아 모델을 바로 구성할 때 사용.
    반환: (model, ckpt_path)
    """
    mcfg = cfg['model']
    dcfg = cfg['data']
    acfg = cfg['artifacts']

    model_type = mcfg['type']
    input_size = int(mcfg.get('input_size', 1))
    dropout    = float(mcfg.get('dropout', 0.2))

    if model_type == 'StackedLSTM':
        h1 = int(mcfg['hidden'].get('stacked_lstm_hidden_1', 64))
        h2 = int(mcfg['hidden'].get('stacked_lstm_hidden_2', 32))
        hidden_sizes = (h1, h2)
    else:
        h = int(mcfg['hidden'].get('lstm_or_gru_hidden', 64))
        hidden_sizes = (h,)

    prediction_days = int(dcfg['prediction_days'])

    model = TimeSeriesModel(
        model_type=model_type,
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        prediction_days=prediction_days,
        dropout=dropout
    ).to(device)

    ckpt_path = acfg['checkpoint_path']
    return model, ckpt_path
