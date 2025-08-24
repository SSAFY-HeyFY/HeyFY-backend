import os
import joblib
import json
import argparse
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from model import LSTMDirectMH
from utils.early_stopping import EarlyStopping
from utils.exchange_dataset import ExchangeRateDataset
from utils.gap_weighted_huber import GapWeightedHuber

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting Model Training')
    parser.add_argument('--tag', type=str, default='return_predict', help='실험을 식별하기 위한 태그') # [변경] 기본 태그 수정
    parser.add_argument('--use_all_features', action='store_true', help='True로 설정 시 모든 피처(요일, diff 포함)를 사용합니다.')
    parser.add_argument('--scaler', type=str, default='minmax', choices=['minmax', 'standard'], help='사용할 스케일러 선택')
    parser.add_argument('--sequence_length', type=int, default=120, help='입력 시퀀스 길이')
    parser.add_argument('--prediction_horizon', type=int, default=5, help='예측할 기간(일)')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM 은닉층의 크기')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM 레이어의 수')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='드롭아웃 확률')
    parser.add_argument('--num_epochs', type=int, default=200, help='학습 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='학습률')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping을 위한 patience')
    
    # jupyter notebook 환경에서는 아래 코드를 활성화하세요.
    # args = parser.parse_args([])
    # production 환경에서는 아래 코드를 활성화하세요.
    args = parser.parse_args()
    return args

# ---------- 모델: Elasticity Multi-Horizon ----------
class LSTMElasticityMH(nn.Module):
    """
    갭-탄성(Elasticity) 헤드: 각 지평 k에 대해 s_k(0~1.2), b_k를 예측
    pred_return[k] = s_k * gap[k] + b_k
    """
    def __init__(self, input_size, hidden_size, num_layers, horizon, dropout_prob, s_max=1.2):
        super().__init__()
        self.horizon = horizon
        self.s_max = s_max
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0.0)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.out = nn.Linear(hidden_size, horizon * 2)  # [z(=s raw), b] per horizon

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.trunk(out[:, -1, :])               # (B, Hdim)
        z = self.out(h).view(-1, self.horizon, 2)   # (B, H, 2)
        z_s = z[..., 0:1]                           # (B, H, 1)
        z_b = z[..., 1:2]                           # (B, H, 1)
        s = torch.sigmoid(z_s) * self.s_max        # (0, s_max)
        b = z_b                                    # unrestricted (작은 규제 추가 권장)
        # squeeze 마지막 차원 제거 → (B, H)
        return s.squeeze(-1), b.squeeze(-1)

# ---------- 손실: residual(H 차원)용 GapWeightedHuber ----------
class GapWeightedHuber(nn.Module):
    def __init__(self, delta=0.5, gamma=1.0, eps=1e-6, shrink=0.0):
        super().__init__()
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.eps   = float(eps)
        self.shrink = float(shrink)

    def forward(self, pred_scaled_residual, true_scaled_residual, gap_unscaled):
        """
        pred/true: (B,H)  # 표준화된 residual
        gap_unscaled: (B,H) # 언스케일 갭 수익률
        """
        e = pred_scaled_residual - true_scaled_residual            # (B,H)
        abs_e = torch.abs(e)
        quad = 0.5 * (e**2)
        lin  = self.delta * (abs_e - 0.5*self.delta)
        huber = torch.where(abs_e <= self.delta, quad, lin)        # (B,H)

        w = torch.pow(torch.abs(gap_unscaled) + self.eps, self.gamma)  # (B,H)
        loss_main = (w * huber).mean()

        loss = loss_main
        if self.shrink > 0:
            loss += self.shrink * (pred_scaled_residual**2).mean()
        return loss

# ---------- 손실: return(H 차원)용 Huber (Elasticity 전용) ----------
class ReturnHuberLoss(nn.Module):
    def __init__(self, delta=1e-2, gamma=0.0, eps=1e-6, b_reg=1e-5):
        """
        delta: 허버 임계 (수익률 스케일에 맞춰 작게)
        gamma: |gap|^gamma 가중 (0이면 가중치 없음)
        b_reg: b L2 규제 강도
        """
        super().__init__()
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.eps   = float(eps)
        self.b_reg = float(b_reg)

    def forward(self, pred_return, true_return, gap_unscaled, b=None):
        """
        pred_return/true_return: (B,H)
        gap_unscaled: (B,H)
        b: (B,H) or None  (있으면 L2 규제)
        """
        err = pred_return - true_return      # (B,H)
        abs_e = torch.abs(err)
        quad = 0.5 * (err**2)
        lin  = self.delta * (abs_e - 0.5*self.delta)
        huber = torch.where(abs_e <= self.delta, quad, lin)    # (B,H)

        if self.gamma > 0:
            w = torch.pow(torch.abs(gap_unscaled) + self.eps, self.gamma)
            loss = (w * huber).mean()
        else:
            loss = huber.mean()
        if b is not None and self.b_reg > 0:
            loss = loss + self.b_reg * (b**2).mean()
        return loss

# ---------- 전처리: H=1/3/5 전부 지원 ----------
def load_and_preprocess_data(args):
    """
    목표:
      - gap_return_t      = Inv_Close_t / ECOS_Close_t - 1
      - next_return_{t+k} = target_{t+k} / ECOS_Close_t - 1   (ECOS 기준으로 k-step 수익률)
      - residual_{t,k}    = next_return_{t,k} - gap_return_{t+k}

    반환:
      X_*: (M, T, F)
      y_*: (M, H)           # residual (표준화 적용 전)
      gap_*: (M, H)         # 언스케일 갭
      target_scaler: residual용 StandardScaler (fit on train)
      test_base_prices: (M,) 기준가 (ECOS_t)
      test_dates: (M,) 예측 anchor 날짜
    """
    print(f"데이터셋 경로: {args.data_path}")

    # 1) 로드 & 정렬
    if args.data_path.endswith('.csv'):
        df = pd.read_csv(args.data_path)
    else:
        df = pd.read_excel(args.data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    ecos = df['ECOS_Close'].astype(float).values
    inv  = df['Inv_Close'].astype(float).values
    nxt  = df[args.target_column].astype(float).values  # 다음날 ECOS (k=1용)

    # gap, next_return(1-step)
    gap_1   = (inv / ecos) - 1.0
    next_1  = (nxt / ecos) - 1.0  # 1-step next return
    # H>1일 때의 next_return_{k}와 gap_{k}를 만들자.
    H = args.prediction_horizon
    N = len(df)

    # k-day ahead next_return: 기준(분모)은 ECOS_t, 분자는 ECOS_{t+k}? 파일에 1일치만 있을 가능성 → 안전하게
    # 'target' 컬럼이 t+1에 해당한다면, k>1은 누적 방식으로 근사:
    #   day-wise 수익률 r_{t,1},..., r_{t,k}를 알 수 있어야 정확. 여기서는 오프라인 평가니까
    #   다음날 실제 ECOS 시계열을 이용해 직접 계산.
    ecos_series = df['ECOS_Close'].astype(float).values
    # day-wise 실제 수익률 r_{t+j} = ECOS_{t+j}/ECOS_{t+j-1} - 1  (j>=1)
    r_daily = np.zeros(N)
    r_daily[1:] = ecos_series[1:]/ecos_series[:-1] - 1.0

    # next_return_{t,k} = (ECOS_{t+k} / ECOS_t) - 1 = prod_{j=1..k} (1+r_{t+j}) - 1
    # gap_{t+k}: inv_{t+k}/ecos_{t+k} - 1  (오프라인에서 미래 갭 사용)
    gaps_all = ((inv / ecos) - 1.0).astype(np.float32)  # 길이 N, 각 t의 gap
    features_df = df[args.feature_columns].copy().astype(np.float32)

    T = args.sequence_length

    X, y_residual, dates, base_prices, gaps_h = [], [], [], [], []
    for i in range(N - T - H + 1):
        # 입력 윈도우
        X.append(features_df.iloc[i:i+T].values)              # (T,F)
        # anchor = t = i+T-1 의 다음 날부터 k일
        t = i + T - 1
        # 기준가: ECOS_t
        base_prices.append(ecos_series[t])
        # 날짜: 예측 anchor = df['Date'][t+1] (t 이후 첫 예측일)
        dates.append(df['Date'].iloc[t+1])

        # k-step 실제 누적수익률
        cum = 1.0
        returns_k = []
        gaps_k = []
        for k in range(1, H+1):
            # day-wise r_{t+k}
            r = r_daily[t + k]
            cum *= (1.0 + r)
            next_k = cum - 1.0                          # (ECOS_{t+k}/ECOS_t - 1)
            returns_k.append(next_k)
            # gap at (t+k) (오프라인 평가용)
            gaps_k.append(gaps_all[t + k])

        returns_k = np.array(returns_k, dtype=np.float32)     # (H,)
        gaps_k = np.array(gaps_k, dtype=np.float32)           # (H,)
        residual_k = returns_k - gaps_k                       # (H,)

        y_residual.append(residual_k)
        gaps_h.append(gaps_k)

    X = np.array(X, dtype=np.float32)                         # (M,T,F)
    y = np.array(y_residual, dtype=np.float32)                # (M,H)
    gaps_h = np.array(gaps_h, dtype=np.float32)               # (M,H)
    dates = pd.Series(dates).reset_index(drop=True)
    base_prices = np.array(base_prices, dtype=np.float32)     # (M,)

    # 4) 날짜 split
    test_split_index = dates[dates < args.test_start_date].index[-1] + 1
    X_rem, X_test = X[:test_split_index], X[test_split_index:]
    y_rem, y_test = y[:test_split_index], y[test_split_index:]
    gaps_rem, gaps_test = gaps_h[:test_split_index], gaps_h[test_split_index:]
    dates_rem, test_dates = dates[:test_split_index], dates[test_split_index:]
    base_prices_rem, test_base_prices = base_prices[:test_split_index], base_prices[test_split_index:]

    val_split_index = int(len(X_rem) * 0.9)
    X_train, X_val = X_rem[:val_split_index], X_rem[val_split_index:]
    y_train, y_val = y_rem[:val_split_index], y_rem[val_split_index:]
    gaps_train, gaps_val = gaps_rem[:val_split_index], gaps_rem[val_split_index:]
    train_dates, val_dates = dates_rem[:val_split_index], dates_rem[val_split_index:]

    # 5) 스케일러: feature는 옵션, target은 항상 표준화
    feature_scaler = StandardScaler() if args.scaler == 'standard' else MinMaxScaler()
    target_scaler  = StandardScaler()

    feature_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    target_scaler.fit(y_train.reshape(-1, 1))     # 모든 horizon 합쳐서 fit

    def scale_X(X_):
        s = X_.reshape(-1, X_.shape[-1])
        s = feature_scaler.transform(s)
        return s.reshape(X_.shape)

    def scale_y(y_):
        s = target_scaler.transform(y_.reshape(-1,1))
        return s.reshape(y_.shape)

    X_train, X_val, X_test = scale_X(X_train), scale_X(X_val), scale_X(X_test)
    y_train, y_val, y_test = scale_y(y_train), scale_y(y_val), scale_y(y_test)
    # gaps_*는 언스케일 그대로 둠

    # 저장
    folder_path = model_folder_name(args)
    joblib.dump(feature_scaler, os.path.join(folder_path, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(folder_path, 'target_scaler.pkl'))

    # 로그
    print("--- 데이터 분할 모니터링 ---")
    print(f"전체 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    if not train_dates.empty:
        print(f"학습: {train_dates.iloc[0].strftime('%Y-%m-%d')} ~ {train_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 {len(X_train)})")
    if not val_dates.empty:
        print(f"검증: {val_dates.iloc[0].strftime('%Y-%m-%d')} ~ {val_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 {len(X_val)})")
    if not test_dates.empty:
        print(f"테스트: {test_dates.iloc[0].strftime('%Y-%m-%d')} ~ {test_dates.iloc[-1].strftime('%Y-%m-%d')} (샘플 {len(X_test)})")
    print("--------------------------")
    print("y_train mean (scaled residual) per H:", np.mean(y_train, axis=0).round(6))
    print("y_train std  (scaled residual) per H:",  np.std (y_train, axis=0).round(6))
    print()

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            target_scaler, test_dates, test_base_prices,
            gaps_train, gaps_val, gaps_test)

# ---------- 학습 루프(그대로 사용 가능): 모델 출력 타입에 따라 분기 ----------
def train_model(args, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
    print("--- 모델 학습 시작 ---")
    save_path = os.path.join(model_folder_name(args), 'best_model.pth')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=save_path)

    for epoch in range(args.num_epochs):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        train_it = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.num_epochs:03d} [Train]", leave=False)

        for batch in train_it:
            if len(batch) == 3:
                Xb, yb, gapb = batch
                gapb = gapb.to(device)
            else:
                Xb, yb = batch
                gapb = None
            Xb = Xb.to(device); yb = yb.to(device)

            out = model(Xb)
            # 모델이 (s,b) 튜플을 내면 elasticity 모드
            if isinstance(out, tuple):
                s, b = out  # (B,H),(B,H)
                # residual true → return true
                mu = torch.tensor(target_scaler.mean_[0], device=device, dtype=torch.float32)
                sd = torch.tensor(target_scaler.scale_[0], device=device, dtype=torch.float32)
                residual_true = yb * sd + mu                      # (B,H)
                true_return   = gapb + residual_true              # (B,H)
                pred_return   = s * gapb + b                      # (B,H)
                loss = criterion(pred_return, true_return, gapb, b)
            else:
                # residual 모드: out=(B,H) scaled residual
                if gapb is None:
                    loss = criterion(out, yb)  # fallback
                else:
                    loss = criterion(out, yb, gapb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_it.set_postfix(loss=f"{loss.item():.6f}")

        # ----- Valid -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_it = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{args.num_epochs:03d} [Valid]", leave=False)
            for batch in val_it:
                if len(batch) == 3:
                    Xb, yb, gapb = batch
                    gapb = gapb.to(device)
                else:
                    Xb, yb = batch
                    gapb = None
                Xb = Xb.to(device); yb = yb.to(device)

                out = model(Xb)
                if isinstance(out, tuple):
                    s, b = out
                    mu = torch.tensor(target_scaler.mean_[0], device=device, dtype=torch.float32)
                    sd = torch.tensor(target_scaler.scale_[0], device=device, dtype=torch.float32)
                    residual_true = yb * sd + mu
                    true_return   = gapb + residual_true
                    pred_return   = s * gapb + b
                    loss = criterion(pred_return, true_return, gapb, b)
                else:
                    if gapb is None:
                        loss = criterion(out, yb)
                    else:
                        loss = criterion(out, yb, gapb)
                val_loss += loss.item()
                val_it.set_postfix(loss=f"{loss.item():.6f}")

        avg_train = train_loss / max(len(train_loader),1)
        avg_val   = val_loss   / max(len(val_loader),1)
        print(f"Epoch {epoch+1:03d}/{args.num_epochs:03d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        if scheduler is not None:
            scheduler.step(avg_val)

        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("--- 모델 학습 완료 ---\n")
    model.load_state_dict(torch.load(early_stopping.path))
    return model

# ---------- 평가: H>=1 지원 ----------
def evaluate_model(model, test_loader, target_scaler, device, test_base_prices, test_gap_returns, mode='residual'):
    """
    residual 모드:
      pred_scaled_residual -> inv -> residual_pred
      return_pred = gap + residual_pred
    elasticity 모드:
      model -> (s,b)
      return_pred = s*gap + b
    이후 returns_to_prices로 누적 복원
    """
    print(f"--- 모델 평가 시작 ({mode}) ---")
    model.eval()
    preds_ret, trues_ret = [], []
    base_all = []

    batch_offset = 0
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                Xb, yb, gapb = batch
                gap_np = gapb.numpy()
            else:
                Xb, yb = batch
                gap_np = test_gap_returns[batch_offset: batch_offset + len(Xb)]
            Xb = Xb.to(device)
            out = model(Xb)

            B = len(Xb)
            H = test_gap_returns.shape[1]

            if isinstance(out, tuple):  # elasticity
                s, b = out
                s = s.cpu().numpy(); b = b.cpu().numpy()          # (B,H)
                mu, sd = target_scaler.mean_[0], target_scaler.scale_[0]
                residual_true = (yb.numpy() * sd + mu)            # (B,H)
                true_return = gap_np + residual_true
                pred_return = s * gap_np + b
            else:  # residual
                out_np = out.cpu().numpy()                        # (B,H) scaled residual
                mu, sd = target_scaler.mean_[0], target_scaler.scale_[0]
                residual_pred = (out_np * sd + mu)
                residual_true = (yb.numpy() * sd + mu)
                true_return = gap_np + residual_true
                pred_return = gap_np + residual_pred

            preds_ret.append(pred_return)     # (B,H)
            trues_ret.append(true_return)
            base_all.append(test_base_prices[batch_offset: batch_offset+B])  # (B,)

            batch_offset += B

    preds_ret = np.vstack(preds_ret)        # (M,H)
    trues_ret = np.vstack(trues_ret)        # (M,H)
    base_all  = np.hstack(base_all)         # (M,)

    # 가격 복원(누적)
    pred_price = returns_to_prices(base_all, preds_ret)  # (M,H)
    true_price = returns_to_prices(base_all, trues_ret)

    # MAE(첫날 및 전체)
    mae_total = np.mean(np.abs(pred_price - true_price))
    print(f"전체 MAE: {mae_total:.2f} 원")
    for k in range(preds_ret.shape[1]):
        mae_k = np.mean(np.abs(pred_price[:,k] - true_price[:,k]))
        print(f"  - {k+1}일 후 MAE: {mae_k:.2f} 원")

    # 베이스라인: 전날 그대로(모든 날 0수익률 가정), 갭100%, 캘리브레이션(α·gap+β)
    # (1) 전날 그대로
    persist_prices = returns_to_prices(base_all, np.zeros_like(trues_ret))
    mae_persist = np.mean(np.abs(persist_prices - true_price))
    print(f"베이스라인(전날 그대로) MAE: {mae_persist:.2f} 원")

    # (2) 갭 100%: returns = gap
    gap100_prices = returns_to_prices(base_all, trues_ret - (trues_ret - preds_ret + (preds_ret - trues_ret)))  # 의미 없는 한 줄 방지용
    gap100_prices = returns_to_prices(base_all, np.copy(test_gap_returns))  # (M,H)
    mae_gap100 = np.mean(np.abs(gap100_prices - true_price))
    print(f"베이스라인(갭 100% 반영) MAE: {mae_gap100:.2f} 원")

    # (3) 캘리브레이션 α·gap+β (훈련구간 OLS → 여기선 간단히 테스트 구간에서도 α,β 산출 가능)
    def calib_alpha_beta(gaps, trues):
        x = gaps.reshape(-1,1); y = trues.reshape(-1)
        X = np.c_[x, np.ones_like(x)]
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(theta[0]), float(theta[1])

    # 각 horizon별 α,β → returns = α*gap + β
    calib_prices = np.zeros_like(true_price)
    for k in range(trues_ret.shape[1]):
        a,b = calib_alpha_beta(test_gap_returns[:,k], trues_ret[:,k])
        pred_k = a*test_gap_returns[:,k] + b
        calib_prices[:,k] = returns_to_prices(base_all, np.column_stack([pred_k if j==k else np.zeros_like(pred_k) for j in range(trues_ret.shape[1])]))[:,k]
        # 위 줄은 k일 수익률만 대체해서 k일가격 산출하는 형태. 간단 비교 목적.

    mae_calib = np.mean(np.abs(calib_prices - true_price))
    print(f"베이스라인(캘리브레이션 α·gap+β) MAE: {mae_calib:.2f} 원")
    print("--- 모델 평가 완료 ---\n")

    return pred_price, true_price

# save_model_config, model_folder_name, plot_test_results 함수는 이전과 동일합니다.
def save_model_config(args):
    folder_path = model_folder_name(args)
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    print(f"✅ 모델 설정이 '{config_path}' 경로에 저장되었습니다.\n")

def model_folder_name(args):
    folder_name = (
        f"seq_{args.sequence_length}-pred_{args.prediction_horizon}"
        f"-hidden_{args.hidden_size}-layers_{args.num_layers}"
        f"-batch_{args.batch_size}-lr_{args.learning_rate}"
        f"-scaler_{args.scaler}-tag_{args.tag}"
    )
    folder_path = os.path.join('models', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"'{folder_path}' 폴더를 생성했습니다.")
    return folder_path

def plot_test_results(args, test_dates, predictions, actuals):
    plt.rc('font', family='Malgun Gothic') # 한글 폰트 설정
    plt.figure(figsize=(15, 8))
    plt.plot(test_dates, actuals[:, 0], label='실제 환율', color='blue', marker='o', markersize=4, linestyle='-')
    plt.plot(test_dates, predictions[:, 0], label='1일 후 예측 환율', color='red', linestyle='--', marker='x', markersize=4)
    
    # 마지막 예측 구간을 상세히 보여주기 위한 로직
    if len(test_dates) >= args.prediction_horizon:
        last_prediction_start_date = test_dates.iloc[-1]
        future_dates = pd.date_range(start=last_prediction_start_date, periods=args.prediction_horizon)
        
        # 마지막 예측에 사용된 실제값들을 표시
        plt.plot(future_dates, actuals[-1, :], 's', color='limegreen', markersize=6, label=f'마지막 {args.prediction_horizon}일 실제값')
        # 마지막 예측값들을 표시
        plt.plot(future_dates, predictions[-1, :], '^', color='orange', markersize=6, label=f'마지막 {args.prediction_horizon}일 예측값')

    plt.title(f'환율 예측 vs 실제값 ({args.prediction_horizon}일 예측)', fontsize=16)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('원/달러', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    folder_path = model_folder_name(args)
    fig_path = os.path.join(folder_path, 'prediction_vs_actual.png')
    plt.savefig(fig_path)
    print(f"✅ 예측 결과 그래프가 '{fig_path}'에 저장되었습니다.")
    plt.show()

# ---------- 누적 가격 복원(멀티홀라이즌) ----------
def returns_to_prices(base_prices, returns):
    """
    base_prices: (M,)
    returns: (M,H)  # day-wise return per horizon
    """
    base_prices = base_prices.reshape(-1,1)
    cumulative = np.cumprod(1.0 + returns, axis=1)
    return base_prices * cumulative


if __name__ == '__main__':
    args = get_args()
    args.mode = 'elasticity'

    
    # 피처 및 기본 설정
    #simple_features = ['Inv_Close', 'ECOS_Close', 'DXY_Close', 'US10Y_Close']
    simple_features = ['Inv_Close', 'ECOS_Close']
    all_features = [
        'Inv_Close', 'Inv_Open', 'Inv_High', 'Inv_Low', 'Inv_Change(%)', 
        'ECOS_Close', 'DXY_Close', 'US10Y_Close',
        'is_Mon', 'is_Tue', 'is_Wed', 'is_Thu', 'is_Fri', 'diff'
    ]
    if args.use_all_features:
        args.feature_columns = all_features
        args.data_path = 'data/train/train_final_with_onehot_20100104_20250812_all.xlsx'
    else:
        args.feature_columns = simple_features
        args.data_path = 'data/train/only_US_KOR_20100104_20250812_simple.csv'

    args.input_size = len(args.feature_columns)
    args.output_size = args.prediction_horizon
    args.target_column = 'target' 
    args.test_start_date = '2025-01-01'
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X_train, y_train, X_val, y_val, X_test, y_test, \
    target_scaler, test_dates, test_base_prices, \
    gap_train, gap_val, test_gap = load_and_preprocess_data(args)

    train_dataset = ExchangeRateDataset(X_train, y_train, gap_train)
    val_dataset = ExchangeRateDataset(X_val, y_val, gap_val)
    test_dataset = ExchangeRateDataset(X_test, y_test, test_gap)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # (A) residual 모드: 기존 LSTMDirectMH 유지
    if args.mode == 'residual':
        model = LSTMDirectMH(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=args.prediction_horizon,
            dropout_prob=args.dropout_prob
        ).to(device)
        criterion = GapWeightedHuber(delta=0.5, gamma=1.0, eps=1e-6, shrink=1e-3)

    # (B) elasticity 모드: 새 헤드 사용 + return 도메인 손실
    else:
        model = LSTMElasticityMH(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            horizon=args.prediction_horizon,
            dropout_prob=args.dropout_prob,
            s_max=1.2
        ).to(device)
        criterion = ReturnHuberLoss(delta=1e-2, gamma=0.5, eps=1e-6, b_reg=1e-5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2)
    
    model = train_model(args, model, train_loader, val_loader, criterion, optimizer, device, scheduler=scheduler)
    
    save_model_config(args)
    
    if len(X_test) > 0:
        # [변경] evaluate_model에 test_base_prices를 전달
        predictions, actuals = evaluate_model(model, test_loader, target_scaler, device, test_base_prices, test_gap)
        plot_test_results(args, test_dates, predictions, actuals)
    else:
        print("테스트 데이터가 없어 평가 및 시각화를 건너뜁니다.")