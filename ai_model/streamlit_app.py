import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model import TimeSeriesModel, build_model_from_config

# streamlit run .\streamlit_app.py 로 실행 가능

# =========================
# 유틸리티 함수 (기존 코드 활용 및 추가)
# =========================

def load_config(config_path: str) -> dict:
    """기존 load_config 함수"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    for key in ['data', 'model', 'artifacts']:
        if key not in cfg:
            raise KeyError(f"config에 '{key}' 섹션이 없습니다: {config_path}")
    return cfg

@st.cache_resource
def load_scaler_or_fit(series_1d: np.ndarray, scaler_path: str | None) -> MinMaxScaler:
    """
    1) scaler_path가 존재하면 로드
    2) 없으면 series를 0~1로 fit 해서 반환
    """
    if scaler_path and os.path.exists(scaler_path):
        print(f"[scaler] 로드: {scaler_path}")
        return joblib.load(scaler_path)
    print("[scaler] 파일이 없어 데이터 전체로 MinMaxScaler를 새로 fit 합니다.")
    scaler = MinMaxScaler()
    scaler.fit(series_1d.reshape(-1, 1))
    return scaler

@st.cache_resource
def load_model(_cfg: dict, device: torch.device):
    """모델을 빌드하고 체크포인트를 로드하는 함수 (Streamlit 캐싱 적용)"""
    model, ckpt_path = build_model_from_config(_cfg, device)
    if not os.path.exists(ckpt_path):
        st.error(f"모델 체크포인트 파일이 없습니다: {ckpt_path}")
        st.stop()
    
    # 가짜 모델일 경우 state_dict 로드를 건너뜁니다 (실제 사용 시 이 if문 제거)
    if 'LSTM_W30_P5.ckpt' not in ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    
    model.eval()
    return model

def calculate_directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """방향성 정확도 계산"""
    true_diff = y_true.diff().dropna()
    pred_diff = (y_pred - y_true.shift(1)).dropna() # 예측값의 변화는 '어제 실제값' 대비로 계산
    
    # 같은 인덱스만 비교
    common_index = true_diff.index.intersection(pred_diff.index)
    if len(common_index) == 0:
        return 0.0
        
    true_diff = true_diff[common_index]
    pred_diff = pred_diff[common_index]

    correct_direction = (np.sign(true_diff) == np.sign(pred_diff)).sum()
    return (correct_direction / len(common_index)) * 100 if len(common_index) > 0 else 0.0


@st.cache_data
def run_rolling_prediction(_model, full_scaled_data: np.ndarray, window_size: int, pred_days: int, start_idx: int, end_idx: int) -> np.ndarray:
    """
    선택된 기간에 대해 Walk-Forward 방식으로 예측을 수행합니다.
    (매일매일 과거 window_size일의 데이터로 미래를 예측하는 과정을 반복)
    """
    predictions = []
    device = next(_model.parameters()).device

    for i in range(start_idx, end_idx):
        if i < window_size:
            # 예측에 필요한 데이터가 부족한 경우 NaN 처리
            pred_for_date = np.full((pred_days,), np.nan)
        else:
            window_data = full_scaled_data[i-window_size:i]
            x_infer = window_data.reshape(1, window_size, 1)
            
            with torch.no_grad():
                x_t = torch.from_numpy(x_infer).to(device)
                pred_scaled = _model(x_t).cpu().numpy()
                pred_for_date = pred_scaled.flatten()

        predictions.append(pred_for_date)
        
    return np.array(predictions)


# =========================
# Streamlit App UI 및 메인 로직
# =========================
def main():
    st.set_page_config(page_title="LSTM 성능 분석기", layout="wide")
    st.title("📈 LSTM 모델 성능 분석 대시보드")
    
    # --- 사이드바 설정 ---
    st.sidebar.header("⚙️ 모델 및 기간 설정")
    
    # 1. 모델(config) 선택
    model_dir = 'models'
    config_files = glob.glob(os.path.join(model_dir, "*.config.json"))
    if not config_files:
        st.error(f"'{model_dir}' 폴더에 .config.json 파일이 없습니다.")
        st.stop()
    
    selected_config_path = st.sidebar.selectbox(
        "분석할 모델의 Config 파일을 선택하세요:",
        config_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    # --- 설정 및 데이터 로드 ---
    cfg = load_config(selected_config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path = cfg['data']['file_path']
    feature = cfg['data'].get('feature_name', 'DATA_VALUE')
    window = int(cfg['data']['window_size'])
    pred_days = int(cfg['data']['prediction_days'])
    
    df_full = pd.read_excel(data_path, index_col=0, parse_dates=True)
    prices = df_full[feature].values.astype(np.float32)
    
    scaler_path = os.path.join(cfg['artifacts']['model_dir'], 'scaler.pkl')
    scaler = load_scaler_or_fit(prices, scaler_path)
    scaled_prices = scaler.transform(prices.reshape(-1, 1)).astype(np.float32)

    model = load_model(cfg, device)

    # 2. 분석 기간 선택
    st.sidebar.subheader("🗓️ 성능 분석 기간")
    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()
    
    # 예측을 위해 최소 window_size일의 데이터가 필요하므로 시작 가능일 조정
    eval_start_date = st.sidebar.date_input("시작일", 
        value=max_date - timedelta(days=90),
        min_value=min_date + timedelta(days=window),
        max_value=max_date
    )
    eval_end_date = st.sidebar.date_input("종료일", 
        value=max_date,
        min_value=eval_start_date,
        max_value=max_date
    )
    
    # --- 메인 대시보드 ---
    st.header(f"📊 분석 결과: _{eval_start_date.strftime('%Y-%m-%d')} ~ {eval_end_date.strftime('%Y-%m-%d')}_", divider='rainbow')

    # 선택된 기간에 대한 인덱스 찾기
    date_index = df_full.index
    start_idx = date_index.searchsorted(pd.to_datetime(eval_start_date))
    end_idx = date_index.searchsorted(pd.to_datetime(eval_end_date))

    # 롤링 예측 수행
    rolling_preds_scaled = run_rolling_prediction(model, scaled_prices, window, pred_days, start_idx, end_idx)
    
    # 결과 역스케일링 및 DataFrame 생성
    # P일 예측 중 첫날(+1일) 예측치만 사용하여 성능을 평가합니다.
    pred_plus_1_day_scaled = rolling_preds_scaled[:, 0]
    pred_plus_1_day = scaler.inverse_transform(pred_plus_1_day_scaled.reshape(-1, 1)).flatten()

    result_df = df_full.iloc[start_idx:end_idx].copy()
    result_df['Predicted'] = pred_plus_1_day
    result_df.dropna(inplace=True) # 예측이 불가능했던 초반 데이터 제거
    
    if result_df.empty:
        st.warning("선택된 기간에 대한 예측 결과가 없습니다. 기간을 확인해주세요.")
        st.stop()

    # 성능 지표 계산
    y_true = result_df['Actual'] = result_df[feature] # 명확성을 위해 'Actual' 컬럼 추가
    y_pred = result_df['Predicted']
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    da = calculate_directional_accuracy(y_true, y_pred)
    
    st.subheader("🎯 핵심 성능 지표")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.2f}", help="모델 예측값과 실제값 사이의 평균 오차 크기입니다.")
    col2.metric("방향성 정확도 (DA)", f"{da:.2f} %", help="실제 등락 방향(상승/하락)을 얼마나 정확하게 예측했는지를 나타냅니다.")

    # 시각화
    st.subheader("📈 실제값 vs. 예측값 비교 차트")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Actual'], mode='lines', name='실제값', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Predicted'], mode='lines', name='예측값 (+1일)', line=dict(color='tomato', dash='dot')))
    fig.update_layout(title='선택 기간 내 실제값과 모델 예측 결과', xaxis_title='날짜', yaxis_title=feature)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🗂️ 예측 결과 상세 데이터 보기"):
        st.dataframe(result_df[['Actual', 'Predicted']])

if __name__ == "__main__":
    main()