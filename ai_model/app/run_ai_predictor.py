import os
import sys
import json
import asyncio
import warnings
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 프로젝트 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 서비스 및 로직 임포트 ---
from predict_rate_model import Predictor
from app.services.exchange_rate_crawler import get_detailed_exchange_rates, ExchangeRateDetail

# --- 설정 ---
MODEL_DIRECTORY = "models/seq_120-pred_1-hidden_128-layers_2-batch_16-lr_0.001-scaler_standard-tag_final_1day"
PREDICTION_CACHE_FILE = "prediction_cache.json"

# --- 스크립트 시작 시 모델 로딩 ---
if not os.path.exists(MODEL_DIRECTORY):
    raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: '{MODEL_DIRECTORY}'")
predictor = Predictor(model_dir=MODEL_DIRECTORY)
print("✅ Predictor가 성공적으로 초기화되었습니다.")

async def run_and_cache_prediction_async():
    """AI 예측을 수행하고 그래프용 데이터를 캐싱합니다."""
    print(f"[{datetime.now()}] 🤖 AI 예측 및 캐싱 작업을 시작합니다...")
    
    try:
        # --- 1. 모든 데이터 소스에서 데이터 가져오기 ---
        seq_len = predictor.config['sequence_length']
        today = datetime.now()
        start_date_fetch = today - timedelta(days=seq_len * 2) # 넉넉하게 조회

        # 1.1. yfinance 및 ECOS 데이터 조회
        df_inv_raw = yf.download("KRW=X", start=start_date_fetch, end=today)
        df_inv = df_inv_raw[['Close']]
        df_inv.columns = ['Inv_Close']

        df_ecos = fdr.DataReader('ECOS-KEYSTAT:K152', start_date_fetch, today)
        df_ecos.rename(columns={df_ecos.columns[0]: 'ECOS_Close'}, inplace=True)

        # 1.2. 실시간 크롤러로 '오늘'의 현재 환율 조회
        realtime_rates = await get_detailed_exchange_rates()
        today_rate_detail = next((r for r in realtime_rates if "USDKRW" in r.currency), None)
        if not today_rate_detail:
            raise ValueError("실시간 USDKRW 환율 정보를 찾을 수 없습니다.")
        today_current_rate = today_rate_detail.rate
        print(f"✅ 실시간 크롤링 완료: 현재 환율 {today_current_rate:.2f}")

        # --- 2. AI 예측 실행 (가장 최신 데이터 사용) ---
        df_merged = pd.merge(df_inv, df_ecos, left_index=True, right_index=True, how='outer')
        df_merged.ffill(inplace=True)
        df_merged['diff'] = df_merged['Inv_Close'] - df_merged['ECOS_Close']
        df_merged.dropna(inplace=True)
        
        input_df_for_ai = df_merged.tail(seq_len)
        if len(input_df_for_ai) < seq_len:
            raise ValueError(f"AI 예측에 필요한 데이터 길이({seq_len})가 부족합니다.")

        predicted_prices = predictor.predict(input_df_for_ai)
        predicted_rate = round(float(predicted_prices[0]), 2)
        print(f"📈 AI 예측 완료: 예측 환율 {predicted_rate:.2f}")

        # --- 3. 그래프용 데이터 생성 ---
        graph_data_points = []

        # 3.1. 과거 데이터 (ECOS 기준, 최근 30일)
        df_historical_30d = df_ecos.tail(30)
        for date, row in df_historical_30d.iterrows():
            graph_data_points.append({
                "date": date.strftime('%Y-%m-%d'),
                "rate": round(row['ECOS_Close'], 2),
                "is_prediction": False
            })

        # 3.2. '오늘' 데이터 (크롤링 기준)
        today_str = today.strftime('%Y-%m-%d')
        today_point = {
            "date": today_str,
            "rate": round(today_current_rate, 2),
            "is_prediction": False
        }
        graph_data_points.append(today_point)

        # 3.3. 그래프 연결용 '브릿지' 데이터
        bridge_point = today_point.copy()
        bridge_point['is_prediction'] = True
        graph_data_points.append(bridge_point)

        # 3.4. AI 예측 데이터 (주말 처리 포함)
        weekday = today.weekday() # 월요일=0, 토요일=5, 일요일=6
        if weekday == 5: # 토요일
            pred_date = today + timedelta(days=2)
        elif weekday == 6: # 일요일
            pred_date = today + timedelta(days=1)
        else: # 평일
            pred_date = today + timedelta(days=1)
        
        prediction_point = {
            "date": pred_date.strftime('%Y-%m-%d'),
            "rate": predicted_rate,
            "is_prediction": True
        }
        graph_data_points.append(prediction_point)
        print(f"🗓️ 예측 날짜: {pred_date.strftime('%Y-%m-%d')} (오늘은 {today.strftime('%A')})")

        # --- 4. 최종 캐시 파일 저장 ---
        cache_data = {
            "updated_at": datetime.now().isoformat(),
            "predictions": graph_data_points
        }
        with open(PREDICTION_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 예측 완료! '{PREDICTION_CACHE_FILE}' 파일에 그래프용 데이터를 저장했습니다.")

    except Exception as e:
        print(f"❌ 작업 실패: {e}")

def run_prediction_job():
    """비동기 함수를 실행하기 위한 동기 래퍼 함수"""
    asyncio.run(run_and_cache_prediction_async())

# --- 스케줄러 설정 ---
sched = BlockingScheduler(timezone='Asia/Seoul')
@sched.scheduled_job('interval', minutes=10)
def scheduled_job():
    run_prediction_job()

if __name__ == "__main__":
    print("🚀 AI 예측 스케줄러를 시작합니다.")
    print("초기 예측을 먼저 1회 실행합니다...")
    run_prediction_job()

    print("\n🗓️ 10분 간격으로 다음 크롤링 작업이 실행됩니다.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("스케줄러를 종료합니다.")
