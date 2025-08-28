### Prophet 모델용 추론 스케줄러
import os
import sys
import json
import asyncio
import warnings
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime, timedelta
# from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# --- 프로젝트 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 서비스 및 로직 임포트 ---
from predict_rate_model import get_hybrid_prophet_forecast
from app.services.exchange_rate_crawler import get_detailed_exchange_rates

# --- 설정 ---
CACHE_BASE_PATH = os.getenv('CACHE_DIR', './logs')
if not os.path.exists(CACHE_BASE_PATH):
    os.makedirs(CACHE_BASE_PATH)
PREDICTION_CACHE_FILE = os.path.join(CACHE_BASE_PATH, 'prediction_cache.json')

print("✅ Prophet 예측 스케줄러가 준비되었습니다.")

async def run_and_cache_prediction_async():
    """AI 예측을 수행하고 그래프용 데이터를 캐싱합니다."""
    print(f"[{datetime.now()}] 🤖 AI 예측 및 캐싱 작업을 시작합니다...")
    
    try:
        # --- 1. AI 예측에 필요한 데이터 준비 ---
        today = datetime.now()
        start_date_fetch = today - timedelta(days=400) # 넉넉하게 과거 데이터 조회

        # 1.1. Inv_Close 과거 데이터 조회 (yfinance)
        df_inv_raw = yf.download("KRW=X", start=start_date_fetch, end=today)
        df_inv = df_inv_raw[['Close']]
        df_inv.columns = ['Inv_Close']
        latest_inv_close = df_inv['Inv_Close'].iloc[-1]
        print(f"✅ yfinance 데이터 로드 완료 (최신 Inv_Close: {latest_inv_close:.2f})")

        # 1.2. ECOS_Close 과거 데이터 조회 (FinanceDataReader)
        df_ecos = fdr.DataReader('ECOS-KEYSTAT:K152', start_date_fetch, today)
        df_ecos.rename(columns={df_ecos.columns[0]: 'ECOS_Close'}, inplace=True)
        print("✅ FinanceDataReader 데이터 로드 완료")

        # 1.3. ECOS_Close '오늘' 데이터 조회 (실시간 크롤러)
        realtime_rates = await get_detailed_exchange_rates()
        today_rate_detail = next((r for r in realtime_rates if "USDKRW" in r.currency), None)
        if not today_rate_detail:
            raise ValueError("실시간 USDKRW 환율 정보를 찾을 수 없습니다.")
        today_current_rate = today_rate_detail.rate
        print(f"✅ 실시간 크롤링 완료 (현재 환율: {today_current_rate:.2f})")

        # 1.4. 데이터 병합 및 최종 데이터프레임 생성
        historical_df = pd.merge(df_inv, df_ecos, left_index=True, right_index=True, how='outer')
        historical_df.ffill(inplace=True) # 누락된 값 채우기
        
        # 'Date' 컬럼을 위해 인덱스 리셋
        historical_df.reset_index(inplace=True)
        historical_df.rename(columns={'index': 'Date'}, inplace=True)
        
        # '오늘' 데이터를 마지막 행으로 추가
        today_row = pd.DataFrame([{
            'Date': pd.to_datetime(today.date()),
            'Inv_Close': latest_inv_close,
            'ECOS_Close': today_current_rate,
        }])
        final_input_df = pd.concat([historical_df, today_row], ignore_index=True)
        print("✅ AI 모델 입력을 위한 최종 데이터프레임 생성 완료.")

        # --- 2. AI 예측 실행 (5일치 결과 수신) ---
        prediction_df = get_hybrid_prophet_forecast(
            historical_df=final_input_df,
            predict_days=5
        )
        print("📈 AI 예측 완료 (5일치):")
        print(prediction_df[['ds', 'yhat']].to_string())

        # --- 3. 그래프용 데이터 생성 ---
        graph_data_points = []

        # 3.1. 과거 데이터 (최근 25일)
        df_historical_25d = historical_df.tail(25)
        for _, row in df_historical_25d.iterrows():
            graph_data_points.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "rate": round(row['ECOS_Close'], 2),
                "is_prediction": False,
                "model_name": None
            })

        # 3.2. '오늘' 데이터 (크롤링 기준)
        today_point = {"date": today.strftime('%Y-%m-%d'), "rate": round(today_current_rate, 2), "is_prediction": False}
        graph_data_points.append(today_point)

        # 3.3. 그래프 연결용 '브릿지' 데이터
        bridge_point = today_point.copy()
        bridge_point['is_prediction'] = True
        # 브릿지 포인트는 첫 예측 모델을 따라감
        bridge_point['model_name'] = "1D Est."
        graph_data_points.append(bridge_point)

        # 3.4. AI 예측 데이터 (모델 이름 수정 및 연결점 추가)
        if not prediction_df.empty:
            # 3.4.1. 1일차 예측 지점 추가
            model_a_pred = prediction_df.iloc[0]
            graph_data_points.append({
                "date": model_a_pred['ds'].strftime('%Y-%m-%d'),
                "rate": round(model_a_pred['yhat'], 2),
                "is_prediction": True,
                "model_name": "1D Est."
            })

            # 3.4.2. 5일 예측 그래프 연결을 위한 시작 지점 추가
            graph_data_points.append({
                "date": model_a_pred['ds'].strftime('%Y-%m-%d'),
                "rate": round(model_a_pred['yhat'], 2),
                "is_prediction": True,
                "model_name": "5D Est."
            })

            # 3.4.3. 2일차 이후 예측 지점들 추가
            for _, row in prediction_df.iloc[1:].iterrows():
                graph_data_points.append({
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "rate": round(row['yhat'], 2),
                    "is_prediction": True,
                    "model_name": "5D Est."
                })
        
        # --- 4. 최종 캐시 파일 저장 ---
        cache_data = {"updated_at": datetime.now().isoformat(), "predictions": graph_data_points}
        with open(PREDICTION_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 예측 완료! '{PREDICTION_CACHE_FILE}' 파일에 그래프용 데이터를 저장했습니다.")

    except Exception as e:
        print(f"❌ 작업 실패: {e}")
        # 에러 추적을 위해 스택 트레이스 출력
        import traceback
        traceback.print_exc()


def run_prediction_job():
    """비동기 함수를 실행하기 위한 동기 래퍼 함수"""
    asyncio.run(run_and_cache_prediction_async())

# --- 스케줄러 설정 (기존과 동일) ---
# sched = BlockingScheduler(timezone='Asia/Seoul')
# @sched.scheduled_job('interval', minutes=10)
# def scheduled_job():
#     run_prediction_job()

if __name__ == "__main__":
    print("🚀 AI 예측 스케줄러를 시작합니다 (Prophet 모델 + 실시간 데이터 처리 버전).")
    # print("초기 예측을 먼저 1회 실행합니다...")
    run_prediction_job()
    # print("\n🗓️ 10분 간격으로 다음 작업이 실행됩니다.")
    # try:
    #     sched.start()
    # except (KeyboardInterrupt, SystemExit):
    #     print("스케줄러를 종료합니다.")