import os
import sys
import json
import asyncio
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr

from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 서비스 및 로직 임포트 ---
from predict_rate_model import Predictor
from app.services.exchange_rate_crawler import get_detailed_exchange_rates

# --- 설정 ---
MODEL_DIRECTORY = "models/seq_120-pred_1-hidden_196-layers_2-batch_16-lr_0.0005-scaler_standard-tag_realGapSuperSimple_15years120days"
PREDICTION_CACHE_FILE = "prediction_cache.json"

# --- 스크립트 시작 시 모델 로딩 (메모리에 한 번만) ---
if not os.path.exists(MODEL_DIRECTORY):
    raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: '{MODEL_DIRECTORY}'")
predictor = Predictor(model_dir=MODEL_DIRECTORY)
print("✅ Predictor가 성공적으로 초기화되었습니다.")

async def run_and_cache_prediction_async():
    """
    비동기 I/O(크롤링)를 포함하여 AI 예측을 수행하고 결과를 캐싱합니다.
    """
    print(f"[{datetime.now()}] 🤖 AI 예측 및 캐싱 작업을 시작합니다...")
    
    # 1. AI 모델 입력 데이터 준비 (과거 + 현재)
    seq_len = predictor.config['sequence_length'] # 모델이 필요로 하는 데이터 길이 (예: 90)
    yesterday = datetime.now() - timedelta(days=1)
    start_date_fetch = yesterday - timedelta(days=seq_len * 2) # 넉넉하게 조회

    try:
        # ⭐️ 데이터 전처리 로직 수정 ⭐️
        # 1.1. yfinance에서 KRW=X 데이터 로드
        df_inv = yf.download("KRW=X", start=start_date_fetch, end=yesterday)
        df_inv.columns = df_inv.columns.droplevel('Ticker')
        df_inv = df_inv[['Close']]
        df_inv.rename(columns={'Close': 'Inv_Close'}, inplace=True)
        df_inv.columns.name = None
        #print(df_inv.tail())
        #quit()
        ##df_inv = fdr.DataReader("USD/KRW", start_date_fetch, yesterday)
        ##df_inv.rename(columns={'Close': 'Inv_Close'}, inplace=True)

        # 1.2. ECOS 한국은행 기준환율 데이터 로드
        df_ecos = fdr.DataReader('ECOS-KEYSTAT:K152', start_date_fetch, yesterday)
        # ECOS 데이터의 컬럼 이름은 날짜마다 다를 수 있으므로, 첫 번째 컬럼을 선택
        df_ecos.rename(columns={df_ecos.columns[0]: 'ECOS_Close'}, inplace=True)
        
        # 1.3. 두 데이터프레임을 날짜 기준으로 병합
        df_merged = pd.merge(df_inv[['Inv_Close']], df_ecos[['ECOS_Close']], left_index=True, right_index=True, how='outer')
        # ECOS 데이터가 최신 날짜에 없는 경우가 많으므로 이전 값으로 채움 (Forward Fill)
        df_merged.ffill(inplace=True)

        # 1.4. 오늘의 실시간 데이터 (크롤러)
        realtime_rates = await get_detailed_exchange_rates()
        today_rate_detail = next((r for r in realtime_rates if r.currency_code == 'USDKRW'), None)
        if not today_rate_detail:
            raise ValueError("실시간 USD/KRW 환율 정보를 찾을 수 없습니다.")
        
        today_index = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))

        ####✍️ 시간대별로 오늘자 'Inv_Close', 'ECOS_Close' 데이터 어떻게 채워넣어야 할지 로직 고민 필요
        # 오늘 데이터에 Inv_Close만 존재, ECOS_Close는 NaN
        df_today = pd.DataFrame({'Inv_Close': [today_rate_detail.rate], 'ECOS_Close': [today_rate_detail.rate]}, index=[today_index])
        print(df_today)
        # 1.5. 과거 데이터와 오늘 데이터 합치기
        df_combined = pd.concat([df_merged, df_today])
        # 다시 한번 Forward Fill을 통해 오늘의 ECOS_Close 값을 어제 값으로 채움
        df_combined.ffill(inplace=True)

        # 1.6. 'diff' 피처 계산
        df_combined['diff'] = df_combined['Inv_Close'] - df_combined['ECOS_Close']
        df_combined.dropna(inplace=True) # 계산 후 NaN이 있을 경우 제거

        # 1.7. 모델 입력에 필요한 최종 데이터 선택
        input_df = df_combined.tail(seq_len)
        if len(input_df) < seq_len:
            print(f"❌ 전처리 후 예측에 필요한 데이터 길이({seq_len})가 부족합니다.")
            return

    except Exception as e:
        print(f"❌ 데이터 준비 실패: {e}")
        return

    # 2. 예측 수행
    print("📈 모델 예측을 수행합니다...")
    predicted_prices = predictor.predict(input_df)
    print(f"📊 예측 결과 (raw): {predicted_prices}")

    # 3. 예측 결과 가공 및 JSON 파일로 저장
    last_date = input_df.index[-1]

    # 3.1. API 응답을 위해 과거 30일치 ECOS 매매기준율 데이터 준비
    df_historical_30d = df_combined[['ECOS_Close']].tail(30)
    
    historical_points = []
    for date, row in df_historical_30d.iterrows():
        historical_points.append({
            "date": date.strftime('%Y-%m-%d'),
            "rate": round(row['ECOS_Close'], 2),
            "is_prediction": False
        })
    
    # [수정됨] 3.2. Inv_Close로 값을 덮어쓰는 로직을 제거했습니다.
    # 이제 historical_points의 마지막 값은 항상 ECOS_Close의 마지막 값이 됩니다.
    print(f"✅ 과거 데이터의 마지막 지점은 ECOS_Close 값({historical_points[-1]['rate']:.2f})을 유지합니다.")

    # 3.3. 예측 결과를 historical_points 리스트에 추가
    predictions_for_api = historical_points

    # [수정됨] 3.3.1. 그래프 연결을 위한 '브릿지' 포인트 추가
    # historical_points에 데이터가 있는 경우, 마지막 포인트를 복사하여 is_prediction=true로 설정한 후 추가합니다.
    if historical_points:
        bridge_point = historical_points[-1].copy()
        bridge_point['is_prediction'] = True
        predictions_for_api.append(bridge_point)
        print(f"✅ 그래프 연결을 위해 브릿지 포인트({bridge_point['date']})를 추가했습니다.")
    
    if predicted_prices.size > 0:
        price = predicted_prices[0]
        pred_date = last_date + timedelta(days=1)
        predictions_for_api.append({
            "date": pred_date.strftime('%Y-%m-%d'),
            "rate": round(float(price), 2),
            "is_prediction": True
        })
        print(f"✅ 예측 결과({pred_date.strftime('%Y-%m-%d')})를 과거 데이터에 추가했습니다.")
    else:
        print("❌ 예측 결과가 비어있어 추가하지 못했습니다.")
            
    # 3.4. 최종 데이터를 JSON 파일로 저장
    cache_data = {
        "updated_at": datetime.now().isoformat(),
        "predictions": predictions_for_api # 이제 이 변수는 과거 데이터 + 예측 데이터를 모두 포함
    }

    with open(PREDICTION_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 예측 완료! '{PREDICTION_CACHE_FILE}' 파일에 저장되었습니다.")

def run_prediction_job():
    """비동기 함수를 실행하기 위한 동기 래퍼 함수"""
    asyncio.run(run_and_cache_prediction_async())

# --- 스케줄러 설정 ---
sched = BlockingScheduler()
@sched.scheduled_job('cron', hour=23, minute=30) # 매일 밤 11시 30분에 실행
def scheduled_job():
    run_prediction_job()

if __name__ == "__main__":
    print("🚀 스케줄러가 시작되었습니다. 초기 예측을 먼저 실행합니다.")
    run_prediction_job()
    sched.start()
