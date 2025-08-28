import os
import sys
import json
import asyncio
from datetime import datetime
# from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# --- 프로젝트 경로 설정 ---
# 'app' 폴더를 찾기 위해 경로를 동적으로 설정합니다.
# 이 스크립트가 프로젝트 루트에 있거나 'scripts' 같은 하위 폴더에 있어도 동작합니다.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir

# 현재 디렉토리에 'app' 폴더가 있는지 확인합니다.
if not os.path.isdir(os.path.join(project_root, 'app')):
    # 없다면, 한 단계 상위 디렉토리를 프로젝트 루트로 간주합니다.
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

# 최종적으로 결정된 프로젝트 루트를 sys.path에 추가합니다.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 서비스 및 로직 임포트 ---
try:
    from app.services.exchange_rate_crawler import get_detailed_exchange_rates
except ModuleNotFoundError:
    print("---! 모듈 로드 오류 !---")
    print(f"프로젝트 경로를 '{project_root}' (으)로 설정했지만 'app' 모듈을 찾을 수 없습니다.")
    print("프로젝트 파일 구조를 확인해주세요. 이 스크립트는 'app' 폴더와 같은 레벨에 있거나, 바로 한 단계 아래 폴더(예: 'scripts/')에 있어야 합니다.")
    sys.exit(1) # 오류 발생 시 스크립트 종료

# --- 설정 ---
CACHE_BASE_PATH = os.getenv('CACHE_DIR', './logs')
if not os.path.exists(CACHE_BASE_PATH):
    os.makedirs(CACHE_BASE_PATH)
REALTIME_CACHE_FILE = os.path.join(CACHE_BASE_PATH, "realtime_cache.json")

def format_rate_data_for_api(raw_data_list, job_timestamp):
    """크롤링된 원본 데이터를 API가 요구하는 최종 Pydantic 모델 형식으로 변환합니다."""
    processed_list = []
    for item in raw_data_list:
        # VND 통화의 경우 정밀도를 유지하기 위해 포맷팅 방식을 변경합니다.
        is_vnd = "VND" in item.currency

        processed_item = {
            # [수정됨] 개별 크롤링 시간이 아닌, 스케줄러 작업 시작 시간으로 통일합니다.
            "updated_at": job_timestamp.isoformat(),
            "currency": item.currency,
            "rate": str(item.rate) if is_vnd else f"{item.rate:.2f}",
            "change_direction": item.change_direction if item.change_direction is not None else "",
            "change_abs": str(item.change_abs) if is_vnd else f"{item.change_abs:.2f}",
            "change_pct": str(item.change_pct) if is_vnd else f"{item.change_pct:.2f}",
            "cash_buy": f"{item.cash_buy:.2f}" if item.cash_buy is not None else None,
            "cash_sell": f"{item.cash_sell:.2f}" if item.cash_sell is not None else None,
            "wire_send": f"{item.wire_send:.2f}" if item.wire_send is not None else None,
            "wire_receive": f"{item.wire_receive:.2f}" if item.wire_receive is not None else None,
            "provider": item.provider,
        }
        processed_list.append(processed_item)
    return processed_list

async def run_and_cache_rates_async():
    """
    실시간 환율을 크롤링하고 결과를 JSON 파일에 캐싱하는 비동기 함수입니다.
    """
    job_start_time = datetime.now()
    print(f"[{job_start_time}] 📈 실시간 환율 크롤링 및 캐싱 작업을 시작합니다...")
    try:
        # 1. 크롤러 실행
        rates_from_crawler = await get_detailed_exchange_rates()
        if not rates_from_crawler:
            print("⚠️ 크롤링된 데이터가 없습니다. 캐시 파일을 업데이트하지 않습니다.")
            return

        # 2. API 형식에 맞게 데이터 가공
        processed_rates = format_rate_data_for_api(rates_from_crawler, job_start_time)

        # 3. JSON 파일로 저장할 최종 데이터 구조 생성
        cache_data = {
            "updated_at": job_start_time.isoformat(),
            "data": processed_rates
        }

        # 4. 파일에 쓰기
        with open(REALTIME_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 크롤링 완료! '{REALTIME_CACHE_FILE}' 파일에 최신 환율 정보를 저장했습니다.")

    except Exception as e:
        print(f"❌ 크롤링 작업 중 오류 발생: {e}")

def run_crawling_job():
    """비동기 크롤링 함수를 실행하기 위한 동기 래퍼 함수"""
    asyncio.run(run_and_cache_rates_async())

# # --- 스케줄러 설정 ---
# sched = BlockingScheduler(timezone='Asia/Seoul')

# 매 10분마다 'run_crawling_job' 함수를 실행합니다.
# @sched.scheduled_job('interval', minutes=10)
# def scheduled_job():
#     run_crawling_job()

if __name__ == "__main__":
    print("🚀 실시간 환율 크롤링 스케줄러를 시작합니다.")
    print(f"📌 크롤링 결과는 '{os.path.abspath(REALTIME_CACHE_FILE)}' 파일에 저장됩니다.")
    
    # 스케줄러 시작 전, 먼저 1회 즉시 실행하여 초기 데이터를 생성합니다.
    # print("초기 크롤링을 먼저 1회 실행합니다...")
    run_crawling_job()
    
    # print("\n🗓️ 10분 간격으로 다음 크롤링 작업이 실행됩니다.")
    # try:
    #     sched.start()
    # except (KeyboardInterrupt, SystemExit):
    #     print("스케줄러를 종료합니다.")
