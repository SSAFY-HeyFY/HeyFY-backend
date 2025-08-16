import os
import time
import argparse
from datetime import datetime, timedelta
from dateutil.parser import parse as dtparse

import requests
import pandas as pd

BASE_URL = "https://oapi.koreaexim.go.kr/site/program/financial/exchangeJSON"

# 숫자 컬럼만!
NUM_COLS = [
    "TTB",             # 전신환(송금) 받을때
    "TTS",             # 전신환(송금) 보낼때
    "DEAL_BAS_R",      # 매매기준율
    "BKPR",            # 장부가격
    "YY_EFEE_R",       # 년환가료율
    "TEN_DD_EFEE_R",   # 10일환가료율
    "KFTC_DEAL_BAS_R", # 서울외국환중개 매매기준율
    "KFTC_BKPR"        # 서울외국환중개 장부가격
]

def yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")

def fetch_exchange(date_str: str, authkey: str, data: str = "AP01", timeout=10, max_retry=3) -> list[dict]:
    params = {"authkey": authkey, "searchdate": date_str, "data": data}
    for attempt in range(1, max_retry + 1):
        try:
            res = requests.get(BASE_URL, params=params, timeout=timeout)
            res.raise_for_status()
            js = res.json()

            # 실패 응답 처리
            if isinstance(js, dict) and js.get("RESULT"):
                if int(js.get("RESULT", 2)) != 1:
                    return []
            if isinstance(js, list) and js and "RESULT" in js[0]:
                if int(js[0].get("RESULT", 2)) != 1:
                    return []
            return js if isinstance(js, list) else []
        except Exception as e:
            if attempt == max_retry:
                print(f"[ERROR] {date_str} 요청 실패: {e}")
                return []
            time.sleep(0.8 * attempt)

def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^\d.\-]", "", regex=True)
                .replace({"": None})
                .astype(float)
            )
    return df

def collect_range(start_date: str, end_date: str, authkey: str, data: str = "AP01") -> pd.DataFrame:
    s = dtparse(start_date).date()
    e = dtparse(end_date).date()
    all_rows = []

    cur = s
    while cur <= e:
        rows = fetch_exchange(yyyymmdd(datetime(cur.year, cur.month, cur.day)), authkey, data=data)
        if rows:
            for r in rows:
                r["_DATE"] = cur.isoformat()
            all_rows.extend(rows)
        cur += timedelta(days=1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.columns = [str(c).upper() for c in df.columns]

    head_cols = [c for c in ["_DATE", "CUR_UNIT", "CUR_NM"] if c in df.columns]
    prefer_cols = head_cols + [c for c in NUM_COLS if c in df.columns]
    other_cols = [c for c in df.columns if c not in prefer_cols]
    df = df[prefer_cols + other_cols]

    df = normalize_numeric(df)

    if "CUR_UNIT" in df.columns:
        df["CUR"] = df["CUR_UNIT"].str.extract(r"^([A-Z]{3})")[0]

    return df

def save_to_excel(df: pd.DataFrame, out_path: str):
    if df.empty:
        print("수집된 데이터가 없습니다.")
        return

    sort_cols = [c for c in ["_DATE", "CUR"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="exchange_rates")
    print(f"Saved: {out_path}  ({len(df):,} rows)")

def main():
    parser = argparse.ArgumentParser(description="한국수출입은행 환율 API → Excel 저장")
    parser.add_argument("--authkey", type=str, default=os.getenv("EXIM_AUTHKEY"))
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--data", type=str, default="AP01", choices=["AP01", "AP02", "AP03"], help="AP01은 환율, 02는 대출금리, 03은 국제금리")
    parser.add_argument("--out", type=str, default=f"Korea_Exim_Bank_exchange_rates.xlsx")
    args = parser.parse_args()

    if not args.authkey:
        raise SystemExit("인증키가 없습니다. --authkey 또는 EXIM_AUTHKEY 환경변수를 설정하세요.")

    df = collect_range(args.start, args.end, args.authkey, data=args.data)
    save_to_excel(df, args.out)

if __name__ == "__main__":
    main()
