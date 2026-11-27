import io
import json
import math
import os
import re
import time
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ==========================================
# 1. 參數配置區
# ==========================================
BENCHMARK = "2800.HK"
BENCHMARK_LABEL = "盈富基金 / 恒指代理"

MIN_AVG_TURNOVER = 30_000_000      # 20日日均成交額下限（3,000 萬港元）
CHUNK_SIZE = 80                    # 每批下載數量，避免觸發 Yahoo 限流
DOCS_DIR = "docs"
CACHE_CSV = os.path.join(DOCS_DIR, "ggt_tickers.csv")

PARAMS = {
    "box_window": 50,              # Stage 2 水平箱體突破天數
    "vol_mult_breakout": 1.5,      # 突破量比門檻（相對 50 日均量）
    "dist_ma50_max": 0.15,         # 防追高：距 50 日線最大乖離率（15%）
    "gap_min": 0.02,               # 有效跳空最小幅度（2%）
    "gap_max_buy": 0.08,           # 首日可追跳空上限（8%）
    "gap_vol_mult": 1.8,           # 放量跳空量比門檻
    "gap_lookback_min": 3,         # 缺口不補回溯最小天數
    "gap_lookback_max": 15,        # 缺口不補回溯最大天數
    "room_ma200_min": 0.15,        # 4B- 距 200 日線最小反彈空間（15%）
    "drawdown_4b_min": 0.30,       # 4B- 距 52 週高點最小回撤（30%）
    "trim_dist_ma50": 0.25,        # 過度延伸分批止盈門檻（25%）
    # Stage 3 見頂派發參數
    "s3_whipsaw_min": 4,           # 近 40 日 50MA 反覆穿插最小次數
    "s3_lousy_window": 10,         # 高位劣質收盤回溯天數
    "s3_island_max_days": 10,      # 島形反轉孤島最大跨度天數
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    )
}

FALLBACK_GGT = {
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴-W", "3690.HK": "美團-W",
    "1810.HK": "小米集團-W", "1211.HK": "比亞迪股份", "0981.HK": "中芯國際",
    "0388.HK": "香港交易所", "2318.HK": "中國平安", "0941.HK": "中國移動",
    "0883.HK": "中國海洋石油", "9618.HK": "京東集團-SW", "9999.HK": "網易-S",
    "2015.HK": "理想汽車-W", "9868.HK": "小鵬汽車-W", "0005.HK": "滙豐控股",
    "1024.HK": "快手-W", "0285.HK": "比亞迪電子", "2382.HK": "舜宇光學科技",
    "0669.HK": "創科實業", "0857.HK": "中國石油股份", "0939.HK": "建設銀行",
    "1398.HK": "工商銀行", "3968.HK": "招商銀行", "2899.HK": "紫金礦業",
    "1816.HK": "中廣核電力", "2269.HK": "藥明生物", "6160.HK": "百濟神州",
    "9926.HK": "康方生物", "0020.HK": "商湯-W", "9880.HK": "優必選",
}


# ==========================================
# 2. 全量港股通名單獲取（四重容錯機制）
# ==========================================
def is_hk_etf_or_rmb_counter(code_num: int, name: str = "") -> bool:
    if code_num >= 80000:
        return True
    if (
        (2800 <= code_num <= 2849)
        or (3000 <= code_num <= 3199)
        or (3400 <= code_num <= 3499)
        or (9000 <= code_num <= 9199)
        or (9400 <= code_num <= 9499)
    ):
        return True
    upper_name = name.upper()
    return ("ETF" in upper_name) or ("盈富基金" in name)


def to_yf_hk_ticker(raw_code: str) -> tuple[str, int] | tuple[None, None]:
    digits = "".join(ch for ch in str(raw_code) if ch.isdigit())
    if not digits:
        return None, None
    num = int(digits)
    if num <= 0:
        return None, None
    ticker = f"{num:04d}.HK" if num < 10000 else f"{num}.HK"
    return ticker, num


def fetch_from_hkexnews() -> dict[str, str]:
    url = "https://www3.hkexnews.hk/sdw/search/mutualmarket_c.aspx?t=hk"
    resp = requests.get(url, headers=HEADERS, timeout=25)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "lxml")
    rows = soup.select("table#mutualmarket-result tbody tr") or soup.select("tbody tr")

    stock_map = {}
    for tr in rows:
        code_td = tr.select_one("td.col-stock-code")
        name_td = tr.select_one("td.col-stock-name")
        if not code_td or not name_td:
            continue

        code_body = code_td.select_one(".mobile-list-body")
        name_body = name_td.select_one(".mobile-list-body")
        raw_code = (code_body or code_td).get_text(strip=True)
        raw_name = (name_body or name_td).get_text(strip=True)
        raw_name = re.sub(r"^(股份名稱|Stock Name)\s*[:：]?\s*", "", raw_name).strip()

        yf_code, num = to_yf_hk_ticker(raw_code)
        if not yf_code or is_hk_etf_or_rmb_counter(num, raw_name):
            continue
        stock_map[yf_code] = raw_name

    return stock_map


def fetch_from_szse_excel() -> dict[str, str]:
    url = (
        "https://www.szse.cn/api/report/ShowReport"
        "?SHOWTYPE=xlsx&CATALOGID=SGT_GGTBDQD&TABKEY=tab1"
    )
    headers = {**HEADERS, "Referer": "https://www.szse.cn/szhk/hkbdsj/bdzq/index.html"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    df = pd.read_excel(io.BytesIO(resp.content), dtype=str)
    code_col = next((c for c in df.columns if "代码" in str(c) or "代碼" in str(c)), df.columns[0])
    name_col = next(
        (c for c in df.columns if ("简称" in str(c) or "簡稱" in str(c)) and "英" not in str(c)),
        df.columns[1],
    )

    stock_map = {}
    for _, row in df.iterrows():
        raw_name = str(row.get(name_col, "")).strip()
        yf_code, num = to_yf_hk_ticker(row.get(code_col, ""))
        if not yf_code or is_hk_etf_or_rmb_counter(num, raw_name):
            continue
        stock_map[yf_code] = raw_name

    return stock_map


def fetch_from_akshare() -> dict[str, str]:
    import akshare as ak

    df_ggt = ak.stock_hk_ggt_components_em()
    code_col = "代码" if "代码" in df_ggt.columns else df_ggt.columns[1]
    name_col = "名称" if "名称" in df_ggt.columns else df_ggt.columns[2]

    stock_map = {}
    for _, row in df_ggt.iterrows():
        raw_name = str(row[name_col]).strip()
        yf_code, num = to_yf_hk_ticker(row[code_col])
        if not yf_code or is_hk_etf_or_rmb_counter(num, raw_name):
            continue
        stock_map[yf_code] = raw_name

    return stock_map


def get_ggt_stock_map() -> tuple[dict[str, str], str]:
    os.makedirs(DOCS_DIR, exist_ok=True)
    sources = [
        ("港交所披露易官方", fetch_from_hkexnews),
        ("深交所官方名單", fetch_from_szse_excel),
        ("東方財富(akshare)", fetch_from_akshare),
    ]

    for source_name, fetcher in sources:
        try:
            stock_map = fetcher()
            if len(stock_map) >= 200:
                pd.DataFrame(
                    [{"ticker": k, "name": v} for k, v in sorted(stock_map.items())]
                ).to_csv(CACHE_CSV, index=False, encoding="utf-8-sig")
                print(f"[名單] {source_name}獲取成功，共 {len(stock_map)} 隻港股通股票。")
                return stock_map, source_name
            print(f"[警告] {source_name}返回數量偏少（{len(stock_map)} 隻），嘗試下一來源...")
        except Exception as e:
            print(f"[警告] {source_name}獲取失敗: {e}")

    if os.path.exists(CACHE_CSV):
        cached = pd.read_csv(CACHE_CSV)
        if len(cached) >= 100:
            stock_map = dict(zip(cached["ticker"], cached["name"]))
            print(f"[名單] 已載入本地快取 CSV，共 {len(stock_map)} 隻港股通股票。")
            return stock_map, "本地快取(港股通)"

    print(f"[名單] 使用內建核心兜底名單（{len(FALLBACK_GGT)} 隻）。")
    return FALLBACK_GGT.copy(), "內建核心兜底"


# ==========================================
# 3. 批量行情下載與指標計算
# ==========================================
def safe_num(val, digits: int = 2) -> float:
    try:
        f = float(val)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, digits)
    except Exception:
        return 0.0


def extract_ohlcv(raw_df: pd.DataFrame, ticker: str)
