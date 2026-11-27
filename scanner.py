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


def extract_ohlcv(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    if isinstance(raw_df.columns, pd.MultiIndex):
        if ticker in raw_df.columns.get_level_values(0):
            df = raw_df[ticker].copy()
        elif ticker in raw_df.columns.get_level_values(1):
            df = raw_df.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        df = raw_df.copy()

    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in cols):
        return pd.DataFrame()
    df = df[cols].dropna(subset=["Close", "Volume"])
    df = df[df["Volume"] > 0]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def download_in_chunks(tickers: list[str], chunk_size: int = 80) -> dict[str, pd.DataFrame]:
    result = {}
    total_batches = (len(tickers) + chunk_size - 1) // chunk_size
    for idx, i in enumerate(range(0, len(tickers), chunk_size), 1):
        batch = tickers[i : i + chunk_size]
        print(f"[行情] 正在下載第 {idx}/{total_batches} 批（{len(batch)} 隻）...")
        try:
            raw = yf.download(
                batch,
                period="2y",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            for t in batch:
                df = extract_ohlcv(raw, t)
                if len(df) >= 260:
                    result[t] = df
        except Exception as e:
            print(f"[警告] 第 {idx} 批下載異常: {e}")
        time.sleep(0.8)
    return result


def add_indicators(df: pd.DataFrame, bench_close: pd.Series) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["MA50"] = c.rolling(50).mean()
    df["MA150"] = c.rolling(150).mean()
    df["MA200"] = c.rolling(200).mean()
    df["Vol_MA50"] = v.rolling(50).mean()
    df["Avg_Turnover_20"] = (c * v).rolling(20).mean()

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    aligned_bench = bench_close.reindex(df.index).ffill()
    rs_raw = c / aligned_bench
    df["MRS"] = (rs_raw / rs_raw.rolling(252).mean() - 1) * 100

    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    fast_k = (c - low14) / (high14 - low14).replace(0, np.nan) * 100
    df["Stoch_K"] = fast_k.rolling(3).mean()

    df["High_60"] = h.rolling(60).max()
    df["High_252"] = h.rolling(252).max()
    df["Dist_52W_High"] = (df["High_252"] - c) / c
    df["Dist_MA50"] = (c - df["MA50"]) / df["MA50"]
    df["Pct_Chg"] = (c - prev_c) / prev_c * 100
    return df


# ==========================================
# 4. 專用偵測模組：缺口不補 & Stage 3 見頂派發
# ==========================================
def check_gap_pullback(df: pd.DataFrame, p: dict):
    latest, prev = df.iloc[-1], df.iloc[-2]
    if not (latest["Close"] > latest["MA150"] and latest["Close"] > latest["MA200"]):
        return None

    max_lb = min(p["gap_lookback_max"], len(df) - 2)
    for offset in range(p["gap_lookback_min"], max_lb + 1):
        idx = len(df) - 1 - offset
        bar_g, bar_prev = df.iloc[idx], df.iloc[idx - 1]
        if pd.isna(bar_g["Vol_MA50"]) or bar_g["Vol_MA50"] <= 0:
            continue

        gap_pct = (bar_g["Open"] - bar_prev["Close"]) / bar_prev["Close"]
        if (
            bar_g["Low"] > bar_prev["High"]
            and bar_g["Volume"] >= p["gap_vol_mult"] * bar_g["Vol_MA50"]
            and gap_pct >= p["gap_min"]
        ):
            gap_top, gap_bottom = bar_g["Low"], bar_prev["High"]
            after_gap = df.iloc[idx + 1 :]
            if (after_gap["Close"] < gap_bottom).any():
                continue

            dist_to_top = (latest["Close"] - gap_top) / gap_top
            if latest["Close"] >= gap_bottom and dist_to_top <= 0.06:
                vol_shrink = latest["Volume"] < latest["Vol_MA50"]
                rebound = latest["Close"] > latest["Open"] and latest["Close"] > prev["High"]
                status = (
                    "缺口上方企穩轉強"
                    if rebound
                    else ("縮量回踩觀察" if vol_shrink else "缺口上方震盪消化")
                )
                return {
                    "gap_date": df.index[idx].strftime("%Y-%m-%d"),
                    "gap_pct": safe_num(gap_pct * 100, 2),
                    "gap_zone": f"{gap_bottom:.2f} - {gap_top:.2f}",
                    "status": status,
                }
    return None


def detect_island_reversal(df: pd.DataFrame, max_span: int = 10) -> str | None:
    """偵測近 2 個交易日內形成的頂部島形反轉"""
    n = len(df)
    for d in (n - 1, n - 2):
        # 嚴格向下跳空缺口
        if not (df["High"].iloc[d] < df["Low"].iloc[d - 1]):
            continue
        down_high = df["High"].iloc[d]

        for span in range(1, max_span + 1):
            u = d - span
            if u <= 60:
                break
            # 前方存在嚴格向上跳空缺口
            if not (df["Low"].iloc[u] > df["High"].iloc[u - 1]):
                continue

            up_prev_high = df["High"].iloc[u - 1]
            island_slice = df.iloc[u:d]
            island_min_low = island_slice["Low"].min()
            island_max_high = island_slice["High"].max()
            ref_high_60 = df["High_60"].iloc[d - 1]

            # 孤島期間所有低點皆懸空於左右兩側缺口之上，且位於 60 日高位區
            if (
                island_min_low > up_prev_high
                and island_min_low > down_high
                and island_max_high >= ref_high_60 * 0.95
            ):
                return f"島形反轉({span}日孤島)"
    return None


def detect_stage3_signals(df: pd.DataFrame, p: dict) -> dict | None:
    """偵測高位劣質收盤、50MA 反覆穿插、島形反轉與 Stage 3B 預警"""
    latest, prev = df.iloc[-1], df.iloc[-2]

    # 前置過濾：排除已深陷第四階段的老弱股（近 80 日曾站上 200MA 且現價未深跌破 200MA）
    had_stage2_context = (df["Close"].iloc[-80:] > df["MA200"].iloc[-80:] * 1.03).any()
    not_deep_stage4 = latest["Close"] >= latest["MA200"] * 0.94
    if not (had_stage2_context and not_deep_stage4):
        return None

    signals = []
    severity = 0
    vol_ratio = float(latest["Volume"] / latest["Vol_MA50"])

    # --- 1. 島形反轉 ---
    island_tag = detect_island_reversal(df, p["s3_island_max_days"])
    if island_tag:
        signals.append(island_tag)
        severity = max(severity, 3)

    # --- 2. 高位劣質收盤（Lousy Closes near Highs） ---
    bar_range = (df["High"] - df["Low"]).replace(0, np.nan)
    close_pos = (df["Close"] - df["Low"]) / bar_range
    bar_amp = (df["High"] - df["Low"]) / df["Close"]
    near_high = df["High"] >= df["High_60"] * 0.96

    lousy_mask = (
        near_high
        & (bar_amp >= 0.025)
        & (close_pos <= 0.35)
        & (df["Volume"] >= 1.2 * df["Vol_MA50"])
    )
    lousy_win = p["s3_lousy_window"]
    lousy_cnt = int(lousy_mask.iloc[-lousy_win:].sum())
    lousy_recent = bool(lousy_mask.iloc[-3:].any())
    lousy_severe_today = bool(
        near_high.iloc[-1]
        and bar_amp.iloc[-1] >= 0.03
        and close_pos.iloc[-1] <= 0.25
        and vol_ratio >= 1.8
        and latest["Close"] < latest["Open"]
    )
    if (lousy_cnt >= 2 and lousy_recent) or lousy_severe_today:
        signals.append(f"高位劣質收盤({max(lousy_cnt, 1)}次)")
        severity = max(severity, 1)

    # --- 3. 50 日線反覆穿插（均線走平寬幅派發） ---
    above_50 = (df["Close"] > df["MA50"]).iloc[-40:]
    cross_50_cnt = int((above_50 != above_50.shift(1)).iloc[1:].sum())
    ma50_slope_20 = abs(latest["MA50"] / df["MA50"].iloc[-21] - 1)
    if (
        cross_50_cnt >= p["s3_whipsaw_min"]
        and ma50_slope_20 <= 0.04
        and latest["High_60"] >= latest["High_252"] * 0.80
    ):
        signals.append(f"50MA反覆穿插({cross_50_cnt}次)")
        severity = max(severity, 1)

    # --- 4. Stage 3B 預警（派發尾聲、瀕臨破位進入 Stage 4） ---
    neckline = float(df["Low"].iloc[-46:-1].min())
    dist_neckline_pct = (latest["Close"] - neckline) / max(neckline, 1e-6) * 100
    ma50_flat_or_down = latest["MA50"] <= df["MA50"].iloc[-11] * 1.005
    gap_down_vol = (latest["High"] < prev["Low"]) and (vol_ratio >= 1.5)
    failed_rally_50 = (
        latest["Close"] < latest["MA50"]
        and latest["Close"] < latest["Open"]
        and latest["High"] >= latest["MA50"] * 0.985
        and prev["Close"] < prev["MA50"]
    )

    is_stage_3b = (
        latest["Close"] < latest["MA50"]
        and ma50_flat_or_down
        and (
            dist_neckline_pct <= 3.5
            or failed_rally_50
            or gap_down_vol
            or cross_50_cnt >= 3
            or (island_tag is not None)
        )
    )
    if is_stage_3b:
        if dist_neckline_pct <= 0:
            signals.append("Stage 3B(跌穿頸線)")
            severity = max(severity, 3)
        elif failed_rally_50:
            signals.append("Stage 3B(反抽50MA受阻)")
            severity = max(severity, 3)
        elif gap_down_vol:
            signals.append("Stage 3B(放量向下跳空)")
            severity = max(severity, 3)
        else:
            signals.append("Stage 3B預警(派發末期)")
            severity = max(severity, 2)

    if not signals:
        return None

    stage_tag = "Stage 3B (高危)" if (is_stage_3b or island_tag) else "Stage 3A (警戒)"
    return {
        "stage_tag": stage_tag,
        "severity": severity,
        "cross_50": cross_50_cnt,
        "neckline": safe_num(neckline, 2),
        "dist_neckline": safe_num(dist_neckline_pct, 1),
        "signals": " | ".join(signals),
    }


def tv_url(yf_ticker: str) -> str:
    num = int(yf_ticker.split(".")[0])
    return f"https://www.tradingview.com/chart/?symbol=HKEX:{num}"


# ==========================================
# 5. 執行全量掃描
# ==========================================
def run_scan() -> dict:
    stock_map, source_name = get_ggt_stock_map()
    tickers = list(stock_map.keys())

    bench_raw = yf.download(BENCHMARK, period="2y", interval="1d", auto_adjust=True, progress=False)
    bench_df = extract_ohlcv(bench_raw, BENCHMARK)
    if len(bench_df) < 220:
        raise RuntimeError(f"基準標的 {BENCHMARK} 資料下載失敗。")

    bench_df["MA50"] = bench_df["Close"].rolling(50).mean()
    bench_df["MA200"] = bench_df["Close"].rolling(200).mean()
    bench_df["Pct_Chg"] = bench_df["Close"].pct_change() * 100
    b_now = bench_df.iloc[-1]
    b_ma200_up = bool(b_now["MA200"] >= bench_df["MA200"].iloc[-21])

    print(f"[行情] 開始全量掃描 {len(tickers)} 隻港股通標的...")
    data_map = download_in_chunks(tickers, CHUNK_SIZE)
    print(f"[行情] 成功獲取有效歷史資料：{len(data_map)} 隻。")

    stage2_list, gap_watch_list, stage4b_list, stage3_list, risk_list = [], [], [], [], []
    liquid_count, above_ma50_count, above_ma200_count = 0, 0, 0
    latest_dates = []
    p = PARAMS

    for ticker, raw_df in data_map.items():
        df = add_indicators(raw_df, bench_df["Close"])
        latest, prev = df.iloc[-1], df.iloc[-2]
        if (
            pd.isna(latest["MRS"])
            or pd.isna(latest["Vol_MA50"])
            or latest["Vol_MA50"] <= 0
            or latest["Avg_Turnover_20"] < MIN_AVG_TURNOVER
        ):
            continue

        latest_dates.append(df.index[-1])
        liquid_count += 1
        if latest["Close"] > latest["MA50"]:
            above_ma50_count += 1
        if latest["Close"] > latest["MA200"]:
            above_ma200_count += 1

        name = stock_map.get(ticker, "")
        vol_ratio = float(latest["Volume"] / latest["Vol_MA50"])
        gap_pct = float((latest["Open"] - prev["Close"]) / prev["Close"])
        is_gap_up = (latest["Low"] > prev["High"]) and (gap_pct >= p["gap_min"])
        turnover_m = safe_num(latest["Avg_Turnover_20"] / 1e6, 1)
        bar_range = max(float(latest["High"] - latest["Low"]), 1e-6)

        # --- 1. Stage 2 突破 ---
        ma150_rising = latest["MA150"] >= df["MA150"].iloc[-21]
        cond_s2_trend = (
            latest["Close"] > latest["MA150"]
            and latest["Close"] > latest["MA200"]
            and latest["MA50"] > latest["MA150"]
            and ma150_rising
            and latest["MRS"] > 0
        )
        box_high = df["Close"].iloc[-(p["box_window"] + 1) : -1].max()
        cond_box = (
            latest["Close"] > box_high
            and vol_ratio >= p["vol_mult_breakout"]
            and latest["Dist_MA50"] <= p["dist_ma50_max"]
        )
        cond_gap_buy = (
            is_gap_up
            and p["gap_min"] <= gap_pct <= p["gap_max_buy"]
            and vol_ratio >= p["gap_vol_mult"]
            and (latest["Close"] - latest["Low"]) / bar_range >= 0.5
        )
        cond_gap_ext = (
            cond_s2_trend
            and is_gap_up
            and gap_pct > p["gap_max_buy"]
            and vol_ratio >= p["gap_vol_mult"]
        )

        if cond_s2_trend and (cond_box or cond_gap_buy):
            stage2_list.append({
                "ticker": ticker,
                "name": name,
                "url": tv_url(ticker),
                "close": safe_num(latest["Close"], 2),
                "pct_chg": safe_num(latest["Pct_Chg"], 2),
                "type": "溫和跳空突破" if cond_gap_buy else "放量箱體突破",
                "vol_ratio": safe_num(vol_ratio, 2),
                "dist_ma50": safe_num(latest["Dist_MA50"] * 100, 1),
                "mrs": safe_num(latest["MRS"], 1),
                "stoch_k": safe_num(latest["Stoch_K"], 1),
                "turnover_m": turnover_m,
                "room": (
                    "創/近52週新高"
                    if latest["Dist_52W_High"] <= 0.03
                    else f"距前高 {latest['Dist_52W_High']*100:.1f}%"
                ),
            })

        # --- 2. 缺口回調觀察 ---
        if cond_gap_ext:
            gap_watch_list.append({
                "ticker": ticker,
                "name": name,
                "url": tv_url(ticker),
                "close": safe_num(latest["Close"], 2),
                "pct_chg": safe_num(latest["Pct_Chg"], 2),
                "gap_date": df.index[-1].strftime("%Y-%m-%d"),
                "gap_pct": safe_num(gap_pct * 100, 2),
                "gap_zone": f"{prev['High']:.2f} - {latest['Low']:.2f}",
                "mrs": safe_num(latest["MRS"], 1),
                "turnover_m": turnover_m,
                "status": "今日巨幅跳空(勿追高,等回踩)",
            })
        else:
            g_info = check_gap_pullback(df, p)
            if g_info:
                gap_watch_list.append({
                    "ticker": ticker,
                    "name": name,
                    "url": tv_url(ticker),
                    "close": safe_num(latest["Close"], 2),
                    "pct_chg": safe_num(latest["Pct_Chg"], 2),
                    "gap_date": g_info["gap_date"],
                    "gap_pct": g_info["gap_pct"],
                    "gap_zone": g_info["gap_zone"],
                    "mrs": safe_num(latest["MRS"], 1),
                    "turnover_m": turnover_m,
                    "status": g_info["status"],
                })

        # --- 3. 4B- 底部反轉 ---
        room_200 = (latest["MA200"] - latest["Close"]) / latest["Close"]
        dd_52w = (latest["High_252"] - latest["Close"]) / latest["High_252"]
        low_20, low_120 = df["Low"].iloc[-20:].min(), df["Low"].iloc[-120:].min()
        high_20 = df["High"].iloc[-20:].max()
        base_high_20 = df["Close"].iloc[-21:-1].max()
        base_amp = (high_20 - low_20) / max(low_20, 1e-6)

        if (
            latest["Close"] < latest["MA200"]
            and room_200 >= p["room_ma200_min"]
            and dd_52w >= p["drawdown_4b_min"]
            and low_20 >= low_120
            and base_amp <= 0.25
            and latest["Close"] > latest["MA50"]
            and latest["Close"] > base_high_20
            and vol_ratio >= p["vol_mult_breakout"]
        ):
            stop_ref = min(float(low_20), float(latest["MA50"] - latest["ATR14"]))
            stage4b_list.append({
                "ticker": ticker,
                "name": name,
                "url": tv_url(ticker),
                "close": safe_num(latest["Close"], 2),
                "pct_chg": safe_num(latest["Pct_Chg"], 2),
                "room_200": safe_num(room_200 * 100, 1),
                "dd_52w": safe_num(dd_52w * 100, 1),
                "vol_ratio": safe_num(vol_ratio, 2),
                "stop_ref": safe_num(stop_ref, 2),
                "target_ref": safe_num(latest["MA200"] * 0.96, 2),
                "turnover_m": turnover_m,
            })

        # --- 4. Stage 3 見頂派發（高位劣質收盤 / 50MA穿插 / 島形反轉 / Stage 3B） ---
        s3_info = detect_stage3_signals(df, p)
        if s3_info:
            stage3_list.append({
                "ticker": ticker,
                "name": name,
                "url": tv_url(ticker),
                "close": safe_num(latest["Close"], 2),
                "pct_chg": safe_num(latest["Pct_Chg"], 2),
                "stage_tag": s3_info["stage_tag"],
                "severity": s3_info["severity"],
                "cross_50": s3_info["cross_50"],
                "neckline": s3_info["neckline"],
                "dist_neckline": s3_info["dist_neckline"],
                "vol_ratio": safe_num(vol_ratio, 2),
                "turnover_m": turnover_m,
                "signals": s3_info["signals"],
            })

        # --- 5. 50MA 風控 / 止盈（過度延伸 Trimming & 剛跌破 50MA） ---
        alerts = []
        is_key_rev = (
            latest["High"] >= df["High"].iloc[-20:].max()
            and latest["Close"] < prev["Close"]
            and (latest["Close"] - latest["Low"]) / bar_range <= 0.25
            and vol_ratio >= 1.5
        )
        if latest["Dist_MA50"] >= p["trim_dist_ma50"] or is_key_rev:
            reason = "關鍵反轉日" if is_key_rev else f"乖離率過高({latest['Dist_MA50']*100:.1f}%)"
            alerts.append(f"分批止盈({reason})")

        recently_above_50 = (df["Close"].iloc[-6:-1] > df["MA50"].iloc[-6:-1]).any()
        ma50_falling = latest["MA50"] < df["MA50"].iloc[-6]
        gap_down_vol = (latest["High"] < prev["Low"]) and (vol_ratio >= 1.5)
        two_below = (
            (latest["Close"] < latest["MA50"])
            and (prev["Close"] < prev["MA50"])
            and (df["Close"].iloc[-3] >= df["MA50"].iloc[-3])
        )
        failed_rally = (
            latest["Close"] < latest["MA50"]
            and latest["Close"] < latest["Open"]
            and latest["High"] >= latest["MA50"] * 0.985
            and prev["Close"] < prev["MA50"]
        )

        if recently_above_50:
            if failed_rally:
                alerts.append("50MA反抽受阻(徹底清倉)")
            elif latest["Close"] < latest["MA50"] and gap_down_vol:
                alerts.append("放量跳空跌破50MA(清倉)")
            elif two_below or (latest["Close"] < latest["MA50"] and ma50_falling and prev["Close"] >= prev["MA50"]):
                alerts.append("剛有效跌破50MA(離場)")

        if alerts:
            risk_list.append({
                "ticker": ticker,
                "name": name,
                "url": tv_url(ticker),
                "close": safe_num(latest["Close"], 2),
                "pct_chg": safe_num(latest["Pct_Chg"], 2),
                "ma50": safe_num(latest["MA50"], 2),
                "dist_ma50": safe_num(latest["Dist_MA50"] * 100, 1),
                "vol_ratio": safe_num(vol_ratio, 2),
                "turnover_m": turnover_m,
                "signal": " | ".join(alerts),
            })

    stage2_list.sort(key=lambda x: x["mrs"], reverse=True)
    gap_watch_list.sort(key=lambda x: x["mrs"], reverse=True)
    stage4b_list.sort(key=lambda x: x["room_200"], reverse=True)
    stage3_list.sort(key=lambda x: (x["severity"], x["vol_ratio"]), reverse=True)
    risk_list.sort(key=lambda x: x["vol_ratio"], reverse=True)

    hk_now = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M HKT")
    trade_date = (
        max(latest_dates).strftime("%Y-%m-%d")
        if latest_dates
        else bench_df.index[-1].strftime("%Y-%m-%d")
    )

    return {
        "updated_at": hk_now,
        "trade_date": trade_date,
        "source_name": source_name,
        "universe": {
            "total_listed": len(tickers),
            "total_fetched": len(data_map),
            "liquid_count": liquid_count,
            "breadth_ma50": safe_num(above_ma50_count / max(liquid_count, 1) * 100, 1),
            "breadth_ma200": safe_num(above_ma200_count / max(liquid_count, 1) * 100, 1),
        },
        "benchmark": {
            "symbol": BENCHMARK,
            "label": BENCHMARK_LABEL,
            "close": safe_num(b_now["Close"], 2),
            "pct_chg": safe_num(b_now["Pct_Chg"], 2),
            "long_bull": bool(b_now["Close"] > b_now["MA200"] and b_ma200_up),
            "short_bull": bool(b_now["Close"] > b_now["MA50"]),
        },
        "stage2": stage2_list,
        "gap_watch": gap_watch_list,
        "stage4b": stage4b_list,
        "stage3": stage3_list,
        "risk": risk_list,
    }


# ==========================================
# 6. 生成綠漲紅跌互動式 HTML 看板
# ==========================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-HK">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>全量港股通 Stan Weinstein 現代版選股看板</title>
  <style>
    :root {
      --bg: #0d1117; --card: #161b22; --border: #30363d;
      --text: #e6edf3; --muted: #8b949e;
      --up: #22c55e; --down: #ef4444; --warn: #f59e0b; --accent: #38bdf8;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang HK", sans-serif; background: var(--bg); color: var(--text); padding: 20px; }
    .container { max-width: 1400px; margin: 0 auto; }
    header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 14px; }
    h1 { margin: 0; font-size: 1.45rem; }
    .meta { color: var(--muted); font-size: 0.86rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 20px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
    .card-label { font-size: 0.8rem; color: var(--muted); margin-bottom: 6px; }
    .card-val { font-size: 1.35rem; font-weight: 700; }
    .card-sub { font-size: 0.82rem; color: var(--muted); margin-top: 4px; }
    .up { color: var(--up); } .down { color: var(--down); } .warn { color: var(--warn); }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; }
    .badge-up { background: rgba(34,197,94,0.16); color: var(--up); }
    .badge-down { background: rgba(239,68,68,0.16); color: var(--down); }
    .badge-warn { background: rgba(245,158,11,0.16); color: var(--warn); }
    .toolbar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; }
    .tab-btn { background: var(--card); color: var(--muted); border: 1px solid var(--border); padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: 0.15s; }
    .tab-btn:hover { color: var(--text); border-color: var(--accent); }
    .tab-btn.active { background: #1f2937; color: var(--text); border-color: var(--accent); }
    .search { background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; width: 250px; font-size: 0.9rem; }
    .table-wrap { background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { padding: 11px 14px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
    th { color: var(--muted); font-weight: 600; background: #121720; cursor: pointer; user-select: none; }
    th:hover { color: var(--text); }
    tr:hover td { background: rgba(56,189,248,0.06); }
    a.ticker-link { color: var(--accent); text-decoration: none; font-weight: 700; }
    a.ticker-link:hover { text-decoration: underline; }
    .empty { text-align: center; padding: 36px; color: var(--muted); }
    footer { margin-top: 22px; text-align: center; color: var(--muted); font-size: 0.8rem; }
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>全量港股通 Stan Weinstein 現代版選股看板</h1>
    <div class="meta" id="metaInfo"></div>
  </header>

  <section class="grid" id="kpiGrid"></section>

  <div class="toolbar">
    <div class="tabs" id="tabs"></div>
    <input class="search" id="searchInput" type="text" placeholder="搜尋代碼、名稱或訊號 (如 島形 / 3B)" />
  </div>

  <div class="table-wrap">
    <table id="dataTable"></table>
  </div>

  <footer>資料來源：港交所/深交所官方港股通名單 & Yahoo Finance ｜ 點擊表頭可排序 ｜ 配色：綠漲紅跌 ｜ 僅供技術面量化復盤參考</footer>
</div>

<script>
const DATA = __DATA_JSON__;

const TABS = [
  { key: "stage2", label: "Stage 2 突破", cols: [
    ["ticker", "代碼 (TV圖表)"], ["name", "名稱"], ["close", "收盤價"], ["pct_chg", "漲跌幅%"],
    ["type", "突破類型"], ["vol_ratio", "量比(50MA)"], ["dist_ma50", "距50MA%"],
    ["mrs", "MRS強度"], ["stoch_k", "StochK"], ["turnover_m", "20日均額(M)"], ["room", "上方空間"]
  ]},
  { key: "gap_watch", label: "缺口回調觀察", cols: [
    ["ticker", "代碼 (TV圖表)"], ["name", "名稱"], ["close", "收盤價"], ["pct_chg", "漲跌幅%"],
    ["gap_date", "跳空日期"], ["gap_pct", "跳空幅度%"], ["gap_zone", "未補缺口區間"],
    ["mrs", "MRS強度"], ["turnover_m", "20日均額(M)"], ["status", "追蹤狀態"]
  ]},
  { key: "stage4b", label: "4B- 底部反轉", cols: [
    ["ticker", "代碼 (TV圖表)"], ["name", "名稱"], ["close", "收盤價"], ["pct_chg", "漲跌幅%"],
    ["room_200", "距200MA空間%"], ["dd_52w", "52週回撤%"], ["vol_ratio", "量比(50MA)"],
    ["stop_ref", "參考止損"], ["target_ref", "近200MA目標"], ["turnover_m", "20日均額(M)"]
  ]},
  { key: "stage3", label: "Stage 3 見頂派發", cols: [
    ["ticker", "代碼 (TV圖表)"], ["name", "名稱"], ["close", "收盤價"], ["pct_chg", "漲跌幅%"],
    ["stage_tag", "階段判定"], ["signals", "見頂 / 派發訊號"], ["cross_50", "50MA穿插(40日)"],
    ["neckline", "頸線支撐參考"], ["dist_neckline", "距頸線%"], ["vol_ratio", "量比(50MA)"], ["turnover_m", "20日均額(M)"]
  ]},
  { key: "risk", label: "50MA 風控 / 止盈", cols: [
    ["ticker", "代碼 (TV圖表)"], ["name", "名稱"], ["close", "收盤價"], ["pct_chg", "漲跌幅%"],
    ["ma50", "50日均線"], ["dist_ma50", "距50MA%"], ["vol_ratio", "量比(50MA)"],
    ["turnover_m", "20日均額(M)"], ["signal", "風控訊號"]
  ]}
];

let activeKey = "stage2";
let sortState = { col: null, asc: false };

function fmtPct(v) {
  const cls = v > 0 ? "up" : (v < 0 ? "down" : "");
  const sign = v > 0 ? "+" : "";
  return `<span class="${cls}">${sign}${v}%</span>`;
}

function renderHeader() {
  const b = DATA.benchmark, u = DATA.universe;
  document.getElementById("metaInfo").innerText =
    `行情日期：${DATA.trade_date} ｜ 更新：${DATA.updated_at} ｜ 名單源：${DATA.source_name}（達標流動股 ${u.liquid_count} / 有效資料 ${u.total_fetched} / 總名單 ${u.total_listed}）`;

  const longTag = b.long_bull ? `<span class="badge badge-up">長牛 (200MA上行)</span>` : `<span class="badge badge-down">防守 (200MA偏弱)</span>`;
  const shortTag = b.short_bull ? `<span class="badge badge-up">短多 (站上50MA)</span>` : `<span class="badge badge-warn">短線回調 (破50MA)</span>`;
  const s3bCount = DATA.stage3.filter(x => x.stage_tag.includes("3B")).length;

  document.getElementById("kpiGrid").innerHTML = `
    <div class="card">
      <div class="card-label">${b.label} (${b.symbol})</div>
      <div class="card-val">${b.close} <span style="font-size:0.95rem">${fmtPct(b.pct_chg)}</span></div>
      <div class="card-sub">${longTag} ${shortTag}</div>
    </div>
    <div class="card">
      <div class="card-label">全量港股通市場廣度</div>
      <div class="card-val ${u.breadth_ma50 >= 50 ? 'up' : 'down'}">${u.breadth_ma50}%</div>
      <div class="card-sub">站上 50MA 佔比 ｜ 站上 200MA：${u.breadth_ma200}%</div>
    </div>
    <div class="card">
      <div class="card-label">多頭與超跌機會</div>
      <div class="card-val up">${DATA.stage2.length + DATA.gap_watch.length} <span style="font-size:0.95rem" class="warn">/ 4B- ${DATA.stage4b.length}</span></div>
      <div class="card-sub">Stage 2 突破 ${DATA.stage2.length} ｜ 缺口觀察 ${DATA.gap_watch.length}</div>
    </div>
    <div class="card">
      <div class="card-label">見頂派發 / 50MA 風控</div>
      <div class="card-val down">${DATA.stage3.length} <span style="font-size:0.95rem" class="warn">/ 風控 ${DATA.risk.length}</span></div>
      <div class="card-sub">Stage 3B 高危 ${s3bCount} ｜ 3A 警戒 ${DATA.stage3.length - s3bCount}</div>
    </div>
  `;
}

function renderTabs() {
  document.getElementById("tabs").innerHTML = TABS.map(t => `
    <button class="tab-btn ${t.key === activeKey ? 'active' : ''}" onclick="switchTab('${t.key}')">
      ${t.label} (${DATA[t.key].length})
    </button>
  `).join("");
}

function switchTab(k) {
  activeKey = k;
  sortState = { col: null, asc: false };
  renderTabs();
  renderTable();
}

function sortBy(col) {
  if (sortState.col === col) sortState.asc = !sortState.asc;
  else { sortState.col = col; sortState.asc = false; }
  renderTable();
}

function renderTable() {
  const tab = TABS.find(t => t.key === activeKey);
  const q = document.getElementById("searchInput").value.trim().toLowerCase();
  let rows = DATA[activeKey].filter(r =>
    !q
    || r.ticker.toLowerCase().includes(q)
    || (r.name && r.name.toLowerCase().includes(q))
    || (r.signals && r.signals.toLowerCase().includes(q))
    || (r.signal && r.signal.toLowerCase().includes(q))
    || (r.stage_tag && r.stage_tag.toLowerCase().includes(q))
  );

  if (sortState.col) {
    const col = sortState.col, dir = sortState.asc ? 1 : -1;
    rows = [...rows].sort((a, b) => {
      const va = a[col], vb = b[col];
      if (typeof va === "number" && typeof vb === "number") return (va - vb) * dir;
      return String(va).localeCompare(String(vb), "zh-HK") * dir;
    });
  }

  const thead = `<thead><tr>${tab.cols.map(([k, label]) => {
    const arrow = sortState.col === k ? (sortState.asc ? " ▲" : " ▼") : "";
    return `<th onclick="sortBy('${k}')">${label}${arrow}</th>`;
  }).join("")}</tr></thead>`;

  if (!rows.length) {
    document.getElementById("dataTable").innerHTML =
      thead + `<tbody><tr><td colspan="${tab.cols.length}" class="empty">當日暫無符合條件的標的</td></tr></tbody>`;
    return;
  }

  const tbody = rows.map(r => `<tr>${tab.cols.map(([k]) => {
    const val = r[k];
    if (k === "ticker") return `<td><a class="ticker-link" href="${r.url}" target="_blank" rel="noopener">${val}</a></td>`;
    if (k === "pct_chg" || k === "dist_ma50" || k === "dist_neckline") return `<td>${fmtPct(val)}</td>`;
    if (k === "mrs") return `<td class="${val >= 0 ? 'up' : 'down'}">${val}</td>`;
    if (k === "type" || k === "status") return `<td><span class="badge badge-up">${val}</span></td>`;
    if (k === "stage_tag") return `<td><span class="badge ${val.includes('3B') ? 'badge-down' : 'badge-warn'}">${val}</span></td>`;
    if (k === "signals") return `<td><span class="badge ${val.includes('島形') || val.includes('3B') ? 'badge-down' : 'badge-warn'}">${val}</span></td>`;
    if (k === "signal") return `<td><span class="badge ${val.includes('清倉') || val.includes('離場') ? 'badge-down' : 'badge-warn'}">${val}</span></td>`;
    return `<td>${val}</td>`;
  }).join("")}</tr>`).join("");

  document.getElementById("dataTable").innerHTML = thead + `<tbody>${tbody}</tbody>`;
}

document.getElementById("searchInput").addEventListener("input", renderTable);
renderHeader(); renderTabs(); renderTable();
</script>
</body>
</html>
"""


def build_dashboard(payload: dict):
    os.makedirs(DOCS_DIR, exist_ok=True)
    json_str = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json_str)
    out_path = os.path.join(DOCS_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[完成] 互動式看板已生成：{out_path}")


if __name__ == "__main__":
    scan_data = run_scan()
    build_dashboard(scan_data)
