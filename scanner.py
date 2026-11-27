import json
import os
import time
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# 1. 参数配置区
# ==========================================
BENCHMARK = "^HSI"                 # 基准指数：恒生指数
MIN_AVG_TURNOVER = 30_000_000      # 20日日均成交额下限（3,000 万港元）
CHUNK_SIZE = 80                    # 分批下载大小，避免触发 Yahoo Finance 限流
DOCS_DIR = "docs"
CACHE_CSV = os.path.join(DOCS_DIR, "ggt_tickers.csv")

PARAMS = {
    "box_window": 50,              # Stage 2 箱体突破天数
    "vol_mult_breakout": 1.5,      # 突破量比阈值（相对 50 日均量）
    "dist_ma50_max": 0.15,         # 防追高：距 50 日线最大乖离率（15%）
    "gap_min": 0.02,               # 有效跳空最小幅度（2%）
    "gap_max_buy": 0.08,           # 首日可追跳空上限（8%）
    "gap_vol_mult": 1.8,           # 放量跳空量比阈值
    "gap_lookback_min": 3,         # 缺口不补回溯最小天数
    "gap_lookback_max": 15,        # 缺口不补回溯最大天数
    "room_ma200_min": 0.15,        # 4B- 距 200 日线最小反弹空间（15%）
    "drawdown_4b_min": 0.30,       # 4B- 距 52 周高点最小回撤（30%）
    "trim_dist_ma50": 0.25,        # 过度延伸分批止盈阈值（25%）
}

# 内置兜底核心港股通名单（当接口与缓存均不可用时启用）
FALLBACK_GGT = {
    "0700.HK": "腾讯控股", "9988.HK": "阿里巴巴-W", "3690.HK": "美团-W",
    "1810.HK": "小米集团-W", "1211.HK": "比亚迪股份", "0981.HK": "中芯国际",
    "0388.HK": "香港交易所", "2318.HK": "中国平安", "0941.HK": "中国移动",
    "0883.HK": "中国海洋石油", "9618.HK": "京东集团-SW", "9999.HK": "网易-S",
    "2015.HK": "理想汽车-W", "9868.HK": "小鹏汽车-W", "0005.HK": "汇丰控股",
    "1024.HK": "快手-W", "0285.HK": "比亚迪电子", "2382.HK": "舜宇光学科技",
    "0669.HK": "创科实业", "0857.HK": "中国石油股份", "0939.HK": "建设银行",
    "1398.HK": "工商银行", "3968.HK": "招商银行", "2899.HK": "紫金矿业",
    "1816.HK": "中广核电力", "2269.HK": "药明生物", "6160.HK": "百济神州",
    "9926.HK": "康方生物", "0020.HK": "商汤-W", "9880.HK": "优必选",
}


# ==========================================
# 2. 获取港股通成分股列表（自动转 yfinance 4位代码）
# ==========================================
def to_yf_hk_ticker(raw_code: str) -> str:
    """将 5 位港股代码（如 00700）转为 yfinance 4 位格式（0700.HK）"""
    digits = "".join(ch for ch in str(raw_code) if ch.isdigit())
    num = int(digits)
    return f"{num:04d}.HK" if num < 10000 else f"{num}.HK"


def get_ggt_stock_map() -> dict[str, str]:
    """优先用 akshare 拉取最新港股通成分股，失败则读取缓存 CSV 或兜底名单"""
    os.makedirs(DOCS_DIR, exist_ok=True)
    try:
        import akshare as ak
        df_ggt = ak.stock_hk_ggt_components_em()
        code_col = "代码" if "代码" in df_ggt.columns else df_ggt.columns[1]
        name_col = "名称" if "名称" in df_ggt.columns else df_ggt.columns[2]

        stock_map = {}
        for _, row in df_ggt.iterrows():
            yf_code = to_yf_hk_ticker(row[code_col])
            stock_map[yf_code] = str(row[name_col]).strip()

        if len(stock_map) >= 100:
            pd.DataFrame(
                [{"ticker": k, "name": v} for k, v in stock_map.items()]
            ).to_csv(CACHE_CSV, index=False, encoding="utf-8-sig")
            print(f"[名单] akshare 获取成功，共 {len(stock_map)} 只港股通标的。")
            return stock_map
    except Exception as e:
        print(f"[警告] akshare 拉取港股通名单失败，尝试本地缓存: {e}")

    if os.path.exists(CACHE_CSV):
        cached = pd.read_csv(CACHE_CSV)
        if not cached.empty:
            stock_map = dict(zip(cached["ticker"], cached["name"]))
            print(f"[名单] 已加载本地缓存，共 {len(stock_map)} 只港股通标的。")
            return stock_map

    print(f"[名单] 使用内置兜底名单，共 {len(FALLBACK_GGT)} 只核心标的。")
    return FALLBACK_GGT.copy()


# ==========================================
# 3. 批量下载与技术指标计算
# ==========================================
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
    return df[df["Volume"] > 0]


def download_in_chunks(tickers: list[str], chunk_size: int = 80) -> dict[str, pd.DataFrame]:
    result = {}
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i : i + chunk_size]
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
            print(f"[警告] 批次 {i}-{i + len(batch)} 下载异常: {e}")
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

    df["High_252"] = h.rolling(252).max()
    df["Dist_52W_High"] = (df["High_252"] - c) / c
    df["Dist_MA50"] = (c - df["MA50"]) / df["MA50"]
    df["Pct_Chg"] = (c - prev_c) / prev_c * 100
    return df


def check_gap_pullback(df: pd.DataFrame, p: dict):
    latest, prev = df.iloc[-1], df.iloc[-2]
    if not (latest["Close"] > latest["MA150"] and latest["Close"] > latest["MA200"]):
        return None

    max_lb = min(p["gap_lookback_max"], len(df) - 2)
    for offset in range(p["gap_lookback_min"], max_lb + 1):
        idx = len(df) - 1 - offset
        bar_g, bar_prev = df.iloc[idx], df.iloc[idx - 1]
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
                status = "缺口上方企稳转强" if rebound else ("缩量回踩观察" if vol_shrink else "缺口上方震荡消化")
                return {
                    "gap_date": df.index[idx].strftime("%Y-%m-%d"),
                    "gap_pct": round(gap_pct * 100, 2),
                    "gap_zone": f"{gap_bottom:.2f} - {gap_top:.2f}",
                    "status": status,
                }
    return None


def tv_url(yf_ticker: str) -> str:
    num = int(yf_ticker.split(".")[0])
    return f"https://www.tradingview.com/chart/?symbol=HKEX:{num}"


# ==========================================
# 4. 运行扫描并生成 JSON 数据包
# ==========================================
def run_scan() -> dict:
    stock_map = get_ggt_stock_map()
    tickers = list(stock_map.keys())

    # 下载恒生指数作为基准
    bench_raw = yf.download(BENCHMARK, period="2y", interval="1d", auto_adjust=True, progress=False)
    bench_df = extract_ohlcv(bench_raw, BENCHMARK)
    if len(bench_df) < 220:
        raise RuntimeError("恒生指数基准数据下载失败。")

    bench_df["MA50"] = bench_df["Close"].rolling(50).mean()
    bench_df["MA200"] = bench_df["Close"].rolling(200).mean()
    bench_df["Pct_Chg"] = bench_df["Close"].pct_change() * 100
    b_now = bench_df.iloc[-1]
    b_ma200_up = bool(b_now["MA200"] >= bench_df["MA200"].iloc[-21])

    print(f"[行情] 开始分批下载 {len(tickers)} 只港股通日线数据...")
    data_map = download_in_chunks(tickers, CHUNK_SIZE)
    print(f"[行情] 有效数据标的：{len(data_map)} 只。")

    stage2_list, gap_watch_list, stage4b_list, risk_list = [], [], [], []
    liquid_count, above_ma50_count, above_ma200_count = 0, 0, 0
    p = PARAMS

    for ticker, raw_df in data_map.items():
        df = add_indicators(raw_df, bench_df["Close"])
        latest, prev = df.iloc[-1], df.iloc[-2]
        if pd.isna(latest["MRS"]) or latest["Avg_Turnover_20"] < MIN_AVG_TURNOVER:
            continue

        liquid_count += 1
        if latest["Close"] > latest["MA50"]:
            above_ma50_count += 1
        if latest["Close"] > latest["MA200"]:
            above_ma200_count += 1

        name = stock_map.get(ticker, "")
        vol_ratio = float(latest["Volume"] / latest["Vol_MA50"])
        gap_pct = float((latest["Open"] - prev["Close"]) / prev["Close"])
        is_gap_up = (latest["Low"] > prev["High"]) and (gap_pct >= p["gap_min"])
        turnover_m = round(float(latest["Avg_Turnover_20"]) / 1e6, 1)

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
        bar_range = max(float(latest["High"] - latest["Low"]), 1e-6)
        cond_gap_buy = (
            is_gap_up
            and p["gap_min"] <= gap_pct <= p["gap_max_buy"]
            and vol_ratio >= p["gap_vol_mult"]
            and (latest["Close"] - latest["Low"]) / bar_range >= 0.5
        )
        cond_gap_ext = cond_s2_trend and is_gap_up and gap_pct > p["gap_max_buy"] and vol_ratio >= p["gap_vol_mult"]

        if cond_s2_trend and (cond_box or cond_gap_buy):
            stage2_list.append({
                "ticker": ticker,
                "name": name,
                "url": tv_url(ticker),
                "close": round(float(latest["Close"]), 2),
                "pct_chg": round(float(latest["Pct_Chg"]), 2),
                "type": "温和跳空突破" if cond_gap_buy else "放量箱体突破",
                "vol_ratio": round(vol_ratio, 2),
                "dist_ma50": round(float(latest["Dist_MA50"]) * 100, 1),
                "mrs": round(float(latest["MRS"]), 1),
                "stoch_k": round(float(latest["Stoch_K"]), 1),
                "turnover_m": turnover_m,
                "room": "创/近52周新高" if latest["Dist_52W_High"] <= 0.03 else f"距前高 {latest['Dist_52W_High']*100:.1f}%",
            })

        # --- 2. 缺口回调观察 ---
        if cond_gap_ext:
            gap_watch_list.append({
                "ticker": ticker,
                "name": name,
                "url": tv_url(ticker),
                "close": round(float(latest["Close"]), 2),
                "pct_chg": round(float(latest["Pct_Chg"]), 2),
                "gap_date": df.index[-1].strftime("%Y-%m-%d"),
                "gap_pct": round(gap_pct * 100, 2),
                "gap_zone": f"{prev['High']:.2f} - {latest['Low']:.2f}",
                "mrs": round(float(latest["MRS"]), 1),
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
                    "close": round(float(latest["Close"]), 2),
                    "pct_chg": round(float(latest["Pct_Chg"]), 2),
                    "gap_date": g_info["gap_date"],
                    "gap_pct": g_info["gap_pct"],
                    "gap_zone": g_info["gap_zone"],
                    "mrs": round(float(latest["MRS"]), 1),
                    "turnover_m": turnover_m,
                    "status": g_info["status"],
                })

        # --- 3. 4B- 底部反转 ---
        room_200 = (latest["MA200"] - latest["Close"]) / latest["Close"]
        dd_52w = (latest["High_252"] - latest["Close"]) / latest["High_252"]
        low_20, low_120 = df["Low"].iloc[-20:].min(), df["Low"].iloc[-120:].min()
        high_20 = df["High"].iloc[-20:].max()
        base_high_20 = df["Close"].iloc[-21:-1].max()
        base_amp = (high_20 - low_20) / low_20

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
                "close": round(float(latest["Close"]), 2),
                "pct_chg": round(float(latest["Pct_Chg"]), 2),
                "room_200": round(float(room_200) * 100, 1),
                "dd_52w": round(float(dd_52w) * 100, 1),
                "vol_ratio": round(vol_ratio, 2),
                "stop_ref": round(stop_ref, 2),
                "target_ref": round(float(latest["MA200"]) * 0.96, 2),
                "turnover_m": turnover_m,
            })

        # --- 4. 现代版风控警报（止盈 / 50日线离场） ---
        alerts = []
        is_key_rev = (
            latest["High"] >= df["High"].iloc[-20:].max()
            and latest["Close"] < prev["Close"]
            and (latest["Close"] - latest["Low"]) / bar_range <= 0.25
            and vol_ratio >= 1.5
        )
        if latest["Dist_MA50"] >= p["trim_dist_ma50"] or is_key_rev:
            reason = "关键反转日" if is_key_rev else f"乖离率过高({latest['Dist_MA50']*100:.1f}%)"
            alerts.append(f"分批止盈({reason})")

        ma50_falling = latest["MA50"] < df["MA50"].iloc[-6]
        gap_down_vol = (latest["High"] < prev["Low"]) and (vol_ratio >= 1.5)
        two_below = (latest["Close"] < latest["MA50"]) and (prev["Close"] < prev["MA50"])
        failed_rally = (
            latest["Close"] < latest["MA50"]
            and latest["Close"] < latest["Open"]
            and latest["High"] >= latest["MA50"] * 0.985
            and prev["Close"] < prev["MA50"]
        )

        # 仅对前期处于上升或刚破位（20日内曾站上50日线）的标的发出离场预警，避免满屏老弱股
        was_above_50 = (df["Close"].iloc[-20:-1] > df["MA50"].iloc[-20:-1]).any()
        if was_above_50:
            if failed_rally:
                alerts.append("50MA反抽受阻(彻底清仓)")
            elif latest["Close"] < latest["MA50"] and gap_down_vol:
                alerts.append("放量跳空跌破50MA(清仓)")
            elif two_below or (latest["Close"] < latest["MA50"] and ma50_falling):
                alerts.append("有效跌破50MA(离场)")

        if alerts:
            risk_list.append({
                "ticker": ticker,
                "name": name,
                "url": tv_url(ticker),
                "close": round(float(latest["Close"]), 2),
                "pct_chg": round(float(latest["Pct_Chg"]), 2),
                "ma50": round(float(latest["MA50"]), 2),
                "dist_ma50": round(float(latest["Dist_MA50"]) * 100, 1),
                "vol_ratio": round(vol_ratio, 2),
                "signal": " | ".join(alerts),
            })

    # 按相对强度或空间降序排序
    stage2_list.sort(key=lambda x: x["mrs"], reverse=True)
    gap_watch_list.sort(key=lambda x: x["mrs"], reverse=True)
    stage4b_list.sort(key=lambda x: x["room_200"], reverse=True)
    risk_list.sort(key=lambda x: x["vol_ratio"], reverse=True)

    hk_now = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M HKT")
    trade_date = bench_df.index[-1].strftime("%Y-%m-%d")

    return {
        "updated_at": hk_now,
        "trade_date": trade_date,
        "universe": {
            "total_fetched": len(data_map),
            "liquid_count": liquid_count,
            "breadth_ma50": round(above_ma50_count / max(liquid_count, 1) * 100, 1),
            "breadth_ma200": round(above_ma200_count / max(liquid_count, 1) * 100, 1),
        },
        "benchmark": {
            "symbol": BENCHMARK,
            "close": round(float(b_now["Close"]), 2),
            "pct_chg": round(float(b_now["Pct_Chg"]), 2),
            "long_bull": bool(b_now["Close"] > b_now["MA200"] and b_ma200_up),
            "short_bull": bool(b_now["Close"] > b_now["MA50"]),
        },
        "stage2": stage2_list,
        "gap_watch": gap_watch_list,
        "stage4b": stage4b_list,
        "risk": risk_list,
    }


# ==========================================
# 5. 生成绿涨红跌交互式 HTML 看板
# ==========================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-HK">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>港股通 Stan Weinstein 现代版选股看板</title>
  <style>
    :root {
      --bg: #0d1117; --card: #161b22; --border: #30363d;
      --text: #e6edf3; --muted: #8b949e;
      --up: #22c55e; --down: #ef4444; --warn: #f59e0b; --accent: #38bdf8;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang HK", sans-serif; background: var(--bg); color: var(--text); padding: 20px; }
    .container { max-width: 1320px; margin: 0 auto; }
    header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 14px; }
    h1 { margin: 0; font-size: 1.45rem; }
    .meta { color: var(--muted); font-size: 0.88rem; }
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
    .search { background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; width: 240px; font-size: 0.9rem; }
    .table-wrap { background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { padding: 11px 14px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
    th { color: var(--muted); font-weight: 600; background: #121720; }
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
    <h1>港股通 Stan Weinstein 现代版选股看板</h1>
    <div class="meta" id="metaInfo"></div>
  </header>

  <section class="grid" id="kpiGrid"></section>

  <div class="toolbar">
    <div class="tabs" id="tabs"></div>
    <input class="search" id="searchInput" type="text" placeholder="搜索代码或名称 (如 0700 / 腾讯)" />
  </div>

  <div class="table-wrap">
    <table id="dataTable"></table>
  </div>

  <footer>数据来源：Yahoo Finance & 东方财富港股通列表 ｜ 配色：绿涨红跌 ｜ 仅供技术面量化复盘参考，不构成投资建议</footer>
</div>

<script>
const DATA = __DATA_JSON__;

const TABS = [
  { key: "stage2", label: "Stage 2 突破", cols: [
    ["ticker", "代码 (TV图表)"], ["name", "名称"], ["close", "收盘价"], ["pct_chg", "涨跌幅%"],
    ["type", "突破类型"], ["vol_ratio", "量比(50MA)"], ["dist_ma50", "距50MA%"],
    ["mrs", "MRS强度"], ["stoch_k", "StochK"], ["turnover_m", "20日均额(M)"], ["room", "上方空间"]
  ]},
  { key: "gap_watch", label: "缺口回调观察", cols: [
    ["ticker", "代码 (TV图表)"], ["name", "名称"], ["close", "收盘价"], ["pct_chg", "涨跌幅%"],
    ["gap_date", "跳空日期"], ["gap_pct", "跳空幅度%"], ["gap_zone", "未补缺口区间"],
    ["mrs", "MRS强度"], ["turnover_m", "20日均额(M)"], ["status", "追踪状态"]
  ]},
  { key: "stage4b", label: "4B- 底部反转", cols: [
    ["ticker", "代码 (TV图表)"], ["name", "名称"], ["close", "收盘价"], ["pct_chg", "涨跌幅%"],
    ["room_200", "距200MA空间%"], ["dd_52w", "52周回撤%"], ["vol_ratio", "量比(50MA)"],
    ["stop_ref", "参考止损"], ["target_ref", "近200MA目标"], ["turnover_m", "20日均额(M)"]
  ]},
  { key: "risk", label: "50MA 风控 / 止盈", cols: [
    ["ticker", "代码 (TV图表)"], ["name", "名称"], ["close", "收盘价"], ["pct_chg", "涨跌幅%"],
    ["ma50", "50日均线"], ["dist_ma50", "距50MA%"], ["vol_ratio", "量比(50MA)"], ["signal", "风控信号"]
  ]}
];

let activeKey = "stage2";

function fmtPct(v) {
  const cls = v > 0 ? "up" : (v < 0 ? "down" : "");
  const sign = v > 0 ? "+" : "";
  return `<span class="${cls}">${sign}${v}%</span>`;
}

function renderHeader() {
  const b = DATA.benchmark, u = DATA.universe;
  document.getElementById("metaInfo").innerText =
    `行情日期：${DATA.trade_date} ｜ 更新时间：${DATA.updated_at} ｜ 有效流动标的：${u.liquid_count} / ${u.total_fetched}`;

  const longTag = b.long_bull ? `<span class="badge badge-up">长牛 (200MA上行)</span>` : `<span class="badge badge-down">防守 (200MA偏弱)</span>`;
  const shortTag = b.short_bull ? `<span class="badge badge-up">短多 (站上50MA)</span>` : `<span class="badge badge-warn">短线回调 (破50MA)</span>`;

  document.getElementById("kpiGrid").innerHTML = `
    <div class="card">
      <div class="card-label">恒生指数 (${b.symbol})</div>
      <div class="card-val">${b.close} <span style="font-size:0.95rem">${fmtPct(b.pct_chg)}</span></div>
      <div class="card-sub">${longTag} ${shortTag}</div>
    </div>
    <div class="card">
      <div class="card-label">港股通市场广度</div>
      <div class="card-val ${u.breadth_ma50 >= 50 ? 'up' : 'down'}">${u.breadth_ma50}%</div>
      <div class="card-sub">站上 50MA 占比 ｜ 站上 200MA：${u.breadth_ma200}%</div>
    </div>
    <div class="card">
      <div class="card-label">多头机会命中</div>
      <div class="card-val up">${DATA.stage2.length + DATA.gap_watch.length} 只</div>
      <div class="card-sub">Stage 2 突破 ${DATA.stage2.length} ｜ 缺口观察 ${DATA.gap_watch.length}</div>
    </div>
    <div class="card">
      <div class="card-label">超跌博弈 / 风险预警</div>
      <div class="card-val warn">${DATA.stage4b.length} / <span class="down">${DATA.risk.length}</span></div>
      <div class="card-sub">4B- 反转 ${DATA.stage4b.length} ｜ 50MA/止盈警报 ${DATA.risk.length}</div>
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

function switchTab(k) { activeKey = k; renderTabs(); renderTable(); }

function renderTable() {
  const tab = TABS.find(t => t.key === activeKey);
  const q = document.getElementById("searchInput").value.trim().toLowerCase();
  const rows = DATA[activeKey].filter(r =>
    !q || r.ticker.toLowerCase().includes(q) || (r.name && r.name.toLowerCase().includes(q))
  );

  const thead = `<thead><tr>${tab.cols.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead>`;
  if (!rows.length) {
    document.getElementById("dataTable").innerHTML = thead + `<tbody><tr><td colspan="${tab.cols.length}" class="empty">当日暂无符合条件的标的</td></tr></tbody>`;
    return;
  }

  const tbody = rows.map(r => `<tr>${tab.cols.map(([k]) => {
    const val = r[k];
    if (k === "ticker") return `<td><a class="ticker-link" href="${r.url}" target="_blank" rel="noopener">${val}</a></td>`;
    if (k === "pct_chg" || k === "dist_ma50") return `<td>${fmtPct(val)}</td>`;
    if (k === "mrs") return `<td class="${val >= 0 ? 'up' : 'down'}">${val}</td>`;
    if (k === "type" || k === "status") return `<td><span class="badge badge-up">${val}</span></td>`;
    if (k === "signal") return `<td><span class="badge ${val.includes('清仓') || val.includes('离场') ? 'badge-down' : 'badge-warn'}">${val}</span></td>`;
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
    print(f"[完成] 交互式看板已生成：{out_path}")


if __name__ == "__main__":
    scan_data = run_scan()
    build_dashboard(scan_data)
