# ENGINE VERSION 2026-01-27 v3
# engine.py
"""
BACKTEST & PROBABILITY ENGINE
"""

import pandas as pd
import numpy as np

from engine_v2 import fetch_data, add_indicators, extract_market_state,latest_candle_info
import os, pickle

# ======================================================
# HELPERS
# ======================================================
CACHE_VERSION = "v3"   # atau versi sesuai kebutuhan
PROB_CACHE = f"prob_cache_{CACHE_VERSION}.pkl"


def load_prob_cache():
    if os.path.exists(PROB_CACHE):
        try:
            return pickle.load(open(PROB_CACHE, "rb"))
        except Exception:
            return {}
    return {}

def save_prob_cache(cache):
    with open(PROB_CACHE, "wb") as f:
        pickle.dump(cache, f)


def clean_number(x):
    if x is None or pd.isna(x):
        return None
    return round(float(x), 2)


# ======================================================
# BACKTEST (TIDAK DIUBAH KONSEPNYA)
# ======================================================
def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = (
        (df["EMA13"] > df["EMA21"]) &
        (df["EMA21"] > df["EMA50"])
    )
    return df

def volume_behavior_at_index(df: pd.DataFrame, idx: int):
    if idx < 1:
        return "VOL_NEUTRAL", np.nan

    last = df.iloc[idx]

    open_ = last["Open"]
    high = last["High"]
    low = last["Low"]
    close = last["Close"]
    volume = last["Volume"]
    vol_ma20 = last["VOL_MA20"] if last["VOL_MA20"] > 0 else 1
    ema21 = last["EMA21"]

    # ===== GUARD =====
    if high == low:
        return "VOL_NEUTRAL", round(volume / vol_ma20, 2)

    range_ = high - low
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    body_ratio = body / range_
    vol_ratio = volume / vol_ma20

    # ===== LOW VOLUME =====
    if vol_ratio < 1.2:
        return "VOL_NEUTRAL", round(vol_ratio, 2)

    # ===== ABSORPTION =====
    if (
        lower_wick >= 1.5 * body and
        body_ratio <= 0.40 and
        close >= ema21
    ):
        return "VOL_ABSORPTION", round(vol_ratio, 2)

    # ===== DISTRIBUTION =====
    if (
        upper_wick >= 1.5 * body and
        body_ratio <= 0.40 and
        close < ema21
    ):
        return "VOL_DISTRIBUTION", round(vol_ratio, 2)

    return "VOL_EXPANSION_NEUTRAL", round(vol_ratio, 2)


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Position"] = df["Signal"].shift(1).fillna(False)
    df["Return"] = df["Close"].pct_change()
    df["StrategyReturn"] = df["Return"] * df["Position"]

    # ==========================
    # HISTORICAL VOLUME BEHAVIOR
    # ==========================
    vol_labels = []
    vol_ratios = []

    for i in range(len(df)):
        label, ratio = volume_behavior_at_index(df, i)
        vol_labels.append(label)
        vol_ratios.append(ratio)

    df["VOL_BEHAVIOR"] = vol_labels
    df["VOL_RATIO"] = vol_ratios

    return df


def backtest_summary(df: pd.DataFrame):
    ret = df["StrategyReturn"].dropna()
    if ret.empty:
        return df, {}

    equity = (1 + ret).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    return df, {
        "TotalReturnPct": round((equity.iloc[-1] - 1) * 100, 2),
        "WinRatePct": round((ret > 0).mean() * 100, 2),
        "Trades": int(df["Position"].sum()),
        "MaxDrawdownPct": round(drawdown.min() * 100, 2)
    }

def backtest_strategy(ticker: str):
    df = fetch_data(ticker)
    if df is None or len(df) < 100:
        return None


    df = add_indicators(df)
    df = generate_signal(df)
    df = run_backtest(df)

    return df

def confidence_level(sample_size: int) -> str:
    if sample_size >= 30:
        return "HIGH"
    if sample_size >= 15:
        return "MEDIUM"
    if sample_size >= 5:
        return "LOW"
    return "VERY_LOW"

def rsi_bucket(rsi: float) -> str:
    if pd.isna(rsi):
        return "UNKNOWN"
    if rsi < 30:
        return "<30"
    if rsi <= 70:
        return "30-70"
    return ">70"



def backtest_decision(ticker: str, lookback: int = 180):
    df = fetch_data(ticker)
    if df is None or len(df) < lookback + 20:
        return None

    df = add_indicators(df)
    i = len(df) - 1

    state_today = extract_market_state(df, i)

    today_state = {
        "MajorTrend": state_today["MajorTrend"],
        "MinorPhase": minor_phase_daily_at_index(df, i),
        "RSI_BUCKET": rsi_bucket(df.iloc[i]["RSI"]),
        "VOL_BEHAVIOR": state_today["VOL_BEHAVIOR"],
        "latest_candle": candle_label_at_index(df, i),
    }

    if today_state["MajorTrend"] != "STRONG":
        return {
            "Bias": "NO_SETUP",
            "Reason": "MajorTrend tidak STRONG",
            "TodayState": today_state
        }

    # ==========================
    # BUILD PROBABILITY TABLE
    # ==========================
    prob = build_probability_table_from_ticker(ticker, lookback)
    if prob is None or prob.empty:
        return {
            "Bias": "NO_MODEL",
            "Reason": "Tidak ada data historis cocok",
            "TodayState": today_state
        }

    # ==========================
    # NORMALIZATION (WAJIB)
    # ==========================
    for col in ["MinorPhase", "RSI_BUCKET", "VOL_BEHAVIOR", "latest_candle"]:
        prob[col] = prob[col].astype(str).str.strip()
        today_state[col] = str(today_state[col]).strip()

    # ==========================
    # 1️⃣ EXACT MATCH
    # ==========================
    exact = prob[
        (prob["MinorPhase"] == today_state["MinorPhase"]) &
        (prob["RSI_BUCKET"] == today_state["RSI_BUCKET"]) &
        (prob["VOL_BEHAVIOR"] == today_state["VOL_BEHAVIOR"]) &
        (prob["latest_candle"] == today_state["latest_candle"])
    ]

    if not exact.empty:
        row = exact.sort_values("Count", ascending=False).iloc[0]
        bias = "HIJAU" if row["%Hijau"] >= row["%Merah"] else "MERAH"

        return {
            "Bias": bias,
            "ProbHijau": float(row["%Hijau"]),
            "ProbMerah": float(row["%Merah"]),
            "Sample": int(row["Count"]),
            "Confidence": confidence_level(int(row["Count"])),

            "DecisionContext": {
                "MajorTrend": today_state["MajorTrend"],
                "MinorPhase": row["MinorPhase"],
                "RSI_BUCKET": row["RSI_BUCKET"],
                "VOL_BEHAVIOR": row["VOL_BEHAVIOR"],
                "latest_candle": row["latest_candle"],
                "AvgVolRatio": row.get("AvgVolRatio"),
                "MatchType": "EXACT"   
            }
        }



    # ==========================
    # 2️⃣ PARTIAL MATCH (DROP latest_candle)
    # ==========================
    partial = prob[
        (prob["MinorPhase"] == today_state["MinorPhase"]) &
        (prob["RSI_BUCKET"] == today_state["RSI_BUCKET"]) &
        (prob["VOL_BEHAVIOR"] == today_state["VOL_BEHAVIOR"])
    ]

    if not partial.empty:
        row = partial.sort_values("Count", ascending=False).iloc[0]
        bias = "HIJAU" if row["%Hijau"] >= row["%Merah"] else "MERAH"

        return {
            "Bias": bias,
            "ProbHijau": float(row["%Hijau"]),
            "ProbMerah": float(row["%Merah"]),
            "Sample": int(row["Count"]),
            "Confidence": confidence_level(int(row["Count"])),
            "DecisionContext": {
                "MajorTrend": today_state["MajorTrend"],
                "MinorPhase": row["MinorPhase"],
                "RSI_BUCKET": row["RSI_BUCKET"],
                "VOL_BEHAVIOR": row["VOL_BEHAVIOR"],
                "latest_candle": row["latest_candle"],
                "AvgVolRatio": row.get("AvgVolRatio"),
                "MatchType": "PARTIAL_NO_CANDLE"
            }
        }


    # ==========================
    # 3️⃣ FALLBACK DEBUG
    # ==========================
    return {
        "Bias": "NO_MATCH",
        "Reason": "Tidak ada exact / partial match historis",
        "TodayState": today_state,
        "DebugCandidates": prob[
            prob["MinorPhase"] == today_state["MinorPhase"]
        ].sort_values("Count", ascending=False).head(5)
    }



def backtest(ticker: str, mode: str = "decision"):
    if mode == "strategy":
        return backtest_strategy(ticker)

    if mode == "decision":
        return backtest_decision(ticker)

    raise ValueError("mode harus 'strategy' atau 'decision'")



# ======================================================
# PROBABILITY MODEL (INI YANG ANDA BUTUHKAN)
# ======================================================
def build_probability_table(df_states: pd.DataFrame) -> pd.DataFrame:
    required = [
        "MajorTrend",
        "MinorPhase",
        "RSI_BUCKET",
        "VOL_BEHAVIOR",
        "VOL_RATIO",
        "latest_candle",
        "Close"
    ]

    missing = [c for c in required if c not in df_states.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_states[df_states["MajorTrend"] == "STRONG"].copy()
    if df.empty:
        return pd.DataFrame()

    # =========================
    # NEXT DAY LABEL
    # =========================
    df["NextCandle"] = df["Close"].shift(-1) > df["Close"]
    df["NextCandle"] = df["NextCandle"].map({True: "HIJAU", False: "MERAH"})
    df = df.iloc[:-1]

    group_cols = [
        "MinorPhase",
        "RSI_BUCKET",
        "VOL_BEHAVIOR",
        "latest_candle"
    ]

    summary = (
        df
        .groupby(group_cols, dropna=False)
        .agg(
            Count=("NextCandle", "count"),
            Hijau=("NextCandle", lambda x: (x == "HIJAU").sum()),
            Merah=("NextCandle", lambda x: (x == "MERAH").sum()),
            AvgVolRatio=("VOL_RATIO", "mean"),
        )
        .reset_index()
    )

    summary["AvgVolRatio"] = summary["AvgVolRatio"].round(2)
    summary["%Hijau"] = (summary["Hijau"] / summary["Count"] * 100).round(2)
    summary["%Merah"] = (summary["Merah"] / summary["Count"] * 100).round(2)
    summary.insert(0, "MajorTrend", "STRONG")

    return summary.sort_values(["%Hijau", "Count"], ascending=False)


def candle_label_at_index(df, idx):
    """
    Ambil label candle untuk bar ke-idx
    menggunakan logic latest_candle_info tanpa mengubah fungsinya
    """
    if idx < 1:
        return "Doji/Netral"

    sub_df = df.iloc[: idx + 1]
    label, _, _ = latest_candle_info(sub_df)
    return label

def minor_phase_daily_at_index(df, idx):
    """
    Daily approximation of MBMA Minor Phase
    Dipakai khusus untuk probability engine
    """
    if idx < 2:
        return "NEUTRAL"

    last = df.iloc[idx]
    prev = df.iloc[idx - 1]

    # guard indikator
    for col in ["EMA13", "EMA21", "EMA50"]:
        if col not in df.columns:
            return "NEUTRAL"

    # 1. EMA Compress
    ema_compress = abs(last["EMA13"] - last["EMA21"]) / last["EMA21"] < 0.003
    if ema_compress:
        return "EMA_COMPRESS_PULLBACK"

    # 2. Pullback Recovered
    pullback_recovered = (
        last["Close"] > last["EMA21"]
        and prev["Close"] <= prev["EMA21"]
    )
    if pullback_recovered:
        return "PULLBACK_RECOVERED"

    # 3. Trend Continue
    is_bullish = (
        last["EMA13"] > last["EMA21"] > last["EMA50"]
        and last["Close"] >= last["EMA13"]
    )
    if is_bullish:
        return "TREND_CONTINUE"

    return "NEUTRAL"



def build_probability_table_from_ticker(ticker: str, lookback: int = 180):
    # cek cache dulu
    prob_cache = load_prob_cache()
    if ticker in prob_cache:
        return prob_cache[ticker]

    df = fetch_data(ticker)
    if df is None or len(df) < lookback + 2:
        return None

    df = add_indicators(df)
    states = []
    start = len(df) - lookback - 1
    end = len(df) - 1

    for i in range(start, end):
        state = extract_market_state(df, i)
        state["latest_candle"] = candle_label_at_index(df, i)
        state["MinorPhase"] = minor_phase_daily_at_index(df, i)
        vol_label, vol_ratio = volume_behavior_at_index(df, i)
        state["VOL_BEHAVIOR"] = vol_label
        state["VOL_RATIO"] = vol_ratio
        if "Close" not in state:
            state["Close"] = df.iloc[i]["Close"]
        states.append(state)

    df_states = pd.DataFrame(states)
    df_prob = build_probability_table(df_states)

    # simpan ke cache
    prob_cache[ticker] = df_prob
    save_prob_cache(prob_cache)

    return df_prob


