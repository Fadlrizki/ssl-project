# ENGINE VERSION 2026-01-27 v3
# engine.py
"""
BACKTEST & PROBABILITY ENGINE
"""

import pandas as pd
import numpy as np
import os
import pickle
import yfinance as yf

# Import dari module lain
from engine_v2 import add_indicators, extract_market_state, latest_candle_info
from utils import data_utils, cache_manager, validation_utils

# ======================================================
# CONFIGURATION
# ======================================================
CACHE_VERSION = "v3"
PROB_CACHE = f"prob_cache_{CACHE_VERSION}.pkl"
LOOKBACK_DAYS = 240  # Default lookback period
MIN_CANDLES = 80  # Minimum candles for backtest

# ======================================================
# CACHE MANAGEMENT (simplified using utils)
# ======================================================
def load_prob_cache():
    """Load probability cache"""
    return cache_manager.load(PROB_CACHE.replace(".pkl", ""), suffix="pkl") or {}

def save_prob_cache(cache):
    """Save probability cache"""
    cache_manager.save(PROB_CACHE.replace(".pkl", ""), cache, suffix="pkl")

# ======================================================
# DATA FETCHING (simplified using utils)
# ======================================================
def fetch_daily_backtest(ticker: str, period: str = "5y", min_candles: int = 150):
    """
    Fetch DAILY data untuk backtest & probability engine
    
    Returns:
    --------
    pd.DataFrame or None
    """
    return data_utils.safe_yf_download(
        ticker=ticker,
        interval="1d",
        period=period,
        max_retries=3
    )

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def clean_number(x):
    """Clean and round number"""
    if x is None or pd.isna(x):
        return None
    return round(float(x), 2)

def confidence_level(sample_size: int) -> str:
    """Determine confidence level based on sample size"""
    if sample_size >= 30:
        return "HIGH"
    if sample_size >= 15:
        return "MEDIUM"
    if sample_size >= 5:
        return "LOW"
    return "VERY_LOW"

def rsi_bucket(rsi: float) -> str:
    """Categorize RSI into buckets"""
    if pd.isna(rsi):
        return "UNKNOWN"
    if rsi < 30:
        return "<30"
    if rsi <= 70:
        return "30-70"
    return ">70"

def volume_behavior_at_index(df: pd.DataFrame, idx: int):
    """
    Determine volume behavior at specific index
    Simplified version - consider moving to engine_v2
    """
    if idx < 1:
        return "VOL_NEUTRAL", np.nan

    last = df.iloc[idx]
    
    # Basic validation
    if not validation_utils.validate_price_data(last["Close"]):
        return "VOL_NEUTRAL", np.nan
    
    open_ = last["Open"]
    high = last["High"]
    low = last["Low"]
    close = last["Close"]
    volume = last["Volume"]
    vol_ma20 = last.get("VOL_MA20", 1) if pd.notna(last.get("VOL_MA20")) else 1
    ema21 = last.get("EMA21", close) if pd.notna(last.get("EMA21")) else close

    # Guard: zero range
    if high == low or vol_ma20 <= 0:
        return "VOL_NEUTRAL", 0.0

    range_ = high - low
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    body_ratio = body / range_ if range_ > 0 else 0
    vol_ratio = volume / vol_ma20

    # Low volume
    if vol_ratio < 1.2:
        return "VOL_NEUTRAL", round(vol_ratio, 2)

    # Absorption (hammer-like with volume)
    if (lower_wick >= 1.5 * body and 
        body_ratio <= 0.40 and 
        close >= ema21):
        return "VOL_ABSORPTION", round(vol_ratio, 2)

    # Distribution (shooting star with volume)
    if (upper_wick >= 1.5 * body and 
        body_ratio <= 0.40 and 
        close < ema21):
        return "VOL_DISTRIBUTION", round(vol_ratio, 2)

    # Expansion (strong move with volume)
    if vol_ratio >= 1.5 and body_ratio >= 0.55:
        return "VOL_EXPANSION", round(vol_ratio, 2)

    return "VOL_EXPANSION_NEUTRAL", round(vol_ratio, 2)

# ======================================================
# BACKTEST ENGINE (Strategy Testing)
# ======================================================
def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals based on EMA structure"""
    df = df.copy()
    df["Signal"] = (
        (df["EMA13"] > df["EMA21"]) &
        (df["EMA21"] > df["EMA50"])
    )
    return df

def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Run backtest on dataframe"""
    df = df.copy()
    
    # Generate signals
    df = generate_signal(df)
    
    # Position is signal from previous day
    df["Position"] = df["Signal"].shift(1).fillna(False)
    
    # Calculate returns
    df["Return"] = df["Close"].pct_change()
    df["StrategyReturn"] = df["Return"] * df["Position"].astype(int)
    
    # Add volume behavior for each day
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
    """Generate backtest summary statistics"""
    if df is None or df.empty:
        return df, {}
    
    ret = df["StrategyReturn"].dropna()
    if ret.empty:
        return df, {}
    
    # Calculate equity curve
    equity = (1 + ret).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    
    # Calculate win rate
    winning_trades = (ret > 0).sum()
    total_trades = len(ret)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    summary = {
        "TotalReturnPct": round((equity.iloc[-1] - 1) * 100, 2),
        "WinRatePct": round(win_rate, 2),
        "Trades": int(total_trades),
        "MaxDrawdownPct": round(drawdown.min() * 100, 2),
        "AvgReturnPct": round(ret.mean() * 100, 2),
        "StdReturnPct": round(ret.std() * 100, 2),
        "SharpeRatio": round(ret.mean() / ret.std() * np.sqrt(252), 2) if ret.std() > 0 else 0
    }
    
    return df, summary

def backtest_strategy(ticker: str):
    """Run full strategy backtest"""
    # Fetch data
    df = fetch_daily_backtest(ticker)
    if df is None or len(df) < 100:
        return None
    
    # Add indicators
    df = add_indicators(df)
    if df is None or df.empty:
        return None
    
    # Run backtest
    df = run_backtest(df)
    
    return df

# ======================================================
# PROBABILITY ENGINE (Decision Making)
# ======================================================
def candle_label_at_index(df, idx):
    """
    Get candle label for specific index
    Uses latest_candle_info from engine_v2
    """
    if idx < 1 or df is None or df.empty:
        return "Doji/Netral"
    
    # Get subset up to idx
    sub_df = df.iloc[:idx + 1]
    
    # Use engine_v2 function
    label, _, _ = latest_candle_info(sub_df)
    return label

def minor_phase_daily_at_index(df, idx):
    """
    Daily approximation of MBMA Minor Phase for probability engine
    """
    if idx < 2 or df is None or len(df) < 3:
        return "NEUTRAL"
    
    last = df.iloc[idx]
    prev = df.iloc[idx - 1]
    
    # Check required columns
    required_cols = ["EMA13", "EMA21", "EMA50", "Close"]
    if not all(col in df.columns for col in required_cols):
        return "NEUTRAL"
    
    # 1. EMA Compress
    if last["EMA21"] > 0:
        ema_compress = abs(last["EMA13"] - last["EMA21"]) / last["EMA21"] < 0.003
        if ema_compress:
            return "EMA_COMPRESS_PULLBACK"
    
    # 2. Pullback Recovered
    pullback_recovered = (
        last["Close"] > last["EMA21"] and
        prev["Close"] <= prev["EMA21"]
    )
    if pullback_recovered:
        return "PULLBACK_RECOVERED"
    
    # 3. Trend Continue
    is_bullish = (
        last["EMA13"] > last["EMA21"] > last["EMA50"] and
        last["Close"] >= last["EMA13"]
    )
    if is_bullish:
        return "TREND_CONTINUE"
    
    return "NEUTRAL"

def build_probability_table(df_states: pd.DataFrame) -> pd.DataFrame:
    """
    Build probability table from historical states
    
    Returns:
    --------
    pd.DataFrame: Probability table
    """
    # Validation
    if df_states is None or df_states.empty:
        return pd.DataFrame()
    
    required = [
        "MajorTrend",
        "MinorPhase",
        "RSI_BUCKET",
        "VOL_BEHAVIOR",
        "VOL_RATIO",
        "latest_candle",
        "Close"
    ]
    
    # Check missing columns
    missing = [c for c in required if c not in df_states.columns]
    if missing:
        print(f"Warning: Missing columns in probability table: {missing}")
        return pd.DataFrame()
    
    # Filter for STRONG trend only
    df = df_states[df_states["MajorTrend"] == "STRONG"].copy()
    if df.empty:
        return pd.DataFrame()
    
    # Determine next day's candle color
    df["NextCandle"] = df["Close"].shift(-1) > df["Close"]
    df["NextCandle"] = df["NextCandle"].map({True: "HIJAU", False: "MERAH"})
    df = df.iloc[:-1]  # Remove last row (no next day)
    
    # Group by market state
    group_cols = ["MinorPhase", "RSI_BUCKET", "VOL_BEHAVIOR", "latest_candle"]
    
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
    
    # Calculate percentages
    summary["AvgVolRatio"] = summary["AvgVolRatio"].round(2)
    summary["%Hijau"] = (summary["Hijau"] / summary["Count"] * 100).round(2)
    summary["%Merah"] = (summary["Merah"] / summary["Count"] * 100).round(2)
    
    # Add trend column
    summary.insert(0, "MajorTrend", "STRONG")
    
    # Sort by highest green probability and sample size
    return summary.sort_values(["%Hijau", "Count"], ascending=False)

def build_probability_table_from_ticker(ticker: str, lookback: int = LOOKBACK_DAYS):
    """
    Build probability table for a specific ticker
    
    Returns:
    --------
    pd.DataFrame or None
    """
    # Check cache first
    prob_cache = load_prob_cache()
    cache_key = f"{ticker}_{lookback}"
    
    if cache_key in prob_cache:
        cached_result = prob_cache[cache_key]
        if cached_result is not None and not cached_result.empty:
            return cached_result
    
    # Fetch data
    df = fetch_daily_backtest(ticker, period=f"{max(lookback+100, 365)}d")
    if df is None or len(df) < lookback + 2:
        print(f"Warning: Insufficient data for {ticker}")
        return None
    
    # Add indicators
    df = add_indicators(df)
    if df is None or df.empty:
        return None
    
    # Extract historical states
    states = []
    start_idx = max(0, len(df) - lookback - 1)
    end_idx = len(df) - 1
    
    for i in range(start_idx, end_idx):
        try:
            # Get market state using engine_v2
            state = extract_market_state(df, i)
            
            # Add additional info
            state["latest_candle"] = candle_label_at_index(df, i)
            state["MinorPhase"] = minor_phase_daily_at_index(df, i)
            
            # Add volume behavior
            vol_label, vol_ratio = volume_behavior_at_index(df, i)
            state["VOL_BEHAVIOR"] = vol_label
            state["VOL_RATIO"] = vol_ratio
            
            # Ensure Close price is included
            if "Close" not in state:
                state["Close"] = df.iloc[i]["Close"]
            
            states.append(state)
            
        except Exception as e:
            print(f"Error extracting state at index {i} for {ticker}: {e}")
            continue
    
    if not states:
        return None
    
    # Build probability table
    df_states = pd.DataFrame(states)
    df_prob = build_probability_table(df_states)
    
    # Cache the result
    prob_cache[cache_key] = df_prob
    save_prob_cache(prob_cache)
    
    return df_prob

def backtest_decision(ticker: str, lookback: int = LOOKBACK_DAYS):
    """
    Make trading decision based on probability model
    
    Returns:
    --------
    dict: Decision result with probabilities
    """
    # Fetch and prepare data
    df = fetch_daily_backtest(ticker, period=f"{max(lookback+100, 365)}d")
    if df is None or len(df) < lookback + 20:
        return {
            "Bias": "NO_DATA",
            "Reason": "Data tidak cukup untuk analisis",
            "TodayState": {}
        }
    
    df = add_indicators(df)
    if df is None or df.empty:
        return {
            "Bias": "NO_DATA",
            "Reason": "Gagal menambahkan indikator",
            "TodayState": {}
        }
    
    # Get today's state (last index)
    i = len(df) - 1
    
    try:
        state_today = extract_market_state(df, i)
        
        today_state = {
            "MajorTrend": state_today.get("MajorTrend", "UNKNOWN"),
            "MinorPhase": minor_phase_daily_at_index(df, i),
            "RSI_BUCKET": rsi_bucket(df.iloc[i]["RSI"]),
            "VOL_BEHAVIOR": state_today.get("VOL_BEHAVIOR", "UNKNOWN"),
            "latest_candle": candle_label_at_index(df, i),
        }
        
        # Check if MajorTrend is STRONG
        if today_state["MajorTrend"] != "STRONG":
            return {
                "Bias": "NO_SETUP",
                "Reason": "MajorTrend tidak STRONG",
                "TodayState": today_state
            }
        
        # Build or load probability table
        prob_table = build_probability_table_from_ticker(ticker, lookback)
        if prob_table is None or prob_table.empty:
            return {
                "Bias": "NO_MODEL",
                "Reason": "Tidak ada data historis yang cukup",
                "TodayState": today_state
            }
        
        # Normalize strings for comparison
        for col in ["MinorPhase", "RSI_BUCKET", "VOL_BEHAVIOR", "latest_candle"]:
            if col in prob_table.columns:
                prob_table[col] = prob_table[col].astype(str).str.strip()
            today_state[col] = str(today_state.get(col, "")).strip()
        
        # 1️⃣ EXACT MATCH
        exact_match = prob_table[
            (prob_table["MinorPhase"] == today_state["MinorPhase"]) &
            (prob_table["RSI_BUCKET"] == today_state["RSI_BUCKET"]) &
            (prob_table["VOL_BEHAVIOR"] == today_state["VOL_BEHAVIOR"]) &
            (prob_table["latest_candle"] == today_state["latest_candle"])
        ]
        
        if not exact_match.empty:
            best_match = exact_match.sort_values("Count", ascending=False).iloc[0]
            
            return {
                "Bias": "HIJAU" if best_match["%Hijau"] >= best_match["%Merah"] else "MERAH",
                "ProbHijau": float(best_match["%Hijau"]),
                "ProbMerah": float(best_match["%Merah"]),
                "Sample": int(best_match["Count"]),
                "Confidence": confidence_level(int(best_match["Count"])),
                "DecisionContext": {
                    "MajorTrend": today_state["MajorTrend"],
                    "MinorPhase": best_match["MinorPhase"],
                    "RSI_BUCKET": best_match["RSI_BUCKET"],
                    "VOL_BEHAVIOR": best_match["VOL_BEHAVIOR"],
                    "latest_candle": best_match["latest_candle"],
                    "AvgVolRatio": float(best_match.get("AvgVolRatio", 0)),
                    "MatchType": "EXACT"
                }
            }
        
        # 2️⃣ PARTIAL MATCH (without candle)
        partial_match = prob_table[
            (prob_table["MinorPhase"] == today_state["MinorPhase"]) &
            (prob_table["RSI_BUCKET"] == today_state["RSI_BUCKET"]) &
            (prob_table["VOL_BEHAVIOR"] == today_state["VOL_BEHAVIOR"])
        ]
        
        if not partial_match.empty:
            best_match = partial_match.sort_values("Count", ascending=False).iloc[0]
            
            return {
                "Bias": "HIJAU" if best_match["%Hijau"] >= best_match["%Merah"] else "MERAH",
                "ProbHijau": float(best_match["%Hijau"]),
                "ProbMerah": float(best_match["%Merah"]),
                "Sample": int(best_match["Count"]),
                "Confidence": confidence_level(int(best_match["Count"])),
                "DecisionContext": {
                    "MajorTrend": today_state["MajorTrend"],
                    "MinorPhase": best_match["MinorPhase"],
                    "RSI_BUCKET": best_match["RSI_BUCKET"],
                    "VOL_BEHAVIOR": best_match["VOL_BEHAVIOR"],
                    "latest_candle": best_match["latest_candle"],
                    "AvgVolRatio": float(best_match.get("AvgVolRatio", 0)),
                    "MatchType": "PARTIAL_NO_CANDLE"
                }
            }
        
        # 3️⃣ NO MATCH FOUND
        return {
            "Bias": "NO_MATCH",
            "Reason": "Tidak ditemukan kondisi historis yang serupa",
            "TodayState": today_state,
            "DebugInfo": {
                "TotalStatesInTable": len(prob_table),
                "TodayState": today_state
            }
        }
        
    except Exception as e:
        print(f"Error in backtest_decision for {ticker}: {e}")
        return {
            "Bias": "ERROR",
            "Reason": f"Error dalam proses decision: {str(e)}",
            "TodayState": {}
        }

def backtest(ticker: str, mode: str = "decision"):
    """
    Main backtest function
    
    Parameters:
    -----------
    ticker : str
        Stock ticker (e.g., "BBCA.JK")
    mode : str
        "decision" for probability-based decision
        "strategy" for full strategy backtest
    
    Returns:
    --------
    dict or pd.DataFrame based on mode
    """
    if mode == "strategy":
        return backtest_strategy(ticker)
    elif mode == "decision":
        return backtest_decision(ticker)
    else:
        raise ValueError(f"Mode tidak valid: {mode}. Gunakan 'decision' atau 'strategy'")

# ======================================================
# MAIN EXECUTION GUARD
# ======================================================
if __name__ == "__main__":
    # Test the engine
    test_ticker = "BBCA.JK"
    
    print(f"Testing engine with {test_ticker}")
    
    # Test decision mode
    result = backtest(test_ticker, mode="decision")
    print("\nDecision Result:")
    print(result)
    
    # Test probability table
    prob_table = build_probability_table_from_ticker(test_ticker)
    print(f"\nProbability Table shape: {prob_table.shape if prob_table is not None else 'None'}")
    
    if prob_table is not None and not prob_table.empty:
        print(f"Top 5 probabilities:")
        print(prob_table.head())