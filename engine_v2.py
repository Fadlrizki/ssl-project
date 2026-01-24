import pandas as pd
import numpy as np
import yfinance as yf
import os

# ======================================================
# CONFIG
# ======================================================
EMA_FAST = 13
EMA_MID = 21
EMA_SLOW = 50
SMA50_PERIOD = 50
VOL_MA_PERIOD = 20
RSI_PERIOD = 14
STOCH_PERIOD = 14
SLOPE_WINDOW = 10
EMA_COMPRESS_TH = 0.003

BODY_THRESHOLD = 0.02
CANDLE_DIST_TH = 5
TOLERANCE = 0.02

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ======================================================
# CACHE
# ======================================================
def cache_path(ticker, interval):
    return os.path.join(CACHE_DIR, f"{ticker}_{interval}.parquet")

def load_cache(ticker, interval):
    p = cache_path(ticker, interval)
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None

def save_cache(df, ticker, interval):
    df.to_parquet(cache_path(ticker, interval))

# ======================================================
# FETCH DATA
# ======================================================
def fetch_data(ticker, interval="1d", period="12mo"):
    try:
        cached = load_cache(ticker, interval)
        if cached is not None and len(cached) > 50:
            return cached.copy()

        df = yf.download(
            ticker,
            interval=interval,
            period=period,
            progress=False,
            threads=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open","High","Low","Close","Volume"]].dropna()

        if len(df) < 20 or df["Volume"].tail(5).sum() == 0:
            return None

        save_cache(df, ticker, interval)
        return df

    except Exception as e:
        print(f"Failed download: {ticker} | {e}")
        return None

# ======================================================
# INDICATORS
# ======================================================
def add_indicators(df):
    df = df.copy()

    df["EMA13"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=EMA_MID, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(SMA50_PERIOD, min_periods=1).mean()
    df["VOL_MA20"] = df["Volume"].rolling(VOL_MA_PERIOD, min_periods=1).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(RSI_PERIOD, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Stochastic %K
    low_min = df["Low"].rolling(STOCH_PERIOD, min_periods=1).min()
    high_max = df["High"].rolling(STOCH_PERIOD, min_periods=1).max()
    df["STOCH"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    return df

def ema_slope(series):
    """Arah EMA: positif = naik"""
    if len(series) <= SLOPE_WINDOW:
        return 0
    return series.iloc[-1] - series.iloc[-SLOPE_WINDOW]

# ======================================================
# MAJOR TREND (1D) — SUDAH OK
# ======================================================
def major_trend_daily(df):
    last = df.iloc[-1]

    if last["EMA21"] <= last["EMA50"]:
        return "INVALID"

    if ema_slope(df["EMA21"]) <= 0:
        return "INVALID"

    return "STRONG"

# ======================================================
# MINOR PHASE (4H) — MBMA ENGINE
# ======================================================
def minor_phase_4h(df):
    """
    MBMA Minor Phase (4H)
    Fokus membaca:
    - struktur EMA
    - transisi EMA (EMA50 → bawah, EMA13 → atas)
    - kondisi sehat / overextend
    """
    # =============================
    # GUARD
    # =============================
    if len(df) < max(6, SLOPE_WINDOW) + 1:
        return "NEUTRAL", ["Data 4H belum cukup"], 0 , 0

    last = df.iloc[-1]
    prev = df.iloc[-2]

    why = []
    confidence = 0

    # =============================
    # HELPER EVALUATION
    # =============================
    ema_compress = abs(last["EMA13"] - last["EMA21"]) / last["EMA21"] < EMA_COMPRESS_TH

    pullback_recovered = (
        last["Close"] > last["EMA21"] and
        prev["Close"] < prev["EMA21"]
    )

    was_bearish = (
        df.iloc[-6]["EMA50"] >= df.iloc[-6]["EMA21"] >= df.iloc[-6]["EMA13"]
    )

    is_bullish = (
        last["EMA13"] > last["EMA21"] > last["EMA50"]
    )

    ema13_cross = (
        prev["EMA13"] <= prev["EMA21"] and
        last["EMA13"] > last["EMA21"]
    )

    ema21_up = ema_slope(df["EMA21"]) > 0
    ema50_flat = ema_slope(df["EMA50"]) >= 0
    price_confirm = last["Close"] >= last["EMA21"]

    # =============================
    # 1. EMA COMPRESS
    # =============================
    if ema_compress:
        why.append("EMA13 dan EMA21 dalam fase kompresi")
        return "EMA_COMPRESS_PULLBACK", why, 1 ,0  

    # =============================
    # 2. PULLBACK RECOVERED
    # =============================
    if pullback_recovered:
        why.append("Harga kembali menutup di atas EMA21 setelah pullback")
        return "PULLBACK_RECOVERED", why, 1 , 0 

    # =============================
    # 3. TREND CONTINUE (NEW)
    # =============================
    trend_new_rules = {
        "Struktur sebelumnya bearish": was_bearish,
        "Struktur EMA sekarang bullish": is_bullish,
        "EMA13 cross ke atas EMA21": ema13_cross,
        "Slope EMA21 positif": ema21_up,
        "EMA50 tidak turun": ema50_flat,
        "Harga bertahan di atas EMA21": price_confirm
    }

    passed = [k for k, v in trend_new_rules.items() if v]
    confidence = len(passed)
    confidence_pct = round(confidence / 6 * 100)


    if confidence == len(trend_new_rules):
        why.extend(passed)
        return "TREND_CONTINUE_NEW", why, confidence,confidence_pct

    # =============================
    # 4. TREND CONTINUE (NORMAL)
    # =============================
    if is_bullish and last["Close"] >= last["EMA13"]:
        why.append("Struktur EMA bullish dan harga di atas EMA13")
        return "TREND_CONTINUE", why, confidence , confidence_pct

    # =============================
    # 5. OVER EXTEND
    # =============================
    dist_ema13 = (last["Close"] - last["EMA13"]) / last["EMA13"]
    dist_ema21 = (last["Close"] - last["EMA21"]) / last["EMA21"]
    dist_ema50 = (last["Close"] - last["EMA50"]) / last["EMA50"]

    if (
        dist_ema13 > 0.05 and
        dist_ema21 > 0.10 and
        dist_ema50 > 0.20 and
        (last["RSI"] >= 70 or last["STOCH"] >= 85)
    ):
        why.append("Harga terlalu jauh dari EMA (overextend)")
        return "TREND_OVEREXTEND", why, confidence , confidence_pct

    return "NEUTRAL", ["Tidak ada struktur minor yang valid"], confidence, confidence_pct

# ======================================================
# SETUP STATE
# ======================================================
def setup_state(minor):
    if minor == "EMA_COMPRESS_PULLBACK":
        return "SETUP_PENDING"

    if minor in (
        "PULLBACK_RECOVERED",
        "TREND_CONTINUE_NEW",
        "TREND_CONTINUE"
    ):
        return "STAGE2_READY"

    return "WAIT"


# ======================================================
# STAGE 2 TRIGGER
# ======================================================
def stage2_trigger(df, setup):
    if setup != "STAGE2_READY":
        return False

    last = df.iloc[-1]
    return last["Close"] > last["EMA13"] and ema_slope(df["EMA13"]) > 0

# ======================================================
# VOLUME STATE
# ======================================================
def volume_behavior(df):
    last = df.iloc[-1]

    # ===== BASIC =====
    vol_ma20 = last["VOL_MA20"] if last["VOL_MA20"] > 0 else 1
    vol_ratio = last["Volume"] / vol_ma20

    open_ = last["Open"]
    close = last["Close"]
    high = last["High"]
    low = last["Low"]

    body = abs(close - open_)
    range_ = high - low
    atr = last["ATR14"] if last["ATR14"] > 0 else range_

    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    body_ratio = body / max(range_, 1e-9)
    cpi = body / max(atr, 1e-9)   # Candle Pressure Index

    # ===== CONDITIONS =====
    is_expansion = vol_ratio >= 1.2
    range_compact = range_ <= 1.4 * atr

    # ===== ABSORPTION =====
    if (
        is_expansion and
        lower_wick >= 1.5 * body and
        body_ratio <= 0.40 and
        cpi >= 0.55 and
        range_compact and
        close >= last["EMA21"]
    ):
        return "ABSORPTION", round(vol_ratio, 2)

    # ===== DISTRIBUTION =====
    if (
        is_expansion and
        upper_wick >= 1.5 * body and
        body_ratio <= 0.40 and
        cpi <= 0.45 and
        range_compact
    ):
        if close < last["EMA21"]:
            return "DISTRIBUTION_STRONG", round(vol_ratio, 2)
        else:
            return "DISTRIBUTION_EARLY", round(vol_ratio, 2)

    return "NORMAL", round(vol_ratio, 2)



# =========================
# CANDLE INFO
# =========================
def latest_candle_info(df):
    if len(df) < 2:
        return "N/A", False, False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body_pct = abs(last["Close"] - last["Open"]) / max(last["Open"], 1)
    is_red = last["Close"] < last["Open"]
    is_green = last["Close"] > last["Open"]
    breakdown = is_red and body_pct > BODY_THRESHOLD and last["Close"] < last["SMA50"] and prev["Close"] >= last["SMA50"]
    approaching = is_green and abs((last["Close"] - last["SMA50"]) / max(last["SMA50"],1) * 100) <= CANDLE_DIST_TH
    if breakdown:
        return "Merah Kuat & Breakdown", -1, False
    if approaching:
        return "Hijau & Mendekati SMA50", False, 1
    if is_red:
        return "Merah Biasa", 0, False
    if is_green:
        return "Hijau Biasa", False, 0
    return "Doji/Netral", 0, 0

# =========================
# DIST SMA50
# =========================

def compute_dist_sma50(df):
    last = df.iloc[-1]
    sma50 = last["SMA50"]
    if pd.isna(sma50) or sma50 == 0:
        return np.nan
    return (last["Close"] - sma50) / sma50 * 100

# ======================================================
# FINAL DECISION
# ======================================================
def final_decision(major, minor, setup, stage2, vol_behavior):
    if major == "INVALID":
        return "SKIP"

    if setup == "SETUP_PENDING":
        return "SETUP_PENDING"

    if (
        major == "STRONG"
        and setup == "STAGE2_READY"
        and stage2
        and vol_behavior in ("ABSORPTION", "NORMAL")
    ):
        return "ENTRY_READY"

    if vol_behavior.startswith("DISTRIBUTION"):
        return "WAIT"

    return "WAIT"


# ======================================================
# PROCESS STOCK
# ======================================================
def process_stock(kode):
    ticker = f"{kode}.JK"

    try:
        d1 = fetch_data(ticker, "1d", "12mo")
        h4 = fetch_data(ticker, "4h", "12mo")

        if d1 is None or h4 is None:
            return None

        d1 = add_indicators(d1)
        h4 = add_indicators(h4)

        major = major_trend_daily(d1)
        minor, why, confidence , confidence_pct= minor_phase_4h(h4)
        setup = setup_state(minor)
        stage2 = stage2_trigger(h4, setup)

        price_today = d1["Close"].iloc[-1]
        price_yesterday = d1["Close"].iloc[-2]
        price_change = round((price_today/price_yesterday - 1)*100, 2)
        vol_behavior, vol_ratio = volume_behavior(d1)
        gap_ema21 = round((price_today/d1["EMA21"].iloc[-1]-1)*100,2)
        gap_ema50 = round((price_today/d1["EMA50"].iloc[-1]-1)*100,2)

        rsi = d1["RSI"].iloc[-1]
        stoch = d1["STOCH"].iloc[-1]
        sma50 = d1["SMA50"].iloc[-1]
        dist_to_sma50 = compute_dist_sma50(d1)

        candle_label, candle_red_breakdown, candle_green_approach = latest_candle_info(d1)
        candle_effect = 1 if candle_green_approach else -1 if candle_red_breakdown else 0

        final_dec = final_decision(major, minor, setup, stage2, vol_behavior)


        return {
            "Kode": kode,
            "Price": price_today,
            "PriceChange%": price_change,
            "Gap_EMA21%": gap_ema21,
            "Gap_EMA50%": gap_ema50,
            "MajorTrend": major,
            "MinorPhase": minor,
            "WHY_MINOR": why,
            "MinorConfidence": confidence,
            "MinorConfidence%" : confidence_pct,
            "SetupState": setup,
            "Stage2Valid": stage2,
            "VOL_BEHAVIOR": vol_behavior,
            "VOL_RATIO": vol_ratio,
            "RSI": round(rsi,2),
            "SMA50": round(sma50,2),
            "Dist_to_SMA50": round(dist_to_sma50,2) if not np.isnan(dist_to_sma50) else np.nan,
            "Stoch_K": round(stoch,2),
            "Latest_Candle": candle_label,
            "Candle_Effect": candle_effect,
            "FinalDecision": final_dec
        }

    except Exception as e:
        print(f"Failed process_stock: {kode} | {e}")
        return None
