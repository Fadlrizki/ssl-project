import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
import datetime


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

BASE_CONF = 6
EXTRA_CONF = 2
MAX_CONF = BASE_CONF + EXTRA_CONF

# ======================================================
# normalize yfinance
# ======================================================
def normalize_yf_df(df, ticker=None):
    """
    Normalize yfinance OHLCV safely.
    Jika MultiIndex, WAJIB ambil level ticker.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if ticker is None:
            raise ValueError("Ticker wajib diberikan untuk MultiIndex dataframe")

        # Ambil kolom milik ticker ini saja
        if ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1)
        else:
            raise ValueError(f"Ticker {ticker} tidak ditemukan di dataframe")

    df = df.loc[:, ~df.columns.duplicated()]
    return df

    

# ======================================================
# CACHE HELPERS
# ======================================================
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(ticker, interval, period):
    safe = ticker.replace(".", "_")
    return os.path.join(
        CACHE_DIR,
        f"{safe}_{interval}_{period}.pkl"
    )


def load_cache(ticker, interval, period):
    path = _cache_path(ticker, interval, period)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            df = pickle.load(f)

        if df is None or df.empty:
            return None

        return df.copy(deep=True)
    except Exception:
        return None


def save_cache(ticker, interval, period, df):
    path = _cache_path(ticker, interval, period)
    with open(path, "wb") as f:
        pickle.dump(df, f)


def is_cache_fresh(df, interval="1d"):
    """
    Cache dianggap fresh jika:
    - tanggal terakhir >= last trading day
    """
    if df is None or df.empty:
        return False

    last_cached_date = df.index[-1].date()
    now = pd.Timestamp.now(tz="Asia/Jakarta")

    # Tentukan last trading day (weekend → Jumat)
    if now.weekday() >= 5:  
        last_trading_day = (now - pd.offsets.BDay(1)).date()
    else:
        last_trading_day = now.date()

    return last_cached_date >= last_trading_day

# ======================================================
# FETCH DATA
# ======================================================
def fetch_data(ticker, interval="1d", period="12mo", force_refresh=True):
    if not force_refresh:
        cached = load_cache(ticker, interval, period)
        if cached is not None and is_cache_fresh(cached):
            return cached.copy(deep=True)

    for attempt in range(2):
        try:
            df = yf.download(
                ticker,
                interval=interval,
                period=period,
                progress=False,
                threads=False,
                auto_adjust=False
            )

            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jakarta")

            df = normalize_yf_df(df, ticker)


            if "Close" not in df.columns and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]

            if "Close" not in df.columns:
                return None

            save_cache(ticker, interval, period, df)
            return df.copy(deep=True)

        except Exception as e:
            if attempt == 0:
                continue
            return None

    return None



# ======================================================
# INDICATORS
# ======================================================
def add_indicators(df):
    df = df.copy(deep=True)

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

    # ======================================================
    # GUARD
    # ======================================================
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        print(f"Invalid columns {ticker}: {df.columns.tolist()}")
        return None

    # Close HARUS Series
    if isinstance(df["Close"], pd.DataFrame):
        print(f"Close still DataFrame {ticker}, skip")
        return None

    return df

def ema_slope(series):
    """Arah EMA: positif = naik"""
    if len(series) <= SLOPE_WINDOW:
        return 0
    return series.iloc[-1] - series.iloc[-SLOPE_WINDOW]

# ======================================================
# MAJOR TREND (1D) 
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
    # 3 & 4. TREND CONTINUE (MERGED)
    # =============================
    trend_rules = {
        "Struktur EMA bullish": is_bullish,
        "Harga bertahan di atas EMA13": last["Close"] >= last["EMA13"],
        "EMA13 cross ke atas EMA21": ema13_cross,
        "Slope EMA21 positif": ema21_up,
        "EMA50 tidak turun": ema50_flat,
        "Struktur sebelumnya bearish": was_bearish,
    }

    passed = [k for k, v in trend_rules.items() if v]
    confidence = len(passed)
    confidence_pct = round(confidence / len(trend_rules) * 100)

    # threshold minimal agar layak disebut TREND_CONTINUE
    if is_bullish and last["Close"] >= last["EMA13"]:
        why.extend(passed)
        return "TREND_CONTINUE", why, confidence, confidence_pct


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
# VOLUME BEHAVIOR (NEW - CANDLE BASED)
# ======================================================
def volume_behavior(df):
    last = df.iloc[-1]

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


# =========================
# CANDLE INFO
# =========================
def latest_candle_info(df):
    if len(df) < 2:
        return "N/A", 0, 0

    last = df.iloc[-1]
    prev = df.iloc[-2]

    open_ = last["Open"]
    close = last["Close"]
    high = last["High"]
    low = last["Low"]

    range_ = high - low
    if range_ == 0:
        return "Doji/Netral", 0, 0

    body = abs(close - open_)
    body_ratio = body / range_

    is_red = close < open_
    is_green = close > open_

    # =============================
    # STRONG GREEN (IMPULSE)
    # =============================
    strong_green = (
        is_green and
        body_ratio >= 0.6 and
        range_ >= 1.2 * last["ATR14"] and
        (high - close) / range_ <= 0.15
    )

    # =============================
    # BREAKDOWN
    # =============================
    breakdown = (
        is_red and
        body_ratio > BODY_THRESHOLD and
        close < last["SMA50"] and
        prev["Close"] >= last["SMA50"]
    )

    # =============================
    # APPROACH SMA50
    # =============================
    approaching = (
        is_green and
        abs((close - last["SMA50"]) / max(last["SMA50"], 1) * 100) <= CANDLE_DIST_TH
    )

    # =============================
    # LABEL PRIORITY
    # =============================
    if strong_green:
        return "Hijau Kuat (Impulse)", 0, 1

    if breakdown:
        return "Merah Kuat & Breakdown", -1, 0

    if approaching:
        return "Hijau & Mendekati SMA50", 0, 1

    if is_red:
        return "Merah Biasa", 0, 0

    if is_green:
        return "Hijau Biasa", 0, 0

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
        # =========================
        # FETCH DATA
        # =========================
        d1 = fetch_data(ticker, "1d", "12mo", force_refresh=True)
        h4 = fetch_data(ticker, "4h", "6mo", force_refresh=True)

        if d1 is None or d1.empty:
            return None

        # =========================
        # DAILY PROCESS
        # =========================
        d1 = add_indicators(d1.copy(deep=True))
        if d1 is None or d1.empty:
            return None

        major = major_trend_daily(d1)

        # =========================
        # 4H PROCESS (GUARDED)
        # =========================
        if h4 is not None and not h4.empty:
            h4 = add_indicators(h4.copy(deep=True))
            if h4 is not None and not h4.empty:
                minor, why, confidence, confidence_pct = minor_phase_4h(h4)
                setup = setup_state(minor)
                stage2 = stage2_trigger(h4, setup)
            else:
                minor, why, confidence, confidence_pct = "NEUTRAL", ["4H invalid"], 0, 0
                setup = "WAIT"
                stage2 = False
        else:
            minor, why, confidence, confidence_pct = "NEUTRAL", ["4H unavailable"], 0, 0
            setup = "WAIT"
            stage2 = False

        # =========================
        # PRICE & METRICS (DAILY ONLY)
        # =========================
        price_today = float(d1["Close"].iloc[-1])
        price_yesterday = float(d1["Close"].iloc[-2])
        price_change = round((price_today / price_yesterday - 1) * 100, 2)

        vol_behavior, vol_ratio = volume_behavior(d1)

        gap_ema13 = round((price_today / d1["EMA13"].iloc[-1] - 1) * 100, 2)
        gap_ema21 = round((price_today / d1["EMA21"].iloc[-1] - 1) * 100, 2)
        gap_ema50 = round((price_today / d1["EMA50"].iloc[-1] - 1) * 100, 2)

        rsi = float(d1["RSI"].iloc[-1])
        stoch = float(d1["STOCH"].iloc[-1])
        sma50 = float(d1["SMA50"].iloc[-1])
        dist_to_sma50 = compute_dist_sma50(d1)

        candle_label, candle_red_breakdown, candle_green_approach = latest_candle_info(d1)
        candle_effect = 1 if candle_green_approach else -1 if candle_red_breakdown else 0

        # =========================
        # CONFIDENCE BOOST
        # =========================
        if minor == "TREND_CONTINUE" and vol_behavior == "VOL_ABSORPTION":
            confidence += 1
            why.append("Volume absorption mendukung kelanjutan trend")
            confidence_pct = round(confidence / 7 * 100)

        if minor == "TREND_CONTINUE" and candle_label == "Hijau Kuat (Impulse)":
            confidence += 1
            why.append("Impulse candle mendukung kelanjutan trend")

        # =========================
        # FINAL DECISION
        # =========================
        final_dec = final_decision(major, minor, setup, stage2, vol_behavior)

        # =========================
        # HARD GUARD (ANTI DATA LEAK)
        # =========================
        if price_today <= 0 or price_today > 1_000_000:
            print(f"INVALID PRICE {kode} {price_today}")
            return None

        # DEBUG TRACE (boleh dihapus nanti)
        print(
            kode,
            ticker,
            d1.index[-1],
            price_today
        )

        # =========================
        # RETURN RESULT
        # =========================
        return {
            "Kode": kode,
            "Ticker": ticker,
            "ProcessTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Price": price_today,
            "PriceChange%": price_change,
            "Gap_EMA13%": gap_ema13,
            "Gap_EMA21%": gap_ema21,
            "Gap_EMA50%": gap_ema50,
            "MajorTrend": major,
            "MinorPhase": minor,
            "WHY_MINOR": why,
            "MinorConfidence": confidence,
            "MinorConfidence%": confidence_pct,
            "SetupState": setup,
            "Stage2Valid": stage2,
            "VOL_BEHAVIOR": vol_behavior,
            "VOL_RATIO": vol_ratio,
            "RSI": round(rsi, 2),
            "SMA50": round(sma50, 2),
            "Dist_to_SMA50": round(dist_to_sma50, 2) if not np.isnan(dist_to_sma50) else np.nan,
            "Stoch_K": round(stoch, 2),
            "Latest_Candle": candle_label,
            "Candle_Effect": candle_effect,
            "FinalDecision": final_dec
        }

    except Exception as e:
        print(f"Failed process_stock: {kode} | {e}")
        return None


# ======================================================
# Market State
# ======================================================
def extract_market_state(df: pd.DataFrame, idx: int) -> dict:
    """
    Extract market state at historical index `idx`
    based on data available UP TO that bar.
    """

    row = df.iloc[: idx + 1].copy()
    last = row.iloc[-1]

    # =========================
    # MAJOR TREND
    # =========================
    if last["EMA13"] > last["EMA21"] > last["EMA50"]:
        major = "STRONG"
    else:
        major = "INVALID"

    # =========================
    # MINOR PHASE
    # (pakai kolom yang SUDAH ADA, fallback aman)
    # =========================
    minor = (
        last["MinorPhase"]
        if "MinorPhase" in row.columns
        else "UNKNOWN"
    )

    # =========================
    # RSI BUCKET
    # =========================
    rsi = last["RSI"]
    if rsi < 30:
        rsi_bucket = "<30"
    elif rsi <= 70:
        rsi_bucket = "30-70"
    else:
        rsi_bucket = ">70"

    # =========================
    # VOL BEHAVIOR (HISTORICAL)
    # =========================
    if "VOL_RATIO" in row.columns:
        vr = last["VOL_RATIO"]
        if vr < 0.8:
            vol_behavior = "VOL_ABSORPTION"
        elif vr > 1.5:
            vol_behavior = "VOL_DISTRIBUTION"
        else:
            vol_behavior = "VOL_NEUTRAL"
    else:
        vol_behavior = "VOL_NEUTRAL"

    # =========================
    # LATEST CANDLE
    # =========================
    latest_candle = (
        last["Latest_Candle"]
        if "Latest_Candle" in row.columns
        else "UNKNOWN"
    )

    return {
        "MajorTrend": major,
        "MinorPhase": minor,
        "RSI_BUCKET": rsi_bucket,
        "VOL_BEHAVIOR": vol_behavior,
        "latest_candle": latest_candle,
        "Close": last["Close"]
    }

