"""
ENGINE V2 - Core Screening Engine
Untuk analisis real-time dan screening saham IDX
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import dari utils module
from utils import data_utils, cache_manager, date_utils, validation_utils

# ======================================================
# CONFIGURATION
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
# CACHE MANAGEMENT (using utils)
# ======================================================
CACHE_DIR = "data_cache"

def get_cache_key(ticker: str, interval: str, period: str) -> str:
    """Generate cache key"""
    safe_ticker = ticker.replace(".", "_").replace(":", "_")
    return f"{safe_ticker}_{interval}_{period}"

def load_cached_data(ticker: str, interval: str, period: str):
    """Load data from cache using cache_manager"""
    cache_key = get_cache_key(ticker, interval, period)
    cached_data = cache_manager.load(cache_key, suffix="pkl")
    return cached_data.copy() if cached_data is not None else None

def save_to_cache(ticker: str, interval: str, period: str, df: pd.DataFrame):
    """Save data to cache using cache_manager"""
    cache_key = get_cache_key(ticker, interval, period)
    cache_manager.save(cache_key, df, suffix="pkl")

def is_cache_fresh(ticker: str, interval: str, period: str, df: pd.DataFrame = None) -> bool:
    """
    Check if cached data is fresh
    Fresh jika: data hari terakhir >= last trading day
    """
    if df is None:
        # Load from cache first
        cached = load_cached_data(ticker, interval, period)
        if cached is None or cached.empty:
            return False
        df = cached
    
    if df.empty:
        return False
    
    # Get last date from data
    last_date = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
    
    # Get last trading day
    last_trading_day = date_utils.get_last_trading_day().date()
    
    return last_date >= last_trading_day

# ======================================================
# DATA FETCHING FUNCTIONS
# ======================================================
def normalize_yf_df(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """
    Normalize yfinance dataframe safely
    """
    if df is None or df.empty:
        return df
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        if ticker is None:
            # Use first available ticker
            available_tickers = df.columns.get_level_values(1).unique()
            if len(available_tickers) > 0:
                ticker = available_tickers[0]
            else:
                raise ValueError("No tickers found in MultiIndex columns")
        
        if ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1)
        else:
            raise ValueError(f"Ticker {ticker} not found in dataframe")
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Ensure proper column names
    column_map = {
        'Adj Close': 'Close',
        'Adj Close': 'Close',
    }
    df.rename(columns=column_map, inplace=True)
    
    return df

def fetch_data(ticker: str, interval: str = "1d", period: str = "12mo", 
               force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch data with caching mechanism
    
    Returns:
    --------
    pd.DataFrame or None
    """
    # Check cache first if not forcing refresh
    if not force_refresh:
        cached_df = load_cached_data(ticker, interval, period)
        if cached_df is not None and is_cache_fresh(ticker, interval, period, cached_df):
            print(f"Using cached data for {ticker}")
            return cached_df.copy()
    
    # Fetch fresh data
    print(f"Fetching fresh data for {ticker} ({interval}, {period})")
    df = data_utils.safe_yf_download(ticker, interval, period)
    
    if df is None or df.empty:
        print(f"Failed to fetch data for {ticker}")
        return None
    
    # Normalize dataframe
    df = normalize_yf_df(df, ticker)
    
    # Ensure required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing required columns for {ticker}: {missing_cols}")
        return None
    
    # Basic data cleaning
    df = df[required_cols].copy()
    df.dropna(inplace=True)
    
    # Check if stock is suspended (no volume for last 10 bars)
    if df['Volume'].iloc[-10:].sum() == 0:
        print(f"Stock {ticker} appears suspended (no volume)")
        return None
    
    # Save to cache
    save_to_cache(ticker, interval, period, df)
    
    return df

def fetch_intraday_safe(
    ticker: str,
    interval: str = "1h",
    period: str = "6mo",
    min_candles: int = 120,
    allow_4h_fallback: bool = True
) -> pd.DataFrame:
    """
    Fetch intraday data safely with fallback options
    
    Returns:
    --------
    pd.DataFrame or None
    """
    cache_key = f"{get_cache_key(ticker, interval, period)}_intraday"
    
    # Check cache first
    cached_data = cache_manager.load(cache_key, suffix="pkl")
    if cached_data is not None and not cached_data.empty:
        # Check if cache is fresh (less than 1 hour old for intraday)
        cache_age = cache_manager.get_cache_age(cache_key, suffix="pkl")
        if cache_age < 3600:  # 1 hour in seconds
            return cached_data.copy()
    
    try:
        # Use safe download from utils
        df = data_utils.safe_yf_download(
            ticker=ticker,
            interval=interval,
            period=period,
            max_retries=2
        )
        
        if df is None or df.empty:
            print(f"[INTRADAY EMPTY] {ticker}")
            
            # Try fallback to 4H if allowed
            if interval == "1h" and allow_4h_fallback:
                print(f"Trying 4H fallback for {ticker}")
                return fetch_intraday_safe(
                    ticker, "4h", "3mo", min_candles//4, False
                )
            return None
        
        # Validate data
        if len(df) < min_candles:
            print(f"[INTRADAY TOO SHORT] {ticker} | {len(df)} bars")
            return None
        
        if df["Volume"].iloc[-10:].sum() == 0:
            print(f"[INTRADAY SUSPENDED] {ticker}")
            return None
        
        # Save to cache
        cache_manager.save(cache_key, df, suffix="pkl")
        
        return df
        
    except Exception as e:
        print(f"[INTRADAY FAIL] {ticker} | {e}")
        return None

# ======================================================
# TECHNICAL INDICATORS
# ======================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to dataframe
    
    Returns:
    --------
    pd.DataFrame with indicators added
    """
    if df is None or df.empty:
        return None
    
    df = df.copy()
    
    # Validate input
    is_valid, error_msg = validation_utils.validate_dataframe(
        df, 
        required_cols=['Open', 'High', 'Low', 'Close', 'Volume']
    )
    
    if not is_valid:
        print(f"Invalid dataframe for indicators: {error_msg}")
        return None
    
    try:
        # EMA Calculations
        df["EMA13"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=EMA_MID, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
        
        # SMA
        df["SMA50"] = df["Close"].rolling(window=SMA50_PERIOD, min_periods=1).mean()
        
        # Volume MA
        df["VOL_MA20"] = df["Volume"].rolling(window=VOL_MA_PERIOD, min_periods=1).mean()
        
        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(window=RSI_PERIOD, min_periods=1).mean()
        loss = (-delta).clip(lower=0).rolling(window=RSI_PERIOD, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Stochastic %K
        low_min = df["Low"].rolling(window=STOCH_PERIOD, min_periods=1).min()
        high_max = df["High"].rolling(window=STOCH_PERIOD, min_periods=1).max()
        df["STOCH"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
        
        # ATR
        tr1 = df["High"] - df["Low"]
        tr2 = abs(df["High"] - df["Close"].shift())
        tr3 = abs(df["Low"] - df["Close"].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR14"] = true_range.rolling(window=14).mean()
        
        # Additional useful indicators
        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        
        # Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
        
        # Price position relative to indicators
        df["Price_vs_EMA13"] = (df["Close"] / df["EMA13"] - 1) * 100
        df["Price_vs_EMA21"] = (df["Close"] / df["EMA21"] - 1) * 100
        df["Price_vs_EMA50"] = (df["Close"] / df["EMA50"] - 1) * 100
        df["Price_vs_SMA50"] = (df["Close"] / df["SMA50"] - 1) * 100
        
        # Volume ratio
        df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]
        
        return df
        
    except Exception as e:
        print(f"Error adding indicators: {e}")
        return None

def ema_slope(series: pd.Series, window: int = SLOPE_WINDOW) -> float:
    """
    Calculate EMA slope (rate of change)
    
    Returns:
    --------
    float: Slope value (positive = uptrend, negative = downtrend)
    """
    if series is None or len(series) < window + 1:
        return 0.0
    
    try:
        # Use simple difference
        return series.iloc[-1] - series.iloc[-window]
    except:
        return 0.0

# ======================================================
# TREND ANALYSIS
# ======================================================
def major_trend_daily(df: pd.DataFrame) -> str:
    """
    Determine major trend based on daily timeframe
    
    Returns:
    --------
    str: "STRONG", "WEAK", or "INVALID"
    """
    if df is None or len(df) < 50:
        return "INVALID"
    
    try:
        last = df.iloc[-1]
        
        # Check if required columns exist
        required = ["EMA21", "EMA50"]
        if not all(col in df.columns for col in required):
            return "INVALID"
        
        # Rule 1: EMA21 must be above EMA50
        if last["EMA21"] <= last["EMA50"]:
            return "INVALID"
        
        # Rule 2: EMA21 slope must be positive
        if ema_slope(df["EMA21"]) <= 0:
            return "INVALID"
        
        # Rule 3: Price should be above EMA21
        if last["Close"] < last["EMA21"]:
            return "WEAK"
        
        return "STRONG"
        
    except Exception as e:
        print(f"Error in major_trend_daily: {e}")
        return "INVALID"

# ======================================================
# MINOR PHASE ANALYSIS (MBMA Engine)
# ======================================================
def minor_phase_4h(df: pd.DataFrame):
    """
    Determine minor phase based on 4H timeframe (MBMA methodology)
    
    Returns:
    --------
    tuple: (phase, reasons, confidence, confidence_pct)
    """
    # Input validation
    if df is None or len(df) < max(10, SLOPE_WINDOW) + 1:
        return "NEUTRAL", ["Data tidak cukup"], 0, 0
    
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        reasons = []
        confidence = 0
        
        # Check required columns
        required_cols = ["EMA13", "EMA21", "EMA50", "Close", "RSI", "STOCH"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return "NEUTRAL", [f"Missing columns: {missing_cols}"], 0, 0
        
        # Helper calculations
        ema_compress = abs(last["EMA13"] - last["EMA21"]) / last["EMA21"] < EMA_COMPRESS_TH
        pullback_recovered = (last["Close"] > last["EMA21"]) and (prev["Close"] <= prev["EMA21"])
        
        # Check past structure (6 bars ago)
        lookback_idx = max(0, len(df) - 7)
        if lookback_idx < len(df):
            past_structure = df.iloc[lookback_idx]
            was_bearish = (
                past_structure["EMA50"] >= past_structure["EMA21"] >= past_structure["EMA13"]
            )
        else:
            was_bearish = False
        
        # Current structure
        is_bullish = last["EMA13"] > last["EMA21"] > last["EMA50"]
        ema13_cross = (prev["EMA13"] <= prev["EMA21"]) and (last["EMA13"] > last["EMA21"])
        ema21_up = ema_slope(df["EMA21"]) > 0
        ema50_flat = ema_slope(df["EMA50"]) >= -0.001  # Slightly negative is okay
        price_above_ema13 = last["Close"] >= last["EMA13"]
        
        # =============================
        # 1. EMA COMPRESS PULLBACK
        # =============================
        if ema_compress:
            reasons.append("EMA13 dan EMA21 dalam fase kompresi (<0.3% distance)")
            return "EMA_COMPRESS_PULLBACK", reasons, 1, 14  # 1/7 ≈ 14%
        
        # =============================
        # 2. PULLBACK RECOVERED
        # =============================
        if pullback_recovered:
            reasons.append("Harga kembali menutup di atas EMA21 setelah pullback")
            return "PULLBACK_RECOVERED", reasons, 1, 14
        
        # =============================
        # 3. TREND CONTINUE
        # =============================
        if is_bullish and price_above_ema13:
            # Calculate confidence based on multiple factors
            trend_factors = {
                "Struktur EMA bullish (13>21>50)": is_bullish,
                "Harga di atas EMA13": price_above_ema13,
                "EMA13 cross atas EMA21": ema13_cross,
                "EMA21 slope positif": ema21_up,
                "EMA50 tidak turun signifikan": ema50_flat,
                "Struktur sebelumnya bearish": was_bearish,
                "RSI dalam range sehat (30-70)": 30 <= last["RSI"] <= 70,
            }
            
            passed_factors = [factor for factor, condition in trend_factors.items() if condition]
            confidence = len(passed_factors)
            confidence_pct = round((confidence / len(trend_factors)) * 100)
            
            reasons.extend(passed_factors)
            return "TREND_CONTINUE", reasons, confidence, confidence_pct
        
        # =============================
        # 4. TREND OVEREXTEND
        # =============================
        dist_ema13 = (last["Close"] - last["EMA13"]) / last["EMA13"]
        dist_ema21 = (last["Close"] - last["EMA21"]) / last["EMA21"]
        dist_ema50 = (last["Close"] - last["EMA50"]) / last["EMA50"]
        
        is_overextended = (
            dist_ema13 > 0.05 and  # >5% dari EMA13
            dist_ema21 > 0.10 and  # >10% dari EMA21
            dist_ema50 > 0.20 and  # >20% dari EMA50
            (last["RSI"] >= 70 or last["STOCH"] >= 85)
        )
        
        if is_overextended:
            reasons.append("Harga terlalu jauh dari EMA support (overextended)")
            reasons.append(f"RSI: {last['RSI']:.1f}, Stoch: {last['STOCH']:.1f}")
            return "TREND_OVEREXTEND", reasons, 2, 29  # 2/7 ≈ 29%
        
        # =============================
        # 5. NEUTRAL / NO CLEAR SIGNAL
        # =============================
        reasons.append("Tidak ada sinyal minor phase yang jelas")
        return "NEUTRAL", reasons, 0, 0
        
    except Exception as e:
        print(f"Error in minor_phase_4h: {e}")
        return "NEUTRAL", [f"Error: {str(e)}"], 0, 0

def setup_state(minor_phase: str) -> str:
    """
    Determine setup state based on minor phase
    """
    if minor_phase == "EMA_COMPRESS_PULLBACK":
        return "SETUP_PENDING"
    
    if minor_phase in ["PULLBACK_RECOVERED", "TREND_CONTINUE"]:
        return "STAGE2_READY"
    
    return "WAIT"

def stage2_trigger(df: pd.DataFrame, setup: str) -> bool:
    """
    Check if stage 2 trigger conditions are met
    """
    if setup != "STAGE2_READY" or df is None or len(df) < 2:
        return False
    
    try:
        last = df.iloc[-1]
        return last["Close"] > last["EMA13"] and ema_slope(df["EMA13"]) > 0
    except:
        return False

# ======================================================
# VOLUME ANALYSIS
# ======================================================
def volume_behavior(df: pd.DataFrame):
    """
    Analyze volume behavior (VSA-inspired)
    
    Returns:
    --------
    tuple: (behavior, vol_ratio, volume, vol_ma20)
    """
    if df is None or len(df) < 2:
        return "VOL_NEUTRAL", 0.0, 0, 0
    
    try:
        last = df.iloc[-1]
        
        # Extract values with safety checks
        open_ = last.get("Open", 0)
        high = last.get("High", 0)
        low = last.get("Low", 0)
        close = last.get("Close", 0)
        volume = last.get("Volume", 0)
        vol_ma20 = last.get("VOL_MA20", 1)
        ema21 = last.get("EMA21", close)
        
        # Validate inputs
        if high <= low or volume <= 0 or vol_ma20 <= 0:
            return "VOL_NEUTRAL", 0.0, volume, vol_ma20
        
        # Calculate metrics
        price_range = high - low
        body_size = abs(close - open_)
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        
        body_ratio = body_size / price_range if price_range > 0 else 0
        vol_ratio = volume / vol_ma20
        
        # =============================
        # 1. LOW ACTIVITY
        # =============================
        if vol_ratio < 1.1:
            return "VOL_NEUTRAL", round(vol_ratio, 2), volume, vol_ma20
        
        # =============================
        # 2. ABSORPTION (Bullish)
        # Volume tinggi, range kecil, lower wick panjang
        # =============================
        absorption = (
            vol_ratio >= 1.3 and
            body_ratio <= 0.35 and
            lower_wick >= body_size * 1.2 and
            close >= ema21
        )
        
        if absorption:
            return "VOL_ABSORPTION", round(vol_ratio, 2), volume, vol_ma20
        
        # =============================
        # 3. DISTRIBUTION (Bearish)
        # Volume tinggi, range kecil, upper wick panjang
        # =============================
        distribution = (
            vol_ratio >= 1.3 and
            body_ratio <= 0.35 and
            upper_wick >= body_size * 1.2 and
            close < ema21
        )
        
        if distribution:
            return "VOL_DISTRIBUTION", round(vol_ratio, 2), volume, vol_ma20
        
        # =============================
        # 4. EXPANSION (Strong Move)
        # Volume tinggi, body besar
        # =============================
        expansion = (
            vol_ratio >= 1.5 and
            body_ratio >= 0.55
        )
        
        if expansion:
            return "VOL_EXPANSION", round(vol_ratio, 2), volume, vol_ma20
        
        # =============================
        # 5. NEUTRAL (Elevated Volume)
        # =============================
        if vol_ratio >= 1.2:
            return "VOL_ELEVATED", round(vol_ratio, 2), volume, vol_ma20
        
        return "VOL_NEUTRAL", round(vol_ratio, 2), volume, vol_ma20
        
    except Exception as e:
        print(f"Error in volume_behavior: {e}")
        return "VOL_NEUTRAL", 0.0, 0, 0

# ======================================================
# CANDLE ANALYSIS
# ======================================================
def latest_candle_info(df: pd.DataFrame):
    """
    Analyze the latest candle
    
    Returns:
    --------
    tuple: (label, is_red, is_green)
    """
    if df is None or len(df) < 2:
        return "N/A", 0, 0
    
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Extract values
        open_ = last.get("Open", 0)
        close = last.get("Close", 0)
        high = last.get("High", 0)
        low = last.get("Low", 0)
        atr14 = last.get("ATR14", 0)
        sma50 = last.get("SMA50", 0)
        
        # Basic validation
        if high <= low:
            return "Doji/Netral", 0, 0
        
        # Calculate metrics
        price_range = high - low
        body_size = abs(close - open_)
        body_ratio = body_size / price_range if price_range > 0 else 0
        
        is_red = close < open_
        is_green = close > open_
        
        # Check for strong green (impulse) candle
        strong_green = (
            is_green and
            body_ratio >= 0.6 and
            price_range >= 1.2 * atr14 and
            (high - close) / price_range <= 0.15
        )
        
        # Check for breakdown candle
        breakdown = (
            is_red and
            body_ratio > BODY_THRESHOLD and
            close < sma50 and
            prev.get("Close", 0) >= sma50
        )
        
        # Check for approach to SMA50
        if sma50 > 0:
            dist_to_sma50 = abs((close - sma50) / sma50 * 100)
            approaching_sma50 = is_green and dist_to_sma50 <= CANDLE_DIST_TH
        else:
            approaching_sma50 = False
        
        # Priority classification
        if strong_green:
            return "Hijau Kuat (Impulse)", 0, 1
        elif breakdown:
            return "Merah Kuat & Breakdown", -1, 0
        elif approaching_sma50:
            return "Hijau & Mendekati SMA50", 0, 1
        elif is_red:
            return "Merah Biasa", 0, 0
        elif is_green:
            return "Hijau Biasa", 0, 0
        else:
            return "Doji/Netral", 0, 0
            
    except Exception as e:
        print(f"Error in latest_candle_info: {e}")
        return "Doji/Netral", 0, 0

def compute_dist_sma50(df: pd.DataFrame) -> float:
    """
    Calculate distance to SMA50 as percentage
    
    Returns:
    --------
    float: Percentage distance (positive = above, negative = below)
    """
    if df is None or len(df) == 0:
        return np.nan
    
    try:
        last = df.iloc[-1]
        sma50 = last.get("SMA50", 0)
        
        if pd.isna(sma50) or sma50 == 0:
            return np.nan
        
        close = last.get("Close", 0)
        return (close - sma50) / sma50 * 100
        
    except:
        return np.nan

# ======================================================
# DECISION ENGINE
# ======================================================
def final_decision(
    major_trend: str,
    minor_phase: str,
    setup_state: str,
    stage2_trigger: bool,
    volume_behavior: str
) -> str:
    """
    Make final trading decision based on all factors
    
    Returns:
    --------
    str: Decision string
    """
    # Basic validation
    if major_trend == "INVALID":
        return "SKIP"
    
    if setup_state == "SETUP_PENDING":
        return "SETUP_PENDING"
    
    # Entry ready conditions
    entry_conditions = (
        major_trend == "STRONG" and
        setup_state == "STAGE2_READY" and
        stage2_trigger and
        volume_behavior in ["VOL_ABSORPTION", "VOL_EXPANSION", "VOL_ELEVATED"]
    )
    
    if entry_conditions:
        return "ENTRY_READY"
    
    # Caution conditions
    if volume_behavior == "VOL_DISTRIBUTION":
        return "WAIT_DISTRIBUTION"
    
    # Default wait
    return "WAIT"

# ======================================================
# MAIN STOCK PROCESSING FUNCTION
# ======================================================
def process_stock(kode: str, use_cache: bool = True):
    """
    Process a single stock for screening
    
    Returns:
    --------
    dict: Stock analysis results or None if failed
    """
    ticker = f"{kode}.JK"
    
    try:
        print(f"Processing {kode}...")
        
        # =========================
        # 1. FETCH DAILY DATA
        # =========================
        d1 = fetch_data(ticker, "1d", "12mo", force_refresh=not use_cache)
        if d1 is None or d1.empty:
            print(f"  No daily data for {kode}")
            return None
        
        # Basic validation
        if "Volume" not in d1.columns or d1["Volume"].iloc[-1] <= 0:
            print(f"  Invalid volume for {kode}")
            return None
        
        # =========================
        # 2. ADD INDICATORS
        # =========================
        d1 = add_indicators(d1.copy())
        if d1 is None or d1.empty:
            print(f"  Failed to add indicators for {kode}")
            return None
        
        # =========================
        # 3. MAJOR TREND
        # =========================
        major = major_trend_daily(d1)
        
        # =========================
        # 4. MINOR PHASE (try intraday first)
        # =========================
        minor = "NEUTRAL"
        why = []
        confidence = 0
        confidence_pct = 0
        setup = "WAIT"
        stage2 = False
        
        # Try intraday (4H equivalent)
        h4 = fetch_intraday_safe(ticker, "1h", "6mo")
        
        if h4 is not None and not h4.empty:
            h4 = add_indicators(h4.copy())
            if h4 is not None and not h4.empty:
                minor, why, confidence, confidence_pct = minor_phase_4h(h4)
                setup = setup_state(minor)
                stage2 = stage2_trigger(h4, setup)
            else:
                # Fallback to daily for minor phase
                minor, why, confidence, confidence_pct = minor_phase_4h(d1)
                why = ["Fallback to Daily"] + why
                setup = setup_state(minor)
                stage2 = stage2_trigger(d1, setup)
        else:
            # Use daily timeframe
            minor, why, confidence, confidence_pct = minor_phase_4h(d1)
            why = ["Using Daily TF"] + why
            setup = setup_state(minor)
            stage2 = stage2_trigger(d1, setup)
        
        # =========================
        # 5. VOLUME ANALYSIS
        # =========================
        vol_behavior, vol_ratio, volume, vol_ma20 = volume_behavior(d1)
        
        # =========================
        # 6. CANDLE ANALYSIS
        # =========================
        candle_label, candle_red, candle_green = latest_candle_info(d1)
        candle_effect = 1 if candle_green else -1 if candle_red else 0
        
        # =========================
        # 7. PRICE METRICS
        # =========================
        last_idx = len(d1) - 1
        price_today = float(d1["Close"].iloc[last_idx])
        price_yesterday = float(d1["Close"].iloc[last_idx - 1]) if last_idx > 0 else price_today
        price_change = round((price_today / price_yesterday - 1) * 100, 2)
        
        # Gap calculations
        gap_ema13 = round((price_today / d1["EMA13"].iloc[last_idx] - 1) * 100, 2)
        gap_ema21 = round((price_today / d1["EMA21"].iloc[last_idx] - 1) * 100, 2)
        gap_ema50 = round((price_today / d1["EMA50"].iloc[last_idx] - 1) * 100, 2)
        
        # Technical readings
        rsi = float(d1["RSI"].iloc[last_idx])
        stoch = float(d1["STOCH"].iloc[last_idx])
        sma50 = float(d1["SMA50"].iloc[last_idx])
        dist_to_sma50 = compute_dist_sma50(d1)
        
        # =========================
        # 8. CONFIDENCE ADJUSTMENTS
        # =========================
        # Boost confidence for favorable conditions
        if minor == "TREND_CONTINUE":
            if vol_behavior in ["VOL_ABSORPTION", "VOL_EXPANSION"]:
                confidence += 1
                why.append("Volume mendukung kelanjutan trend")
            
            if candle_label == "Hijau Kuat (Impulse)":
                confidence += 1
                why.append("Impulse candle terdeteksi")
        
        # Recalculate confidence percentage
        confidence_pct = round((confidence / 7) * 100) if 7 > 0 else 0
        
        # =========================
        # 9. FINAL DECISION
        # =========================
        final_dec = final_decision(major, minor, setup, stage2, vol_behavior)
        
        # =========================
        # 10. VALIDATION GUARDS
        # =========================
        if not validation_utils.validate_price_data(price_today):
            print(f"  Invalid price for {kode}: {price_today}")
            return None
        
        if price_today > 1_000_000:  # Unrealistic price for IDX
            print(f"  Suspicious price for {kode}: {price_today}")
            return None
        
        # =========================
        # 11. RETURN RESULTS
        # =========================
        result = {
            "Kode": kode,
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
            "Volume": volume,
            "Vol_20MA": vol_ma20,
            "VOL_BEHAVIOR": vol_behavior,
            "VOL_RATIO": vol_ratio,
            "RSI": round(rsi, 2),
            "SMA50": round(sma50, 2),
            "Dist_to_SMA50": round(dist_to_sma50, 2) if not np.isnan(dist_to_sma50) else np.nan,
            "Stoch_K": round(stoch, 2),
            "Latest_Candle": candle_label,
            "Candle_Effect": candle_effect,
            "FinalDecision": final_dec,
            "ProcessTimestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"  ✓ Processed {kode}: {major}/{minor}/{final_dec}")
        return result
        
    except Exception as e:
        print(f"  ✗ Error processing {kode}: {str(e)}")
        return None

# ======================================================
# MARKET STATE EXTRACTION
# ======================================================
def extract_market_state(df: pd.DataFrame, idx: int) -> dict:
    """
    Extract market state at historical index
    
    Returns:
    --------
    dict: Market state parameters
    """
    if df is None or idx < 0 or idx >= len(df):
        return {}
    
    try:
        # Get data up to the index
        subset = df.iloc[:idx + 1].copy()
        last = subset.iloc[-1]
        
        # Major Trend
        if ("EMA13" in subset.columns and "EMA21" in subset.columns and 
            "EMA50" in subset.columns):
            major_trend = "STRONG" if (
                last["EMA13"] > last["EMA21"] > last["EMA50"]
            ) else "INVALID"
        else:
            major_trend = "UNKNOWN"
        
        # Minor Phase (if available)
        minor_phase = last.get("MinorPhase", "UNKNOWN")
        
        # RSI Bucket
        rsi = last.get("RSI", 50)
        if rsi < 30:
            rsi_bucket = "<30"
        elif rsi <= 70:
            rsi_bucket = "30-70"
        else:
            rsi_bucket = ">70"
        
        # Volume Behavior
        vol_ratio = last.get("VOL_RATIO", 1.0)
        if vol_ratio < 0.8:
            vol_behavior = "VOL_ABSORPTION"
        elif vol_ratio > 1.5:
            vol_behavior = "VOL_DISTRIBUTION"
        else:
            vol_behavior = "VOL_NEUTRAL"
        
        # Latest Candle
        latest_candle = last.get("Latest_Candle", "UNKNOWN")
        
        return {
            "MajorTrend": major_trend,
            "MinorPhase": minor_phase,
            "RSI_BUCKET": rsi_bucket,
            "VOL_BEHAVIOR": vol_behavior,
            "latest_candle": latest_candle,
            "Close": last.get("Close", 0),
            "RSI": rsi,
            "VOL_RATIO": vol_ratio
        }
        
    except Exception as e:
        print(f"Error extracting market state at index {idx}: {e}")
        return {}

# ======================================================
# MAIN EXECUTION GUARD
# ======================================================
if __name__ == "__main__":
    # Test the engine
    test_codes = ["BBCA", "BBRI", "TLKM"]
    
    print("Testing engine_v2.py")
    print("=" * 50)
    
    for kode in test_codes:
        print(f"\nProcessing {kode}...")
        result = process_stock(kode, use_cache=True)
        
        if result:
            print(f"  Price: {result['Price']:,}")
            print(f"  Major Trend: {result['MajorTrend']}")
            print(f"  Minor Phase: {result['MinorPhase']}")
            print(f"  Final Decision: {result['FinalDecision']}")
            print(f"  RSI: {result['RSI']}, Stoch: {result['Stoch_K']}")
        else:
            print(f"  Failed to process {kode}")
    
    print("\n" + "=" * 50)
    print("Engine test completed")