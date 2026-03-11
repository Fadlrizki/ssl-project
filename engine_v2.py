"""
ENGINE V2 - Core Screening Engine
Analisis real-time dan screening saham IDX dengan Wyckoff Phase Detection
"""

import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import yfinance as yf

from utils import data_utils, cache_manager, date_utils, validation_utils

warnings.filterwarnings('ignore')

# ======================================================
# CONFIGURATION CONSTANTS
# ======================================================

# EMA Periods
EMA_FAST = 13
EMA_MID = 21
EMA_SLOW = 50
SMA50_PERIOD = 50

# Volume
VOL_MA_PERIOD = 20

# Oscillators
RSI_PERIOD = 14
STOCH_PERIOD = 14

# Trend Analysis
SLOPE_WINDOW = 10
EMA_COMPRESS_TH = 0.003

# Candle Analysis
BODY_THRESHOLD = 0.02
CANDLE_DIST_TH = 5
TOLERANCE = 0.02

# Confidence Scoring
BASE_CONF = 6
EXTRA_CONF = 2
MAX_CONF = BASE_CONF + EXTRA_CONF

# Wyckoff Constants
WYCKOFF_LOOKBACK = 50
SPRING_THRESHOLD = 0.98  # 2% below support
UPTHRUST_THRESHOLD = 1.02  # 2% above resistance
VOLUME_SPIKE_THRESHOLD = 1.5

# ======================================================
# CACHE MANAGEMENT
# ======================================================

CACHE_DIR = "data_cache"

def get_cache_key(ticker: str, interval: str, period: str) -> str:
    """Generate cache key for data caching"""
    safe_ticker = ticker.replace(".", "_").replace(":", "_")
    return f"{safe_ticker}_{interval}_{period}"

def load_cached_data(ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Load data from cache using cache_manager"""
    cache_key = get_cache_key(ticker, interval, period)
    cached_data = cache_manager.load(cache_key, suffix="pkl")
    return cached_data.copy() if cached_data is not None else None

def save_to_cache(ticker: str, interval: str, period: str, df: pd.DataFrame) -> None:
    """Save data to cache using cache_manager"""
    cache_key = get_cache_key(ticker, interval, period)
    cache_manager.save(cache_key, df, suffix="pkl")

def is_cache_fresh(
    ticker: str, 
    interval: str, 
    period: str, 
    df: Optional[pd.DataFrame] = None
) -> bool:
    """
    Check if cached data is fresh
    Fresh if: last data date >= last trading day
    """
    if df is None:
        cached = load_cached_data(ticker, interval, period)
        if cached is None or cached.empty:
            return False
        df = cached
    
    if df.empty:
        return False
    
    # Get last date from data
    last_date = df.index[-1]
    last_date = last_date.date() if hasattr(last_date, 'date') else last_date
    
    # Get last trading day
    last_trading_day = date_utils.get_last_trading_day().date()
    
    return last_date >= last_trading_day

# ======================================================
# DATA FETCHING FUNCTIONS
# ======================================================

def normalize_yf_df(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize yfinance dataframe safely
    
    Args:
        df: Raw yfinance dataframe
        ticker: Stock ticker for MultiIndex handling
    
    Returns:
        Normalized dataframe
    """
    if df is None or df.empty:
        return df
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        if ticker is None:
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
    
    # Standardize column names
    column_map = {'Adj Close': 'Close'}
    df.rename(columns=column_map, inplace=True)
    
    return df

def fetch_data(
    ticker: str, 
    interval: str = "1d", 
    period: str = "12mo", 
    force_refresh: bool = False
) -> Optional[pd.DataFrame]:
    """
    Fetch data with caching mechanism
    
    Args:
        ticker: Stock ticker (e.g., "BBCA.JK")
        interval: Data interval ("1d", "1h", etc.)
        period: Data period ("12mo", "6mo", etc.)
        force_refresh: Force fresh download ignoring cache
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    # Check cache first if not forcing refresh
    if not force_refresh:
        cached_df = load_cached_data(ticker, interval, period)
        if cached_df is not None and is_cache_fresh(ticker, interval, period, cached_df):
            print(f"  Using cached data for {ticker}")
            return cached_df.copy()
    
    # Fetch fresh data
    print(f"  Fetching fresh data for {ticker} ({interval}, {period})")
    df = data_utils.safe_yf_download(ticker, interval, period)
    
    if df is None or df.empty:
        print(f"  Failed to fetch data for {ticker}")
        return None
    
    # Normalize dataframe
    df = normalize_yf_df(df, ticker)
    
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"  Missing required columns for {ticker}: {missing_cols}")
        return None
    
    # Basic data cleaning
    df = df[required_cols].copy()
    df.dropna(inplace=True)
    
    # Check if stock is suspended (no volume for last 10 bars)
    if len(df) >= 10 and df['Volume'].iloc[-10:].sum() == 0:
        print(f"  Stock {ticker} appears suspended (no volume)")
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
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday data safely with fallback options
    
    Args:
        ticker: Stock ticker
        interval: Data interval
        period: Data period
        min_candles: Minimum required candles
        allow_4h_fallback: Allow fallback to 4H data
    
    Returns:
        DataFrame with intraday data or None
    """
    cache_key = f"{get_cache_key(ticker, interval, period)}_intraday"
    
    # Check cache first
    cached_data = cache_manager.load(cache_key, suffix="pkl")
    if cached_data is not None and not cached_data.empty:
        try:
            last_date = cached_data.index[-1]
            today = pd.Timestamp.today().date()
            
            # Use cache if from today
            if hasattr(last_date, 'date'):
                if last_date.date() == today:
                    return cached_data.copy()
            elif isinstance(last_date, str):
                cache_date = pd.to_datetime(last_date).date()
                if cache_date == today:
                    return cached_data.copy()
        except Exception:
            pass  # If error, fetch fresh data
    
    try:
        # Use safe download from utils
        df = data_utils.safe_yf_download(
            ticker=ticker,
            interval=interval,
            period=period,
            max_retries=2
        )
        
        if df is None or df.empty:
            print(f"  [INTRADAY EMPTY] {ticker}")
            
            # Try fallback to 4H if allowed
            if interval == "1h" and allow_4h_fallback:
                print(f"  Trying 4H fallback for {ticker}")
                return fetch_intraday_safe(
                    ticker, "4h", "3mo", min_candles // 4, False
                )
            return None
        
        # Validate data
        if len(df) < min_candles:
            print(f"  [INTRADAY TOO SHORT] {ticker} | {len(df)} bars")
            return None
        
        if df["Volume"].iloc[-10:].sum() == 0:
            print(f"  [INTRADAY SUSPENDED] {ticker}")
            return None
        
        # Save to cache
        cache_manager.save(cache_key, df, suffix="pkl")
        
        return df
        
    except Exception as e:
        print(f"  [INTRADAY FAIL] {ticker} | {e}")
        return None

# ======================================================
# VALUE TRX FUNCTIONS
# ======================================================

def calculate_value_trx_from_1m(
    ticker: str, 
    date: Optional[str] = None, 
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate value transaction from 1-minute data
    
    Args:
        ticker: Stock ticker (BBCA.JK)
        date: Date (YYYY-MM-DD), defaults to today
        use_cache: Use cache for faster access
    
    Returns:
        Dictionary with value transaction metrics
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Generate cache key
    cache_key = f"valuetrx_{ticker.replace('.', '_')}_{date}"
    
    # Check cache first if enabled
    if use_cache:
        cached = cache_manager.load(cache_key, suffix="pkl")
        if cached is not None:
            print(f"    Using cached value trx for {ticker} on {date}")
            return cached
    
    try:
        print(f"    Fetching 1m data for value trx: {ticker} ({date})")
        
        # Format dates for yfinance
        start_date = pd.Timestamp(date).strftime("%Y-%m-%d")
        end_date = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        df_1m = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1m",
            progress=False,
            threads=False,
            timeout=10
        )
        
        if df_1m.empty:
            print(f"      No 1m data available for {ticker} on {date}")
            return {
                'total_value': 0,
                'avg_price': 0,
                'volume_total': 0,
                'bars_count': 0,
                'vwap': 0,
                'date': date,
                'status': 'NO_DATA'
            }
        
        # Handle insufficient bars
        if len(df_1m) < 10:
            print(f"      Insufficient 1m bars for {ticker}: {len(df_1m)} bars")
            return {
                'total_value': 0,
                'avg_price': 0,
                'volume_total': 0,
                'bars_count': len(df_1m),
                'vwap': 0,
                'date': date,
                'status': 'INSUFFICIENT_BARS'
            }
        
        # Normalize MultiIndex if needed
        if isinstance(df_1m.columns, pd.MultiIndex):
            if ticker in df_1m.columns.get_level_values(1):
                df_1m = df_1m.xs(ticker, axis=1, level=1)
            else:
                available_tickers = df_1m.columns.get_level_values(1).unique()
                if len(available_tickers) > 0:
                    df_1m = df_1m.xs(available_tickers[0], axis=1, level=1)
                else:
                    return {
                        'total_value': 0,
                        'avg_price': 0,
                        'volume_total': 0,
                        'bars_count': 0,
                        'vwap': 0,
                        'date': date,
                        'status': 'NO_TICKER_DATA'
                    }
        
        # Standardize column names
        column_map = {'Adj Close': 'Close'}
        df_1m.rename(columns=column_map, inplace=True)
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df_1m.columns]
        
        if missing_cols:
            print(f"      Missing columns for {ticker}: {missing_cols}")
            return {
                'total_value': 0,
                'avg_price': 0,
                'volume_total': 0,
                'bars_count': 0,
                'vwap': 0,
                'date': date,
                'status': f'MISSING_COLS'
            }
        
        # Clean data
        df_1m = df_1m[required_cols].copy()
        df_1m.dropna(inplace=True)
        
        if len(df_1m) == 0:
            return {
                'total_value': 0,
                'avg_price': 0,
                'volume_total': 0,
                'bars_count': 0,
                'vwap': 0,
                'date': date,
                'status': 'NO_VALID_DATA'
            }
        
        # Calculate metrics
        volume_total = df_1m['Volume'].sum()
        
        # Use typical price for more accurate calculation
        df_1m['TypicalPrice'] = (df_1m['High'] + df_1m['Low'] + df_1m['Close']) / 3
        vwap = (df_1m['TypicalPrice'] * df_1m['Volume']).sum() / volume_total if volume_total > 0 else 0
        
        # Value per bar and total value
        df_1m['ValuePerBar'] = df_1m['TypicalPrice'] * df_1m['Volume']
        total_value = df_1m['ValuePerBar'].sum()
        
        # Average price
        avg_price = df_1m['TypicalPrice'].mean()
        
        # Additional metrics
        first_price = df_1m['Close'].iloc[0] if len(df_1m) > 0 else 0
        last_price = df_1m['Close'].iloc[-1] if len(df_1m) > 0 else 0
        price_change = ((last_price / first_price) - 1) * 100 if first_price > 0 else 0
        
        # Volume distribution
        if len(df_1m) > 0:
            high_volume_bars = len(df_1m[df_1m['Volume'] > df_1m['Volume'].median()])
        else:
            high_volume_bars = 0
        
        result = {
            'total_value': total_value,
            'total_value_rp': f"Rp {total_value:,.0f}",
            'total_value_m': round(total_value / 1_000_000, 2),
            'total_value_b': round(total_value / 1_000_000_000, 3),
            'vwap': vwap,
            'avg_price': avg_price,
            'volume_total': volume_total,
            'volume_total_formatted': f"{volume_total:,.0f}",
            'bars_count': len(df_1m),
            'first_price': first_price,
            'last_price': last_price,
            'price_change_pct': round(price_change, 2),
            'high_volume_bars': high_volume_bars,
            'high_volume_ratio': round(high_volume_bars / len(df_1m), 2) if len(df_1m) > 0 else 0,
            'date': date,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'SUCCESS'
        }
        
        # Save to cache
        if use_cache:
            cache_manager.save(cache_key, result, suffix="pkl")
        
        return result
        
    except Exception as e:
        print(f"    Error calculating value trx from 1m for {ticker}: {e}")
        return {
            'total_value': 0,
            'avg_price': 0,
            'volume_total': 0,
            'bars_count': 0,
            'vwap': 0,
            'date': date,
            'status': f'ERROR: {str(e)[:50]}'
        }

def calculate_daily_value_trx(df_daily: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate value transaction from daily data (fallback method)
    
    Args:
        df_daily: Daily OHLCV dataframe
    
    Returns:
        Dictionary with value transaction metrics
    """
    if df_daily.empty or len(df_daily) < 1:
        return {}
    
    try:
        # Get last day data
        last_day = df_daily.iloc[-1]
        
        # Calculate typical price
        typical_price = (
            last_day['High'] + last_day['Low'] + last_day['Close']
        ) / 3
        
        # Calculate value transaction
        value_trx = last_day['Volume'] * typical_price
        
        # Format date
        if hasattr(last_day.name, 'strftime'):
            date_str = last_day.name.strftime("%Y-%m-%d")
        else:
            date_str = str(last_day.name)
        
        return {
            'total_value': value_trx,
            'total_value_rp': f"Rp {value_trx:,.0f}",
            'total_value_m': round(value_trx / 1_000_000, 2),
            'total_value_b': round(value_trx / 1_000_000_000, 3),
            'vwap': typical_price,
            'avg_price': typical_price,
            'volume_total': last_day['Volume'],
            'volume_total_formatted': f"{last_day['Volume']:,.0f}",
            'bars_count': 1,
            'first_price': last_day['Open'],
            'last_price': last_day['Close'],
            'price_change_pct': round(((last_day['Close'] / last_day['Open']) - 1) * 100, 2),
            'date': date_str,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'DAILY_APPROXIMATION',
            'status': 'SUCCESS_DAILY'
        }
        
    except Exception as e:
        print(f"  Error in calculate_daily_value_trx: {e}")
        return {}

def get_sector_info(ticker: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Get sector information for a ticker with caching
    
    Args:
        ticker: Stock ticker (BBCA.JK)
        use_cache: Use cache for faster access
    
    Returns:
        Dictionary with sector information
    """
    # Generate cache key
    cache_key = f"sector_{ticker.replace('.', '_')}"
    
    # Check cache first if enabled
    if use_cache:
        cached = cache_manager.load(cache_key, suffix="pkl")
        if cached is not None:
            return cached
    
    try:
        # Create yfinance ticker object
        yf_ticker = yf.Ticker(ticker)
        
        # Get info with retry mechanism
        max_retries = 2
        for attempt in range(max_retries):
            try:
                info = yf_ticker.info
                
                # Extract sector information
                sector_info = {
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'marketCap': info.get('marketCap', 0),
                    'company_name': info.get('longName', ticker),
                    'country': info.get('country', 'ID')
                }
                
                # Save to cache
                if use_cache:
                    cache_manager.save(cache_key, sector_info, suffix="pkl")
                
                return sector_info
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Wait before retry
                    continue
                else:
                    raise e
                    
    except Exception as e:
        print(f"  Error getting sector info for {ticker}: {e}")
        # Return default values
        return {
            'sector': 'Unknown',
            'industry': 'Unknown',
            'marketCap': 0,
            'company_name': ticker,
            'country': 'ID'
        }

def get_value_trx_metrics(
    ticker: str, 
    df_daily: Optional[pd.DataFrame] = None, 
    use_1m_preferred: bool = True, 
    date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get value transaction metrics with fallback strategy
    
    Args:
        ticker: Stock ticker
        df_daily: Daily dataframe (for fallback)
        use_1m_preferred: Prefer 1-minute data
        date: Specific date to analyze
    
    Returns:
        Dictionary with value transaction metrics
    """
    if date is None and df_daily is not None and not df_daily.empty:
        last_date = df_daily.index[-1]
        if hasattr(last_date, 'strftime'):
            date = last_date.strftime("%Y-%m-%d")
        else:
            date = str(last_date)
    
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Adjust for weekends
    check_date = pd.Timestamp(date)
    if check_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        days_back = check_date.weekday() - 4  # 4 = Friday
        check_date = check_date - pd.Timedelta(days=days_back)
        date = check_date.strftime("%Y-%m-%d")
        print(f"    Adjusted to last trading day: {date}")
    
    # Try 1m data first if preferred
    if use_1m_preferred:
        result_1m = calculate_value_trx_from_1m(ticker, date, use_cache=True)
        
        if (result_1m.get('status') in ['SUCCESS', 'INSUFFICIENT_BARS'] and 
            result_1m.get('total_value', 0) > 0 and
            result_1m.get('bars_count', 0) >= 5):
            result_1m['method'] = '1M_ACCURATE'
            return result_1m
        
        print(f"    1m data insufficient for {ticker}, falling back to daily approximation")
    
    # Fallback to daily approximation
    if df_daily is not None and not df_daily.empty:
        result_daily = calculate_daily_value_trx(df_daily)
        if result_daily:
            result_daily['method'] = 'DAILY_APPROXIMATION'
            return result_daily
    
    # Final fallback
    return {
        'total_value': 0,
        'total_value_rp': "Rp 0",
        'total_value_m': 0,
        'total_value_b': 0,
        'vwap': 0,
        'avg_price': 0,
        'volume_total': 0,
        'volume_total_formatted': "0",
        'bars_count': 0,
        'date': date,
        'method': 'FALLBACK',
        'status': 'NO_DATA_AVAILABLE'
    }

# ======================================================
# TECHNICAL INDICATORS
# ======================================================

def add_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Add technical indicators to dataframe
    
    Args:
        df: OHLCV dataframe
    
    Returns:
        DataFrame with indicators added, or None if failed
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
        print(f"  Invalid dataframe for indicators: {error_msg}")
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
        print(f"  Error adding indicators: {e}")
        return None

def ema_slope(series: pd.Series, window: int = SLOPE_WINDOW) -> float:
    """
    Calculate EMA slope (rate of change)
    
    Args:
        series: EMA series
        window: Lookback window
    
    Returns:
        Slope value (positive = uptrend, negative = downtrend)
    """
    if series is None or len(series) < window + 1:
        return 0.0
    
    try:
        return series.iloc[-1] - series.iloc[-window]
    except Exception:
        return 0.0

# ======================================================
# TREND ANALYSIS
# ======================================================

def major_trend_daily(df: pd.DataFrame) -> str:
    """
    Determine major trend based on daily timeframe
    
    Args:
        df: Dataframe with indicators
    
    Returns:
        "STRONG", "WEAK", or "INVALID"
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
        print(f"  Error in major_trend_daily: {e}")
        return "INVALID"

# ======================================================
# MINOR PHASE ANALYSIS
# ======================================================

def minor_phase_4h(df: pd.DataFrame) -> Tuple[str, List[str], int, int]:
    """
    Determine minor phase based on 4H timeframe (MBMA methodology)
    
    Args:
        df: Dataframe with indicators
    
    Returns:
        Tuple of (phase, reasons, confidence, confidence_pct)
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
        
        # 1. EMA COMPRESS PULLBACK
        if ema_compress:
            reasons.append("EMA13 dan EMA21 dalam fase kompresi (<0.3% distance)")
            return "EMA_COMPRESS_PULLBACK", reasons, 1, 14
        
        # 2. PULLBACK RECOVERED
        if pullback_recovered:
            reasons.append("Harga kembali menutup di atas EMA21 setelah pullback")
            return "PULLBACK_RECOVERED", reasons, 1, 14
        
        # 3. TREND CONTINUE
        if is_bullish and price_above_ema13:
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
        
        # 4. TREND OVEREXTEND
        dist_ema13 = (last["Close"] - last["EMA13"]) / last["EMA13"]
        dist_ema21 = (last["Close"] - last["EMA21"]) / last["EMA21"]
        dist_ema50 = (last["Close"] - last["EMA50"]) / last["EMA50"]
        
        is_overextended = (
            dist_ema13 > 0.05 and
            dist_ema21 > 0.10 and
            dist_ema50 > 0.20 and
            (last["RSI"] >= 70 or last["STOCH"] >= 85)
        )
        
        if is_overextended:
            reasons.append("Harga terlalu jauh dari EMA support (overextended)")
            reasons.append(f"RSI: {last['RSI']:.1f}, Stoch: {last['STOCH']:.1f}")
            return "TREND_OVEREXTEND", reasons, 2, 29
        
        # 5. NEUTRAL / NO CLEAR SIGNAL
        reasons.append("Tidak ada sinyal minor phase yang jelas")
        return "NEUTRAL", reasons, 0, 0
        
    except Exception as e:
        print(f"  Error in minor_phase_4h: {e}")
        return "NEUTRAL", [f"Error: {str(e)}"], 0, 0

def setup_state(minor_phase: str) -> str:
    """
    Determine setup state based on minor phase
    
    Args:
        minor_phase: Minor phase string
    
    Returns:
        Setup state string
    """
    if minor_phase == "EMA_COMPRESS_PULLBACK":
        return "SETUP_PENDING"
    
    if minor_phase in ["PULLBACK_RECOVERED", "TREND_CONTINUE"]:
        return "STAGE2_READY"
    
    return "WAIT"

def stage2_trigger(df: pd.DataFrame, setup: str) -> bool:
    """
    Check if stage 2 trigger conditions are met
    
    Args:
        df: Dataframe with indicators
        setup: Setup state
    
    Returns:
        True if trigger conditions met
    """
    if setup != "STAGE2_READY" or df is None or len(df) < 2:
        return False
    
    try:
        last = df.iloc[-1]
        return last["Close"] > last["EMA13"] and ema_slope(df["EMA13"]) > 0
    except Exception:
        return False

# ======================================================
# VOLUME ANALYSIS
# ======================================================

def volume_behavior(df: pd.DataFrame) -> Tuple[str, float, float, float]:
    """
    Analyze volume behavior (VSA-inspired)
    
    Args:
        df: Dataframe with indicators
    
    Returns:
        Tuple of (behavior, vol_ratio, volume, vol_ma20)
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
        
        # 1. ABSORPTION (Bullish)
        absorption = (
            vol_ratio >= 1.3 and
            body_ratio <= 0.35 and
            lower_wick >= body_size * 1.2 and
            close >= ema21
        )
        
        if absorption:
            return "VOL_ABSORPTION", round(vol_ratio, 2), volume, vol_ma20
        
        # 2. DISTRIBUTION (Bearish)
        distribution = (
            vol_ratio >= 1.3 and
            body_ratio <= 0.35 and
            upper_wick >= body_size * 1.2 and
            close < ema21
        )
        
        if distribution:
            return "VOL_DISTRIBUTION", round(vol_ratio, 2), volume, vol_ma20
        
        # 3. EXPANSION (Strong Move)
        expansion = (
            vol_ratio >= 1.5 and
            body_ratio >= 0.55
        )
        
        if expansion:
            return "VOL_EXPANSION", round(vol_ratio, 2), volume, vol_ma20
        
        # 4. NEUTRAL (Default)
        return "VOL_NEUTRAL", round(vol_ratio, 2), volume, vol_ma20
        
    except Exception as e:
        print(f"  Error in volume_behavior: {e}")
        return "VOL_NEUTRAL", 0.0, 0, 0

# ======================================================
# CANDLE ANALYSIS
# ======================================================

def latest_candle_info(df: pd.DataFrame) -> Tuple[str, int, int]:
    """
    Analyze the latest candle
    
    Args:
        df: Dataframe with indicators
    
    Returns:
        Tuple of (label, is_red, is_green)
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
        print(f"  Error in latest_candle_info: {e}")
        return "Doji/Netral", 0, 0

def compute_dist_sma50(df: pd.DataFrame) -> float:
    """
    Calculate distance to SMA50 as percentage
    
    Args:
        df: Dataframe with indicators
    
    Returns:
        Percentage distance (positive = above, negative = below)
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
        
    except Exception:
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
    
    Args:
        major_trend: Major trend from daily
        minor_phase: Minor phase from 4H
        setup_state: Setup state
        stage2_trigger: Stage 2 trigger status
        volume_behavior: Volume behavior
    
    Returns:
        Decision string
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
# WYCKOFF PHASE DETECTION
# ======================================================

def add_wyckoff_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Wyckoff-specific indicators to dataframe"""
    df = df.copy()
    
    # Volume indicators
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    
    # Price position relative to EMAs
    df['Dist_to_EMA13'] = (df['Close'] - df['EMA13']) / df['EMA13'] * 100
    df['Dist_to_EMA21'] = (df['Close'] - df['EMA21']) / df['EMA21'] * 100
    df['Dist_to_EMA50'] = (df['Close'] - df['EMA50']) / df['EMA50'] * 100
    
    # EMA alignment
    df['EMA_Alignment'] = 0
    df.loc[df['EMA13'] > df['EMA21'], 'EMA_Alignment'] += 1
    df.loc[df['EMA21'] > df['EMA50'], 'EMA_Alignment'] += 1
    
    # Volume trend (up/down volume)
    df['Volume_Up'] = df['Volume'].where(df['Close'] > df['Open'], 0)
    df['Volume_Down'] = df['Volume'].where(df['Close'] < df['Open'], 0)
    
    # Accumulation/Distribution indicators
    df['ADI'] = (2*df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    df['ADI_Cum'] = df['ADI'].cumsum()
    
    # Volume price trend
    df['VPT'] = df['Volume'] * ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1))
    df['VPT_Cum'] = df['VPT'].cumsum()
    
    # Money Flow Index
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_sum = positive_flow.rolling(14).sum()
    negative_sum = negative_flow.rolling(14).sum()
    
    money_ratio = positive_sum / (negative_sum + 1e-9)
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    # Detect swing points
    df['Swing_High'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['Swing_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    
    # Volume trend
    df['Volume_Trend'] = 'NEUTRAL'
    df.loc[df['Volume_Ratio'] > 1.3, 'Volume_Trend'] = 'HIGH'
    df.loc[df['Volume_Ratio'] < 0.7, 'Volume_Trend'] = 'LOW'
    
    return df

def detect_springs(df: pd.DataFrame, lookback: int = 30) -> Dict[str, Any]:
    """
    Detect Spring (shakeout) patterns
    
    Spring characteristics:
    1. Break below support
    2. Quick reversal back above
    3. Volume spike on breakdown
    4. Lower wick on reversal
    """
    if len(df) < 20:
        return {'detected': False, 'confidence': 0, 'description': ''}
    
    # Find potential support level (previous lows)
    support_level = df['Low'].iloc[-20:-5].min()
    
    # Look for springs in last 10 bars
    for i in range(1, 11):
        if i + 2 >= len(df):
            continue
        
        # Check if price broke below support
        if df['Low'].iloc[-i] < support_level * SPRING_THRESHOLD:
            
            # Check if it reversed back above support within 2-3 days
            if i > 1:
                days_after = min(3, i - 1)
                if df['Close'].iloc[-i + days_after] > support_level * 1.01:
                    
                    # Check volume confirmation
                    if df['Volume'].iloc[-i] > df['Volume_MA20'].iloc[-i] * VOLUME_SPIKE_THRESHOLD:
                        confidence = 8
                        desc = f"Spring at {df.index[-i].strftime('%Y-%m-%d')}: broke {support_level:,.0f} on high vol, reversed quickly"
                        return {
                            'detected': True,
                            'confidence': confidence,
                            'description': desc,
                            'date': df.index[-i],
                            'support': support_level
                        }
    
    return {'detected': False, 'confidence': 0, 'description': ''}

def detect_upthrusts(df: pd.DataFrame, lookback: int = 30) -> Dict[str, Any]:
    """
    Detect Upthrust patterns
    
    Upthrust characteristics:
    1. Break above resistance
    2. Quick reversal back below
    3. Volume spike on breakout
    4. Upper wick on reversal
    """
    if len(df) < 20:
        return {'detected': False, 'confidence': 0, 'description': ''}
    
    # Find potential resistance level (previous highs)
    resistance_level = df['High'].iloc[-20:-5].max()
    
    # Look for upthrusts in last 10 bars
    for i in range(1, 11):
        if i + 2 >= len(df):
            continue
        
        # Check if price broke above resistance
        if df['High'].iloc[-i] > resistance_level * UPTHRUST_THRESHOLD:
            
            # Check if it reversed back below resistance within 2-3 days
            if i > 1:
                days_after = min(3, i - 1)
                if df['Close'].iloc[-i + days_after] < resistance_level * 0.99:
                    
                    # Check volume confirmation
                    if df['Volume'].iloc[-i] > df['Volume_MA20'].iloc[-i] * VOLUME_SPIKE_THRESHOLD:
                        confidence = 8
                        desc = f"Upthrust at {df.index[-i].strftime('%Y-%m-%d')}: broke {resistance_level:,.0f} on high vol, failed"
                        return {
                            'detected': True,
                            'confidence': confidence,
                            'description': desc,
                            'date': df.index[-i],
                            'resistance': resistance_level
                        }
    
    return {'detected': False, 'confidence': 0, 'description': ''}

def detect_accumulation_phase(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Phase A / Accumulation: Smart money buying quietly, price sideways
    """
    reasons = []
    confidence = 0
    max_confidence = 10
    
    # 1. Price action: Sideways/trading range
    price_range_pct = (df['High'].max() - df['Low'].min()) / df['Low'].min() * 100
    if 10 < price_range_pct < 30:
        confidence += 2
        reasons.append(f"Trading range: {price_range_pct:.1f}% (ideal for accumulation)")
    
    # 2. Volume: Increasing on dips, decreasing on rallies
    vol_correlation = 0
    for i in range(1, 21):
        if i < len(df):
            if df['Close'].iloc[-i] < df['Close'].iloc[-i-1]:  # Down day
                if df['Volume'].iloc[-i] > df['Volume_MA20'].iloc[-i]:
                    vol_correlation += 1
    
    if vol_correlation > 12:
        confidence += 3
        reasons.append(f"Volume increases on dips ({vol_correlation}/20 days)")
    
    # 3. Support/resistance: Price testing lows with reduced selling
    recent_lows = df.nsmallest(5, 'Low')
    avg_vol_at_lows = recent_lows['Volume'].mean()
    avg_vol_overall = df['Volume'].mean()
    
    if avg_vol_at_lows < avg_vol_overall * 0.8:
        confidence += 2
        reasons.append("Lows formed on below-average volume (selling drying up)")
    
    # 4. Spring detection
    springs = detect_springs(df)
    if springs['detected']:
        confidence += springs['confidence']
        reasons.append(f"Spring detected: {springs['description']}")
    
    # 5. EMA alignment: EMAs flattening/compressing
    ema13_slope_val = df['EMA13'].diff().iloc[-5:].mean()
    ema21_slope_val = df['EMA21'].diff().iloc[-5:].mean()
    
    if abs(ema13_slope_val) < 0.01 and abs(ema21_slope_val) < 0.01:
        confidence += 2
        reasons.append("EMAs flattening (loss of trend momentum)")
    
    # 6. MFI/RSI: Often oversold then recovering
    mfi_avg = df['MFI'].iloc[-10:].mean()
    if 30 < mfi_avg < 50:
        confidence += 1
        reasons.append(f"MFI recovering from oversold: {mfi_avg:.1f}")
    
    # Calculate confidence percentage
    confidence_pct = min(100, int((confidence / max_confidence) * 100))
    
    return {
        'phase': 'ACC',
        'confidence': confidence_pct,
        'reasons': reasons
    }

def detect_markup_phase(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Phase B / Markup: Price trending up, volume confirming
    """
    last = df.iloc[-1]
    
    reasons = []
    confidence = 0
    max_confidence = 10
    
    # 1. Price action: Higher highs, higher lows
    hh_count = 0
    hl_count = 0
    
    for i in range(1, 11):
        if i + 5 < len(df):
            if df['High'].iloc[-i] > df['High'].iloc[-i-5:].max():
                hh_count += 1
            if df['Low'].iloc[-i] > df['Low'].iloc[-i-5:].min():
                hl_count += 1
    
    if hh_count >= 5 and hl_count >= 5:
        confidence += 3
        reasons.append(f"Higher highs ({hh_count}/10) and higher lows ({hl_count}/10)")
    
    # 2. Volume: Expanding on up days
    up_volume_ratio = 0
    for i in range(1, 11):
        if i < len(df):
            if df['Close'].iloc[-i] > df['Close'].iloc[-i-1]:  # Up day
                if df['Volume'].iloc[-i] > df['Volume_MA20'].iloc[-i]:
                    up_volume_ratio += 1
    
    if up_volume_ratio >= 6:
        confidence += 2
        reasons.append(f"Volume expands on up days ({up_volume_ratio}/10)")
    
    # 3. EMA alignment: Bullish
    if last['EMA13'] > last['EMA21'] > last['EMA50']:
        confidence += 2
        reasons.append("Bullish EMA alignment (13>21>50)")
    
    # 4. Price above key EMAs
    if last['Close'] > last['EMA21']:
        confidence += 1
        reasons.append(f"Price above EMA21 ({last['Close']/last['EMA21']-1:.2%})")
    
    # 5. Volume expanding overall
    recent_vol = df['Volume'].iloc[-10:].mean()
    prior_vol = df['Volume'].iloc[-30:-10].mean() if len(df) > 30 else recent_vol
    
    if recent_vol > prior_vol * 1.2:
        confidence += 1
        reasons.append("Volume expanding compared to prior period")
    
    # 6. MFI confirming
    if 50 < last.get('MFI', 50) < 80:
        confidence += 1
        reasons.append("MFI in bullish zone (50-80)")
    
    confidence_pct = min(100, int((confidence / max_confidence) * 100))
    
    return {
        'phase': 'MU',
        'confidence': confidence_pct,
        'reasons': reasons
    }

def detect_distribution_phase(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Phase C / Distribution: Smart money selling, price still looks healthy
    """
    reasons = []
    confidence = 0
    max_confidence = 10
    
    # 1. Price action: Long upper wicks
    wick_count = 0
    for i in range(1, 11):
        if i < len(df):
            candle = df.iloc[-i]
            body = abs(candle['Close'] - candle['Open'])
            upper_wick = candle['High'] - max(candle['Close'], candle['Open'])
            
            if upper_wick > body * 1.5:
                wick_count += 1
    
    if wick_count >= 5:
        confidence += 3
        reasons.append(f"Long upper wicks indicating selling pressure ({wick_count}/10)")
    
    # 2. Volume: Increasing on down days
    down_volume_ratio = 0
    for i in range(1, 11):
        if i < len(df):
            if df['Close'].iloc[-i] < df['Close'].iloc[-i-1]:  # Down day
                if df['Volume'].iloc[-i] > df['Volume_MA20'].iloc[-i]:
                    down_volume_ratio += 1
    
    if down_volume_ratio >= 6:
        confidence += 2
        reasons.append(f"Volume expands on down days ({down_volume_ratio}/10)")
    
    # 3. Upthrust detection
    upthrusts = detect_upthrusts(df)
    if upthrusts['detected']:
        confidence += upthrusts['confidence']
        reasons.append(f"Upthrust detected: {upthrusts['description']}")
    
    # 4. Divergence: Price higher but indicators lower
    if len(df) > 20:
        price_high_20 = df['High'].iloc[-20:].max()
        price_high_10 = df['High'].iloc[-10:].max()
        mfi_20 = df['MFI'].iloc[-20:].max()
        mfi_10 = df['MFI'].iloc[-10:].max()
        
        if price_high_10 >= price_high_20 * 0.98 and mfi_10 < mfi_20 * 0.9:
            confidence += 2
            reasons.append("Bearish divergence: Price high but MFI lower")
    
    # 5. Volume drying up on rallies
    recent_up_vol = 0
    recent_up_days = 0
    for i in range(1, 11):
        if i < len(df) and df['Close'].iloc[-i] > df['Close'].iloc[-i-1]:
            recent_up_vol += df['Volume'].iloc[-i]
            recent_up_days += 1
    
    if recent_up_days > 0:
        avg_up_vol = recent_up_vol / recent_up_days
        if avg_up_vol < df['Volume_MA20'].iloc[-1] * 0.8:
            confidence += 1
            reasons.append("Rallies on below-average volume")
    
    confidence_pct = min(100, int((confidence / max_confidence) * 100))
    
    return {
        'phase': 'DIS',
        'confidence': confidence_pct,
        'reasons': reasons
    }

def detect_markdown_phase(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Phase D / Markdown: Price trending down, selling pressure
    """
    reasons = []
    confidence = 0
    max_confidence = 10
    
    # 1. Price action: Lower highs, lower lows
    lh_count = 0
    ll_count = 0
    
    for i in range(1, 11):
        if i + 5 < len(df):
            if df['High'].iloc[-i] < df['High'].iloc[-i-5:].max():
                lh_count += 1
            if df['Low'].iloc[-i] < df['Low'].iloc[-i-5:].min():
                ll_count += 1
    
    if lh_count >= 5 and ll_count >= 5:
        confidence += 3
        reasons.append(f"Lower highs ({lh_count}/10) and lower lows ({ll_count}/10)")
    
    # 2. Volume: High on down days
    down_volume_ratio = 0
    for i in range(1, 11):
        if i < len(df) and df['Close'].iloc[-i] < df['Close'].iloc[-i-1]:
            if df['Volume'].iloc[-i] > df['Volume_MA20'].iloc[-i] * 1.2:
                down_volume_ratio += 1
    
    if down_volume_ratio >= 5:
        confidence += 2
        reasons.append(f"High volume on down days ({down_volume_ratio}/10)")
    
    # 3. EMA alignment: Bearish
    if df['EMA13'].iloc[-1] < df['EMA21'].iloc[-1] < df['EMA50'].iloc[-1]:
        confidence += 2
        reasons.append("Bearish EMA alignment (13<21<50)")
    
    # 4. Price below key EMAs
    if df['Close'].iloc[-1] < df['EMA21'].iloc[-1]:
        confidence += 1
        reasons.append("Price below EMA21")
    
    # 5. No buying absorption
    absorption_score = 0
    for i in range(1, 6):
        if i < len(df):
            body = abs(df['Close'].iloc[-i] - df['Open'].iloc[-i])
            lower_wick = min(df['Open'].iloc[-i], df['Close'].iloc[-i]) - df['Low'].iloc[-i]
            
            if lower_wick > body * 2:
                absorption_score += 1
    
    if absorption_score <= 2:
        confidence += 1
        reasons.append("No significant buying absorption detected")
    
    # 6. MFI in oversold but still falling
    if df['MFI'].iloc[-1] < 30 and df['MFI'].iloc[-1] < df['MFI'].iloc[-5]:
        confidence += 1
        reasons.append("MFI in oversold and still declining")
    
    confidence_pct = min(100, int((confidence / max_confidence) * 100))
    
    return {
        'phase': 'MD',
        'confidence': confidence_pct,
        'reasons': reasons
    }

def get_historical_phases(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate historical phase labels for chart overlay
    Returns list of dict with date ranges and phases
    """
    phases_history = []
    
    if len(df) < 30:
        return phases_history
    
    # Gunakan window yang lebih kecil untuk deteksi yang lebih sensitif
    min_phase_days = 5  # Minimal 5 hari untuk sebuah fase (sebelumnya lebih panjang)
    
    current_phase = 'UNKNOWN'
    phase_start = df.index[0]
    
    for i in range(len(df)):
        date = df.index[i]
        
        if i > 5:  # Mulai deteksi setelah 5 hari (sebelumnya 10)
            # Ambil data untuk analisis
            lookback = min(10, i)
            recent_df = df.iloc[i-lookback:i+1]
            
            # Deteksi fase berdasarkan multiple faktor
            phase = detect_phase_at_point(recent_df)
            
            if phase != 'UNKNOWN':
                if phase != current_phase:
                    # Simpan fase sebelumnya
                    if current_phase != 'UNKNOWN' and current_phase != phase:
                        days_in_phase = (df.index[i-1] - phase_start).days
                        if days_in_phase >= min_phase_days:
                            phases_history.append({
                                'phase': current_phase,
                                'start': phase_start,
                                'end': df.index[i-1],
                                'days': days_in_phase
                            })
                    # Mulai fase baru
                    current_phase = phase
                    phase_start = date
    
    # Tambahkan fase terakhir
    if current_phase != 'UNKNOWN':
        days_in_phase = (df.index[-1] - phase_start).days
        if days_in_phase >= min_phase_days:
            phases_history.append({
                'phase': current_phase,
                'start': phase_start,
                'end': df.index[-1],
                'days': days_in_phase
            })
    
    return phases_history


def detect_phase_at_point(df_slice: pd.DataFrame) -> str:
    """
    Detect Wyckoff phase at a specific point in time
    """
    if len(df_slice) < 5:
        return 'UNKNOWN'
    
    last = df_slice.iloc[-1]
    
    # 1. Cek Accumulation
    vol_decreasing = df_slice['Volume'].iloc[-3:].mean() < df_slice['Volume'].iloc[:3].mean() * 0.8
    price_range_small = (df_slice['High'].max() - df_slice['Low'].min()) / df_slice['Low'].min() < 0.05
    ema_flat = abs(df_slice['EMA13'].diff().mean()) < 0.005
    
    if price_range_small and vol_decreasing and ema_flat:
        return 'ACC'
    
    # 2. Cek Markup
    price_up = df_slice['Close'].pct_change().mean() > 0.005
    vol_increasing = df_slice['Volume'].iloc[-3:].mean() > df_slice['Volume'].iloc[:3].mean() * 1.2
    ema_bullish = last['EMA13'] > last['EMA21'] > last['EMA50']
    
    if price_up and vol_increasing and ema_bullish:
        return 'MU'
    
    # 3. Cek Distribution
    long_upper_wick = (last['High'] - max(last['Close'], last['Open'])) > (abs(last['Close'] - last['Open']) * 1.5)
    vol_high = last['Volume'] > df_slice['Volume_MA20'].iloc[-1] * 1.3
    
    if long_upper_wick and vol_high:
        return 'DIS'
    
    # 4. Cek Markdown
    price_down = df_slice['Close'].pct_change().mean() < -0.005
    vol_high_down = last['Volume'] > df_slice['Volume_MA20'].iloc[-1] * 1.2
    ema_bearish = last['EMA13'] < last['EMA21'] < last['EMA50']
    
    if price_down and (vol_high_down or ema_bearish):
        return 'MD'
    
    return 'UNKNOWN'

def detect_wyckoff_phase(df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
    """
    Detect Wyckoff phases (ACC, MU, DIS, MD) based on price structure and volume
    """
    if df is None or len(df) < lookback:
        return {
            'phase': 'UNKNOWN',
            'confidence': 0,
            'reasons': ['Insufficient data'],
            'phases_history': [],
            'price_range': {'low': 0, 'high': 0},  # FIXED: Always include both keys
            'duration': 0
        }
    
    # Use last 'lookback' bars
    df_analysis = df.iloc[-lookback:].copy()
    
    # Calculate additional Wyckoff-specific indicators
    df_analysis = add_wyckoff_indicators(df_analysis)
    
    # Detect phases
    phase_results = [
        detect_accumulation_phase(df_analysis),
        detect_markup_phase(df_analysis),
        detect_distribution_phase(df_analysis),
        detect_markdown_phase(df_analysis)
    ]
    
    # Find the phase with highest confidence
    best_phase = max(phase_results, key=lambda x: x['confidence'])
    
    # Get historical phases for chart overlay
    phases_history = get_historical_phases(df_analysis)
    
    # Safe price range calculation - FIXED: Always provide both keys
    try:
        price_low = float(df_analysis['Low'].min()) if not df_analysis.empty and 'Low' in df_analysis.columns else 0
        price_high = float(df_analysis['High'].max()) if not df_analysis.empty and 'High' in df_analysis.columns else 0
    except (ValueError, TypeError):
        price_low = 0
        price_high = 0
    
    return {
        'phase': best_phase['phase'],
        'confidence': best_phase['confidence'],
        'reasons': best_phase['reasons'],
        'phases_history': phases_history,
        'price_range': {  # FIXED: Always include both keys
            'low': price_low,
            'high': price_high
        },
        'duration': len(df_analysis),
        'volume_trend': df_analysis['Volume_Trend'].iloc[-1] if 'Volume_Trend' in df_analysis.columns and not df_analysis.empty else 'NEUTRAL',
        'supply_demand': 0
    }

# ======================================================
# MAIN STOCK PROCESSING FUNCTION
# ======================================================

def process_stock(
    kode: str, 
    use_cache: bool = True, 
    include_value_trx: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Process a single stock for screening
    
    Args:
        kode: Stock code (e.g., "BBCA")
        use_cache: Use cache for data
        include_value_trx: Include value transaction calculation
    
    Returns:
        Dictionary with stock analysis results or None if failed
    """
    ticker = f"{kode}.JK"
    
    try:
        print(f"Processing {kode}...")
        
        # =========================
        # 1. FETCH DAILY DATA
        # =========================
        sector_info = get_sector_info(ticker, use_cache=use_cache)
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
        # 3. VALUE TRX CALCULATION
        # =========================
        value_trx_metrics = {}
        if include_value_trx:
            print(f"  Calculating value trx for {kode}...")
            
            if not d1.empty:
                last_trading_date = d1.index[-1].strftime("%Y-%m-%d")
                value_trx_metrics = get_value_trx_metrics(
                    ticker=ticker,
                    df_daily=d1,
                    use_1m_preferred=True,
                    date=last_trading_date
                )
            else:
                value_trx_metrics = get_value_trx_metrics(
                    ticker=ticker,
                    df_daily=d1,
                    use_1m_preferred=True,
                    date=None
                )
        
        # =========================
        # 4. WYCKOFF DETECTION
        # =========================
        try:
            wyckoff_result = detect_wyckoff_phase(d1, lookback=WYCKOFF_LOOKBACK)
        except Exception as e:
            print(f"  Wyckoff detection failed: {e}")
            wyckoff_result = {
                'phase': 'UNKNOWN',
                'confidence': 0,
                'reasons': [f'Detection error: {str(e)}'],
                'phases_history': []
            }
        
        # =========================
        # 5. MAJOR TREND
        # =========================
        major = major_trend_daily(d1)
        
        # =========================
        # 6. MINOR PHASE (try intraday first)
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
        # 7. VOLUME ANALYSIS
        # =========================
        vol_behavior, vol_ratio, volume, vol_ma20 = volume_behavior(d1)
        
        # =========================
        # 8. CANDLE ANALYSIS
        # =========================
        candle_label, candle_red, candle_green = latest_candle_info(d1)
        candle_effect = 1 if candle_green else -1 if candle_red else 0
        
        # =========================
        # 9. PRICE METRICS
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
        # 10. CONFIDENCE ADJUSTMENTS
        # =========================
        if minor == "TREND_CONTINUE":
            if vol_behavior in ["VOL_ABSORPTION", "VOL_EXPANSION"]:
                confidence += 1
                why.append("Volume mendukung kelanjutan trend")
            
            if candle_label == "Hijau Kuat (Impulse)":
                confidence += 1
                why.append("Impulse candle terdeteksi")
        
        # Tambah confidence jika value trx tinggi
        if include_value_trx and value_trx_metrics:
            value_trx_b = value_trx_metrics.get('total_value_b', 0)
            if value_trx_b > 5:
                confidence += 1
                why.append(f"Likuiditas tinggi ({value_trx_b:.1f}B)")
        
        # Recalculate confidence percentage
        confidence_pct = round((confidence / 7) * 100) if 7 > 0 else 0
        
        # =========================
        # 11. FINAL DECISION
        # =========================
        final_dec = final_decision(major, minor, setup, stage2, vol_behavior)
        
        # =========================
        # 12. VALIDATION GUARDS
        # =========================
        if not validation_utils.validate_price_data(price_today):
            print(f"  Invalid price for {kode}: {price_today}")
            return None
        
        if price_today > 1_000_000:
            print(f"  Suspicious price for {kode}: {price_today}")
            return None
        
        # =========================
        # 13. BUILD RESULT
        # =========================
        result = {
            # Basic Info
            "Kode": kode,
            "Sector": sector_info.get('sector', 'Unknown'),
            "Industry": sector_info.get('industry', 'Unknown'),
            
            # Price Metrics
            "Price": price_today,
            "PriceChange%": price_change,
            "Gap_EMA13%": gap_ema13,
            "Gap_EMA21%": gap_ema21,
            "Gap_EMA50%": gap_ema50,
            
            # Trend Analysis
            "MajorTrend": major,
            "MinorPhase": minor,
            "WHY_MINOR": why,
            "MinorConfidence": confidence,
            "MinorConfidence%": confidence_pct,
            "SetupState": setup,
            "Stage2Valid": stage2,
            
            # Volume Analysis
            "Volume": volume,
            "Vol_20MA": vol_ma20,
            "VOL_BEHAVIOR": vol_behavior,
            "VOL_RATIO": vol_ratio,
            
            # Technical Indicators
            "RSI": round(rsi, 2),
            "SMA50": round(sma50, 2),
            "Dist_to_SMA50": round(dist_to_sma50, 2) if not np.isnan(dist_to_sma50) else np.nan,
            "Stoch_K": round(stoch, 2),
            
            # Candle Analysis
            "Latest_Candle": candle_label,
            "Candle_Effect": candle_effect,
            
            # Final Decision
            "FinalDecision": final_dec,
            
            # Wyckoff Analysis
            "Wyckoff_Phase": wyckoff_result.get('phase', 'UNKNOWN'),
            "Wyckoff_Confidence": wyckoff_result.get('confidence', 0),
            "Wyckoff_Reasons": wyckoff_result.get('reasons', []),
            "Wyckoff_History": wyckoff_result.get('phases_history', []),
            "Wyckoff_Price_Range": wyckoff_result.get('price_range', {'low': 0, 'high': 0}),
            "Wyckoff_Duration": wyckoff_result.get('duration', 0),
            "Spring_Detected": any('spring' in r.lower() for r in wyckoff_result.get('reasons', [])),
            "Upthrust_Detected": any('upthrust' in r.lower() for r in wyckoff_result.get('reasons', [])),
            
            # Metadata
            "ProcessTimestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add value transaction metrics if requested
        if include_value_trx and value_trx_metrics:
            result.update({
                "ValueTrx": value_trx_metrics.get('total_value', 0),
                "ValueTrx_Rp": value_trx_metrics.get('total_value_rp', 'Rp 0'),
                "ValueTrx_B": value_trx_metrics.get('total_value_b', 0),
                "VWAP": round(value_trx_metrics.get('vwap', 0), 2),
                "AvgPrice": round(value_trx_metrics.get('avg_price', 0), 2),
                "ValueTrx_Volume": value_trx_metrics.get('volume_total', 0),
                "ValueTrx_Bars": value_trx_metrics.get('bars_count', 0),
                "ValueTrx_Method": value_trx_metrics.get('method', 'N/A'),
                "ValueTrx_Status": value_trx_metrics.get('status', 'UNKNOWN')
            })
        
        print(f"  ✓ Processed {kode}: {major}/{minor}/{final_dec} | Wyckoff: {result['Wyckoff_Phase']}")
        
        if include_value_trx and value_trx_metrics.get('total_value', 0) > 0:
            print(f"    Value Trx: {value_trx_metrics.get('total_value_rp', 'Rp 0')}")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error processing {kode}: {str(e)}")
        return None

# ======================================================
# BATCH PROCESSING FUNCTIONS
# ======================================================

def batch_process_stocks(
    stock_list: List[str], 
    include_value_trx: bool = False
) -> pd.DataFrame:
    """
    Process multiple stocks in batch mode
    
    Args:
        stock_list: List of stock codes
        include_value_trx: Include value transaction calculation
    
    Returns:
        DataFrame with results
    """
    results = []
    
    print(f"\nBatch processing {len(stock_list)} stocks...")
    print(f"Include Value Trx: {include_value_trx}")
    print("=" * 60)
    
    for i, kode in enumerate(stock_list, 1):
        print(f"\n[{i}/{len(stock_list)}] ", end="")
        
        result = process_stock(
            kode=kode,
            use_cache=True,
            include_value_trx=include_value_trx
        )
        
        if result:
            results.append(result)
    
    # Convert to DataFrame
    if results:
        df_results = pd.DataFrame(results)
        
        # Sort by Value Trx if available
        if include_value_trx and "ValueTrx_B" in df_results.columns:
            df_results = df_results.sort_values("ValueTrx_B", ascending=False)
            print(f"\n\nTop 5 by Value Trx:")
            for _, row in df_results.head().iterrows():
                print(f"  {row['Kode']}: {row.get('ValueTrx_Rp', 'Rp 0')}")
        
        return df_results
    
    return pd.DataFrame()

# ======================================================
# MARKET STATE EXTRACTION
# ======================================================

def extract_market_state(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """
    Extract market state at historical index
    
    Args:
        df: DataFrame with indicators
        idx: Historical index
    
    Returns:
        Dictionary with market state parameters
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
        print(f"  Error extracting market state at index {idx}: {e}")
        return {}

# ======================================================
# MAIN EXECUTION GUARD
# ======================================================

if __name__ == "__main__":
    # Test the engine
    test_codes = ["BBCA", "BBRI", "TLKM"]
    
    print("=" * 60)
    print("TESTING ENGINE V2")
    print("=" * 60)
    
    # Test without value trx
    print("\n📊 TEST WITHOUT VALUE TRX")
    print("-" * 40)
    
    for kode in test_codes:
        print(f"\n🔍 Processing {kode}...")
        result = process_stock(kode, use_cache=True, include_value_trx=False)
        
        if result:
            print(f"  ✅ Price: {result['Price']:,.0f}")
            print(f"  ✅ Major Trend: {result['MajorTrend']}")
            print(f"  ✅ Minor Phase: {result['MinorPhase']}")
            print(f"  ✅ Final Decision: {result['FinalDecision']}")
            print(f"  ✅ Wyckoff Phase: {result['Wyckoff_Phase']} ({result['Wyckoff_Confidence']}%)")
        else:
            print(f"  ❌ Failed to process {kode}")
    
    # Test with value trx
    print("\n\n💰 TEST WITH VALUE TRX")
    print("-" * 40)
    
    for kode in test_codes:
        print(f"\n🔍 Processing {kode} with value trx...")
        result = process_stock(kode, use_cache=True, include_value_trx=True)
        
        if result:
            print(f"  ✅ Price: {result['Price']:,.0f}")
            print(f"  ✅ Major Trend: {result['MajorTrend']}")
            if 'ValueTrx_Rp' in result:
                print(f"  ✅ Value Trx: {result['ValueTrx_Rp']}")
                print(f"  ✅ Method: {result.get('ValueTrx_Method', 'N/A')}")
        else:
            print(f"  ❌ Failed to process {kode}")
    
    print("\n" + "=" * 60)
    print("✅ ENGINE TEST COMPLETED")
    print("=" * 60)