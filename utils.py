"""
Utility functions for Trading Screener
Helper functions, data processing, and common utilities
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import pickle
import json
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataUtils:
    """Utility class for data operations"""
    
    @staticmethod
    def safe_yf_download(ticker: str, interval: str = "1d", period: str = "12mo", 
                         max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Safe Yahoo Finance download with retry logic
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        interval : str
            Time interval
        period : str
            Period length
        max_retries : int
            Maximum number of retries
        
        Returns:
        --------
        pd.DataFrame or None
        """
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    ticker,
                    interval=interval,
                    period=period,
                    progress=False,
                    threads=False,
                    auto_adjust=False
                )
                
                if df is None or df.empty:
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                # Flatten multi-index columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Ensure required columns
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing = [col for col in required if col not in df.columns]
                
                if missing:
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                # Clean data
                df = df[required].copy()
                df.dropna(inplace=True)
                
                # Check for suspended stock
                if df['Volume'].iloc[-10:].sum() == 0:
                    return None
                
                return df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                print(f"Failed to download {ticker} after {max_retries} attempts: {e}")
                return None
        
        return None
    
    @staticmethod
    def normalize_dataframe(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
        """
        Normalize dataframe columns and index
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        ticker : str, optional
            Ticker for multi-index handling
        
        Returns:
        --------
        pd.DataFrame
        """
        df = df.copy()
        
        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            if ticker is not None:
                if ticker in df.columns.get_level_values(1):
                    df = df.xs(ticker, axis=1, level=1)
                else:
                    raise ValueError(f"Ticker {ticker} not found in dataframe")
            else:
                # Use first ticker if available
                available_tickers = df.columns.get_level_values(1).unique()
                if len(available_tickers) > 0:
                    df = df.xs(available_tickers[0], axis=1, level=1)
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Normalize column names
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV dataframe
        period : int
            ATR period
        
        Returns:
        --------
        pd.Series : ATR values
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, 
                                 std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Returns:
        --------
        Tuple of (upper_band, middle_band, lower_band)
        """
        middle = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD
        
        Returns:
        --------
        Tuple of (macd_line, signal_line, histogram)
        """
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window: int = 20, 
                                 threshold: float = 0.02) -> Dict:
        """
        Detect support and resistance levels
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLC dataframe
        window : int
            Lookback window
        threshold : float
            Threshold for level identification
        
        Returns:
        --------
        Dict with support and resistance levels
        """
        high = df['High']
        low = df['Low']
        
        # Find local maxima and minima
        local_max = high[(high.shift(1) < high) & (high.shift(-1) < high)]
        local_min = low[(low.shift(1) > low) & (low.shift(-1) > low)]
        
        # Cluster nearby levels
        def cluster_levels(levels, threshold):
            if len(levels) == 0:
                return []
            
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for price in levels[1:]:
                if price <= current_cluster[-1] * (1 + threshold):
                    current_cluster.append(price)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [price]
            
            clusters.append(current_cluster)
            
            # Return average of each cluster
            return [sum(cluster) / len(cluster) for cluster in clusters]
        
        support_levels = cluster_levels(local_min.tolist(), threshold)
        resistance_levels = cluster_levels(local_max.tolist(), threshold)
        
        # Current price position
        current_price = df['Close'].iloc[-1]
        
        # Find nearest support and resistance
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'distance_to_support': (current_price - nearest_support) / current_price * 100 if nearest_support else None,
            'distance_to_resistance': (nearest_resistance - current_price) / current_price * 100 if nearest_resistance else None
        }


class CacheManager:
    """Manage data caching"""
    
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, key: str, suffix: str = "pkl") -> str:
        """Generate cache file path"""
        safe_key = key.replace(".", "_").replace(":", "_")
        return os.path.join(self.cache_dir, f"{safe_key}.{suffix}")
    
    def save(self, key: str, data, suffix: str = "pkl"):
        """Save data to cache"""
        cache_path = self.get_cache_path(key, suffix)
        
        if suffix == "pkl":
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        elif suffix == "json":
            with open(cache_path, "w") as f:
                json.dump(data, f, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported cache suffix: {suffix}")
        
        return cache_path
    
    def load(self, key: str, suffix: str = "pkl"):
        """Load data from cache"""
        cache_path = self.get_cache_path(key, suffix)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            if suffix == "pkl":
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            elif suffix == "json":
                with open(cache_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}")
            return None
    
    def is_fresh(self, key: str, max_age_hours: int = 24) -> bool:
        """Check if cache is fresh"""
        cache_path = self.get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age_hours = (datetime.now() - file_time).total_seconds() / 3600
        
        return age_hours < max_age_hours
    
    def clear_old_cache(self, max_age_days: int = 7):
        """Clear old cache files"""
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            age_days = (datetime.now() - file_time).days
            
            if age_days > max_age_days:
                os.remove(filepath)
                print(f"Removed old cache: {filename}")


class DateUtils:
    """Date and time utilities"""
    
    @staticmethod
    def get_trading_dates(start_date: str, end_date: str = None) -> List[str]:
        """
        Get list of trading dates (exclude weekends)
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format (default: today)
        
        Returns:
        --------
        List[str] : List of trading dates
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return [date.strftime("%Y-%m-%d") for date in dates]
    
    @staticmethod
    def get_last_trading_day(date: datetime = None) -> datetime:
        """Get last trading day (skip weekends)"""
        if date is None:
            date = datetime.now()
        
        # Move back to last weekday
        while date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            date -= timedelta(days=1)
        
        return date
    
    @staticmethod
    def is_market_hours(dt: datetime = None) -> bool:
        """
        Check if current time is within IDX market hours
        Market hours: 09:00-12:00 and 13:30-15:30
        """
        if dt is None:
            dt = datetime.now()
        
        hour = dt.hour
        minute = dt.minute
        
        # First session: 09:00-12:00
        morning_session = (9 <= hour < 12) or (hour == 12 and minute == 0)
        
        # Second session: 13:30-15:30
        afternoon_session = (13 <= hour < 15) or (hour == 15 and minute <= 30)
        
        return morning_session or afternoon_session


class FormatUtils:
    """Formatting utilities"""
    
    @staticmethod
    def format_currency(value: float, currency: str = "IDR") -> str:
        """Format currency value"""
        if value is None or pd.isna(value):
            return "-"
        
        if currency == "IDR":
            if abs(value) >= 1e9:
                return f"IDR {value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"IDR {value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                return f"IDR {value/1e3:.1f}K"
            else:
                return f"IDR {value:,.0f}"
        else:
            return f"{currency} {value:,.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format percentage value"""
        if value is None or pd.isna(value):
            return "-"
        
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_number(value: float, decimals: int = 2) -> str:
        """Format number with commas"""
        if value is None or pd.isna(value):
            return "-"
        
        return f"{value:,.{decimals}f}"
    
    @staticmethod
    def color_cell(value, threshold_green=0, threshold_red=0):
        """Return CSS color for cell based on value"""
        if value is None or pd.isna(value):
            return ""
        
        if value > threshold_green:
            return "color: green; font-weight: bold;"
        elif value < threshold_red:
            return "color: red; font-weight: bold;"
        else:
            return ""


class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, str]:
        """
        Validate dataframe structure
        
        Returns:
        --------
        Tuple of (is_valid, error_message)
        """
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        # Check for NaN values in required columns
        nan_check = df[required_cols].isna().sum()
        if nan_check.any():
            nan_cols = nan_check[nan_check > 0].index.tolist()
            return False, f"NaN values found in: {nan_cols}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_price_data(price: float) -> bool:
        """Validate price data"""
        if price is None or pd.isna(price):
            return False
        
        if price <= 0:
            return False
        
        if price > 1_000_000:  # Unrealistic price for IDX
            return False
        
        return True
    
    @staticmethod
    def validate_volume(volume: float) -> bool:
        """Validate volume data"""
        if volume is None or pd.isna(volume):
            return False
        
        if volume < 0:
            return False
        
        if volume == 0:  # Suspended stock
            return False
        
        return True


# Singleton instances for easy access
data_utils = DataUtils()
cache_manager = CacheManager()
date_utils = DateUtils()
format_utils = FormatUtils()
validation_utils = ValidationUtils()