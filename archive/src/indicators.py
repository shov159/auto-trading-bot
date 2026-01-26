import numpy as np
import pandas as pd
from typing import Tuple, Union

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: Series of prices
        window: Rolling window size
        
    Returns:
        Series containing SMA values
    """
    return prices.rolling(window=window).mean()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of prices
        period: RSI period (default 14)
        
    Returns:
        Series containing RSI values (0-100)
    """
    delta = prices.diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Calculate initial average gain/loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Handle the first 'period' values which are NaN/inaccurate
    # Standard smoothing for Wilders RSI (optional, but using Simple MA for simplicity/robustness here)
    # For closer adherence to standard trading platforms, we could use EMA for subsequent steps,
    # but rolling mean is acceptable for this MVP.
    
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices: Series of prices
        fast: Fast period EMA
        slow: Slow period EMA
        signal: Signal line period EMA
        
    Returns:
        Tuple of (macd_line, signal_line)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line, signal_line

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Series of prices
        window: Rolling window for SMA
        num_std: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = prices.rolling(window=window).mean()
    std_dev = prices.rolling(window=window).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        window: Smoothing period (default 14)
        
    Returns:
        Series containing ATR values
    """
    # Calculate True Range (TR)
    # TR is max of:
    # 1. High - Low
    # 2. Abs(High - Previous Close)
    # 3. Abs(Low - Previous Close)
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR (typically using Wilder's Smoothing, effectively an EMA)
    atr = tr.ewm(span=window, adjust=False).mean()
    
    return atr
