"""
Tests for data processing functions.
"""
import pandas as pd
import numpy as np
from src.data import fix_yf_columns

def test_fix_yf_columns_simple():
    """Test fixing columns when no multiindex exists."""
    df = pd.DataFrame({'Close': [1, 2, 3], 'Open': [1, 2, 3]})
    fixed_df = fix_yf_columns(df)
    assert list(fixed_df.columns) == ['Close', 'Open']

def test_fix_yf_columns_multiindex():
    """Test fixing columns with yfinance style MultiIndex."""
    # Create MultiIndex: (Price, Ticker)
    arrays = [
        ['Close', 'Close', 'Open', 'Open'],
        ['SPY', 'NVDA', 'SPY', 'NVDA']
    ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['Price', 'Ticker'])
    df = pd.DataFrame(np.random.randn(3, 4), columns=index)

    # We simulate passing just one ticker's worth of data usually,
    # but the function is designed to handle the droplevel.
    # If we pass a df with multiple tickers mixed, droplevel(1) might create duplicates.
    # The function handles duplicates by keeping the first.

    fixed_df = fix_yf_columns(df)

    # After dropping level 1 (Ticker), we get Close, Close, Open, Open.
    # The function removes duplicates.
    assert list(fixed_df.columns) == ['Close', 'Open']
