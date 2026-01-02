"""
Data processing module for the trading bot.
Contains functions for generating mock data and fixing data format issues.
"""
import numpy as np
import pandas as pd

def generate_mock_data(ticker, start, end):
    """Generates random walk data if yfinance fails."""
    print(f"Generating mock data for {ticker}...")
    date_range = pd.date_range(start=start, end=end, freq='B') # Business days
    n = len(date_range)

    # Random walk
    start_price = 100 if ticker == 'SPY' else 50
    returns = np.random.normal(0.0005, 0.02, n) # Mean drift, vol
    price_path = start_price * (1 + returns).cumprod()

    df = pd.DataFrame(index=date_range)
    df['Open'] = price_path
    df['High'] = price_path * 1.01
    df['Low'] = price_path * 0.99
    df['Close'] = price_path
    df['Volume'] = np.random.randint(1000000, 5000000, n)

    return df

def fix_yf_columns(df):
    """
    Fixes yfinance MultiIndex columns by flattening them.
    Expects columns to be (Price, Ticker) or just Price.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the 'Ticker' level, keeping only 'Price' (e.g. 'Close', 'Open')
        df.columns = df.columns.droplevel(1)
        # Verify if we have unique columns now
        if not df.columns.is_unique:
            # If duplicates exist (unlikely for single ticker), just take the first set
            df = df.loc[:, ~df.columns.duplicated()]
    return df
