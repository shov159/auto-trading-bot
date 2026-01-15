import unittest
import pandas as pd
import numpy as np
from src.indicators import (
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr
)

class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Create 50 days of dummy data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        self.prices = pd.Series(np.linspace(100, 150, 50) + np.random.normal(0, 2, 50), index=dates)
        
        # For ATR
        self.high = self.prices + 2
        self.low = self.prices - 2
        self.close = self.prices

    def test_calculate_sma(self):
        sma = calculate_sma(self.prices, window=10)
        # First 9 should be NaN
        self.assertTrue(pd.isna(sma.iloc[8]))
        self.assertFalse(pd.isna(sma.iloc[9]))
        # Size should match
        self.assertEqual(len(sma), 50)

    def test_calculate_rsi(self):
        rsi = calculate_rsi(self.prices, period=14)
        # Should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())
        # Last value should not be NaN (given enough data)
        self.assertFalse(pd.isna(rsi.iloc[-1]))

    def test_calculate_macd(self):
        macd, signal = calculate_macd(self.prices)
        self.assertEqual(len(macd), 50)
        self.assertEqual(len(signal), 50)
        self.assertFalse(pd.isna(macd.iloc[-1]))
        self.assertFalse(pd.isna(signal.iloc[-1]))

    def test_calculate_bollinger_bands(self):
        upper, mid, lower = calculate_bollinger_bands(self.prices, window=20)
        
        # Check integrity
        valid_indices = slice(19, None) # After warmup
        # Upper should be >= Middle
        self.assertTrue((upper[valid_indices] >= mid[valid_indices]).all())
        # Lower should be <= Middle
        self.assertTrue((lower[valid_indices] <= mid[valid_indices]).all())

    def test_calculate_atr(self):
        atr = calculate_atr(self.high, self.low, self.close, window=14)
        self.assertEqual(len(atr), 50)
        # ATR should be positive
        self.assertTrue((atr.dropna() > 0).all())

if __name__ == '__main__':
    unittest.main()
