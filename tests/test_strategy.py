import unittest
import pandas as pd
import numpy as np
from src.strategy_logic import StrategyLogic
from src.risk_manager import RiskManager

class TestStrategyLogic(unittest.TestCase):
    def setUp(self):
        # Config
        self.config = {
            'risk': {
                'max_position_pct': 0.02,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            },
            'strategy': {
                'rsi_period': 14,
                'sma_fast': 50
            }
        }
        self.rm = RiskManager(self.config)
        self.strategy = StrategyLogic(self.rm, self.config)

        # Create 60 days of data to satisfy SMA(50) requirement
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        
        # Base price uptrend
        prices = np.linspace(100, 150, 60)
        
        # DataFrame
        self.data = pd.DataFrame({
            'close': prices,
            'high': prices + 1,
            'low': prices - 1,
            'volume': np.full(60, 1000)
        }, index=dates)

    def test_generate_buy_signal(self):
        # ... (Same setup logic) ...
        # Creating a more robust dataset that GUARANTEES all conditions.
        # This is tricky with strictly generated data because indicators conflict 
        # (e.g., Deep RSI dip usually kills MACD).
        
        # Instead of fighting the math, let's Mock the indicator functions 
        # to return exactly what we need for the LOGIC test.
        # This verifies the Strategy Logic class, not the Indicator Math (which is tested separately).
        
        from unittest.mock import patch
        
        # Create dummy data just to satisfy type checks and length checks
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'close': np.full(60, 100.0),
            'high': np.full(60, 105.0),
            'low': np.full(60, 95.0),
            'volume': np.full(60, 1000.0)
        }, index=dates)
        
        # Set latest values to be plausible
        df.iloc[-1, df.columns.get_loc('close')] = 100.0
        df.iloc[-1, df.columns.get_loc('volume')] = 2000.0 # High volume
        
        # Patch the indicators imported in src.strategy_logic
        with patch('src.strategy_logic.calculate_rsi') as mock_rsi, \
             patch('src.strategy_logic.calculate_sma') as mock_sma, \
             patch('src.strategy_logic.calculate_macd') as mock_macd:
            
            # Setup returns to satisfy BUY conditions
            # 1. RSI < 45
            mock_rsi.return_value = pd.Series(np.full(60, 30.0), index=dates) 
            
            # 2. Price (100) > SMA (90)
            mock_sma.return_value = pd.Series(np.full(60, 90.0), index=dates)
            
            # 3. MACD > Signal
            mock_macd.return_value = (
                pd.Series(np.full(60, 5.0), index=dates), # MACD Line
                pd.Series(np.full(60, 2.0), index=dates)  # Signal Line
            )
            
            sentiment_score = 0.8 # > 0.5
            account_value = 100000
            
            signal = self.strategy.generate_signal("TEST", df, sentiment_score, account_value)
            
            self.assertEqual(signal['action'], 'BUY', f"Signal failed: {signal.get('reason')}")
            self.assertTrue('stop_loss' in signal)
            self.assertTrue('take_profit' in signal)

    def test_generate_sell_signal(self):
        # Test Sentiment Sell Trigger
        df = self.data.copy()
        sentiment_score = -0.5 # < -0.2
        
        signal = self.strategy.generate_signal("TEST", df, sentiment_score, 100000)
        
        self.assertEqual(signal['action'], 'SELL')
        self.assertIn("Negative Sentiment", signal['reason'])

if __name__ == '__main__':
    unittest.main()
