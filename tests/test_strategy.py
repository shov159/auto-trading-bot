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

        # Create 250 days of data to satisfy SMA(200) requirement
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        
        # Base price uptrend
        prices = np.linspace(100, 150, 250)
        
        # DataFrame
        self.data = pd.DataFrame({
            'close': prices,
            'high': prices + 1,
            'low': prices - 1,
            'volume': np.full(250, 1000)
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
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        df = pd.DataFrame({
            'close': np.full(250, 100.0),
            'high': np.full(250, 105.0),
            'low': np.full(250, 95.0),
            'volume': np.full(250, 1000.0)
        }, index=dates)
        
        # Set latest values to be plausible
        df.iloc[-1, df.columns.get_loc('close')] = 100.0
        df.iloc[-1, df.columns.get_loc('volume')] = 2000.0 # High volume
        
        # Patch the indicators imported in src.strategy_logic
        with patch('src.strategy_logic.calculate_rsi') as mock_rsi, \
             patch('src.strategy_logic.calculate_sma') as mock_sma, \
             patch('src.strategy_logic.calculate_macd') as mock_macd, \
             patch('src.strategy_logic.calculate_atr') as mock_atr:
            
            # Setup returns to satisfy BUY conditions
            # 1. RSI < 40 (Value +20)
            mock_rsi.return_value = pd.Series(np.full(250, 30.0), index=dates) 
            
            # 2. Price (100) > SMA_200 (90) (Trend +30)
            mock_sma.return_value = pd.Series(np.full(250, 90.0), index=dates)
            
            # 3. MACD > Signal (Momentum +20)
            mock_macd.return_value = (
                pd.Series(np.full(250, 5.0), index=dates), # MACD Line
                pd.Series(np.full(250, 2.0), index=dates)  # Signal Line
            )

            # ATR for SL calculation
            mock_atr.return_value = pd.Series(np.full(250, 2.0), index=dates)
            
            # Sentiment > 0.2 (Sentiment +30)
            sentiment_score = 0.8 
            # Total Score = 100
            
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

    def test_score_threshold_hold(self):
        """Test that a score below 70 results in HOLD."""
        # Setup:
        # Trend: Price > SMA (+30)
        # Momentum: MACD < Signal (0)
        # Value: RSI > 40 (0)
        # Sentiment: > 0.2 (+30)
        # Total = 60. Result should be HOLD.
        
        from unittest.mock import patch
        
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        df = pd.DataFrame({
            'close': np.full(250, 100.0),
            'high': np.full(250, 105.0),
            'low': np.full(250, 95.0),
            'volume': np.full(250, 1000.0)
        }, index=dates)
        
        with patch('src.strategy_logic.calculate_rsi') as mock_rsi, \
             patch('src.strategy_logic.calculate_sma') as mock_sma, \
             patch('src.strategy_logic.calculate_macd') as mock_macd, \
             patch('src.strategy_logic.calculate_atr') as mock_atr:
            
            # 1. RSI > 40 (0 pts)
            mock_rsi.return_value = pd.Series(np.full(250, 50.0), index=dates) 
            
            # 2. Price (100) > SMA_200 (90) (+30 pts)
            mock_sma.return_value = pd.Series(np.full(250, 90.0), index=dates)
            
            # 3. MACD < Signal (0 pts)
            mock_macd.return_value = (
                pd.Series(np.full(250, 2.0), index=dates),
                pd.Series(np.full(250, 5.0), index=dates)
            )

            mock_atr.return_value = pd.Series(np.full(250, 2.0), index=dates)
            
            # 4. Sentiment > 0.2 (+30 pts)
            sentiment_score = 0.5 
            
            # Total Score = 60
            
            signal = self.strategy.generate_signal("TEST", df, sentiment_score, 100000)
            
            self.assertEqual(signal['action'], 'HOLD')
            self.assertIn("Low Score (60)", signal['reason'])

    def test_calculate_trade_score_logic(self):
        """Directly test the score calculation method."""
        # perfect score
        s = self.strategy.calculate_trade_score(
            price=100, sma_200=90,     # +30
            macd=5, signal_line=2,     # +20
            rsi=30,                    # +20
            sentiment_score=0.5        # +30
        )
        self.assertEqual(s, 100)
        
        # zero score
        s = self.strategy.calculate_trade_score(
            price=80, sma_200=90,      # 0
            macd=2, signal_line=5,     # 0
            rsi=50,                    # 0
            sentiment_score=0.0        # 0
        )
        self.assertEqual(s, 0)


if __name__ == '__main__':
    unittest.main()
