"""
Strategy Logic
Defines buy/sell signals for the backtester.
Baseline: SMA Crossover + RSI Filter.
"""
class StrategyLogic:
    def __init__(self, risk_manager, config):
        self.risk_manager = risk_manager
        self.config = config

        # Params
        self.sma_fast = 20
        self.sma_slow = 50
        self.rsi_threshold = 40 # Buy dip in trend

        # State
        self.history = {} # {ticker: [prices]}

    def generate_signal(self, ticker, candle, current_equity):
        """
        Returns dict: {'action': 'BUY', 'stop_loss': float, 'take_profit': float} or None
        """
        price = candle['Close']

        # Update history
        if ticker not in self.history: self.history[ticker] = []
        self.history[ticker].append(price)

        # Context (simulated logic inputs)
        hist = self.history[ticker]
        if len(hist) < 55: return None

        # Calculate Indicators (Simplified for speed vs using TA-lib)
        import pandas as pd
        series = pd.Series(hist)
        sma20 = series.rolling(20).mean().iloc[-1]
        sma50 = series.rolling(50).mean().iloc[-1]

        # Logic: Trend Following
        # Buy if Price > SMA20 > SMA50 (Strong Trend)
        if price > sma20 and sma20 > sma50:

            # Risk Management
            atr_sim = price * 0.02 # Assume 2% volatility for baseline
            stop_loss = price - (atr_sim * 2)
            take_profit = price + (atr_sim * 4) # 1:2 Risk/Reward

            return {
                "action": "BUY",
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }

        return None

    def should_exit(self, ticker, candle):
        # Exit if trend breaks (Price < SMA20) - Trailing stop logic simulated
        # For this baseline, we stick to fixed TP/SL handled by Backtester
        return False
