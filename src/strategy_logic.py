import pandas as pd
from typing import Dict, Any, Optional
from src.indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_macd,
    calculate_atr
)
from src.risk_manager import RiskManager

class StrategyLogic:
    """
    Core strategy logic combining technical indicators and sentiment analysis
    to generate trading signals.
    """

    def __init__(self, risk_manager: RiskManager, config: Dict[str, Any]):
        """
        Initialize the StrategyLogic.

        Args:
            risk_manager: Instance of RiskManager for sizing and limits.
            config: Dictionary containing strategy parameters.
        """
        self.risk_manager = risk_manager
        self.config = config
        self.strat_config = config.get('strategy', {})
        
        # Load parameters
        self.rsi_period = self.strat_config.get('rsi_period', 14)
        self.rsi_buy_thresh = 60 # Relaxed from 45
        self.rsi_sell_thresh = 75
        self.sma_period = self.strat_config.get('sma_fast', 50)
        
        # Sentiment thresholds
        self.sentiment_buy_thresh = 0.0 # Relaxed from 0.5
        self.sentiment_sell_thresh = -0.2

    def generate_signal(self, symbol: str, data: pd.DataFrame, sentiment_score: float, account_value: float) -> Dict[str, Any]:
        """
        Analyze data and return a trading signal.

        Args:
            symbol: Ticker symbol.
            data: DataFrame with OHLCV data.
            sentiment_score: Sentiment score (-1.0 to 1.0).
            account_value: Total account equity (for position sizing).

        Returns:
            Dictionary containing action, price, SL/TP details, and reasoning.
        """
        if data.empty or len(data) < self.sma_period:
            return {"action": "HOLD", "reason": "Insufficient Data"}

        # 1. Calculate Indicators
        # We process the whole series but mostly care about the last value
        rsi = calculate_rsi(data['close'], period=self.rsi_period)
        sma_50 = calculate_sma(data['close'], window=self.sma_period)
        macd_line, signal_line = calculate_macd(data['close'])
        atr = calculate_atr(data['high'], data['low'], data['close']) # Default window 14
        
        # Volume MA for confirmation
        vol_ma = data['volume'].rolling(window=20).mean()

        # Get latest values
        current_close = data['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_sma = sma_50.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_atr = atr.iloc[-1]
        current_vol = data['volume'].iloc[-1]
        current_vol_ma = vol_ma.iloc[-1]

        # 2. Logic Check
        
        # SELL Conditions (Priority Check)
        sell_reasons = []
        if sentiment_score < self.sentiment_sell_thresh:
            sell_reasons.append("Negative Sentiment")
        if current_rsi > self.rsi_sell_thresh:
            sell_reasons.append("RSI Overbought")
        if current_close < current_sma * 0.95:
            sell_reasons.append("Trend Broken (Below SMA)")
            
        if sell_reasons:
            return {
                "action": "SELL",
                "price": current_close,
                "reason": " + ".join(sell_reasons)
            }

        # BUY Conditions (ALL must be true)
        buy_reasons = []
        conditions_met = True
        
        # Sentiment
        if sentiment_score > self.sentiment_buy_thresh:
            buy_reasons.append("Positive Sentiment")
        else:
            conditions_met = False

        # RSI Dip
        if current_rsi < self.rsi_buy_thresh:
            buy_reasons.append("RSI Dip")
        else:
            conditions_met = False
            
        # Uptrend
        if current_close > current_sma:
            buy_reasons.append("Uptrend")
        else:
            conditions_met = False
            
        # Momentum (MACD Crossover or simply MACD > Signal for this snapshot)
        # Note: "Crosses above" implies comparing t-1 and t. 
        # For simplicity in this snapshot check, we strictly check if MACD > Signal.
        # A stricter crossover check would require: prev_macd < prev_signal and current_macd > current_signal
        if current_macd > current_signal:
            buy_reasons.append("MACD Bullish")
        else:
            conditions_met = False
            
        # Volume Confirmation
        # if current_vol > current_vol_ma:
        #     buy_reasons.append("High Volume")
        # else:
        #     conditions_met = False
        
        # Disabled Volume check for testing
        pass

        if conditions_met:
            # Calculate Risk Parameters
            quantity = self.risk_manager.calculate_position_size(account_value, current_close)
            stop_loss = self.risk_manager.get_stop_loss_price(current_close, "buy", current_atr)
            take_profit = self.risk_manager.get_take_profit_price(current_close, "buy", current_atr)
            
            if quantity > 0:
                return {
                    "action": "BUY",
                    "price": current_close,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reason": " + ".join(buy_reasons),
                    "debug": {
                        "rsi": current_rsi,
                        "sma": current_sma,
                        "macd": current_macd,
                        "signal": current_signal,
                        "sentiment": sentiment_score
                    }
                }
            else:
                return {"action": "HOLD", "reason": "Insufficient Capital for Risk Sizing"}

        return {
            "action": "HOLD", 
            "reason": "No Signal",
            "debug": {
                "rsi": current_rsi,
                "sma": current_sma,
                "macd": current_macd,
                "signal": current_signal,
                "sentiment": sentiment_score
            }
        }

