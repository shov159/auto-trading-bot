import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from src.indicators import TechnicalIndicators
from src.risk_manager import RiskManager

class EnhancedStrategy:
    """
    Implements the Enhanced AI Trading Strategy.
    Combines Technical Indicators with Sentiment Analysis.
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager, is_backtest: bool = False):
        """
        Initialize the strategy.

        Args:
            config: Configuration dictionary.
            risk_manager: Instance of RiskManager.
            is_backtest: Flag to enable mock sentiment for backtesting.
        """
        self.config = config
        self.risk_manager = risk_manager
        self.is_backtest = is_backtest
        
        # Strategy Parameters
        self.strat_params = config.get('strategy', {})
        self.max_holding_days = self.strat_params.get('max_holding_days', 10)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all technical indicators and adds them to the DataFrame.
        
        Args:
            data: Raw OHLCV DataFrame.
            
        Returns:
            pd.DataFrame: Enriched DataFrame with indicators.
        """
        # Ensure data is sorted
        df = data.sort_index()
        
        # Group by symbol to calculate indicators per stock
        # Assuming index is (symbol, timestamp) or column 'symbol' exists.
        # If MultiIndex (symbol, timestamp), we can groupby level 0.
        
        # Helper to apply indicators
        def apply_indicators(group):
            # RSI
            group['rsi'] = TechnicalIndicators.calculate_rsi(group['close'], self.strat_params.get('rsi_period', 14))
            
            # MACD
            macd, signal, _ = TechnicalIndicators.calculate_macd(
                group['close'],
                self.strat_params.get('macd_fast', 12),
                self.strat_params.get('macd_slow', 26),
                self.strat_params.get('macd_signal', 9)
            )
            group['macd'] = macd
            group['macd_signal'] = signal
            
            # SMA 50 & 200
            group['sma_50'] = TechnicalIndicators.calculate_sma(group['close'], self.strat_params.get('sma_fast', 50))
            group['sma_200'] = TechnicalIndicators.calculate_sma(group['close'], self.strat_params.get('sma_slow', 200))
            
            # Bollinger Bands
            upper, _, lower = TechnicalIndicators.calculate_bollinger_bands(
                group['close'],
                self.strat_params.get('bollinger_period', 20),
                self.strat_params.get('bollinger_std', 2)
            )
            group['bb_upper'] = upper
            group['bb_lower'] = lower
            
            # Volume SMA
            group['vol_sma_20'] = TechnicalIndicators.calculate_sma(group['volume'], 20)
            
            # Pre-calculate Crossover (MACD > Signal)
            # We want "MACD line crosses above signal line in last 3 bars"
            # Logic: (MACD > Signal) AND (Prev MACD < Prev Signal) happened in [t, t-1, t-2]
            # Simple check: Just check if MACD > Signal now and was < Signal recently? 
            # Or just check if MACD > Signal. 
            # The requirement "crosses above ... in last 3 bars" requires rolling check.
            # We'll create a boolean series 'macd_cross_up'
            
            prev_macd = group['macd'].shift(1)
            prev_signal = group['macd_signal'].shift(1)
            
            cross_up = (group['macd'] > group['macd_signal']) & (prev_macd <= prev_signal)
            
            # Check if cross_up happened in last 3 periods
            group['macd_cross_recent'] = cross_up.rolling(window=3).max() > 0
            
            return group

        # Apply to each symbol
        # Check if index has levels
        if isinstance(df.index, pd.MultiIndex):
            df = df.groupby(level=0, group_keys=False).apply(apply_indicators)
        else:
            # Assume single symbol dataframe
            df = apply_indicators(df)
            
        return df

    def get_sentiment(self, symbol: str) -> str:
        """
        Fetches sentiment. Mocked if is_backtest=True.
        """
        if self.is_backtest:
            # Randomly simulate sentiment for backtest to allow trades
            # Bias towards BUY/SELL to see some action
            choice = np.random.choice(["BUY", "SELL", "HOLD"], p=[0.4, 0.4, 0.2])
            return choice
            
        # TODO: Implement Live Gemini call here (refactoring from ai_trader.py)
        # For now, return HOLD to be safe until live logic is integrated
        return "HOLD"

    def generate_signal(self, symbol: str, data_row: pd.Series, current_qty: int, portfolio_value: float, avg_entry_price: float = 0.0, entry_time: datetime = None) -> str:
        """
        Decides whether to BUY, SELL, or HOLD based on enriched data row.
        
        Args:
            symbol: Ticker symbol.
            data_row: Row from dataframe containing price and indicators.
            current_qty: Current position quantity.
            portfolio_value: Total portfolio value.
            avg_entry_price: Average entry price of the position (0.0 if none).
            entry_time: Timestamp when the position was entered (None if none).
            
        Returns:
            str: "BUY", "SELL", or "HOLD"
        """
        price = data_row['close']
        timestamp = data_row.name[1] if isinstance(data_row.name, tuple) else data_row.name
        
        # Risk Manager: Check Drawdown Halt
        if self.risk_manager.should_stop_trading(portfolio_value):
            return "SELL" # Close everything if max drawdown hit

        sentiment = self.get_sentiment(symbol)
        
        # ENTRY LOGIC
        # 1. Gemini Sentiment = BUY
        # 2. RSI < 40
        # 3. Price > SMA(50)
        # 4. MACD Cross Up Recent
        # 5. Volume > Vol SMA 20
        
        is_buy_signal = (
            sentiment == "BUY" and
            data_row['rsi'] < self.strat_params.get('rsi_oversold', 40) and
            price > data_row['sma_50'] and
            data_row['macd_cross_recent'] and
            data_row['volume'] > data_row['vol_sma_20']
        )
        
        if is_buy_signal and current_qty == 0:
            return "BUY"

        # EXIT LOGIC
        if current_qty > 0:
            # Calculate days held
            days_held = 0
            if entry_time:
                delta = timestamp - entry_time
                days_held = delta.days

            # Stop Loss & Take Profit Checks
            sl_price = self.risk_manager.get_stop_loss_price(avg_entry_price, 'buy')
            tp_price = self.risk_manager.get_take_profit_price(avg_entry_price, 'buy')
            
            stop_loss_hit = price <= sl_price
            take_profit_hit = price >= tp_price

            is_sell_signal = (
                sentiment == "SELL" or
                stop_loss_hit or
                take_profit_hit or
                data_row['rsi'] > self.strat_params.get('rsi_overbought', 70) or
                days_held > self.max_holding_days
            )
            
            if is_sell_signal:
                return "SELL"
                
        return "HOLD"

