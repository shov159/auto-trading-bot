import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.strategy_logic import StrategyLogic
from src.risk_manager import RiskManager
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
import os

class Backtester:
    """
    Simple event-driven backtesting engine for StrategyLogic.
    """

    def __init__(self, 
                 strategy: StrategyLogic, 
                 risk_manager: RiskManager, 
                 initial_capital: float = 100000.0,
                 transaction_cost_pct: float = 0.001):
        """
        Initialize the Backtester.
        
        Args:
            strategy: Instance of StrategyLogic.
            risk_manager: Instance of RiskManager.
            initial_capital: Starting cash.
            transaction_cost_pct: Fee/slippage per trade (0.001 = 0.1%).
        """
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.equity = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        
        # Portfolio: {symbol: {'quantity': int, 'entry_price': float, 'sl': float, 'tp': float}}
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # History
        self.equity_curve: List[Dict[str, Any]] = []
        self.trade_log: List[Dict[str, Any]] = []

    def load_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data using Alpaca (requires API keys in env) or Mock if fails.
        """
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            print("[Backtest] No API Keys found. Generating MOCK data.")
            return self._generate_mock_data(symbols, start_date, end_date)
            
        try:
            client = StockHistoricalDataClient(api_key, secret_key)
            # Use feed param for free tier compatibility (removed DataFeed enum usage as it caused issues or feed param not supported by client directly in this version)
            # Actually, alpaca-py client.get_stock_bars accepts request_params which HAS 'feed' attribute.
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                feed=DataFeed.IEX # Specify feed here in the Request object
            )
            bars = client.get_stock_bars(request_params)
            
            if bars.df.empty:
                print("[Backtest] Alpaca returned empty dataframe. Generating MOCK data.")
                return self._generate_mock_data(symbols, start_date, end_date)

            # Convert to dictionary of DataFrames
            data_dict = {}
            for symbol in symbols:
                if symbol in bars.df.index.levels[0]:
                    df = bars.df.loc[symbol].copy()
                    # Ensure columns are lowercase
                    df.columns = [c.lower() for c in df.columns]
                    data_dict[symbol] = df
                else:
                    print(f"[Backtest] No data found for {symbol} in response.")
            
            return data_dict
            
        except Exception as e:
            print(f"[Backtest] Error fetching data: {e}. Falling back to MOCK.")
            return self._generate_mock_data(symbols, start_date, end_date)

    def _generate_mock_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Generate mock data with clear trends and tradable dips.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(dates)
        data = {}
        
        for symbol in symbols:
            # 1. Create Uptrend (Linear)
            trend = np.linspace(100, 200, n)
            
            # 2. Add Waves (Sine) to create tradable dips (RSI < 45)
            # 3 cycles over the period
            waves = 10 * np.sin(np.linspace(0, 3 * 2 * np.pi, n))
            
            # 3. Add Noise
            noise = np.random.normal(0, 2, n)
            
            prices = trend + waves + noise
            
            # Ensure no negative prices
            prices = np.maximum(prices, 10)
            
            df = pd.DataFrame({
                'open': prices,
                'high': prices + 2,
                'low': prices - 2,
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, n) # High volume
            }, index=dates)
            data[symbol] = df
            
        print("[Backtest] Generated MOCK data with trends.")
        return data

    def run(self, symbols: List[str], start_date: datetime, end_date: datetime, mock_sentiment: bool = True):
        """
        Execute the backtest simulation.
        """
        print(f"--- Starting Backtest ({start_date.date()} to {end_date.date()}) ---")
        
        # 1. Load Data
        market_data = self.load_data(symbols, start_date, end_date)
        if not market_data:
            print("No data available.")
            return

        # Align all dates (Trading Calendar)
        all_dates = sorted(list(set().union(*[df.index for df in market_data.values()])))
        
        days_counter = 0

        for current_date in all_dates:
            days_counter += 1
            # Update Equity (Cash + Positions Value)
            portfolio_value = 0
            
            # Iterate over a COPY of keys to allow deletion during iteration
            for sym in list(self.positions.keys()):
                pos = self.positions[sym]
                
                # Get current price
                if current_date in market_data[sym].index:
                    curr_price = market_data[sym].loc[current_date]['close']
                    portfolio_value += pos['quantity'] * curr_price
                    
                    # Check Stops (SL/TP)
                    # We check Low/High of TODAY to see if we got stopped out intraday
                    day_high = market_data[sym].loc[current_date]['high']
                    day_low = market_data[sym].loc[current_date]['low']
                    
                    exit_price = None
                    reason = None
                    
                    if day_low <= pos['sl']:
                        exit_price = pos['sl']
                        reason = "Stop Loss"
                    elif day_high >= pos['tp']:
                        exit_price = pos['tp']
                        reason = "Take Profit"
                        
                    if exit_price:
                        # EXECUTE SELL
                        proceeds = exit_price * pos['quantity']
                        cost = proceeds * self.transaction_cost_pct
                        self.current_cash += (proceeds - cost)
                        
                        self.trade_log.append({
                            "date": current_date,
                            "symbol": sym,
                            "action": "SELL",
                            "price": exit_price,
                            "quantity": pos['quantity'],
                            "reason": reason,
                            "pnl": (exit_price - pos['entry_price']) * pos['quantity']
                        })
                        del self.positions[sym]
                        continue # Position closed, skip to next symbol

            self.equity = self.current_cash + portfolio_value
            self.equity_curve.append({"date": current_date, "equity": self.equity})

            # Check Risk (Drawdown)
            peak = max([x['equity'] for x in self.equity_curve])
            if self.risk_manager.check_drawdown(self.equity, peak):
                print(f"[Risk] Max Drawdown hit at {current_date}. Halting trading.")
                break

            # Generate Signals for Next Day
            for symbol in symbols:
                if symbol in self.positions:
                    continue # Already holding, skip (simplified for MVP)
                
                if current_date not in market_data[symbol].index:
                    continue
                    
                # Slice data up to current_date
                hist_data = market_data[symbol].loc[:current_date]
                if len(hist_data) < 50:
                    continue
                
                # Mock Sentiment
                # Generate varied sentiment instead of 0.0
                sentiment = np.random.uniform(-1, 1) if mock_sentiment else 0.0
                
                # Generate Signal
                signal = self.strategy.generate_signal(symbol, hist_data, sentiment, self.equity)
                
                # Debug logging every 50 days or if HOLDing to see indicators
                if days_counter % 50 == 0 and signal['action'] == 'HOLD' and symbol == symbols[0]:
                    d = signal.get('debug', {})
                    print(f"Day {current_date.date()} Skipped {symbol}: {signal.get('reason')} | "
                          f"RSI={d.get('rsi', 0):.1f}, Sent={d.get('sentiment', 0):.1f}, "
                          f"Price={hist_data['close'].iloc[-1]:.1f}, SMA={d.get('sma', 0):.1f}")

                if signal['action'] == 'BUY':
                    qty = signal['quantity']
                    price = signal['price']
                    cost = qty * price
                    fee = cost * self.transaction_cost_pct
                    
                    if self.current_cash >= (cost + fee):
                        self.current_cash -= (cost + fee)
                        self.positions[symbol] = {
                            'quantity': qty,
                            'entry_price': price,
                            'sl': signal['stop_loss'],
                            'tp': signal['take_profit']
                        }
                        self.trade_log.append({
                            "date": current_date,
                            "symbol": symbol,
                            "action": "BUY",
                            "price": price,
                            "quantity": qty,
                            "reason": signal['reason']
                        })

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from equity curve and trade log.
        """
        if not self.equity_curve:
            return {}

        df = pd.DataFrame(self.equity_curve).set_index('date')
        df['returns'] = df['equity'].pct_change()
        
        total_return_pct = (self.equity - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe (Annualized) - assuming 252 trading days
        mean_ret = df['returns'].mean()
        std_ret = df['returns'].std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0
        
        # Max Drawdown
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
        max_dd_pct = df['drawdown'].min() * 100
        
        # Win Rate
        trades = [t for t in self.trade_log if t['action'] == 'SELL']
        if trades:
            wins = len([t for t in trades if t['pnl'] > 0])
            win_rate = (wins / len(trades)) * 100
        else:
            win_rate = 0.0
            
        return {
            "Total Return (%)": round(total_return_pct, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown (%)": round(max_dd_pct, 2),
            "Win Rate (%)": round(win_rate, 2),
            "Trades Executed": len(trades),
            "Final Equity": round(self.equity, 2)
        }
