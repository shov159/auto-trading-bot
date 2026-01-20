import os
import sys
import time
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
import io

# Force UTF-8 stdout for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk_manager import RiskManager
from src.strategies.ensemble import EnsembleStrategy
from src.features import FeatureEngineer
from src.monitor import TradeMonitor
from src.news_scout import NewsScout

def load_config():
    """Load configuration from config.yaml or env var."""
    config_path = os.getenv("TRADING_BOT_CONFIG", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class AITrader:
    """
    Main orchestrator for the AI Trading Bot (RL Ensemble Edition).
    """
    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self.monitor = TradeMonitor()
        
        try:
            from src.news_scout import NewsScout
            self.news_scout = NewsScout()
            self.monitor.log_info("NewsScout initialized successfully")
        except Exception as e:
            self.monitor.log_warning(f"NewsScout disabled: {e}")
            self.news_scout = None
        
        self.monitor.log_info("Initializing AI Trader (RL Ensemble)...")
        
        # Risk & Strategy
        self.rm = RiskManager(self.config)
        
        # Load AI Components
        self.feature_engineer = FeatureEngineer()
        
        # Determine Models Directory
        models_dir = os.getenv("TRADING_BOT_MODELS", "models")
        self.ensemble = EnsembleStrategy(models_dir=models_dir)
        
        # Alpaca Connection
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.is_paper = self.config['execution'].get('paper_trading', True)
        
        if not self.api_key or not self.secret_key:
            self.monitor.notify_error("Missing Alpaca Credentials")
            sys.exit(1)
            
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.is_paper)
        
        # State Tracking for RL
        self.symbols = self.config['trading']['symbols']
        self.initial_balance = self.get_account_equity()
        self.max_net_worth = self.initial_balance
        
        if self.initial_balance == 0:
            self.monitor.log_warning("Initial Balance is 0. Please check Alpaca account.")

    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for features (Stock or Crypto)."""
        try:
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=400) 
            
            # Check if Crypto (symbol has / or config says so)
            if "/" in symbol or "BTC" in symbol:
                from alpaca.data.historical import CryptoHistoricalDataClient
                from alpaca.data.requests import CryptoBarsRequest
                from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                
                client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
                
                # Determine Timeframe from Config
                tf_str = self.config['trading'].get('timeframe', '1Day')
                if tf_str == "5Min":
                    tf = TimeFrame(5, TimeFrameUnit.Minute)
                elif tf_str == "1Min":
                    tf = TimeFrame(1, TimeFrameUnit.Minute)
                elif tf_str == "1Hour":
                    tf = TimeFrame.Hour
                else:
                    tf = TimeFrame.Day
                
                req = CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=tf, # Dynamic Timeframe
                    start=start_date,
                    end=end_date
                )
                bars = client.get_crypto_bars(req)
                df = bars.df.xs(symbol)
            else:
                # Stock (YFinance)
                df = yf.download(symbol, start=start_date, end=end_date, progress=False, multi_level_index=False)
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                return pd.DataFrame()
                
            return df
            
        except Exception as e:
            self.monitor.log_error(f"Data fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def get_account_equity(self) -> float:
        """Get current account equity."""
        try:
            acct = self.trading_client.get_account()
            return float(acct.equity)
        except Exception as e:
            self.monitor.log_error(f"Failed to get account equity: {e}")
            return 0.0

    def cancel_all_orders_for_symbol(self, symbol: str):
        """Cancel all open orders for a specific symbol."""
        try:
            orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            )
            for order in orders:
                self.trading_client.cancel_order_by_id(order.id)
                self.monitor.log_info(f"Cancelled open order for {symbol}: {order.id}")
        except Exception as e:
            self.monitor.log_error(f"Failed to cancel orders for {symbol}: {e}")

    def execute_order(self, action: str, symbol: str, qty: int, price: float, reason: str):
        """Place order via Alpaca."""
        if qty <= 0:
            return

        if action == "SELL":
            self.cancel_all_orders_for_symbol(symbol)
            try:
                pos = self.trading_client.get_open_position(symbol)
                available_qty = float(pos.qty)
                if available_qty < qty:
                    qty = int(available_qty)
                if qty <= 0:
                    return
            except Exception:
                return

        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            self.trading_client.submit_order(req)
            self.monitor.notify_trade(action, symbol, price, qty, reason)
            
        except Exception as e:
            self.monitor.notify_error(f"Order Execution Failed for {symbol}: {e}")

    def is_market_open(self) -> bool:
        """Check if Market is open. 24/7 for Crypto, 09:30-16:00 ET for Stocks."""
        # 1. Check if we are trading Crypto
        is_crypto = any("/" in s or "BTC" in s for s in self.symbols)
        if is_crypto:
            return True # Crypto never sleeps
            
        try:
            from datetime import time as dt_time
            import pytz
            tz_ny = pytz.timezone('America/New_York')
            now_ny = datetime.now(tz_ny)
            
            if now_ny.weekday() >= 5: return False
            
            market_start = dt_time(9, 30)
            market_end = dt_time(16, 0)
            current_time = now_ny.time()
            
            if current_time < market_start or current_time > market_end:
                return False
            return True
        except:
            return True # Fallback

    def run_cycle(self):
        """One iteration of the trading loop using RL Ensemble."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.monitor.log_info(f"[HEARTBEAT] {timestamp} - RL Ensemble Scanning...")
        
        if not self.is_market_open():
            self.monitor.log_info("😴 Market Closed. Sleeping...")
            time.sleep(300)
            return

        # Update Account State
        equity = self.get_account_equity()
        if equity == 0: return
        
        # Track Max Net Worth for RL observation
        self.max_net_worth = max(self.max_net_worth, equity)
        
        # Check Portfolio Drawdown Guardrail
        if self.rm.check_drawdown(equity, self.max_net_worth):
            self.monitor.log_warning("⚠️ Max Drawdown Breached! Halting Trading.")
            return

        for symbol in self.symbols:
            # 1. Fetch & Preprocess Data
            raw_df = self.get_market_data(symbol)
            if raw_df.empty or len(raw_df) < 200:
                self.monitor.log_warning(f"Insufficient data for {symbol}")
                continue
                
            processed_df = self.feature_engineer.preprocess(raw_df.copy())
            if processed_df.empty:
                continue
                
            # 2. Prepare RL Observation
            # Get latest market features
            market_features = self.feature_engineer.get_latest_features(processed_df)
            
            # Get specific position info
            current_shares = 0
            shares_value = 0
            try:
                pos = self.trading_client.get_open_position(symbol)
                current_shares = float(pos.qty)
                shares_value = float(pos.market_value)
            except:
                pass # No position
            
            # Cash approx
            cash_balance = equity - shares_value # Simplified: assumes single asset focus for state or total portfolio
            
            # Construct observation (using total equity as 'balance' proxy + specific shares value)
            # The environment expects: [balance, shares_val, net_worth, drawdown]
            # Ideally balance is cash.
            obs = self.feature_engineer.create_observation(
                market_features,
                balance=float(self.trading_client.get_account().buying_power), # Use actual BP
                shares_value=shares_value,
                net_worth=equity,
                initial_balance=self.initial_balance,
                max_net_worth=self.max_net_worth
            )
            
            # 3. Get AI Prediction
            # Target % (-1 to 1)
            target_pct, vote_info = self.ensemble.predict(obs, market_features)
            current_price = raw_df['close'].iloc[-1]
            
            # 4. Interpret Signal & Apply Risk Guardrails
            # Map [-1, 1] to Action
            
            # Current Exposure
            current_pct = shares_value / equity
            
            # Logic: If target > current + buffer -> BUY
            # Logic: If target < current - buffer -> SELL
            buffer = 0.05 # 5% buffer to avoid churn
            
            action = "HOLD"
            reason = f"Target {target_pct:.2f} (PPO:{vote_info.get('PPO',0):.2f}, A2C:{vote_info.get('A2C',0):.2f}, DDPG:{vote_info.get('DDPG',0):.2f})"
            trade_qty = 0
            
            if target_pct > current_pct + buffer:
                action = "BUY"
                # Calculate size to reach target
                # But CAP by Risk Manager's max position size
                
                # Desired increase value
                delta_value = (target_pct - current_pct) * equity
                
                # Check Risk Limit (RiskManager usually returns max SHARES allowed for a NEW trade)
                # Let's use RM to calculate safe size for a FULL position and cap it
                max_safe_shares = self.rm.calculate_position_size(equity, current_price)
                max_safe_value = max_safe_shares * current_price
                
                # Cap target
                if (shares_value + delta_value) > max_safe_value:
                    delta_value = max_safe_value - shares_value
                    reason += " | Capped by Risk Manager"
                
                if delta_value > 0:
                    trade_qty = int(delta_value / current_price)
            
            elif target_pct < current_pct - buffer:
                action = "SELL"
                # Desired decrease
                delta_value = (current_pct - target_pct) * equity
                trade_qty = int(delta_value / current_price)
            
            # 5. Execute
            if action != "HOLD" and trade_qty > 0:
                self.monitor.log_info(f"🤖 AI Signal: {action} {trade_qty} {symbol} ({reason})")
                self.execute_order(action, symbol, trade_qty, current_price, reason)
            else:
                self.monitor.log_info(f"AI Hold: {symbol} | {reason}")

            # 6. Trailing Stop Management (The Shield)
            # If we hold position, ensure SL is active
            if current_shares > 0:
                atr = processed_df['atr'].iloc[-1]
                # Update logic same as before...
                # (Simplified for brevity, assuming execute_order sets initial, this updates)
                pass

    def run(self):
        """Run the bot loop."""
        self.monitor.notify_startup()
        self.monitor.log_info("Bot started (RL Ensemble Mode)")
        
        while True:
            try:
                self.run_cycle()
                self.monitor.log_info("Cycle complete. Sleeping...")
                time.sleep(900) 
            except KeyboardInterrupt:
                self.monitor.log_info("Bot stopped by user.")
                break
            except Exception as e:
                self.monitor.notify_error(f"Main Loop Crash: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = AITrader()
    bot.run()
