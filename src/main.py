import os
import sys
import time
from datetime import datetime
import yaml
import pandas as pd
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
# from alpaca.data.historical import StockHistoricalDataClient
# from alpaca.data.requests import StockBarsRequest
# from alpaca.data.timeframe import TimeFrame
# from alpaca.data.enums import DataFeed

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk_manager import RiskManager
from src.strategy_logic import StrategyLogic
from src.monitor import TradeMonitor
from src.news_scout import NewsScout

def load_config():
    """Load configuration from config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class AITrader:
    """
    Main orchestrator for the AI Trading Bot.
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
        
        self.monitor.log_info("Initializing AI Trader...")
        
        # Risk & Strategy
        self.rm = RiskManager(self.config)
        self.strategy = StrategyLogic(self.rm, self.config)
        
        # Alpaca Connection
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.is_paper = self.config['execution'].get('paper_trading', True)
        
        if not self.api_key or not self.secret_key:
            self.monitor.notify_error("Missing Alpaca Credentials")
            sys.exit(1)
            
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.is_paper)
        # self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # State
        self.symbols = self.config['trading']['symbols']

    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch last 250 days of data for indicators using yfinance."""
        try:
            # Using yfinance for better availability/free access as requested
            # Fetch roughly 1 year to be safe for SMA200
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=400) 
            
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, multi_level_index=False)
            
            if df.empty:
                return pd.DataFrame()
            
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                self.monitor.log_warning(f"Incomplete data for {symbol}")
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
        """Cancel all open orders for a specific symbol to free up shares."""
        try:
            orders = self.trading_client.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[symbol]
                )
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

        # Cancel existing orders before selling to free up shares (Fix "Insufficient Qty" error)
        if action == "SELL":
            # 1. Cancel open orders
            self.cancel_all_orders_for_symbol(symbol)
            
            # 2. Double Check Position Exists (Ghost Selling Fix)
            try:
                pos = self.trading_client.get_open_position(symbol)
                available_qty = float(pos.qty)
                if available_qty < qty:
                    self.monitor.log_warning(f"⚠️ Adjusting SELL qty for {symbol}: {qty} -> {available_qty}")
                    qty = int(available_qty)
                
                if qty <= 0:
                    self.monitor.log_info(f"⚠️ Skipping SELL for {symbol}: No position found (Qty=0).")
                    return
            except Exception:
                # If get_open_position fails (404), we don't hold it
                self.monitor.log_info(f"⚠️ Skipping SELL for {symbol}: No position found.")
                return

        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            # Place Order
            self.trading_client.submit_order(req)
            
            # Notify
            self.monitor.notify_trade(action, symbol, price, qty, reason)
            
        except Exception as e:
            self.monitor.notify_error(f"Order Execution Failed for {symbol}: {e}")

    def is_market_open(self) -> bool:
        """Check if US Market is open (09:30 - 16:00 ET, Mon-Fri)."""
        # For MVP/testing, we often return True. But to stop weekend trading:
        try:
            from datetime import time
            import pytz
            
            # Define ET timezone
            tz_ny = pytz.timezone('America/New_York')
            now_ny = datetime.now(tz_ny)
            
            # 1. Check Weekday (Mon=0, Sun=6)
            if now_ny.weekday() >= 5: # Saturday or Sunday
                return False
                
            # 2. Check Time (09:30 - 16:00)
            market_start = time(9, 30)
            market_end = time(16, 0)
            current_time = now_ny.time()
            
            if current_time < market_start or current_time > market_end:
                return False
                
            return True
        except ImportError:
            # Fallback if pytz not installed (though pandas usually has it)
            # Just verify it's not weekend in local time as a crude check
            if datetime.now().weekday() >= 5:
                return False
            return True 
        except Exception as e:
            self.monitor.log_error(f"Market hours check failed: {e}")
            return True # Fail open to allow trading if uncertain, or False to be safe.
            # Let's return True to not block if there's a weird error, but log it. 

    def run_cycle(self):
        """One iteration of the trading loop."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.monitor.log_info(f"[HEARTBEAT] {timestamp} - Starting Market Scan...")
        
        if not self.is_market_open():
            self.monitor.log_info("😴 Market Closed. Sleeping...")
            # Sleep 5 minutes (300s) as requested
            time.sleep(300)
            return

        # ------------------------------
        
        # --- News Scout Integration ---
        try:
            news_report = self.news_scout.get_market_brief()
            if news_report:
                self.monitor.send_telegram_message(news_report)
                self.monitor.log_info("News update sent")
        except Exception as e:
            self.monitor.log_error(f"News error: {e}")
        # ------------------------------

        equity = self.get_account_equity()
        if equity == 0:
            return

        # 1. Check Market Open? (Optional, skipping for MVP to allow testing anytime)
        
        for symbol in self.symbols:
            # 2. Get Data
            df = self.get_market_data(symbol)
            if df.empty or len(df) < 50:
                self.monitor.log_warning(f"Insufficient data for {symbol}")
                continue
                
            # 3. Get Sentiment & Regime
            # Use real news regime instead of placeholder
            try:
                regime = self.news_scout.get_regime()
                sentiment_score = regime.get("overall", 0.0)
                
                # Determine Confidence Multiplier & Gating
                confidence_mult = 1.0
                block_buys = False
                
                if sentiment_score <= -0.30:
                    block_buys = True
                    self.monitor.log_warning(f"News Regime Bearish ({sentiment_score:.2f}). Blocking BUYs.")
                elif -0.30 < sentiment_score < 0.10:
                    confidence_mult = 0.5
                elif 0.10 <= sentiment_score < 0.40:
                    confidence_mult = 1.0
                elif sentiment_score >= 0.40:
                    confidence_mult = 1.25
                    
            except Exception as e:
                self.monitor.log_error(f"Regime check failed: {e}")
                sentiment_score = 0.0
                confidence_mult = 1.0
                block_buys = False
            
            # 4. Generate Signal
            signal = self.strategy.generate_signal(symbol, df, sentiment_score, equity)
            
            self.monitor.log_info(f"Analysis {symbol}: {signal['action']} | {signal.get('reason')}")

            # --- Trailing Stop Logic (The Shield) ---
            try:
                # Check if we hold a position
                position = None
                try:
                    position = self.trading_client.get_open_position(symbol)
                except:
                    pass # No position found

                if position:
                    # We have a position, check/update trailing stop
                    current_price = float(position.current_price)
                    atr = signal['debug'].get('atr', 0.0)
                    
                    if atr > 0:
                        # Find existing Stop Loss Order
                        # We search for open SELL STOP orders for this symbol.
                        orders_req = GetOrdersRequest(
                            status=QueryOrderStatus.OPEN,
                            symbols=[symbol],
                            side=OrderSide.SELL
                        )
                        open_orders = self.trading_client.get_orders(orders_req)
                        
                        sl_order = None
                        for order in open_orders:
                            if order.type in ['stop', 'stop_limit']:
                                sl_order = order
                                break
                        
                        if sl_order:
                            current_sl = float(sl_order.stop_price)
                            new_sl = self.rm.update_trailing_stop(current_price, current_sl, atr)
                            
                            if new_sl and new_sl > current_sl:
                                self.monitor.log_info(f"Updating Trailing Stop for {symbol}: {current_sl} -> {new_sl:.2f}")
                                self.trading_client.replace_order(
                                    order_id=sl_order.id,
                                    stop_price=new_sl
                                )
                        else:
                             # No SL order found. In a real system, we might create one here.
                             pass
            except Exception as e:
                 self.monitor.log_error(f"Trailing Stop Update Failed for {symbol}: {e}")
            # ----------------------------------------
            
            # 5. Execute
            if signal['action'] == 'BUY':
                if block_buys:
                    self.monitor.log_info(f"Skipping BUY {symbol}: Blocked by News Regime")
                    # Optional: Send Telegram Alert about block (throttled)
                    # self.monitor.send_telegram_message(f"🛑 חסימת קניות: סנטימנט חדשות שלילי ({sentiment_score:.2f})")
                    continue

                # Check if we already hold it? (Simplified: No position check in MVP StrategyLogic yet)
                # But we should check positions here to avoid doubling up.
                try:
                    positions = self.trading_client.get_all_positions()
                    current_qty = 0
                    for p in positions:
                        if p.symbol == symbol:
                            current_qty = float(p.qty)
                            break
                    
                    if current_qty == 0:
                        # Recalculate quantity with confidence multiplier
                        # Note: StrategyLogic already calculated base quantity. 
                        # We need to apply multiplier here or pass it into StrategyLogic earlier.
                        # Since generate_signal is decoupled, let's re-calculate qty using RiskManager directly here.
                        
                        base_qty = signal['quantity']
                        # Re-run sizing with multiplier
                        adjusted_qty = self.rm.calculate_position_size(equity, signal['price'], confidence_mult)
                        
                        # Double check we don't already have a position (race condition)
                        # Actually, we checked `current_qty == 0` above.
                        # But let's log specifically.
                        if adjusted_qty > 0:
                            self.execute_order("BUY", symbol, adjusted_qty, signal['price'], signal['reason'])
                        else:
                            self.monitor.log_info(f"Skipping BUY {symbol}: Calc Qty is 0 (Risk Mgmt)")
                    else:
                        self.monitor.log_info(f"Skipping BUY {symbol}: Already hold {current_qty}")
                        
                except Exception as e:
                    self.monitor.log_error(f"Position check failed: {e}")

            elif signal['action'] == 'SELL':
                # Check if we hold it
                try:
                    positions = self.trading_client.get_all_positions()
                    current_qty = 0
                    for p in positions:
                        if p.symbol == symbol:
                            current_qty = float(p.qty)
                            break
                    
                    if current_qty > 0:
                        self.execute_order("SELL", symbol, int(current_qty), signal['price'], signal['reason'])
                except Exception as e:
                    self.monitor.log_error(f"Position check failed: {e}")

    def run(self):
        """Run the bot loop."""
        self.monitor.notify_startup()
        self.monitor.log_info("Bot started in Live Mode (Paper Trading)")
        self.monitor.log_info(f"Strategy: Score Threshold > {self.strategy.buy_threshold}, TP: {self.rm.tp_atr_mult}x ATR")

        while True:
            try:
                self.run_cycle()
                self.monitor.log_info("Cycle complete. Sleeping for 15 minutes...")
                # Sleep 15 minutes
                time.sleep(900) 
            except KeyboardInterrupt:
                self.monitor.log_info("Bot stopped by user.")
                break
            except Exception as e:
                self.monitor.notify_error(f"Main Loop Crash: {e}")
                time.sleep(60) # Wait before restart

if __name__ == "__main__":
    bot = AITrader()
    bot.run()
