import os
import sys
import time
from datetime import datetime
import yaml
import pandas as pd
from dotenv import load_dotenv

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

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
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # State
        self.symbols = self.config['trading']['symbols']

    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch last 60 days of data for indicators."""
        try:
            now = datetime.now()
            start = now - pd.Timedelta(days=100) # Buffer
            
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=now,
                feed=DataFeed.IEX # Free tier friendly
            )
            bars = self.data_client.get_stock_bars(req)
            
            if bars.df.empty:
                return pd.DataFrame()
            
            df = bars.df.loc[symbol].copy()
            df.columns = [c.lower() for c in df.columns]
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

    def execute_order(self, action: str, symbol: str, qty: int, price: float, reason: str):
        """Place order via Alpaca."""
        if qty <= 0:
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

    def run_cycle(self):
        """One iteration of the trading loop."""
        self.monitor.log_info("Starting Trading Cycle...")
        
        # ------------------------------
        
        # --- News Scout Integration ---
        try:
            news_report = self.news_scout.get_market_brief()
            if news_report:
                self.monitor.send_telegram_message(news_report)
                self.monitor.log_info("📰 News update sent")
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
                        
                        self.execute_order("BUY", symbol, adjusted_qty, signal['price'], signal['reason'])
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
        
        while True:
            try:
                self.run_cycle()
                self.monitor.log_info("Cycle complete. Sleeping for 15 minutes...")
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
