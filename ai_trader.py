"""
AI Trader - Standalone Version (No Lumibot)
Directly uses alpaca-py for Trading, Data, and News.
"""
import os
import time
from datetime import datetime
import requests
from dotenv import load_dotenv

# Google Gemini
from google import genai

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockLatestTradeRequest, NewsRequest

# Load environment variables
load_dotenv()

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Portfolio Settings
SYMBOLS = ["SPY", "NVDA", "TSLA", "AAPL", "AMD", "MSFT"]
CASH_AT_RISK = 0.15  # Allocate 15% of available cash per trade

class AITrader:
    def __init__(self):
        print("--- Initializing AI Trader (Standalone) ---")
        
        # 1. Initialize Google Gemini
        if GOOGLE_API_KEY:
            self.gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            self.gemini_client = None
            print("[WARNING] No Google API Key found.")

        # 2. Initialize Alpaca Clients
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            self.trading_client = TradingClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                paper=ALPACA_PAPER
            )
            self.data_client = StockHistoricalDataClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY
            )
            self.news_client = NewsClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY
            )
        else:
            raise ValueError("Alpaca API Keys are missing!")

        self.post_telegram_message("🚀 AI Trader Started (Standalone Mode)")

    def post_telegram_message(self, message):
        """Send notification to Telegram."""
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
                requests.post(url, json=payload, timeout=5)
            except Exception as e:
                print(f"[ERROR] Telegram failed: {e}")
        else:
            print(f"[TELEGRAM MOCK] {message}")

    def check_market_open(self):
        """Check if the market is open."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"[ERROR] Market check failed: {e}")
            return False

    def get_price(self, symbol):
        """Fetch latest price via Alpaca Data API."""
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            res = self.data_client.get_stock_latest_trade(req)
            return res[symbol].price
        except Exception as e:
            print(f"[ERROR] Price fetch failed for {symbol}: {e}")
            return None

    def get_position(self, symbol):
        """Check if we currently hold a position in the symbol."""
        try:
            # get_all_positions returns a list of Position objects
            positions = self.trading_client.get_all_positions()
            for p in positions:
                if p.symbol == symbol:
                    return float(p.qty)
            return 0
        except Exception as e:
            print(f"[ERROR] Position check failed: {e}")
            return 0

    def get_account_cash(self):
        """Get available cash from the account."""
        try:
            account = self.trading_client.get_account()
            # use 'cash' or 'buying_power'
            return float(account.cash)
        except Exception as e:
            print(f"[ERROR] Account check failed: {e}")
            return 0.0

    def calculate_quantity(self, price):
        """Calculate shares to buy based on risk management."""
        cash = self.get_account_cash()
        if cash <= 0:
            return 0
        
        allocation = cash * CASH_AT_RISK
        if price <= 0:
            return 0
            
        qty = int(allocation // price)
        return qty

    def get_sentiment(self, symbol):
        """Fetch news and analyze with Gemini."""
        news_headlines = []
        try:
            # Fetch News
            req = NewsRequest(symbols=symbol, limit=10)
            news_set = self.news_client.get_news(req)
            
            # Handle Alpaca news response structure (list of objects)
            if hasattr(news_set, 'news'):
                items = news_set.news
            else:
                items = news_set # It might be a list directly

            for item in items:
                if hasattr(item, 'headline'):
                    news_headlines.append(item.headline)
                elif isinstance(item, dict):
                    news_headlines.append(item.get('headline', str(item)))
            
            print(f"[DEBUG] Fetched {len(news_headlines)} headlines for {symbol}")

        except Exception as e:
            print(f"[ERROR] News fetch failed: {e}")

        # Mock fallback if empty
        if not news_headlines:
            print(f"[INFO] No news for {symbol}, using mock.")
            news_headlines = [
                f"{symbol} shows strong technicals.",
                f"Analysts upgrade {symbol}.",
                "Market rally continues."
            ]

        # Analyze with Gemini
        prompt = f"""
        You are a financial analyst. Analyze the sentiment of these headlines for {symbol}:
        {news_headlines}
        Return a single word: BUY, SELL, or HOLD.
        """
        
        if not self.gemini_client:
            return "BUY", 0.9 # Fallback

        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-flash-latest', contents=prompt
            )
            decision = response.text.strip().upper()
            
            # Retry on 503
            if "503" in decision or "UNAVAILABLE" in decision: 
                # (Though usually it raises an exception, sometimes text might reflect error)
                raise Exception("Service Unavailable")

        except Exception as e:
            print(f"[Gemini Error] {e}")
            if "503" in str(e):
                time.sleep(5)
                try:
                    print("Retrying Gemini...")
                    response = self.gemini_client.models.generate_content(
                        model='gemini-flash-latest', contents=prompt
                    )
                    decision = response.text.strip().upper()
                except:
                    return "HOLD", 0.0
            else:
                return "HOLD", 0.0

        # Parse logic
        if "BUY" in decision:
            return "BUY", 0.9
        if "SELL" in decision:
            return "SELL", 0.9
        return "HOLD", 0.5

    def submit_buy_order(self, symbol, qty, price):
        """Submit a Market Buy Order."""
        print(f" -> Submitting BUY for {symbol} ({qty} shares)...")
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            self.trading_client.submit_order(order_data)
            self.post_telegram_message(
                f"✅ BUY Executed for {symbol}: {qty} shares @ ~${price:.2f}"
            )
        except Exception as e:
            print(f"[ERROR] Buy Order Failed: {e}")
            self.post_telegram_message(f"⚠️ Buy Failed for {symbol}: {e}")

    def submit_sell_order(self, symbol, qty):
        """Submit a Market Sell Order (Close Position)."""
        print(f" -> Submitting SELL for {symbol} ({qty} shares)...")
        try:
            # We can use close_position to sell all
            self.trading_client.close_position(symbol_or_asset_id=symbol)
            self.post_telegram_message(
                f"🛑 SELL Executed for {symbol} (Sentiment Shift)"
            )
        except Exception as e:
            print(f"[ERROR] Sell Order Failed: {e}")
            self.post_telegram_message(f"⚠️ Sell Failed for {symbol}: {e}")

    def run_trading_cycle(self):
        """Execute one pass through the portfolio."""
        print(f"\n--- Starting Trading Cycle: {datetime.now()} ---")
        
        # Check Market Status
        if not self.check_market_open():
            print("[MARKET CLOSED] Waiting for open...")
            # We continue anyway to test logic, or you can return here.
            # For this task, we'll proceed but log it.

        for symbol in SYMBOLS:
            print(f"\nProcessing {symbol}...")
            
            # 1. Get Price
            price = self.get_price(symbol)
            if price is None:
                continue
            
            # 2. Get Sentiment
            sentiment, confidence = self.get_sentiment(symbol)
            print(f"[{symbol}] Price: ${price:.2f} | Signal: {sentiment} ({confidence})")

            # 3. Check Position
            current_qty = self.get_position(symbol)
            print(f"   Current Position: {current_qty} shares")

            # 4. Decision Logic
            if sentiment == "BUY":
                if current_qty == 0:
                    # Calculate size
                    buy_qty = self.calculate_quantity(price)
                    if buy_qty > 0:
                        self.submit_buy_order(symbol, buy_qty, price)
                    else:
                        print("   [INFO] Insufficient cash.")
                else:
                    print("   [INFO] Already holding. Holding.")

            elif sentiment == "SELL":
                if current_qty > 0:
                    self.submit_sell_order(symbol, current_qty)
                else:
                    print("   [INFO] No position to sell.")

            # Rate Limit Sleep
            time.sleep(2)

        print("\n--- Cycle Complete ---")

if __name__ == "__main__":
    try:
        trader = AITrader()
        
        while True:
            trader.run_trading_cycle()
            print("Sleeping for 1 hour...")
            time.sleep(3600)

    except KeyboardInterrupt:
        print("\n[STOP] AI Trader stopped by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
