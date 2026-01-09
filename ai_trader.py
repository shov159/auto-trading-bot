"""
AI Trader Module
"""
import os
from datetime import datetime
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from openai import OpenAI
from dotenv import load_dotenv
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from alpaca.trading.client import TradingClient

# Load environment variables
load_dotenv()

# Constants
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class SentimentStrategy(Strategy):
    """
    A Strategy that combines Technical Analysis (RSI) with AI-based Sentiment Analysis.
    """
    # pylint: disable=attribute-defined-outside-init, arguments-differ

    def initialize(self):
        """
        Initialize the strategy with the trading symbol and risk parameter.
        """
        self.symbol = self.parameters.get("symbol", "SPY")
        self.cash_at_risk = self.parameters.get("cash_at_risk", 0.5)
        self.sleeptime = "1H" # Trading frequency
        self.last_trade = None

        # Initialize OpenAI Client (Mock if key is missing)
        if OPENAI_API_KEY:
            self.api = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.api = None
            print("WARNING: No OpenAI API Key found. Using mock sentiment.")

        # Initialize Alpaca Clients
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            self.news_client = NewsClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY
            )
            self.trading_client = TradingClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                paper=True
            )
        else:
            self.news_client = None
            self.trading_client = None
            print("WARNING: No Alpaca Credentials found. News/Trading checks will fail.")

    def position_sizing(self):
        """
        Calculate the quantity of shares to buy based on cash at risk.
        """
        # pylint: disable=no-member
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return last_price, quantity

    def get_sentiment(self):
        """
        Fetch news from Alpaca and analyze sentiment using GPT-4o.
        Returns: Tuple(Signal, Confidence)
        """
        # 1. Fetch News from Alpaca
        news_headlines = []
        if self.news_client:
            try:
                request = NewsRequest(
                    symbols=self.symbol,
                    limit=10
                )
                # pylint: disable=no-member
                news_items = self.news_client.get_news(request).news
                news_headlines = [item.headline for item in news_items]
                print(f"[DEBUG] Headlines fetched for {self.symbol}:")
                for _, h in enumerate(news_headlines[:5]): # Print first 5
                    print(f"  - {h}")
            except Exception as e: # pylint: disable=broad-exception-caught
                print(f"Error fetching Alpaca news: {e}")

        # Fallback if no news found or client missing
        if not news_headlines:
            print("No real news found. Using mock data for fallback.")
            news_headlines = [
                f"Market data for {self.symbol} suggests strong upward momentum.",
                f"Analysts upgrade {self.symbol} price target based on recent earnings beat.",
                "Tech sector showing resilience despite inflation concerns."
            ]

        prompt = f"""
        You are a financial analyst. Analyze the sentiment of the following news headlines for {self.symbol}:
        {news_headlines}

        Return a single word: BUY, SELL, or HOLD.
        """

        # 2. Call LLM
        if not self.api:
            return "BUY", 0.95 # Mock fallback

        try:
            response = self.api.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful financial trading assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] OpenAI Raw Response: {content}")

            if "BUY" in content:
                return "BUY", 0.9
            if "SELL" in content:
                return "SELL", 0.9
            return "HOLD", 0.5
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Error fetching sentiment: {e}")
            return "HOLD", 0.0

    def on_trading_iteration(self):
        """
        Main trading logic executed every iteration (sleeptime).
        """
        try:
            # Check Market Status
            is_market_open = True
            if self.trading_client:
                try:
                    clock = self.trading_client.get_clock()
                    is_market_open = clock.is_open
                except Exception as e: # pylint: disable=broad-exception-caught
                    print(f"Error checking market clock: {e}")

            # pylint: disable=no-member
            last_price, quantity = self.position_sizing()
            sentiment, confidence = self.get_sentiment()

            # 3. Calculate Technical Indicators (RSI)
            # Fetch 30 days of daily data
            bars = self.get_historical_prices(self.symbol, 30, "day")
            df = bars.df

            # Simple RSI Calculation (14-period)
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50

            dt_str = self.get_datetime().strftime("%Y-%m-%d %H:%M")
            print(
                f"[{dt_str}] {self.symbol} Price: {last_price:.2f} | "
                f"RSI: {current_rsi:.2f} | AI: {sentiment} ({confidence})"
            )

            # 4. Hybrid Decision Logic
            # BUY: AI says BUY AND RSI is not Overbought (>70)
            if sentiment == "BUY" and current_rsi < 70:
                if self.last_trade == "sell":
                    self.sell_all()

                if self.get_position(self.symbol) is None: # Only buy if no position
                    if is_market_open:
                        order = self.create_order(
                            self.symbol,
                            quantity,
                            "buy",
                            take_profit_price=last_price * 1.05, # 5% Take Profit
                            stop_loss_price=last_price * 0.98   # 2% Stop Loss
                        )
                        self.submit_order(order)
                        self.last_trade = "buy"
                        print(f" -> EXECUTING BUY ORDER ({quantity} shares)")
                    else:
                        print(
                            f"[MARKET CLOSED] AI Signal is BUY. "
                            f"If market were open, I would buy {quantity} shares of {self.symbol}."
                        )

            # SELL: AI says SELL OR RSI is Overbought (>70)
            elif sentiment == "SELL" or current_rsi > 70:
                if self.last_trade == "buy" or self.get_position(self.symbol):
                    if is_market_open:
                        self.sell_all()
                        self.last_trade = "sell"
                        print(" -> EXECUTING SELL/CLOSE ORDER")
                    else:
                        print(
                            "[MARKET CLOSED] Signal is SELL. "
                            "If market were open, I would close position."
                        )

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"[ERROR] Strategy Loop Failed: {e}")

if __name__ == "__main__":
    # Configuration
    IS_LIVE = True # Change to True to connect to Alpaca

    if IS_LIVE:
        # Live / Paper Trading Mode
        print("Starting Live Trading Agent...")
        broker = Alpaca({
            "API_KEY": ALPACA_API_KEY,
            "API_SECRET": ALPACA_SECRET_KEY,
            "PAPER": True  # Force Paper Trading
        })

        strategy = SentimentStrategy(
            broker=broker,
            parameters={"symbol": "SPY", "cash_at_risk": 0.5}
        )

        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()

    else:
        # Backtesting Mode
        print("Starting AI Strategy Backtest...")
        backtesting_start = datetime(2023, 1, 1)
        backtesting_end = datetime(2023, 12, 31)

        SentimentStrategy.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            parameters={"symbol": "SPY", "cash_at_risk": 0.5}
        )
