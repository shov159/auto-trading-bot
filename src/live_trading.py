"""
Live/Paper Trading Script for Alpaca.
Connects the RegimeStrategy to Alpaca Brokerage.
"""
import os
from datetime import datetime, timedelta
import backtrader as bt
from alpaca_backtrader_api import AlpacaStore
from dotenv import load_dotenv
from src.strategy import RegimeStrategy
from src.scanner import MarketScanner

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

TIMEFRAME = bt.TimeFrame.Days
COMPRESSION = 1

def run_live_trading():
    """Runs the strategy in live/paper mode."""
    print("Starting Live Trading Bot...")

    # Check credentials
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not found in .env file.")
        return

    # 1. Market Scanner
    print("Initializing Market Scanner...")
    scanner = MarketScanner()
    top_picks = scanner.get_top_picks(top_n=10)

    # Always include SPY for Regime Filter
    if 'SPY' not in top_picks:
        tickers = ['SPY'] + top_picks
    else:
        # Ensure SPY is first
        top_picks.remove('SPY')
        tickers = ['SPY'] + top_picks

    print(f"Trading Universe: {tickers}")

    cerebro = bt.Cerebro()

    # 2. Configure Alpaca Store
    print(f"Connecting to Alpaca ({'PAPER' if ALPACA_PAPER else 'LIVE'})...")
    # pylint: disable=unexpected-keyword-arg
    store = AlpacaStore(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=ALPACA_PAPER,
        usePolygon=False # Use Alpaca data
    )

    # 3. Set Broker
    # use_positions=True allows Cerebro to read existing positions from Alpaca on start
    cerebro.setbroker(store.getbroker())

    # 4. Add Data Feeds
    # We need historical data for indicators (SMA200, ATR14)
    # Fetching 1 year of history
    from_date = datetime.now() - timedelta(days=365)

    print("Adding Data Feeds...")
    for ticker in tickers:
        # pylint: disable=unexpected-keyword-arg
        data = store.getdata(
            dataname=ticker,
            name=ticker,
            timeframe=TIMEFRAME,
            compression=COMPRESSION,
            fromdate=from_date,
            historical=True, # Backfill historical data then switch to live
            use_polygon=False
        )
        cerebro.adddata(data)
        print(f"  - Added {ticker}")

    # 5. Add Strategy
    # We pass explicit params if needed, or rely on defaults/logic in strategy
    print("Initializing Strategy...")
    cerebro.addstrategy(RegimeStrategy)

    # 6. Run
    print("Bot is running. Press Ctrl+C to stop.")
    try:
        cerebro.run()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_live_trading()
