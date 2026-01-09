import os
import backtrader as bt
from alpaca_backtrader_api import AlpacaStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

class TestBuyStrategy(bt.Strategy):
    """
    Simple strategy to test order execution.
    Buys 1 share of SPY if no position exists.
    """
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')

    def next(self):
        # Simply buy 1 share if we don't have a position
        if not self.position:
            self.log("No position found. Sending TEST BUY ORDER for 1 share of SPY...")
            self.buy(size=1)
            self.log("TEST BUY ORDER SENT")
        else:
            # If we already have a position, just log it occasionally
            self.log(f"Position confirmed: {self.position.size} shares of SPY.")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            self.log(f"Order Status: {order.getstatusname()}")
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED at: ${order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED at: ${order.executed.price:.2f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Failed: {order.getstatusname()}")

def run_test_execution():
    print("Starting Live Execution Test...")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not found in .env file.")
        return

    cerebro = bt.Cerebro()

    # Configure Alpaca Store
    print(f"Connecting to Alpaca ({'PAPER' if ALPACA_PAPER else 'LIVE'})...")
    store = AlpacaStore(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=ALPACA_PAPER,
        usePolygon=False
    )
    
    # Set Broker
    cerebro.setbroker(store.getbroker())

    # Add Data Feed
    # We use Minute timeframe to get updates quickly if market is open
    print("Adding Data Feed for SPY (Live)...")
    data = store.getdata(
        dataname='SPY',
        timeframe=bt.TimeFrame.Minutes,
        historical=False, # Stream live data
        use_polygon=False
    )
    cerebro.adddata(data)

    cerebro.addstrategy(TestBuyStrategy)

    print("Waiting for market data (Market must be OPEN for live updates)...")
    print("Watch your Alpaca Dashboard for the order.")
    print("Press Ctrl+C to stop.")
    
    try:
        cerebro.run()
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_test_execution()

