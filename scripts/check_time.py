import os
import sys
import yaml
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_time():
    load_dotenv()
    
    # 1. Local Machine Time
    print(f"Local System Time: {datetime.now()}")
    
    # 2. Calculated NY Time (Current Logic)
    try:
        import pytz
        tz_ny = pytz.timezone('America/New_York')
        now_ny = datetime.now(tz_ny)
        print(f"Calculated NY Time: {now_ny}")
    except Exception as e:
        print(f"Error calculating NY Time: {e}")

    # 3. Alpaca Server Clock
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key:
        print("Missing Alpaca Keys")
        return

    try:
        client = TradingClient(api_key, secret_key, paper=True)
        clock = client.get_clock()
        print(f"Alpaca Server Time: {clock.timestamp}")
        print(f"Alpaca Is Open: {clock.is_open}")
        print(f"Next Open: {clock.next_open}")
        print(f"Next Close: {clock.next_close}")
    except Exception as e:
        print(f"Alpaca API Error: {e}")

if __name__ == "__main__":
    check_time()

