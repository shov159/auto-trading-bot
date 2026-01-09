import os
import asyncio
from dotenv import load_dotenv
from alpaca_trade_api.stream import Stream

load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

async def print_quote(q):
    print(f"Quote: {q}")

async def run_connection():
    print("Testing direct WebSocket connection...")
    base_url = 'https://paper-api.alpaca.markets' if ALPACA_PAPER else 'https://api.alpaca.markets'
    feed = 'iex' # Try IEX explicitly which is free
    
    stream = Stream(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        base_url=base_url,
        data_feed=feed,
        raw_data=True
    )

    print(f"Subscribing to SPY quotes on {feed} feed...")
    stream.subscribe_quotes(print_quote, 'SPY')

    print("Starting stream (Press Ctrl+C to stop)...")
    await stream._run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(run_connection())
    except KeyboardInterrupt:
        print("Stopped.")


