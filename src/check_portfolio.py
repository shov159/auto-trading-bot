import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

def check_portfolio():
    """
    Connects to Alpaca and prints current account status, positions, and orders.
    """
    print(f"Connecting to Alpaca ({'PAPER' if ALPACA_PAPER else 'LIVE'})...")
    
    # Initialize REST API
    base_url = 'https://paper-api.alpaca.markets' if ALPACA_PAPER else 'https://api.alpaca.markets'
    
    try:
        api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=base_url,
            api_version='v2'
        )
    except Exception as e:
        print(f"Failed to initialize Alpaca API: {e}")
        return

    try:
        # 1. Account Info
        account = api.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        cash = float(account.cash)
        
        print("\n=== Account Snapshot ===")
        print(f"Status:       {account.status}")
        print(f"Equity:       ${equity:,.2f}")
        print(f"Buying Power: ${buying_power:,.2f}")
        print(f"Cash:         ${cash:,.2f}")
        print(f"Day Trades:   {account.daytrade_count}")

        # 2. Open Positions
        positions = api.list_positions()
        print(f"\n=== Open Positions ({len(positions)}) ===")
        
        if positions:
            # Header
            print(f"{'Symbol':<8} | {'Qty':<8} | {'Entry':<10} | {'Current':<10} | {'P/L ($)':<10} | {'P/L (%)':<10}")
            print("-" * 70)
            
            total_pl = 0.0
            for p in positions:
                symbol = p.symbol
                qty = float(p.qty)
                entry = float(p.avg_entry_price)
                current = float(p.current_price)
                pl = float(p.unrealized_pl)
                pl_pct = float(p.unrealized_plpc) * 100
                total_pl += pl
                
                print(f"{symbol:<8} | {qty:<8} | ${entry:<9.2f} | ${current:<9.2f} | ${pl:<9.2f} | {pl_pct:+.2f}%")
            
            print("-" * 70)
            print(f"Total Open P/L: ${total_pl:,.2f}")
        else:
            print("No open positions.")

        # 3. Open Orders
        orders = api.list_orders(status='open')
        print(f"\n=== Open Orders ({len(orders)}) ===")
        
        if orders:
            print(f"{'Symbol':<8} | {'Side':<6} | {'Qty':<8} | {'Type':<8} | {'Status':<10}")
            print("-" * 50)
            for o in orders:
                qty = o.qty if o.qty else 'N/A'
                print(f"{o.symbol:<8} | {o.side:<6} | {qty:<8} | {o.type:<8} | {o.status:<10}")
        else:
            print("No open orders.")
            
    except Exception as e:
        print(f"Error fetching portfolio data: {e}")

if __name__ == "__main__":
    check_portfolio()

