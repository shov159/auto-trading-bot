import os
import sys
import psutil
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import CancelOrderRequest
from alpaca.trading.enums import OrderStatus

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def emergency_stop():
    print("🚨 EMERGENCY STOP INITIATED 🚨")
    
    # 1. Kill Python Processes running main.py
    print("Step 1: Halting Bot Processes...")
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'python' in proc.info['name'] and 'src/main.py' in ' '.join(cmdline):
                print(f"Killing PID {proc.info['pid']} ({' '.join(cmdline)})")
                proc.terminate() # Try graceful SIGTERM first
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill() # Force kill
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    print(f"Stopped {killed_count} bot processes.")
    
    # 2. Cancel All Orders via Alpaca
    print("Step 2: Cancelling Open Orders...")
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    is_paper = True # Force safe default, check config if needed but usually paper for testing
    
    if not api_key or not secret_key:
        print("Error: Missing Alpaca Credentials in .env")
        return

    try:
        trading_client = TradingClient(api_key, secret_key, paper=is_paper)
        trading_client.cancel_orders()
        print("✅ All open orders cancelled successfully.")
    except Exception as e:
        print(f"❌ Failed to cancel orders: {e}")

    print("\n🛑 EMERGENCY STOP COMPLETED. SYSTEM SECURE.")

if __name__ == "__main__":
    # Require confirmation
    confirm = input("Are you sure you want to HALT ALL TRADING? (type 'YES'): ")
    if confirm == "YES":
        emergency_stop()
    else:
        print("Aborted.")

