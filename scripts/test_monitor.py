import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.monitor import TradeMonitor
from dotenv import load_dotenv

def test_notifications():
    load_dotenv()
    
    print("Initializing Monitor...")
    monitor = TradeMonitor()
    
    # 1. Test Console/File Logging
    monitor.log_info("Test Info Message")
    monitor.log_warning("Test Warning Message")
    monitor.log_error("Test Error Message")
    print("[OK] Logging test complete (check logs/trading_bot.log)")
    
    # 2. Test Telegram
    print("Sending Telegram Startup...")
    monitor.notify_startup()
    
    print("Sending Telegram Trade...")
    monitor.notify_trade("BUY", "TEST", 150.00, 10, "Test Script Execution")
    
    print("Sending Telegram Error...")
    monitor.notify_error("This is a simulated error for testing.")
    
    print("[OK] Telegram test complete (check your phone)")

if __name__ == "__main__":
    test_notifications()

