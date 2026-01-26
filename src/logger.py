"""
Rich Logger - Hacker Dashboard Style Logging
Provides colorful, structured console output for debugging.
"""
import os
import logging
import traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ==================== COLOR SETUP ====================
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback - no colors
    class Fore:
        CYAN = YELLOW = GREEN = RED = MAGENTA = WHITE = BLUE = RESET = ""
    class Back:
        BLACK = RESET = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""

# ==================== DEBUG MODE ====================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


class RichLogger:
    """
    Rich console logger with colors and structured output.
    
    Categories:
    - AI: Cyan - AI thinking, prompts, responses
    - WARN: Yellow - Warnings, validation adjustments
    - OK: Green - Success, valid trades, orders filled
    - ERROR: Red - Errors, failures, low conviction
    - INFO: White - General information
    - DEBUG: Magenta - Debug details (only in debug mode)
    """
    
    def __init__(self, name: str = "Bot"):
        self.name = name
        self.debug_mode = DEBUG_MODE
        
        # Also setup standard logger for file output
        self.file_logger = logging.getLogger(f"Rich_{name}")
        self.file_logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
    
    def _timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%H:%M:%S")
    
    def _print(self, category: str, color: str, emoji: str, message: str, 
               bright: bool = False, include_trace: bool = False):
        """Internal print method with formatting."""
        timestamp = self._timestamp()
        style = Style.BRIGHT if bright else ""
        
        # Format: [HH:MM:SS] [CATEGORY] emoji message
        prefix = f"{Fore.WHITE}[{timestamp}]{Style.RESET_ALL} "
        cat_str = f"{color}{style}[{category}]{Style.RESET_ALL} "
        msg_str = f"{color}{emoji} {message}{Style.RESET_ALL}"
        
        print(prefix + cat_str + msg_str)
        
        # Log to file as well
        self.file_logger.info(f"[{category}] {emoji} {message}")
        
        # Include stack trace in debug mode
        if include_trace and self.debug_mode:
            trace = traceback.format_exc()
            if trace and "NoneType" not in trace:
                print(f"{Fore.RED}{Style.DIM}{trace}{Style.RESET_ALL}")
    
    # ==================== PUBLIC METHODS ====================
    
    def ai(self, message: str):
        """AI-related messages (thinking, prompts, responses)."""
        self._print("AI", Fore.CYAN, "🧠", message, bright=True)
    
    def ai_raw(self, json_str: str):
        """Print raw AI response JSON."""
        self._print("AI", Fore.CYAN, "📩", "Raw JSON Response:")
        # Print JSON with dim formatting
        print(f"{Fore.CYAN}{Style.DIM}{json_str[:1000]}{'...' if len(json_str) > 1000 else ''}{Style.RESET_ALL}")
    
    def warn(self, message: str):
        """Warning messages."""
        self._print("WARN", Fore.YELLOW, "⚠️", message)
    
    def ok(self, message: str):
        """Success messages."""
        self._print("OK", Fore.GREEN, "✅", message, bright=True)
    
    def error(self, message: str, include_trace: bool = True):
        """Error messages."""
        self._print("ERROR", Fore.RED, "❌", message, bright=True, include_trace=include_trace)
    
    def info(self, message: str):
        """General info messages."""
        self._print("INFO", Fore.WHITE, "ℹ️", message)
    
    def debug(self, message: str):
        """Debug messages (only shown in debug mode)."""
        if self.debug_mode:
            self._print("DEBUG", Fore.MAGENTA, "🔍", message)
    
    def trade(self, action: str, ticker: str, details: str = ""):
        """Trade-related messages."""
        if action.upper() == "BUY":
            self._print("TRADE", Fore.GREEN, "🟢", f"BUY {ticker} {details}", bright=True)
        elif action.upper() == "SELL":
            self._print("TRADE", Fore.RED, "🔴", f"SELL {ticker} {details}", bright=True)
        else:
            self._print("TRADE", Fore.YELLOW, "⏸️", f"PASS {ticker} {details}")
    
    def validation(self, passed: bool, message: str):
        """Kill switch validation messages."""
        if passed:
            self._print("KILL-SWITCH", Fore.GREEN, "✅", message, bright=True)
        else:
            self._print("KILL-SWITCH", Fore.RED, "🚫", message)
    
    def telegram(self, message: str):
        """Telegram-related messages."""
        self._print("TELEGRAM", Fore.BLUE, "📤", message)
    
    def alpaca(self, message: str, success: bool = True):
        """Alpaca/execution messages."""
        if success:
            self._print("ALPACA", Fore.GREEN, "🚀", message, bright=True)
        else:
            self._print("ALPACA", Fore.RED, "💥", message)
    
    def scanner(self, message: str):
        """Market scanner messages."""
        self._print("SCANNER", Fore.MAGENTA, "🔎", message)
    
    def divider(self, title: str = ""):
        """Print a visual divider."""
        line = "═" * 50
        if title:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}╔{line}╗{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}║  {title:^46}  ║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}╚{line}╝{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.CYAN}{line}{Style.RESET_ALL}")


# ==================== SINGLETON INSTANCE ====================
_logger_instance = None

def get_logger(name: str = "Bot") -> RichLogger:
    """Get or create logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = RichLogger(name)
    return _logger_instance


# ==================== QUICK ACCESS FUNCTIONS ====================
def log_ai(msg: str): get_logger().ai(msg)
def log_ai_raw(json_str: str): get_logger().ai_raw(json_str)
def log_warn(msg: str): get_logger().warn(msg)
def log_ok(msg: str): get_logger().ok(msg)
def log_error(msg: str, trace: bool = True): get_logger().error(msg, trace)
def log_info(msg: str): get_logger().info(msg)
def log_debug(msg: str): get_logger().debug(msg)
def log_trade(action: str, ticker: str, details: str = ""): get_logger().trade(action, ticker, details)
def log_validation(passed: bool, msg: str): get_logger().validation(passed, msg)
def log_telegram(msg: str): get_logger().telegram(msg)
def log_alpaca(msg: str, success: bool = True): get_logger().alpaca(msg, success)
def log_scanner(msg: str): get_logger().scanner(msg)
def log_divider(title: str = ""): get_logger().divider(title)


if __name__ == "__main__":
    # Demo the logger
    logger = get_logger("Demo")
    
    logger.divider("RICH LOGGER DEMO")
    
    logger.ai("Analyzing NVDA...")
    logger.ai_raw('{"action": "BUY", "ticker": "NVDA", "plan": {"buy_zone": "$125-$127"}}')
    logger.validation(True, "Trade Validated (R/R: 3.5)")
    logger.trade("BUY", "NVDA", "@ $126.50 | 10 shares")
    logger.telegram("Alert sent to user")
    logger.alpaca("Order submitted: #abc123")
    
    logger.divider()
    
    logger.warn("Volume below average")
    logger.error("API connection failed")
    logger.info("Waiting for market open...")
    logger.debug("This only shows in debug mode")
    
    logger.divider("END DEMO")

