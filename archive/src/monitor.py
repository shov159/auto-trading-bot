import os
import logging
import requests
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

class TradeMonitor:
    """
    Handles logging and user notifications via Telegram.
    """

    def __init__(self, log_file: str = "trading_bot.log"):
        # 1. Setup Logging
        self.logger = logging.getLogger("AITrader")
        self.logger.setLevel(logging.INFO)
        
        # Ensure logs directory exists
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # Formatter
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

        # File Handler (Rotating)
        file_handler = RotatingFileHandler(f"logs/{log_file}", maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Force UTF-8 for Windows Console to support Emojis
        if os.name == 'nt':
            try:
                import sys
                import io
                # Reconfigure stdout to utf-8 if possible, or wrap it
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8')
                else:
                    # Fallback for older python or specific envs
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                
                # Update handler stream just in case
                console_handler.stream = sys.stdout
            except Exception:
                pass
                
        self.logger.addHandler(console_handler)

        # 2. Setup Telegram
        self.tg_token = os.getenv("TELEGRAM_TOKEN")
        self.tg_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.tg_token or not self.tg_chat_id:
            self.logger.warning("Telegram credentials missing in .env. Notifications disabled.")

    def log_info(self, msg: str):
        self.logger.info(msg)

    def log_warning(self, msg: str):
        self.logger.warning(msg)

    def log_error(self, msg: str):
        self.logger.error(msg)

    def send_telegram_message(self, text: str):
        """Send message to Telegram channel."""
        if not self.tg_token or not self.tg_chat_id:
            return

        try:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            payload = {
                "chat_id": self.tg_chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            # Timeout is crucial to not block trading loop
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")

    def notify_startup(self):
        self.log_info("Bot Starting Up...")
        msg = (
            "🤖 **הבוט של שובל התעורר!**\n"
            "🚀 המערכת מוכנה למסחר\n"
            "📊 אסטרטגיה: Enhanced V2 (ATR)\n"
            "👀 סורק את השוק..."
        )
        self.send_telegram_message(msg)

    def notify_trade(self, action: str, symbol: str, price: float, qty: int, reason: str):
        """
        Send a trade alert.
        action: 'BUY' or 'SELL'
        """
        if action == "BUY":
            emoji = "🟢"
            title = "נכנסים לעסקה!"
        elif action == "SELL":
            emoji = "🔴"
            title = "יוצאים מעסקה!"
        else:
            emoji = "⚠️"
            title = "פעולה"
        
        msg = (
            f"{emoji} **{title}**\n"
            f"🏷️ מניה: `{symbol}`\n"
            f"💵 מחיר: `${price:.2f}`\n"
            f"📦 כמות: `{qty}`\n"
            f"💡 סיבה: {reason}\n"
            f"⏰ זמן: {datetime.now().strftime('%H:%M:%S')}"
        )
        
        self.log_info(f"TRADE EXECUTION: {action} {symbol} {qty}@{price} ({reason})")
        self.send_telegram_message(msg)

    def notify_error(self, error_msg: str):
        self.log_error(error_msg)
        msg = (
            f"🚨 **שגיאה במערכת!** 🚨\n"
            f"הבוט נתקל בבעיה ועצר רגע לחשוב.\n\n"
            f"השגיאה:\n`{error_msg}`"
        )
        self.send_telegram_message(msg)
