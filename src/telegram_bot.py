"""
Telegram Bot - CIO Trading Assistant
Handles /analyze commands, autonomous watchlist scanning, and ONE-CLICK EXECUTION.
"""
import os
import sys
import json
import logging
import threading
import uuid
import time as time_module
from datetime import datetime, time as dt_time
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import requests

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("TelegramBot")

# Import AI Brain & Market Scanner
from src.ai_brain import get_brain, AIBrain
from src.market_scanner import get_scanner, MarketScanner
from src.logger import (
    get_logger, log_ai, log_ok, log_error, log_warn, log_info,
    log_debug, log_telegram, log_alpaca, log_scanner, log_divider,
    log_trade, log_validation
)
from src.rate_limiter import (
    get_scan_lock, get_deduplicator, ScanLock, TickerDeduplicator
)
from src.analysis_cache import get_analysis_cache, AnalysisCache
from src.news_brain import get_news_brain, NewsBrain
from src.learning_engine import get_learning_engine


class TelegramCIOBot:
    """
    Telegram bot for CIO trading analysis with ONE-CLICK EXECUTION.

    Supports:
    - /analyze TICKER - Manual analysis with execution buttons
    - /watchlist - Show current watchlist
    - /add TICKER - Add to watchlist
    - /remove TICKER - Remove from watchlist
    - /scan - Force scan all watchlist
    - /autoscan - Auto-detect stocks in play and analyze
    - /status - Bot status
    - /risk [amount] - Set risk per trade
    - /positions - Show open positions

    Button callbacks:
    - EXECUTE_{trade_id} - Execute the pending trade
    - DISMISS_{trade_id} - Dismiss the alert

    Scheduled:
    - Pre-market scan at 16:00 Israel time (9:00 AM NY)
    """

    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.token or not self.chat_id:
            logger.error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in environment")
            raise ValueError("Telegram credentials not configured")

        self.api_base = f"https://api.telegram.org/bot{self.token}"
        self.brain: AIBrain = get_brain()
        self.scanner: MarketScanner = get_scanner()

        # ==================== TURBO MODE / AUTO-TRADING (must be first!) ====================
        self.auto_trade_enabled = os.getenv("AUTO_TRADE", "false").lower() == "true"
        self.turbo_mode = os.getenv("TURBO_MODE", "false").lower() == "true"
        self.auto_trade_min_conviction = os.getenv("AUTO_TRADE_MIN_CONVICTION", "HIGH").upper()

        # ==================== RATE LIMITING & DEDUPLICATION ====================
        self.scan_lock: ScanLock = get_scan_lock()
        self.deduplicator: TickerDeduplicator = get_deduplicator(cooldown=120.0)

        # Cache settings (TURBO MODE = 10 min, Normal = 60 min)
        cache_age = 10.0 if self.turbo_mode else 60.0
        self.analysis_cache: AnalysisCache = get_analysis_cache(
            max_age_minutes=cache_age,
            price_threshold_pct=0.5,
            cache_file="cache/analysis_cache.json"
        )
        self.news_brain: NewsBrain = get_news_brain()

        # ==================== ALPACA SETUP ====================
        self.alpaca_client = None
        self._init_alpaca()

        # ==================== PENDING TRADES (for callback buttons) ====================
        self.pending_trades: Dict[str, Dict[str, Any]] = {}

        # ==================== RISK SETTINGS ====================
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", "50"))
        self.risk_percentage = float(os.getenv("RISK_PERCENTAGE", "0.005"))
        self.max_risk_amount = float(os.getenv("MAX_RISK_AMOUNT", "2000"))
        self.use_dynamic_risk = os.getenv("USE_DYNAMIC_RISK", "true").lower() == "true"
        self.is_paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

        # Cached account info for dynamic risk
        self._cached_equity = None
        self._equity_cache_time = None

        # Default watchlist (can be extended)
        self.watchlist = [
            "SPY", "QQQ", "NVDA", "TSLA", "AMD",
            "AAPL", "MSFT", "AMZN", "META", "GOOGL"
        ]

        # Autonomous scan settings (TURBO MODE = 5 min, Normal = 30 min)
        self.scan_interval_minutes = 5 if self.turbo_mode else 30
        self.last_scan_time = None
        self.last_autoscan_time = None
        self.premarket_scan_done_today = False
        self.is_running = False
        self.last_update_id = 0

        logger.info(f"TelegramCIOBot initialized with {len(self.watchlist)} symbols")
        logger.info(f"AI Provider: {self.brain.provider}")
        # Log risk settings
        if self.use_dynamic_risk:
            logger.info(f"Risk Mode: DYNAMIC ({self.risk_percentage * 100:.2f}% of equity, max ${self.max_risk_amount})")
        else:
            logger.info(f"Risk Mode: FIXED (${self.risk_per_trade})")
        logger.info(f"Alpaca Mode: {'PAPER' if self.is_paper else 'LIVE'}")

        # Log turbo/auto-trade settings
        if self.turbo_mode:
            logger.info(f"⚡ TURBO MODE: ENABLED (scan every {self.scan_interval_minutes} min)")
        if self.auto_trade_enabled:
            logger.info(f"🤖 AUTO-TRADE: ENABLED (min conviction: {self.auto_trade_min_conviction})")
        logger.info("Market Scanner initialized")

    def _init_alpaca(self):
        """Initialize Alpaca trading client."""
        try:
            from alpaca.trading.client import TradingClient

            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            is_paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

            if api_key and secret_key:
                self.alpaca_client = TradingClient(api_key, secret_key, paper=is_paper)
                logger.info("Alpaca client initialized successfully")
            else:
                logger.warning("Alpaca credentials not found - execution disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca: {e}")
            self.alpaca_client = None

    def _generate_trade_id(self) -> str:
        """Generate short unique ID for pending trade."""
        return str(uuid.uuid4())[:8]

    def _get_account_equity(self) -> Optional[float]:
        """
        Fetch current account equity from Alpaca.
        Caches the result for 60 seconds to avoid excessive API calls.
        """
        if not self.alpaca_client:
            return None

        # Check cache (60 second TTL)
        now = datetime.now()
        if self._cached_equity and self._equity_cache_time:
            age = (now - self._equity_cache_time).total_seconds()
            if age < 60:
                return self._cached_equity

        try:
            account = self.alpaca_client.get_account()
            equity = float(account.equity)

            # Update cache
            self._cached_equity = equity
            self._equity_cache_time = now

            return equity
        except Exception as e:
            log_error(f"Failed to fetch account equity: {e}")
            return self._cached_equity  # Return stale cache on error

    def _calculate_dynamic_risk(self) -> tuple[float, str]:
        """
        Calculate risk amount based on account equity percentage.

        Returns:
            (risk_amount, description) - The risk amount and a description string
        """
        if not self.use_dynamic_risk:
            return self.risk_per_trade, f"Fixed ${self.risk_per_trade:.2f}"

        equity = self._get_account_equity()

        if not equity or equity <= 0:
            # Fallback to fixed risk if we can't get equity
            log_warn("Could not fetch equity, using fixed risk amount")
            return self.risk_per_trade, f"Fixed ${self.risk_per_trade:.2f} (fallback)"

        # Calculate percentage-based risk
        raw_risk = equity * self.risk_percentage

        # Cap at max risk amount
        risk_amount = min(raw_risk, self.max_risk_amount)

        # Format description
        pct_str = f"{self.risk_percentage * 100:.2f}%"
        desc = f"${risk_amount:.2f} ({pct_str} of ${equity:,.2f})"

        if raw_risk > self.max_risk_amount:
            desc += f" [CAPPED from ${raw_risk:.2f}]"

        log_info(f"Dynamic Risk: {desc}")

        return risk_amount, desc

    def _calculate_position_size(self, entry: float, stop_loss: float) -> tuple[int, float, str]:
        """
        Calculate position size based on dynamic or fixed risk amount.

        Args:
            entry: Entry price
            stop_loss: Stop loss price

        Returns:
            (qty, risk_amount, risk_description) - Position size, risk used, and description
        """
        if entry <= 0 or stop_loss <= 0:
            return 0, 0, "Invalid prices"

        risk_per_share = abs(entry - stop_loss)
        if risk_per_share <= 0:
            return 0, 0, "Invalid risk per share"

        # Get dynamic risk amount
        risk_amount, risk_desc = self._calculate_dynamic_risk()

        # Calculate quantity
        qty = int(risk_amount / risk_per_share)

        # Minimum 1 share, but warn if position is too expensive
        if qty < 1:
            log_warn(f"Risk too small for position: ${risk_amount:.2f} / ${risk_per_share:.2f} per share = {risk_amount/risk_per_share:.2f} shares")
            qty = 1

        return qty, risk_amount, risk_desc

    # ==================== TELEGRAM API METHODS ====================

    def send_message(self, text: str, parse_mode: str = "Markdown",
                     reply_markup: Optional[dict] = None) -> Optional[int]:
        """Send message to Telegram. Returns message_id if successful."""
        try:
            url = f"{self.api_base}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            if reply_markup:
                payload["reply_markup"] = json.dumps(reply_markup)

            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("result", {}).get("message_id")
            return None
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None

    def edit_message(self, message_id: int, text: str,
                     parse_mode: str = "Markdown",
                     reply_markup: Optional[dict] = None) -> bool:
        """Edit an existing message."""
        try:
            url = f"{self.api_base}/editMessageText"
            payload = {
                "chat_id": self.chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode
            }
            if reply_markup:
                payload["reply_markup"] = json.dumps(reply_markup)

            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to edit message: {e}")
            return False

    def answer_callback(self, callback_id: str, text: str = "", show_alert: bool = False) -> bool:
        """Answer a callback query (acknowledge button press)."""
        try:
            url = f"{self.api_base}/answerCallbackQuery"
            payload = {
                "callback_query_id": callback_id,
                "text": text,
                "show_alert": show_alert
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to answer callback: {e}")
            return False

    def get_updates(self, offset: int = 0) -> List[dict]:
        """Get new messages and callbacks from Telegram."""
        try:
            url = f"{self.api_base}/getUpdates"
            params = {
                "offset": offset,
                "timeout": 30,
                "allowed_updates": ["message", "callback_query"]
            }
            response = requests.get(url, params=params, timeout=35)
            if response.status_code == 200:
                data = response.json()
                return data.get("result", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get updates: {e}")
            return []

    # ==================== INLINE KEYBOARD BUILDER ====================

    def _build_execution_keyboard(self, trade_id: str, ticker: str,
                                   action: str, qty: int, risk: float) -> dict:
        """Build inline keyboard with Execute and Dismiss buttons."""

        action_emoji = "🟢" if action == "BUY" else "🔴"

        keyboard = {
            "inline_keyboard": [
                [
                    {
                        "text": f"{action_emoji} {action} {ticker} (~{qty} shares, ${risk:.0f} risk)",
                        "callback_data": f"EXEC_{trade_id}"
                    }
                ],
                [
                    {
                        "text": "❌ Dismiss",
                        "callback_data": f"DISMISS_{trade_id}"
                    }
                ]
            ]
        }
        return keyboard

    # ==================== TRADE EXECUTION ====================

    def execute_bracket_order(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a bracket order via Alpaca.
        Returns dict with success status and order details.
        """
        if not self.alpaca_client:
            return {"success": False, "error": "Alpaca client not initialized"}

        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

            ticker = trade_data.get("ticker")
            action = trade_data.get("action", "BUY").upper()
            qty = trade_data.get("qty", 1)
            stop_loss = trade_data.get("stop_loss")
            take_profit = trade_data.get("take_profit")

            # Determine side
            side = OrderSide.BUY if action == "BUY" else OrderSide.SELL

            # Build bracket order
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss={"stop_price": round(stop_loss, 2)},
                take_profit={"limit_price": round(take_profit, 2)}
            )

            # Submit order
            order = self.alpaca_client.submit_order(order_data)

            logger.info(f"Order submitted: {order.id} - {action} {qty} {ticker}")

            return {
                "success": True,
                "order_id": str(order.id),
                "ticker": ticker,
                "action": action,
                "qty": qty,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== CALLBACK HANDLER ====================

    def handle_callback(self, callback_query: dict):
        """Handle button click callbacks."""
        callback_id = callback_query.get("id")
        data = callback_query.get("data", "")
        message = callback_query.get("message", {})
        message_id = message.get("message_id")
        chat_id = message.get("chat", {}).get("id")

        # Security check
        if str(chat_id) != str(self.chat_id):
            self.answer_callback(callback_id, "⛔ Unauthorized", show_alert=True)
            return

        logger.info(f"Callback received: {data}")

        # Parse callback data
        if data.startswith("EXEC_"):
            trade_id = data.replace("EXEC_", "")
            self._handle_execute(callback_id, message_id, trade_id)

        elif data.startswith("DISMISS_"):
            trade_id = data.replace("DISMISS_", "")
            self._handle_dismiss(callback_id, message_id, trade_id)

        else:
            self.answer_callback(callback_id, "Unknown action")

    def _handle_execute(self, callback_id: str, message_id: int, trade_id: str):
        """Handle EXECUTE button click."""
        log_divider("TRADE EXECUTION")
        log_info(f"Execute button clicked for trade_id: {trade_id}")

        # Retrieve pending trade
        trade_data = self.pending_trades.get(trade_id)

        if not trade_data:
            log_warn(f"Trade not found: {trade_id}")
            self.answer_callback(callback_id, "⚠️ Trade expired or not found", show_alert=True)
            return

        ticker = trade_data.get("ticker")
        action = trade_data.get("action")
        qty = trade_data.get("qty")
        is_test = trade_data.get("is_test", False)

        if is_test:
            log_warn("🧪 TEST MODE - This is a test signal!")

        log_trade(action, ticker, f"{qty} shares @ Entry={trade_data.get('entry')} Stop={trade_data.get('stop_loss')}")

        # Acknowledge click immediately
        self.answer_callback(callback_id, "⏳ Submitting order...")

        # Execute the trade
        log_alpaca(f"Submitting bracket order to Alpaca...", success=True)
        result = self.execute_bracket_order(trade_data)

        if result["success"]:
            log_alpaca(f"Order submitted successfully! ID: {result['order_id']}", success=True)
            log_ok(f"🎉 Trade executed: {action} {qty}x {ticker}")

            # Get risk description
            risk_desc = trade_data.get("risk_desc", f"${trade_data.get('risk', 0):.2f}")

            success_msg = f"""✅ **ORDER SUBMITTED!**

🎯 **{action}** {qty} x **{ticker}**
📋 Order ID: `{result['order_id']}`

🛡️ Bracket Order Set:
• Stop Loss: ${result['stop_loss']:.2f}
• Take Profit: ${result['take_profit']:.2f}

💰 Risk: {risk_desc}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            if is_test:
                success_msg = "🧪 **TEST MODE**\n\n" + success_msg

            self.edit_message(message_id, success_msg)
            log_telegram("Success message sent")

            # LOG TRADE TO LEARNING ENGINE
            try:
                get_learning_engine().log_trade_entry(trade_data, result['order_id'])
            except Exception as e:
                log_error(f"Failed to log trade to Learning Engine: {e}")

            # Remove from pending
            del self.pending_trades[trade_id]

        else:
            log_alpaca(f"Order FAILED: {result.get('error')}", success=False)
            log_error(f"Execution failed: {result.get('error')}")

            # Show error
            error_msg = f"""❌ **ORDER FAILED**

Error: `{result.get('error', 'Unknown error')}`

Please check Alpaca dashboard.
"""
            self.edit_message(message_id, error_msg)

    def _handle_dismiss(self, callback_id: str, message_id: int, trade_id: str):
        """Handle DISMISS button click."""

        # Remove from pending trades
        if trade_id in self.pending_trades:
            ticker = self.pending_trades[trade_id].get("ticker", "???")
            del self.pending_trades[trade_id]

            # Update message
            dismiss_msg = f"🚫 **Alert Dismissed** - {ticker}\n\n_No action taken._"
            self.edit_message(message_id, dismiss_msg)
            self.answer_callback(callback_id, "Dismissed ✓")
        else:
            self.answer_callback(callback_id, "Already dismissed")

    # ==================== COMMAND HANDLERS ====================

    def handle_command(self, message: dict):
        """Process incoming commands."""
        text = message.get("text", "").strip()
        chat_id = message.get("chat", {}).get("id")

        # Security: Only respond to authorized chat
        if str(chat_id) != str(self.chat_id):
            logger.warning(f"Unauthorized chat_id: {chat_id}")
            return

        if not text.startswith("/"):
            return

        parts = text.split()
        command = parts[0].lower()
        args_list = parts[1:] if len(parts) > 1 else []
        args_str = " ".join(args_list)  # String version for some handlers

        logger.info(f"Command received: {command} {args_str}")

        # Route commands
        if command == "/analyze":
            self._cmd_analyze(args_str)
        elif command == "/watchlist":
            self._cmd_watchlist()
        elif command == "/add":
            self._cmd_add(args_list)
        elif command == "/remove":
            self._cmd_remove(args_list)
        elif command == "/scan":
            self._cmd_scan()
        elif command == "/status":
            self._cmd_status()
        elif command == "/help":
            self._cmd_help()
        elif command == "/start":
            self._cmd_help()
        elif command == "/risk":
            self._cmd_risk(args_list)
        elif command == "/positions":
            self._cmd_positions()
        elif command == "/autoscan":
            self._cmd_autoscan(args_str)
        elif command == "/movers":
            self._cmd_movers()
        elif command == "/test_signal":
            self._cmd_test_signal(args_str)
        elif command == "/debug":
            self._cmd_debug()
        elif command == "/cache":
            self._cmd_cache(args_str)
        elif command == "/fresh":
            self._cmd_fresh_analyze(args_str)
        elif command == "/news":
            self._cmd_news(args_str)
        elif command == "/macro":
            self._cmd_macro()
        elif command == "/turbo":
            self._cmd_turbo(args_str)
        elif command == "/autotrade":
            self._cmd_autotrade(args_str)
        elif command == "/learn":
            self._cmd_learn()
        else:
            self.send_message(f"❓ פקודה לא מוכרת: `{command}`\nשלח /help לעזרה")

    def _cmd_analyze(self, args: List[str]):
        """Handle /analyze TICKER command with execution buttons."""
        if not args:
            self.send_message("⚠️ שימוש: `/analyze TICKER`\nדוגמה: `/analyze NVDA`")
            return

        ticker = args[0].upper()
        self.send_message(f"🔍 מנתח את **{ticker}**...\nאנא המתן.")

        try:
            analysis = self.brain.analyze_ticker(ticker)
            response = self.brain.format_hebrew_response(analysis)
            action = analysis.get("action", "PASS").upper()

            # If actionable trade, add execution buttons
            if action in ["BUY", "SELL"]:
                validation = analysis.get("validation", {})
                entry = validation.get("entry", 0)
                stop_loss = validation.get("stop_loss", 0)
                take_profit = validation.get("target", 0)
                risk = validation.get("risk", 0)

                # Calculate position size
                qty, risk_amount, risk_desc = self._calculate_position_size(entry, stop_loss)

                if qty > 0 and entry > 0:
                    # Store pending trade
                    trade_id = self._generate_trade_id()
                    self.pending_trades[trade_id] = {
                        "ticker": ticker,
                        "action": action,
                        "entry": entry,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "qty": qty,
                        "risk": risk_amount,
                        "risk_desc": risk_desc,
                        "analysis": analysis,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Build keyboard
                    keyboard = self._build_execution_keyboard(
                        trade_id, ticker, action, qty, risk_amount
                    )

                    # Send with buttons
                    self.send_message(response, reply_markup=keyboard)
                    logger.info(f"Trade alert sent with buttons: {trade_id}")
                else:
                    # No valid position size, send without buttons
                    self.send_message(response)
            else:
                # PASS action, no buttons needed
                self.send_message(response)

            logger.info(f"Analysis for {ticker}: Action={action}, Conviction={analysis.get('conviction')}")

        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {e}")
            self.send_message(f"❌ שגיאה בניתוח {ticker}:\n`{str(e)}`")

    def _cmd_watchlist(self):
        """Show current watchlist."""
        symbols = ", ".join(self.watchlist)
        msg = f"""📋 **רשימת מעקב נוכחית:**

{symbols}

📊 סה"כ: {len(self.watchlist)} מניות
⏰ סריקה אוטומטית: כל {self.scan_interval_minutes} דקות
💰 סיכון נוכחי: {self._calculate_dynamic_risk()[1]}

פקודות:
• `/add TICKER` - הוסף לרשימה
• `/remove TICKER` - הסר מהרשימה
• `/scan` - סרוק עכשיו
• `/risk` - הגדרות סיכון
"""
        self.send_message(msg)

    def _cmd_add(self, args: List[str]):
        """Add ticker to watchlist."""
        if not args:
            self.send_message("⚠️ שימוש: `/add TICKER`")
            return

        ticker = args[0].upper()
        if ticker in self.watchlist:
            self.send_message(f"ℹ️ **{ticker}** כבר ברשימה")
        else:
            self.watchlist.append(ticker)
            self.send_message(f"✅ **{ticker}** נוסף לרשימת המעקב")

    def _cmd_remove(self, args: List[str]):
        """Remove ticker from watchlist."""
        if not args:
            self.send_message("⚠️ שימוש: `/remove TICKER`")
            return

        ticker = args[0].upper()
        if ticker in self.watchlist:
            self.watchlist.remove(ticker)
            self.send_message(f"✅ **{ticker}** הוסר מרשימת המעקב")
        else:
            self.send_message(f"ℹ️ **{ticker}** לא נמצא ברשימה")

    def _cmd_scan(self):
        """Force scan all watchlist."""
        # Check if scan already running
        if self.scan_lock.is_scanning():
            scan_info = self.scan_lock.get_scan_info()
            msg = f"⚠️ סריקה כבר רצה! ({scan_info['type'] if scan_info else 'unknown'}) - אנא המתן."
            self.send_message(msg)
            return

        self.send_message(f"🔄 מתחיל סריקה של {len(self.watchlist)} מניות...")
        self._run_watchlist_scan(force=True, scan_type="manual")

    def _cmd_status(self):
        """Show bot status."""
        last_scan = self.last_scan_time.strftime('%H:%M:%S') if self.last_scan_time else "לא בוצע"
        last_autoscan = self.last_autoscan_time.strftime('%d/%m %H:%M') if self.last_autoscan_time else "לא בוצע"
        alpaca_status = "מחובר ✅" if self.alpaca_client else "לא מחובר ❌"
        pending_count = len(self.pending_trades)

        # Scan lock status
        scan_info = self.scan_lock.get_scan_info()
        if scan_info:
            scan_status = f"🔄 רץ ({scan_info['type']}, {scan_info['duration']:.0f}s)"
        else:
            scan_status = "⏸️ לא רץ"

        # In-flight tickers
        inflight = self.deduplicator.get_inflight()
        inflight_str = ", ".join(inflight) if inflight else "אין"

        # Cache stats
        cache_stats = self.analysis_cache.get_stats()
        cache_entries = cache_stats['entries']
        cache_hit_rate = cache_stats['hit_rate_pct']

        # Dynamic risk info
        equity = self._get_account_equity()
        equity_str = f"${equity:,.2f}" if equity else "N/A"
        risk_amount, risk_desc = self._calculate_dynamic_risk()
        risk_mode = "Dynamic 📊" if self.use_dynamic_risk else "Fixed 💵"

        # Show watchlist preview
        watchlist_preview = ", ".join(self.watchlist[:5])
        if len(self.watchlist) > 5:
            watchlist_preview += f"... (+{len(self.watchlist) - 5})"

        # Turbo/Auto-trade status
        turbo_status = "⚡ ON" if self.turbo_mode else "🐢 OFF"
        auto_status = "🤖 ON" if self.auto_trade_enabled else "🔴 OFF"

        msg = f"""🤖 **סטטוס הבוט:**

🧠 AI Provider: `{self.brain.provider or 'לא מוגדר'}`
📈 Alpaca: {alpaca_status} ({'PAPER' if self.is_paper else '⚠️ LIVE'})
🔎 Scanner: מוכן ✅

⚡ **Turbo Mode:** {turbo_status}
🤖 **Auto-Trade:** {auto_status} (min: {self.auto_trade_min_conviction})

📋 **רשימת מעקב ({len(self.watchlist)}):**
{watchlist_preview}

🔄 **מצב סריקה:** {scan_status}
📡 **In-Flight:** {inflight_str}
📦 **Cache:** {cache_entries} entries, {cache_hit_rate:.0f}% hit rate

💰 **Risk Management:** {risk_mode}
📊 **Equity:** {equity_str}
🎯 **Current Risk:** {risk_desc}

⏰ סריקה אחרונה: {last_scan}
🌅 Autoscan אחרון: {last_autoscan}
🔄 מרווח סריקה: {self.scan_interval_minutes} דקות
📊 התראות ממתינות: {pending_count}
✅ פעיל: {'כן' if self.is_running else 'לא'}

📆 **סריקה מתוזמנת:** 16:00 (IL) / 9:00 AM (NY)
🛡️ **Rate Limit:** 5s min interval + jitter
"""
        self.send_message(msg)


    def _cmd_learn(self):
        """Trigger the Post-Mortem Analysis manually."""
        self.send_message("🧠 **Post-Mortem Engine**\n\nStarting analysis of recent closed trades...")

        try:
            engine = get_learning_engine()

            # 1. Update outcomes
            self.send_message("🔄 Syncing trade outcomes with Alpaca...")
            count = engine.update_trade_outcomes(self.alpaca_client)

            # 2. Analyze
            if count >= 0:
                self.send_message(f"found {count} updated trades. Running critique agent...")
                lessons = engine.analyze_past_performance()

                if lessons:
                    lesson_text = "\n".join(lessons)
                    self.send_message(f"🎓 **New Lessons Learned:**\n\n{lesson_text}")
                else:
                    self.send_message("✅ No new lessons generated (no fresh closed trades or nothing to learn).")
            else:
                 self.send_message("⚠️ Failed to sync outcomes.")

        except Exception as e:
            logger.error(f"Learn command failed: {e}")
            self.send_message(f"❌ Error during learning: {e}")

    def _cmd_risk(self, args: List[str]):
        """
        Set risk per trade or percentage.

        Usage:
            /risk - Show current risk settings
            /risk 100 - Set fixed risk to $100
            /risk 0.5% - Set percentage risk to 0.5%
            /risk dynamic - Enable dynamic risk
            /risk fixed - Enable fixed risk
        """
        if not args:
            # Show current settings
            equity = self._get_account_equity()
            equity_str = f"${equity:,.2f}" if equity else "N/A"

            risk_amount, risk_desc = self._calculate_dynamic_risk()

            mode = "Dynamic 📊" if self.use_dynamic_risk else "Fixed 💵"

            msg = f"""💰 **Risk Management Settings**

**Mode:** {mode}
**Account Equity:** {equity_str}

**Dynamic Settings:**
• Risk Percentage: `{self.risk_percentage * 100:.2f}%`
• Max Risk Cap: `${self.max_risk_amount:,.2f}`

**Fixed Fallback:** `${self.risk_per_trade:,.2f}`

**Current Calculated Risk:** `{risk_desc}`

**Commands:**
• `/risk 100` - Set fixed risk to $100
• `/risk 0.5%` - Set percentage to 0.5%
• `/risk dynamic` - Enable dynamic mode
• `/risk fixed` - Enable fixed mode
"""
            self.send_message(msg)
            return

        arg = args[0].lower().strip()

        # Toggle modes
        if arg == "dynamic":
            self.use_dynamic_risk = True
            self.send_message("✅ **Dynamic Risk Mode Enabled**\nRisk will be calculated as percentage of equity.")
            return

        if arg == "fixed":
            self.use_dynamic_risk = False
            self.send_message(f"✅ **Fixed Risk Mode Enabled**\nRisk set to ${self.risk_per_trade:.2f} per trade.")
            return

        try:
            # Check if it's a percentage (ends with %)
            if arg.endswith('%'):
                pct = float(arg.rstrip('%'))
                if pct < 0.1:
                    self.send_message("⚠️ Minimum risk: 0.1%")
                    return
                if pct > 5.0:
                    self.send_message("⚠️ Maximum risk: 5% (safety limit)")
                    return

                self.risk_percentage = pct / 100.0
                self.use_dynamic_risk = True

                # Show new calculated risk
                risk_amount, risk_desc = self._calculate_dynamic_risk()

                self.send_message(f"✅ **Risk Percentage Updated**\n\nNew: `{pct:.2f}%`\nCalculated Risk: `{risk_desc}`")
            else:
                # Fixed dollar amount
                new_risk = float(arg)
                if new_risk < 10:
                    self.send_message("⚠️ סיכון מינימלי: $10")
                    return
                if new_risk > 5000:
                    self.send_message("⚠️ סיכון מקסימלי: $5000")
                    return

                self.risk_per_trade = new_risk
                self.send_message(f"✅ Fixed risk updated to **${self.risk_per_trade:,.2f}**\n\n_Note: Using {'dynamic' if self.use_dynamic_risk else 'fixed'} mode_")
        except ValueError:
            self.send_message("⚠️ Invalid input.\n\nExamples:\n• `/risk 100` - Fixed $100\n• `/risk 0.5%` - 0.5% of equity")

    def _cmd_positions(self):
        """Show open positions from Alpaca."""
        if not self.alpaca_client:
            self.send_message("❌ Alpaca לא מחובר")
            return

        try:
            positions = self.alpaca_client.get_all_positions()

            if not positions:
                self.send_message("📭 אין פוזיציות פתוחות")
                return

            lines = ["📊 **פוזיציות פתוחות:**\n"]
            total_pnl = 0

            for pos in positions:
                pnl = float(pos.unrealized_pl)
                total_pnl += pnl
                emoji = "🟢" if pnl >= 0 else "🔴"
                pnl_pct = float(pos.unrealized_plpc) * 100

                lines.append(
                    f"{emoji} **{pos.symbol}**: {pos.qty} @ ${float(pos.avg_entry_price):.2f}\n"
                    f"   P/L: ${pnl:.2f} ({pnl_pct:+.1f}%)"
                )

            lines.append(f"\n💰 **סה\"כ P/L:** ${total_pnl:.2f}")
            self.send_message("\n".join(lines))

        except Exception as e:
            self.send_message(f"❌ שגיאה בטעינת פוזיציות:\n`{e}`")

    def _cmd_autoscan(self, args: List[str]):
        """
        Auto-scan for stocks in play and add to watchlist.
        Usage: /autoscan [limit] [replace]
        - limit: number of stocks (default 5)
        - replace: 'replace' to clear watchlist, else append
        """
        # Parse arguments
        limit = 5
        replace_mode = False

        for arg in args:
            if arg.isdigit():
                limit = min(int(arg), 15)  # Max 15
            elif arg.lower() in ["replace", "clear", "new"]:
                replace_mode = True

        self.send_message(f"🔎 **מתחיל סריקת שוק אוטומטית...**\nמחפש {limit} מניות בתנועה.")

        try:
            # Run the scanner
            tickers, movers = self.scanner.get_top_gainers_and_losers(limit)

            if not tickers:
                self.send_message("❌ לא נמצאו מניות בתנועה משמעותית כרגע.\nנסה שוב בשעות המסחר.")
                return

            # Update watchlist
            if replace_mode:
                old_watchlist = self.watchlist.copy()
                self.watchlist = tickers
                mode_msg = "החלפת רשימת מעקב"
            else:
                # Append unique tickers
                added = []
                for ticker in tickers:
                    if ticker not in self.watchlist:
                        self.watchlist.append(ticker)
                        added.append(ticker)
                tickers = added  # Only show newly added
                mode_msg = "הוספה לרשימת מעקב"

            self.last_autoscan_time = datetime.now()

            # Format results message
            results_msg = self.scanner.format_scan_results(movers)

            summary = f"""
{results_msg}

✅ **{mode_msg}:** {len(tickers)} מניות
📋 רשימה חדשה: {', '.join(self.watchlist[:10])}{'...' if len(self.watchlist) > 10 else ''}

⏳ מתחיל ניתוח CIO על המניות החדשות...
"""
            self.send_message(summary)

            # Trigger analysis on new tickers
            if tickers:
                for i, ticker in enumerate(tickers[:5]):  # Limit to 5 for speed
                    try:
                        log_info(f"Auto-analyzing {ticker} ({i+1}/5)...")
                        self._cmd_analyze([ticker])
                        time_module.sleep(4)  # Rate limit protection
                    except Exception as e:
                        logger.error(f"Analysis failed for {ticker}: {e}")
                        time_module.sleep(2)

        except Exception as e:
            logger.error(f"Autoscan failed: {e}")
            self.send_message(f"❌ שגיאה בסריקה:\n`{str(e)}`")

    def _cmd_movers(self):
        """Show current market movers without adding to watchlist."""
        log_scanner("Running movers scan...")
        self.send_message("🔎 סורק מניות בתנועה...")

        try:
            tickers, movers = self.scanner.get_top_gainers_and_losers(10)

            if not movers:
                self.send_message("❌ לא נמצאו מניות בתנועה משמעותית כרגע.")
                return

            results_msg = self.scanner.format_scan_results(movers)
            self.send_message(results_msg + "\n\n💡 שלח `/autoscan` להוספה לרשימת מעקב")
            log_ok(f"Movers scan complete: {len(movers)} found")

        except Exception as e:
            log_error(f"Movers scan failed: {e}")
            self.send_message(f"❌ שגיאה בסריקה:\n`{str(e)}`")

    def _cmd_test_signal(self, args: List[str]):
        """
        🧪 TEST MODE: Generate a fake signal to test the execution pipeline.
        Bypasses AI and market data - sends hardcoded valid trade.
        """
        log_divider("TEST SIGNAL MODE")
        log_warn("⚠️ TEST MODE: Generating fake signal for pipeline testing")

        ticker = args[0].upper() if args else "TSLA"

        # Generate hardcoded dummy trade with perfect R/R
        dummy_entry = 100.00
        dummy_stop = 95.00  # $5 risk
        dummy_target = 115.00  # $15 reward = 3:1 R/R

        fake_analysis = {
            "bluf": f"🧪 TEST SIGNAL - Pipeline verification for {ticker}",
            "logic_engine": "TEST_MODE",
            "action": "BUY",
            "ticker": ticker,
            "plan": {
                "buy_zone": f"${dummy_entry:.2f}",
                "targets": [f"${dummy_target:.2f}", f"${dummy_target + 5:.2f}"],
                "invalidation": f"${dummy_stop:.2f}"
            },
            "conviction": "HIGH",
            "is_push_alert": True,
            "reasoning": "🧪 This is a TEST SIGNAL to verify Telegram → Alpaca pipeline. DO NOT trade based on this!",
            "validation": {
                "entry": dummy_entry,
                "target": dummy_target,
                "stop_loss": dummy_stop,
                "risk": dummy_entry - dummy_stop,
                "reward": dummy_target - dummy_entry,
                "rr_ratio": (dummy_target - dummy_entry) / (dummy_entry - dummy_stop),
                "passed": True,
                "notes": ["✅ TEST MODE - Hardcoded valid R/R"]
            }
        }

        log_ok(f"Fake signal generated for {ticker}")
        log_trade("BUY", ticker, f"Entry=${dummy_entry} Stop=${dummy_stop} Target=${dummy_target}")

        # Calculate position size with dynamic risk
        qty, risk_amount, risk_desc = self._calculate_position_size(dummy_entry, dummy_stop)
        log_info(f"Position size calculated: {qty} shares ({risk_desc})")

        # Store pending trade
        trade_id = self._generate_trade_id()
        self.pending_trades[trade_id] = {
            "ticker": ticker,
            "action": "BUY",
            "entry": dummy_entry,
            "stop_loss": dummy_stop,
            "take_profit": dummy_target,
            "qty": qty,
            "risk": risk_amount,
            "risk_desc": risk_desc,
            "analysis": fake_analysis,
            "timestamp": datetime.now().isoformat(),
            "is_test": True  # Mark as test
        }

        # Format and send message with buttons
        response = self.brain.format_hebrew_response(fake_analysis)

        # Add TEST warning banner
        test_banner = f"""⚠️ **🧪 TEST MODE 🧪** ⚠️
_This is a fake signal for testing the execution pipeline._
_The button below will attempt to send a REAL order to Alpaca!_

💰 **Dynamic Risk:** {risk_desc}
📊 **Position:** {qty} shares

"""
        response = test_banner + response

        keyboard = self._build_execution_keyboard(
            trade_id, ticker, "BUY", qty, risk_amount
        )

        log_telegram(f"Sending test alert with buttons (trade_id: {trade_id})")
        self.send_message(response, reply_markup=keyboard)
        log_ok("Test signal sent to Telegram")

    def _cmd_debug(self):
        """Show debug information."""
        import os
        from src.logger import DEBUG_MODE

        debug_status = "🟢 ON" if DEBUG_MODE else "🔴 OFF"
        pending_count = len(self.pending_trades)

        # Show pending trades
        pending_info = ""
        if self.pending_trades:
            for tid, trade in list(self.pending_trades.items())[:3]:
                pending_info += f"\n• `{tid}`: {trade.get('action')} {trade.get('ticker')} ({trade.get('qty')} shares)"
        else:
            pending_info = "\nאין עסקאות ממתינות"

        msg = f"""🔧 **Debug Information:**

**Mode:** {debug_status}
**Set DEBUG_MODE=true in .env for verbose output**

**AI Provider:** `{self.brain.provider}`
**Alpaca:** {'Connected ✅' if self.alpaca_client else 'Not Connected ❌'}
**Paper Mode:** {'Yes' if self.is_paper else 'No - LIVE!'}

**Pending Trades ({pending_count}):**{pending_info}

**Commands:**
`/test_signal TICKER` - Test execution pipeline
`/debug` - Show this info
`/cache` - Show cache stats
`/fresh TICKER` - Force fresh analysis (skip cache)
"""
        self.send_message(msg)

    # =========================================================================
    # CACHE COMMANDS
    # =========================================================================

    def _cmd_cache(self, args: str):
        """
        Cache management command.

        Usage:
            /cache - Show cache statistics
            /cache clear - Clear all cached entries
            /cache list - List all cached tickers
        """
        sub_cmd = args.strip().lower() if args else ""

        if sub_cmd == "clear":
            self.analysis_cache.clear()
            self.send_message("🗑️ **Cache Cleared**\nכל הניתוחים השמורים נמחקו.")
            return

        if sub_cmd == "list":
            cached = self.analysis_cache.get_cached_tickers()

            if not cached:
                self.send_message("📦 **Cache Empty**\nאין ניתוחים שמורים.")
                return

            lines = ["📦 **Cached Analyses:**\n"]
            for ticker, info in sorted(cached.items()):
                age = info['age_minutes']
                action = info['action']
                conviction = info['conviction']
                price = info['price']
                hits = info['hit_count']

                emoji = "🟢" if action == "BUY" else "🔴" if action in ["SELL", "SHORT"] else "⚪"
                lines.append(f"{emoji} `{ticker}` @ ${price:.2f}")
                lines.append(f"   ├ {action} ({conviction}) | Age: {age:.0f}m | Hits: {hits}")

            self.send_message("\n".join(lines))
            return

        # Default: Show stats
        stats = self.analysis_cache.get_stats()
        cached_count = stats['entries']
        hits = stats['hits']
        misses = stats['misses']
        hit_rate = stats['hit_rate_pct']
        total = stats['total_requests']

        # Calculate savings estimate
        avg_api_cost = 0.002  # ~$0.002 per API call estimate
        saved_calls = hits
        saved_cost = saved_calls * avg_api_cost

        msg = f"""📦 **Analysis Cache Stats**

**Performance:**
• Entries: `{cached_count}`
• Hit Rate: `{hit_rate:.1f}%` ({hits}/{total})
• Cache Hits: `{hits}` ✅
• Cache Misses: `{misses}` ❌

**Estimated Savings:**
• API Calls Saved: `~{saved_calls}`
• Est. Cost Saved: `~${saved_cost:.3f}`

**Settings:**
• Max Age: `60 minutes`
• Price Threshold: `0.5%`
• Persistence: `cache/analysis_cache.json`

**Commands:**
• `/cache list` - Show cached tickers
• `/cache clear` - Clear all cache
• `/fresh TICKER` - Skip cache, force analysis
"""
        self.send_message(msg)

    def _cmd_fresh_analyze(self, args: str):
        """
        Force fresh analysis, skipping cache.
        Usage: /fresh TICKER
        """
        if not args:
            self.send_message("❓ **שימוש:** `/fresh TICKER`\nניתוח חדש בלי שימוש בקאש")
            return

        ticker = args.upper().strip()
        log_info(f"Force fresh analysis for {ticker} (cache skipped)")

        self.send_message(f"🔄 **Fresh Analysis:** מנתח {ticker} (ללא קאש)...")

        try:
            # Invalidate existing cache for this ticker
            self.analysis_cache.invalidate(ticker)

            # Run fresh analysis with skip_cache=True
            analysis = self.brain.analyze_ticker(ticker, skip_cache=True)

            if analysis.get("error"):
                self.send_message(f"❌ שגיאה: {analysis.get('error')}")
                return

            self._send_analysis_with_buttons(ticker, analysis)

        except Exception as e:
            log_error(f"Fresh analysis error: {e}")
            self.send_message(f"❌ שגיאה בניתוח: `{str(e)}`")

    # =========================================================================
    # NEWS INTELLIGENCE COMMANDS
    # =========================================================================

    def _cmd_news(self, args: str):
        """
        Analyze news sentiment for a ticker.
        Usage: /news TICKER
        """
        if not args:
            self.send_message("""📰 **News Intelligence**

**שימוש:** `/news TICKER`

**דוגמאות:**
• `/news TSLA` - ניתוח חדשות Tesla
• `/news NVDA` - ניתוח חדשות NVIDIA
• `/news AAPL` - ניתוח חדשות Apple

הפקודה תאסוף את 3 הכותרות האחרונות ותנתח אותן עם AI לקבלת ציון סנטימנט (-10 עד +10).""")
            return

        ticker = args.upper().strip()
        log_info(f"News analysis requested for {ticker}")

        self.send_message(f"📰 **מנתח חדשות עבור {ticker}...**\n_זה עשוי לקחת מספר שניות_")

        try:
            # Get full news analysis
            analysis = self.news_brain.get_full_news_analysis(ticker)

            if analysis.get("news_count", 0) == 0:
                self.send_message(f"📭 לא נמצאו חדשות עדכניות עבור {ticker}")
                return

            # Format and send
            message = self.news_brain.format_news_for_telegram(analysis)
            self.send_message(message)

            log_ok(f"News analysis sent for {ticker}: Score={analysis.get('overall_score')}")

        except Exception as e:
            log_error(f"News analysis error: {e}")
            self.send_message(f"❌ שגיאה בניתוח חדשות: `{str(e)}`")

    def _cmd_macro(self):
        """
        Get overall market sentiment from macro news.
        Usage: /macro
        """
        log_info("Market macro sentiment requested")

        self.send_message("🌍 **מנתח סנטימנט שוק...**\n_אוסף ומנתח חדשות מאקרו_")

        try:
            # Get market sentiment
            analysis = self.news_brain.get_market_sentiment(limit=5)

            if not analysis.get("headlines"):
                self.send_message("📭 לא נמצאו חדשות שוק עדכניות")
                return

            # Format and send
            message = self.news_brain.format_macro_for_telegram(analysis)
            self.send_message(message)

            log_ok(f"Macro sentiment sent: {analysis.get('mood')}")

        except Exception as e:
            log_error(f"Macro analysis error: {e}")
            self.send_message(f"❌ שגיאה בניתוח מאקרו: `{str(e)}`")

    # =========================================================================
    # TURBO MODE & AUTO-TRADE COMMANDS
    # =========================================================================

    def _cmd_turbo(self, args: str):
        """
        Toggle Turbo Mode (faster scanning).
        Usage: /turbo [on|off]
        """
        arg = args.strip().lower() if args else ""

        if arg == "on":
            self.turbo_mode = True
            self.scan_interval_minutes = 5
            log_info("⚡ TURBO MODE ENABLED")
            self.send_message("""⚡ **TURBO MODE: ON**

**Changes Applied:**
• Scan interval: **5 minutes** (was 30)
• Cache TTL: Reduced for fresher data

_Bot is now in aggressive hunting mode._

⚠️ **Warning:** Higher API usage
""")
        elif arg == "off":
            self.turbo_mode = False
            self.scan_interval_minutes = 30
            log_info("⚡ TURBO MODE DISABLED")
            self.send_message("""🐢 **TURBO MODE: OFF**

**Changes Applied:**
• Scan interval: **30 minutes**
• Normal operation resumed

_Conserving API quota._
""")
        else:
            # Show status
            status = "⚡ ON" if self.turbo_mode else "🐢 OFF"
            msg = f"""⚡ **Turbo Mode Status:** {status}

**Current Settings:**
• Scan Interval: `{self.scan_interval_minutes} min`
• Auto-Trade: `{'ON' if self.auto_trade_enabled else 'OFF'}`

**Commands:**
• `/turbo on` - Enable turbo (5 min scans)
• `/turbo off` - Normal mode (30 min scans)
"""
            self.send_message(msg)

    def _cmd_autotrade(self, args: str):
        """
        Toggle Auto-Trade mode.
        Usage: /autotrade [on|off|high|med]
        """
        arg = args.strip().lower() if args else ""

        if arg == "on":
            self.auto_trade_enabled = True
            log_info(f"🤖 AUTO-TRADE ENABLED (min: {self.auto_trade_min_conviction})")
            self.send_message(f"""🤖 **AUTO-TRADE: ENABLED**

**Settings:**
• Min Conviction: `{self.auto_trade_min_conviction}`
• Alpaca: `{'Connected ✅' if self.alpaca_client else 'Not Connected ❌'}`
• Mode: `{'PAPER' if self.is_paper else '⚠️ LIVE'}`

⚠️ **Warning:** Bot will execute trades automatically when:
1. Action is BUY/SELL
2. Conviction >= {self.auto_trade_min_conviction}
3. R/R >= 2.5
4. During market hours

_You are responsible for all auto-executed trades._
""")
        elif arg == "off":
            self.auto_trade_enabled = False
            log_info("🤖 AUTO-TRADE DISABLED")
            self.send_message("""🛑 **AUTO-TRADE: DISABLED**

Bot will show buttons for manual execution.
No trades will be executed automatically.
""")
        elif arg == "high":
            self.auto_trade_min_conviction = "HIGH"
            self.send_message(f"""🎯 **Auto-Trade Conviction: HIGH**

Only HIGH conviction trades will auto-execute.
_More conservative, fewer trades._
""")
        elif arg == "med":
            self.auto_trade_min_conviction = "MED"
            self.send_message(f"""🎯 **Auto-Trade Conviction: MED**

MED and HIGH conviction trades will auto-execute.
_More aggressive, more trades._

⚠️ Higher risk!
""")
        else:
            # Show status
            status = "🟢 ON" if self.auto_trade_enabled else "🔴 OFF"
            alpaca_status = "Connected ✅" if self.alpaca_client else "Not Connected ❌"
            mode = "⚠️ LIVE" if not self.is_paper else "PAPER"

            msg = f"""🤖 **Auto-Trade Status:** {status}

**Current Settings:**
• Min Conviction: `{self.auto_trade_min_conviction}`
• Turbo Mode: `{'ON' if self.turbo_mode else 'OFF'}`
• Alpaca: `{alpaca_status}`
• Trading Mode: `{mode}`

**Commands:**
• `/autotrade on` - Enable auto-trading
• `/autotrade off` - Disable auto-trading
• `/autotrade high` - Only HIGH conviction
• `/autotrade med` - MED+ conviction

**Safety Checks (always on):**
✓ R/R >= 2.5 required
✓ Validation must pass
✓ Market hours only
"""
            self.send_message(msg)

    def _cmd_help(self):
        """Show help message."""
        msg = """🤖 **CIO Trading Bot - עוזר המסחר שלך**

📌 **פקודות ניתוח:**
`/analyze TICKER` - ניתוח מלא + כפתור ביצוע
`/watchlist` - הצג רשימת מעקב
`/add TICKER` - הוסף מניה לרשימה
`/remove TICKER` - הסר מניה מהרשימה
`/scan` - סרוק את כל הרשימה עכשיו

📌 **סריקה אוטומטית:**
`/autoscan` - מצא מניות בתנועה והוסף לרשימה
`/autoscan 10` - מצא 10 מניות
`/autoscan replace` - החלף את כל הרשימה
`/movers` - הצג מניות בתנועה (ללא הוספה)

📌 **פקודות מסחר:**
`/risk` - הצג/שנה הגדרות סיכון
`/risk 0.5%` - סיכון אחוזי מההון
`/risk dynamic` - מצב דינמי
`/positions` - הצג פוזיציות פתוחות
`/status` - סטטוס הבוט

🔧 **Debug & Testing:**
`/test_signal TSLA` - 🧪 בדיקת pipeline ביצוע
`/debug` - מידע טכני ומצב debug

📦 **Cache Management:**
`/cache` - סטטיסטיקות קאש
`/cache list` - רשימת ניתוחים שמורים
`/cache clear` - מחק קאש
`/fresh TICKER` - ניתוח חדש ללא קאש

📰 **News Intelligence:**
`/news TICKER` - ניתוח חדשות עם AI
`/macro` - סנטימנט שוק כללי

⚡ **Turbo & Auto-Trade:**
`/turbo [on|off]` - מצב טורבו (סריקה כל 5 דק')
`/autotrade [on|off]` - ביצוע אוטומטי
`/autotrade high` - רק HIGH conviction
`/autotrade med` - MED+ conviction

📊 **Logic Engines:**
• Sympathy - מסחר סימפטיה
• Squeeze - Short/Gamma Squeeze
• Gamma - אופציות וזרימת כסף
• Macro - מאקרו ומשטר שוק

🛡️ **כללי סיכון:**
• R/R מינימום 2.5:1
• Stop Loss טכני
• Conviction: HIGH/MED/LOW

🎯 **One-Click Execution:**
כשמתקבלת התראה עם action = BUY/SELL,
יופיע כפתור לביצוע מיידי עם Bracket Order!

⏰ **סריקה אוטומטית:**
כל יום ב-16:00 (שעון ישראל) מתבצעת סריקה אוטומטית
"""
        self.send_message(msg)

    def _run_watchlist_scan(self, force: bool = False, scan_type: str = "scheduled"):
        """
        Scan all watchlist symbols with:
        - Scan lock to prevent overlapping scans
        - Ticker deduplication
        - Rate limiting between API calls
        """
        # Try to acquire scan lock
        if not self.scan_lock.try_acquire(scan_type):
            scan_info = self.scan_lock.get_scan_info()
            if scan_info:
                msg = f"⚠️ סריקה כבר רצה ({scan_info['type']}) כבר {scan_info['duration']:.0f} שניות"
            else:
                msg = "⚠️ סריקה כבר רצה. אנא המתן."
            self.send_message(msg)
            return

        try:
            self.last_scan_time = datetime.now()
            high_conviction_alerts = []
            total = len(self.watchlist)
            skipped = 0
            analyzed = 0

            log_scanner(f"Starting {scan_type} scan: {total} symbols")

            for i, ticker in enumerate(self.watchlist):
                try:
                    # Check for duplicate (recently analyzed or in-flight)
                    if self.deduplicator.is_duplicate(ticker):
                        log_warn(f"Skipping {ticker} (duplicate)")
                        skipped += 1
                        continue

                    # Mark as in-flight
                    self.deduplicator.mark_started(ticker)

                    try:
                        log_info(f"Analyzing {ticker} ({i+1}/{total})...")
                        analysis = self.brain.analyze_ticker(ticker)
                        analyzed += 1

                        # Only alert on HIGH conviction or if forced
                        if force or (analysis.get("conviction") == "HIGH" and analysis.get("action") != "PASS"):
                            if analysis.get("is_push_alert") or force:
                                high_conviction_alerts.append((ticker, analysis))

                        logger.info(f"Scanned {ticker}: {analysis.get('action')} ({analysis.get('conviction')})")

                        # Mark as completed (updates cooldown timer)
                        self.deduplicator.mark_completed(ticker)

                    except Exception as e:
                        # Clear in-flight on error
                        self.deduplicator.clear_inflight(ticker)
                        raise e

                except Exception as e:
                    logger.error(f"Scan failed for {ticker}: {e}")
                    log_error(f"Scan failed for {ticker}: {e}")

            log_ok(f"Scan complete: {analyzed} analyzed, {skipped} skipped, {len(high_conviction_alerts)} alerts")

            # Send alerts with buttons
            if high_conviction_alerts:
                for ticker, analysis in high_conviction_alerts:
                    self._send_analysis_with_buttons(ticker, analysis)
                    time_module.sleep(1)  # Small delay between Telegram messages
            elif force:
                msg = f"✅ סריקה הושלמה.\n📊 נותחו: {analyzed} | דולגו: {skipped}\n🔔 אין התראות HIGH conviction."
                self.send_message(msg)

        finally:
            # Always release scan lock
            self.scan_lock.release()

            # Cleanup old deduplication entries
            self.deduplicator.cleanup_old_entries(max_age_seconds=300.0)

    def _send_analysis_with_buttons(self, ticker: str, analysis: Dict[str, Any]):
        """
        Send analysis message with execution buttons if applicable.

        AUTO-TRADE LOGIC:
        If AUTO_TRADE is enabled AND conviction meets threshold:
        - Execute trade immediately (bypass button)
        - Send notification
        """
        response = self.brain.format_hebrew_response(analysis)
        action = analysis.get("action", "PASS").upper()
        conviction = analysis.get("conviction", "LOW").upper()

        if action in ["BUY", "SELL"]:
            validation = analysis.get("validation", {})
            entry = validation.get("entry", 0)
            stop_loss = validation.get("stop_loss", 0)
            take_profit = validation.get("target", 0)

            qty, risk_amount, risk_desc = self._calculate_position_size(entry, stop_loss)

            if qty > 0 and entry > 0:
                trade_id = self._generate_trade_id()
                trade_data = {
                    "ticker": ticker,
                    "action": action,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "qty": qty,
                    "risk": risk_amount,
                    "risk_desc": risk_desc,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }

                # =========================================================
                # AUTO-TRADE CHECK
                # =========================================================
                should_auto_trade = self._should_auto_trade(conviction, validation)

                if should_auto_trade:
                    # Execute trade immediately!
                    log_ok(f"🤖⚡ AUTO-TRADE TRIGGERED: {action} {ticker} (Conviction: {conviction})")
                    self._execute_auto_trade(trade_data, response)
                    return

                # Normal flow - show button
                self.pending_trades[trade_id] = trade_data

                keyboard = self._build_execution_keyboard(
                    trade_id, ticker, action, qty, risk_amount
                )
                self.send_message(response, reply_markup=keyboard)
                return

        self.send_message(response)

    def _should_auto_trade(self, conviction: str, validation: Dict) -> bool:
        """
        Determine if trade should be auto-executed.

        Conditions:
        1. AUTO_TRADE must be enabled
        2. Conviction must meet minimum threshold
        3. Trade must pass validation (R/R check)
        4. Must be during market hours (safety)
        """
        if not self.auto_trade_enabled:
            return False

        if not self.alpaca_client:
            log_warn("Auto-trade skipped: Alpaca not connected")
            return False

        # Check conviction threshold
        conviction_levels = {"HIGH": 3, "MED": 2, "LOW": 1}
        min_level = conviction_levels.get(self.auto_trade_min_conviction, 3)
        current_level = conviction_levels.get(conviction.upper(), 0)

        if current_level < min_level:
            log_debug(f"Auto-trade skipped: Conviction {conviction} < {self.auto_trade_min_conviction}")
            return False

        # Check validation passed
        if not validation.get("passed", False):
            log_debug("Auto-trade skipped: Validation failed")
            return False

        # R/R must be good
        rr_ratio = validation.get("rr_ratio", 0)
        if rr_ratio < 2.5:
            log_debug(f"Auto-trade skipped: R/R {rr_ratio} < 2.5")
            return False

        # Market hours check (optional safety - can be disabled)
        if not self._is_market_hours():
            log_debug("Auto-trade skipped: Outside market hours")
            return False

        return True

    def _execute_auto_trade(self, trade_data: Dict[str, Any], analysis_msg: str):
        """
        Execute trade automatically and send notification.
        """
        ticker = trade_data.get("ticker")
        action = trade_data.get("action")
        qty = trade_data.get("qty")
        risk_desc = trade_data.get("risk_desc", "")

        log_alpaca(f"🤖 AUTO-EXECUTING: {action} {qty}x {ticker}")

        # Send pre-execution notification
        pre_msg = f"""🤖⚡ **AUTO-TRADE INITIATED**

{analysis_msg}

━━━━━━━━━━━━━━━━━━━━
🚀 **Executing {action} {qty}x {ticker}...**
💰 Risk: {risk_desc}
"""
        self.send_message(pre_msg)

        # Execute the trade
        result = self.execute_bracket_order(trade_data)

        if result["success"]:
            log_ok(f"🤖✅ AUTO-TRADE SUCCESS: {action} {qty}x {ticker}")

            success_msg = f"""🤖✅ **AUTO-TRADE EXECUTED!**

🎯 **{action}** {qty} x **{ticker}**
📋 Order ID: `{result['order_id']}`

🛡️ Bracket Order Set:
• Stop Loss: ${result['stop_loss']:.2f}
• Take Profit: ${result['take_profit']:.2f}

💰 Risk: {risk_desc}
⚡ Mode: **AUTONOMOUS**
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            self.send_message(success_msg)
        else:
            log_error(f"🤖❌ AUTO-TRADE FAILED: {result.get('error')}")

            fail_msg = f"""🤖❌ **AUTO-TRADE FAILED**

**Ticker:** {ticker}
**Action:** {action}
**Error:** `{result.get('error', 'Unknown')}`

_Trade was NOT executed. Manual intervention may be required._
"""
            self.send_message(fail_msg)

    def _is_market_hours(self) -> bool:
        """Check if it's US market hours."""
        try:
            import pytz
            tz_ny = pytz.timezone('America/New_York')
            now_ny = datetime.now(tz_ny)

            # Weekend check
            if now_ny.weekday() >= 5:
                return False

            # Market hours: 9:30 AM - 4:00 PM ET
            market_open = dt_time(9, 30)
            market_close = dt_time(16, 0)
            current_time = now_ny.time()

            return market_open <= current_time <= market_close
        except Exception:
            return True  # Optimistic fallback

    def _is_premarket_scan_time(self) -> bool:
        """
        Check if it's time for pre-market auto-scan.
        Target: 16:00 Israel time = 9:00 AM NY (30 mins before open)
        """
        try:
            import pytz

            # Israel timezone
            tz_israel = pytz.timezone('Asia/Jerusalem')
            now_israel = datetime.now(tz_israel)

            # Check if it's a weekday
            if now_israel.weekday() >= 5:
                return False

            # Target time: 16:00 Israel (9:00 AM NY)
            target_hour = 16
            target_minute = 0

            # Check if we're within the scan window (16:00-16:05)
            current_time = now_israel.time()
            scan_start = dt_time(target_hour, target_minute)
            scan_end = dt_time(target_hour, target_minute + 5)

            return scan_start <= current_time <= scan_end

        except Exception as e:
            logger.error(f"Premarket time check error: {e}")
            return False

    def _run_premarket_autoscan(self):
        """Run automatic pre-market scan and update watchlist."""
        logger.info("🌅 Running scheduled pre-market auto-scan...")

        try:
            self.send_message("🌅 **סריקת טרום-שוק אוטומטית מתחילה...**\n⏰ 30 דקות לפתיחת השוק!")

            # Get top movers
            tickers, movers = self.scanner.get_top_gainers_and_losers(7)

            if not tickers:
                self.send_message("❌ לא נמצאו מניות בתנועה משמעותית בטרום-שוק.")
                return

            # Keep core watchlist + add movers
            core_watchlist = ["SPY", "QQQ"]  # Always keep indices

            # Add new movers (avoiding duplicates)
            for ticker in tickers:
                if ticker not in self.watchlist:
                    self.watchlist.append(ticker)

            # Limit watchlist size
            if len(self.watchlist) > 20:
                self.watchlist = core_watchlist + self.watchlist[2:20]

            self.last_autoscan_time = datetime.now()

            # Format results
            results_msg = self.scanner.format_scan_results(movers)

            summary = f"""🌅 **סריקת טרום-שוק הושלמה!**

{results_msg}

📋 **רשימת מעקב עודכנה:** {len(self.watchlist)} מניות
🔔 התראות יישלחו אוטומטית עבור HIGH conviction

⏳ מתחיל ניתוח CIO...
"""
            self.send_message(summary)

            # Analyze top 3 movers immediately
            for i, ticker in enumerate(tickers[:3]):
                try:
                    log_info(f"Pre-market analyzing {ticker} ({i+1}/3)...")
                    self._cmd_analyze([ticker])
                    time_module.sleep(4)  # Rate limit protection
                except Exception as e:
                    logger.error(f"Pre-market analysis failed for {ticker}: {e}")
                    time_module.sleep(2)

        except Exception as e:
            logger.error(f"Pre-market autoscan failed: {e}")
            self.send_message(f"❌ שגיאה בסריקת טרום-שוק:\n`{str(e)}`")

    def _autonomous_scan_loop(self):
        """Background thread for autonomous scanning and scheduled tasks."""
        logger.info("Autonomous scan loop started")

        while self.is_running:
            try:
                now = datetime.now()

                # ========== SCHEDULED PRE-MARKET SCAN (16:00 Israel / 9:00 NY) ==========
                if self._is_premarket_scan_time():
                    # Check if we already ran today
                    if self.last_autoscan_time:
                        last_scan_date = self.last_autoscan_time.date()
                        today = now.date()
                        if last_scan_date == today:
                            logger.debug("Pre-market scan already done today")
                        else:
                            self._run_premarket_autoscan()
                    else:
                        self._run_premarket_autoscan()

                # ========== REGULAR WATCHLIST SCAN (during market hours) ==========
                if self._is_market_hours():
                    # Check if enough time passed since last scan
                    should_scan = True
                    if self.last_scan_time:
                        minutes_since_scan = (now - self.last_scan_time).total_seconds() / 60
                        should_scan = minutes_since_scan >= self.scan_interval_minutes

                    if should_scan:
                        logger.info("Running scheduled watchlist scan...")
                        self._run_watchlist_scan(force=False, scan_type="scheduled")
                else:
                    logger.debug("Market closed, skipping regular scan")

                # Sleep for 1 minute then check again
                for _ in range(60):
                    if not self.is_running:
                        break
                    time_module.sleep(1)

            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                time_module.sleep(60)

    def _cleanup_expired_trades(self):
        """Remove pending trades older than 1 hour."""
        now = datetime.now()
        expired = []

        for trade_id, trade_data in self.pending_trades.items():
            try:
                timestamp = datetime.fromisoformat(trade_data.get("timestamp", ""))
                if (now - timestamp).total_seconds() > 3600:  # 1 hour
                    expired.append(trade_id)
            except (ValueError, TypeError):
                expired.append(trade_id)

        for trade_id in expired:
            del self.pending_trades[trade_id]
            logger.info(f"Expired pending trade: {trade_id}")

    def run(self):
        """Main bot loop - poll for messages and handle commands."""
        self.is_running = True

        # Send startup message
        startup_msg = """🚀 **CIO Trading Bot התחיל!**

🧠 AI Provider: `{provider}`
📈 Alpaca: {alpaca} ({mode})
📋 מניות במעקב: {count}
💰 סיכון לעסקה: ${risk}
⏰ סריקה אוטומטית: כל {interval} דקות

🎯 **One-Click Execution מופעל!**
שלח /help לרשימת פקודות
""".format(
            provider=self.brain.provider or "לא מוגדר",
            alpaca="מחובר ✅" if self.alpaca_client else "לא מחובר ❌",
            mode="PAPER" if self.is_paper else "LIVE",
            count=len(self.watchlist),
            risk=self.risk_per_trade,
            interval=self.scan_interval_minutes
        )
        self.send_message(startup_msg)

        # Start autonomous scan thread
        scan_thread = threading.Thread(target=self._autonomous_scan_loop, daemon=True)
        scan_thread.start()

        logger.info("Starting Telegram polling loop...")

        cleanup_counter = 0

        while self.is_running:
            try:
                updates = self.get_updates(offset=self.last_update_id + 1)

                for update in updates:
                    self.last_update_id = update.get("update_id", 0)

                    # Handle regular messages
                    message = update.get("message")
                    if message:
                        self.handle_command(message)

                    # Handle callback queries (button clicks)
                    callback_query = update.get("callback_query")
                    if callback_query:
                        self.handle_callback(callback_query)

                # Periodic cleanup of expired trades
                cleanup_counter += 1
                if cleanup_counter >= 100:  # Every ~100 polling cycles
                    self._cleanup_expired_trades()
                    cleanup_counter = 0

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.is_running = False
                break

            except Exception as e:
                logger.error(f"Polling error: {e}")
                time_module.sleep(5)

        self.send_message("🛑 **הבוט נכבה.** להתראות!")


def main():
    """Entry point for the Telegram bot."""
    try:
        bot = TelegramCIOBot()
        bot.run()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
