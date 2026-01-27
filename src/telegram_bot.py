"""
Telegram Bot Interface - CIO Trading System
Handles commands, user interaction, and trade confirmations.
"""
import os
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv

from src.ai_brain import get_brain
from src.alpaca_manager import get_alpaca
from src.logger import log_info, log_error, log_ok, log_warn, log_trade
from src.analysis_cache import get_analysis_cache

# Import Learning Engine
from src.learning_engine import get_learning_engine

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))

class TelegramBot:
    def __init__(self):
        if not TELEGRAM_TOKEN:
            log_error("TELEGRAM_TOKEN not set!")
            return
            
        self.bot = telebot.TeleBot(TELEGRAM_TOKEN)
        self.brain = get_brain()
        self.alpaca = get_alpaca()
        self.learning_engine = get_learning_engine()
        self.cache = get_analysis_cache()
        
        # State storage for pending confirmations
        # key: message_id, value: analysis_json
        self.pending_trades = {}
        
        self._setup_handlers()
        log_ok("Telegram Bot initialized")

    def _setup_handlers(self):
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            if not self._check_auth(message): return
            self.bot.reply_to(message, """
🤖 **CIO Trading Bot**
---------------------
/analyze [TICKER] - Analyze a stock
/trade [TICKER] - Force trade setup
/portfolio - View open positions
/close [TICKER] - Close position
/learn - Trigger learning feedback loop
/clear - Clear cache
""")

        @self.bot.message_handler(commands=['analyze'])
        def handle_analyze(message):
            if not self._check_auth(message): return
            self._handle_analysis_command(message)
            
        @self.bot.message_handler(commands=['trade'])
        def handle_trade(message):
            if not self._check_auth(message): return
            # Similar to analyze but forces skip_cache=True
            self._handle_analysis_command(message, skip_cache=True)

        @self.bot.message_handler(commands=['portfolio'])
        def handle_portfolio(message):
            if not self._check_auth(message): return
            self._send_portfolio_status(message)

        @self.bot.message_handler(commands=['close'])
        def handle_close(message):
            if not self._check_auth(message): return
            self._handle_close_command(message)

        @self.bot.message_handler(commands=['clear'])
        def handle_clear(message):
            if not self._check_auth(message): return
            self.cache.clear()
            self.bot.reply_to(message, "🧹 Cache cleared.")

        @self.bot.message_handler(commands=['learn'])
        def handle_learn(message):
            if not self._check_auth(message): return
            self._cmd_learn(message)

        @self.bot.callback_query_handler(func=lambda call: True)
        def handle_callback(call):
            self._handle_button_click(call)

    def _check_auth(self, message):
        if message.from_user.id != ALLOWED_USER_ID:
            self.bot.reply_to(message, "⛔ Unauthorized Access")
            return False
        return True

    def _handle_analysis_command(self, message, skip_cache=False):
        try:
            parts = message.text.split()
            if len(parts) < 2:
                self.bot.reply_to(message, "⚠️ Usage: /analyze TICKER")
                return
            
            ticker = parts[1].upper()
            self.bot.send_message(message.chat.id, f"🔍 Analyzing {ticker}...")
            
            # Run AI Analysis
            analysis = self.brain.analyze_ticker(ticker, skip_cache=skip_cache)
            
            if analysis.get("error"):
                self.bot.reply_to(message, f"❌ Error: {analysis['error']}")
                return

            # Format Response
            response_text = self.brain.format_hebrew_response(analysis)
            
            # Create Keyboard if Action is BUY/SELL
            markup = None
            action = analysis.get("action", "PASS")
            
            if action in ["BUY", "SELL"]:
                markup = InlineKeyboardMarkup()
                
                # Check validation status
                validation = analysis.get("validation", {})
                passed_validation = validation.get("passed", False)
                
                if passed_validation:
                    btn_execute = InlineKeyboardButton(f"🚀 Execute {action}", callback_data=f"exec_{action}_{ticker}")
                    markup.add(btn_execute)
                else:
                    # If failed validation, show warning button (or maybe still allow override?)
                    btn_override = InlineKeyboardButton(f"⚠️ Force {action} (Risk!)", callback_data=f"force_{action}_{ticker}")
                    markup.add(btn_override)
                
                btn_cancel = InlineKeyboardButton("❌ Cancel", callback_data="cancel")
                markup.add(btn_cancel)

            # Send Message
            sent_msg = self.bot.send_message(message.chat.id, response_text, reply_markup=markup, parse_mode="Markdown")
            
            # Store analysis for execution callback
            if markup:
                self.pending_trades[sent_msg.message_id] = analysis
                
                # Auto-Trade Logic (if configured)
                # For safety, we only auto-trade if specifically enabled in env and validation passed
                if os.getenv("AUTO_TRADE_ENABLED") == "true" and passed_validation and analysis.get("conviction") == "HIGH":
                     self._execute_auto_trade(message.chat.id, analysis, sent_msg.message_id)

        except Exception as e:
            log_error(f"Telegram command failed: {e}")
            self.bot.reply_to(message, f"❌ System Error: {e}")

    def _handle_button_click(self, call):
        try:
            # Parse callback data
            data = call.data
            chat_id = call.message.chat.id
            message_id = call.message.message_id
            
            if data == "cancel":
                self.bot.edit_message_reply_markup(chat_id, message_id, reply_markup=None)
                self.bot.send_message(chat_id, "❌ Trade cancelled.")
                return

            if data.startswith("exec_") or data.startswith("force_"):
                parts = data.split("_")
                # action = parts[1] # BUY/SELL
                # ticker = parts[2]
                
                # Retrieve original analysis
                # Note: This relies on message_id mapping which might persist only in memory
                # If bot restarts, this breaks. In prod, use Redis or similar.
                analysis = self.pending_trades.get(message_id)
                
                if not analysis:
                    self.bot.answer_callback_query(call.id, "⚠️ Trade data expired.")
                    return
                
                # Execute Trade
                self._execute_trade_action(chat_id, analysis, message_id)
                
                # Remove buttons
                self.bot.edit_message_reply_markup(chat_id, message_id, reply_markup=None)

        except Exception as e:
            log_error(f"Callback handler failed: {e}")

    def _execute_trade_action(self, chat_id, analysis, message_id):
        """Execute the trade via Alpaca."""
        ticker = analysis.get("ticker")
        action = analysis.get("action")
        plan = analysis.get("plan", {})
        
        self.bot.send_message(chat_id, f"🚀 Executing {action} {ticker}...")
        
        try:
            # Convert analysis to order parameters
            qty = self.alpaca.calculate_position_size(analysis.get("price", 0), plan.get("invalidation"))
            if qty <= 0:
                self.bot.send_message(chat_id, "⚠️ Position size too small (Risk too high?)")
                return

            side = "buy" if action == "BUY" else "sell"
            
            # Extract prices safely
            take_profit = plan.get("targets", [None])[0]
            stop_loss = plan.get("invalidation")
            
            # Place Order
            result = self.alpaca.place_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type="market",
                time_in_force="gtc",
                take_profit=take_profit,
                stop_loss=stop_loss
            )
            
            if result:
                self.bot.send_message(chat_id, f"✅ Order Placed!\nID: `{result['id']}`\nQty: {qty}", parse_mode="Markdown")
                
                # LOG TO LEARNING ENGINE
                analysis['qty'] = qty
                self.learning_engine.log_trade_entry(analysis, result['id'])
                
            else:
                self.bot.send_message(chat_id, "❌ Order Failed (Check logs)")

        except Exception as e:
            log_error(f"Trade execution failed: {e}")
            self.bot.send_message(chat_id, f"❌ Execution Error: {e}")

    def _execute_auto_trade(self, chat_id, analysis, message_id):
        """Handle auto-trading execution."""
        log_info(f"🤖 Auto-Trading triggered for {analysis['ticker']}")
        self.bot.send_message(chat_id, "🤖 Auto-Trade Initiated...")
        self._execute_trade_action(chat_id, analysis, message_id)

    def _send_portfolio_status(self, message):
        """Show current open positions."""
        positions = self.alpaca.get_positions()
        if not positions:
            self.bot.reply_to(message, "📂 Portfolio is empty.")
            return

        msg = "📂 **Portfolio Status:**\n\n"
        total_pnl = 0.0
        
        for p in positions:
            pnl = float(p.unrealized_pl)
            total_pnl += pnl
            emoji = "🟢" if pnl >= 0 else "🔴"
            msg += f"• **{p.symbol}**: {p.qty} shares | {emoji} ${pnl:.2f}\n"
        
        msg += f"\n💰 **Total PnL:** ${total_pnl:.2f}"
        self.bot.reply_to(message, msg, parse_mode="Markdown")

    def _handle_close_command(self, message):
        """Close a specific position."""
        parts = message.text.split()
        if len(parts) < 2:
            self.bot.reply_to(message, "⚠️ Usage: /close TICKER")
            return
        
        ticker = parts[1].upper()
        if self.alpaca.close_position(ticker):
            self.bot.reply_to(message, f"✅ Closed position for {ticker}")
        else:
            self.bot.reply_to(message, f"❌ Failed to close {ticker} (Position exists?)")

    def _cmd_learn(self, message):
        """Manual trigger for learning feedback loop."""
        self.bot.reply_to(message, "🧠 Syncing trade history & analyzing lessons...")
        
        try:
            # 1. Sync outcomes
            updated = self.learning_engine.update_trade_outcomes(self.alpaca.api)
            
            # 2. Run critique
            report = self.learning_engine.analyze_past_performance()
            
            response = f"✅ Learning Cycle Complete.\n\n🔄 Trades Updated: {updated}\n📝 Report: {report}"
            self.bot.send_message(message.chat.id, response)
            
        except Exception as e:
            log_error(f"Learning command failed: {e}")
            self.bot.reply_to(message, f"❌ Learning Error: {e}")

    def run(self):
        log_ok("Telegram Bot polling started...")
        self.bot.infinity_polling()

if __name__ == "__main__":
    bot = TelegramBot()
    bot.run()
