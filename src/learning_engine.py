"""
Learning Engine - The Post-Mortem Analysis System
=================================================
Handles trade logging, outcome tracking, and AI critique generation.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.logger import log_info, log_error, log_ok, log_warn, log_ai
from src.ai_brain import get_brain

# Paths
DATA_DIR = "data"
TRADE_HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
LESSONS_FILE = os.path.join("config", "lessons_learned.txt")

class LearningEngine:
    def __init__(self):
        self._ensure_paths()

    def _ensure_paths(self):
        """Ensure data directories and files exist."""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        if not os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE, 'w') as f:
                json.dump([], f)

        if not os.path.exists(os.path.dirname(LESSONS_FILE)):
            os.makedirs(os.path.dirname(LESSONS_FILE), exist_ok=True)

        if not os.path.exists(LESSONS_FILE):
            with open(LESSONS_FILE, 'w') as f:
                f.write("")

    def log_trade_entry(self, trade_data: Dict[str, Any], order_id: str):
        """
        Log a new trade entry with full context.
        """
        try:
            trades = self._load_trades()

            # Enrich trade data
            entry_record = {
                "order_id": order_id,
                "ticker": trade_data.get("ticker"),
                "action": trade_data.get("action"),
                "entry_price": trade_data.get("entry"),
                "stop_loss": trade_data.get("stop_loss"),
                "take_profit": trade_data.get("take_profit"),
                "entry_time": datetime.now().isoformat(),
                "status": "OPEN",
                "pnl_pct": 0.0,
                "outcome_reason": None,
                "ai_analysis": trade_data.get("analysis", {}),  # Full AI JSON
                "market_data_snapshot": trade_data.get("analysis", {}).get("raw_data", {}) # Snapshot if available
            }

            trades.append(entry_record)
            self._save_trades(trades)
            log_info(f"📝 Trade logged to memory: {entry_record['ticker']} ({order_id})")

        except Exception as e:
            log_error(f"Failed to log trade entry: {e}")

    def update_trade_outcomes(self, alpaca_client):
        """
        Sync with Alpaca to find closed trades and update outcomes.
        """
        if not alpaca_client:
            log_error("Cannot update outcomes: Alpaca client not matched")
            return 0

        try:
            from alpaca.trading.enums import OrderStatus

            trades = self._load_trades()
            updated_count = 0

            # Filter for OPEN trades
            open_trades = [t for t in trades if t.get("status") == "OPEN"]

            if not open_trades:
                log_info("No open trades to sync.")
                return 0

            # Check status for each open trade
            for trade in open_trades:
                order_id = trade.get("order_id")
                try:
                    order = alpaca_client.get_order(order_id)

                    if order.status in [OrderStatus.FILLED, OrderStatus.CLOSED]:
                        # Check if it was entry or exist.
                        # Actually we need to check if the POSITION is closed.
                        # This is a simplification. For a real system we'd check positions or linked exit orders.
                        # Assuming 'order' here is the ENTRY order.
                        # If entry is plain filled, we need to find the exit.

                        # Simplified logic: Check if we have a closed position or recent filled install
                        # Better: Check closed orders for this symbol after the entry time
                        entry_time = order.created_at

                        # Get closed orders for this symbol
                        closed_orders = alpaca_client.get_orders(
                            status=OrderStatus.CLOSED,
                            symbols=[trade['ticker']],
                            limit=5,
                            after=entry_time
                        )

                        # Look for a compliant exit
                        exit_order = None
                        for co in closed_orders:
                            if co.side != order.side: # Opposite side
                                exit_order = co
                                break

                        if exit_order:
                            # Calculate PnL
                            fill_price = float(exit_order.filled_avg_price) if exit_order.filled_avg_price else 0
                            entry_price = float(trade['entry_price'])
                            qty = float(exit_order.filled_qty)

                            if trade['action'] == 'BUY':
                                pnl = (fill_price - entry_price) * qty
                                pnl_pct = (fill_price - entry_price) / entry_price * 100
                            else:
                                pnl = (entry_price - fill_price) * qty
                                pnl_pct = (entry_price - fill_price) / entry_price * 100

                            trade['status'] = "CLOSED"
                            trade['exit_price'] = fill_price
                            trade['exit_time'] = exit_order.filled_at.isoformat() if exit_order.filled_at else datetime.now().isoformat()
                            trade['pnl_abs'] = round(pnl, 2)
                            trade['pnl_pct'] = round(pnl_pct, 2)

                            # Determine reason
                            if trade['action'] == 'BUY':
                                if fill_price >= trade.get('take_profit', float('inf')):
                                    trade['outcome_reason'] = "TAKE_PROFIT"
                                elif fill_price <= trade.get('stop_loss', float('-inf')):
                                    trade['outcome_reason'] = "STOP_LOSS"
                                else:
                                    trade['outcome_reason'] = "MANUAL_EXIT"

                            updated_count += 1
                            log_ok(f"🔄 updated outcome for {trade['ticker']}: {trade['pnl_pct']}% ({trade['outcome_reason']})")

                except Exception as e:
                    log_warn(f"Could not check order {order_id}: {e}")

            if updated_count > 0:
                self._save_trades(trades)

            return updated_count

        except Exception as e:
            log_error(f"Error updating outcomes: {e}")
            return 0

    def analyze_past_performance(self):
        """
        The Critique Agent: Analyzes recent closed trades and generates lessons.
        """
        trades = self._load_trades()

        # Get recently closed trades that haven't been analyzed yet
        # (For this simplified version, we just take last 5 closed)
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
        if not closed_trades:
            log_info("No closed trades to analyze.")
            return []

        target_trades = closed_trades[-5:]
        new_lessons = []

        brain = get_brain()

        for trade in target_trades:
            # Skip if already analyzed (you might want to add a flag)
            if trade.get("feedback_generated"):
                continue

            ticker = trade['ticker']
            pnl = trade.get('pnl_pct', 0)
            reason = trade.get('outcome_reason', 'UNKNOWN')

            log_ai(f"🕵️ Analyzing trade execution for {ticker} (PnL: {pnl}%)")

            # Construct Prompt
            prompt = f"""
            # POST-MORTEM ANALYSIS

            You previously analyzed {ticker} and executed a {trade['action']}.

            ## YOUR ORIGINAL ANALYSIS:
            {json.dumps(trade.get('ai_analysis', {}).get('reasoning', 'N/A'))}

            ## ACTUAL OUTCOME:
            - PnL: {pnl}%
            - Result: {reason}
            - Entry: {trade.get('entry_price')}
            - Exit: {trade.get('exit_price')}

            ## TASK:
            Analyze WHY you were wrong (if negative) or what worked (if positive).
            Provide a single, concise "Lesson Learned" rule to add to your system prompt.

            Format: "LESSON: [The lesson text]"
            Example: "LESSON: Avoid buying breakout stocks when RSI > 80."
            """

            try:
                # Call AI
                response = brain._call_ai_api(prompt, ticker=ticker)

                # Extract Lesson
                if "LESSON:" in response:
                    lesson = response.split("LESSON:")[1].strip().split("\n")[0]
                    self._append_lesson(lesson)
                    new_lessons.append(f"{ticker}: {lesson}")
                    trade['feedback_generated'] = True
                    log_ok(f"💡 New Lesson: {lesson}")

            except Exception as e:
                log_error(f" critique failed: {e}")

        self._save_trades(trades)
        return new_lessons

    def _append_lesson(self, lesson: str):
        """Append a validated lesson to the config file."""
        try:
            with open(LESSONS_FILE, 'r', encoding='utf-8') as f:
                content = f.read()

            if lesson not in content:
                with open(LESSONS_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"\n- {lesson}")

        except Exception as e:
            log_error(f"Could not save lesson: {e}")

    def _load_trades(self) -> List[Dict]:
        if os.path.exists(TRADE_HISTORY_FILE):
             with open(TRADE_HISTORY_FILE, 'r') as f:
                try:
                    return json.load(f)
                except:
                    return []
        return []

    def _save_trades(self, trades: List[Dict]):
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(trades, f, indent=2)

# Singleton
_learning_engine = None

def get_learning_engine():
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = LearningEngine()
    return _learning_engine
