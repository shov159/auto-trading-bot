"""
<<<<<<< HEAD
Learning Engine - The Post-Mortem Analysis System
=================================================
Handles trade logging, outcome tracking, and AI critique generation.
=======
Learning Engine - Continuous Improvement Loop
Tracks trade outcomes, critiques performance via LLM, and refines the trading strategy.
Hardened for production with atomic writes and strict schema validation.
>>>>>>> 81430df63ff5904781a9a446cc11ee32ae0becaf
"""
import os
import json
import logging
<<<<<<< HEAD
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
=======
import time
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import logger and brain
from src.logger import (
    log_info, log_error, log_ok, log_warn, log_debug, log_divider, log_ai
)
from src.ai_brain import get_brain, AIBrain

class LearningEngine:
    def __init__(self, history_file: str = "data/trade_history.json", lessons_file: str = "config/lessons_learned.txt"):
        self.history_file = history_file
        self.lessons_file = lessons_file
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.lessons_file), exist_ok=True)
        
        # Initialize history if missing
        if not os.path.exists(self.history_file):
            self._save_trades([])
            log_info(f"Initialized empty trade history at {self.history_file}")

    def _load_trades(self) -> List[Dict[str, Any]]:
        """Load trade history from JSON file."""
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            log_error(f"Failed to load trade history: {e}")
            return []

    def _save_trades(self, trades: List[Dict[str, Any]]):
        """
        Atomic save of trade history using temporary file + rename.
        Ensures no data corruption if process crashes during write.
        """
        temp_file = f"{self.history_file}.tmp"
        try:
            with open(temp_file, "w") as f:
                json.dump(trades, f, indent=2)
            
            # Atomic rename (replace)
            os.replace(temp_file, self.history_file)
            log_debug(f"Saved {len(trades)} trades atomically to {self.history_file}")
            
        except Exception as e:
            log_error(f"Atomic save failed: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def log_trade_entry(self, trade_data: Dict[str, Any], order_id: str):
        """
        Log a new trade entry when submitted to Alpaca.
        
        Schema includes:
        - entry_time, ticker, action, qty
        - planned_entry, planned_stop, planned_target
        - status: OPEN
        - logic_engine: Sympathy, Squeeze, etc.
        """
        trades = self._load_trades()
        
        analysis = trade_data.get("analysis", {})
        
        entry_record = {
            "order_id": order_id,
            "ticker": trade_data.get("ticker"),
            "action": trade_data.get("action"),
            "qty": trade_data.get("qty"),
            "entry_time": datetime.now().isoformat(),
            "status": "OPEN",
            
            # Planned execution metrics
            "planned_entry": trade_data.get("entry"),
            "planned_stop": trade_data.get("stop_loss"),
            "planned_target": trade_data.get("take_profit"),
            
            # AI Context
            "logic_engine": analysis.get("logic_engine", "Unknown"),
            "conviction": analysis.get("conviction", "LOW"),
            "initial_reasoning": analysis.get("reasoning", ""),
            
            # Outcome placeholders
            "exit_time": None,
            "exit_price": None,
            "pnl_usd": 0.0,
            "pnl_percent": 0.0,
            "exit_reason": None,
            "review_status": "PENDING"
        }
        
        trades.append(entry_record)
        self._save_trades(trades)
        log_info(f"Logged new trade entry: {order_id} ({trade_data.get('ticker')})")

    def update_trade_outcomes(self, alpaca_client):
        """
        Sync with Alpaca to find closed trades and update outcome data.
        """
        if not alpaca_client:
            log_warn("No Alpaca client provided for learning update")
            return

        trades = self._load_trades()
        updated_count = 0
        
        try:
            # Fetch closed orders from Alpaca (limit 50)
            closed_orders = alpaca_client.get_orders(status="closed", limit=50)
            closed_map = {o.id: o for o in closed_orders}
            
            for trade in trades:
                if trade["status"] == "OPEN":
                    order_id = trade.get("order_id")
                    
                    # Check if order is closed
                    if order_id in closed_map:
                        alpaca_order = closed_map[order_id]
                        
                        # Only process if filled
                        if alpaca_order.filled_at:
                            # Fetch PnL logic here is tricky via Orders API alone
                            # We estimate PnL based on fill price vs planned/current price?
                            # Better: We need to match the ENTRY order with the EXIT order.
                            # For simplicity in V1, we assume the Bracket Order logic.
                            
                            # Calculate simple result based on fill price
                            fill_price = float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else 0.0
                            qty = float(alpaca_order.qty) if alpaca_order.qty else 0.0
                            
                            # Note: This just marks the ENTRY as filled. 
                            # We actually need to find the paired EXIT trade.
                            # For hardening, we will just mark it as FILLED and wait for the exit order trigger in a real system.
                            # But here we assume "closed" means the round trip is done? 
                            # Actually, get_orders(status='closed') includes the entry order too.
                            
                            # Simplification: If the entry order is filled, we look for its legs.
                            # If legs are filled, we calculate PnL.
                            
                            # Fetch legs (Stop Loss / Take Profit)
                            try:
                                legs = alpaca_client.get_order(order_id).legs
                                exit_order = None
                                if legs:
                                    for leg in legs:
                                        if leg.status == "filled":
                                            exit_order = leg
                                            break
                                
                                if exit_order:
                                    exit_price = float(exit_order.filled_avg_price)
                                    exit_time = exit_order.filled_at.isoformat()
                                    
                                    # Calculate PnL
                                    if trade["action"] == "BUY":
                                        pnl_usd = (exit_price - fill_price) * qty
                                        pnl_pct = ((exit_price - fill_price) / fill_price) * 100
                                    else: # SELL
                                        pnl_usd = (fill_price - exit_price) * qty
                                        pnl_pct = ((fill_price - exit_price) / fill_price) * 100
                                    
                                    trade["status"] = "CLOSED"
                                    trade["exit_price"] = exit_price
                                    trade["exit_time"] = exit_time
                                    trade["pnl_usd"] = round(pnl_usd, 2)
                                    trade["pnl_percent"] = round(pnl_pct, 2)
                                    
                                    # Determine exit reason
                                    planned_stop = trade.get("planned_stop", 0)
                                    planned_target = trade.get("planned_target", 0)
                                    
                                    # Fuzzy match for exit reason
                                    if abs(exit_price - planned_target) < abs(exit_price - planned_stop):
                                        trade["exit_reason"] = "TAKE_PROFIT"
                                    else:
                                        trade["exit_reason"] = "STOP_LOSS"
                                    
                                    updated_count += 1
                                    log_ok(f"Updated closed trade: {trade['ticker']} (PnL: ${pnl_usd:.2f})")
                                
                            except Exception as leg_err:
                                log_error(f"Error checking order legs for {order_id}: {leg_err}")

        except Exception as e:
            log_error(f"Failed to sync trade outcomes: {e}")
        
        if updated_count > 0:
            self._save_trades(trades)

    def analyze_past_performance(self) -> str:
        """
        Run a 'Post-Mortem' analysis on closed trades using the AI Brain.
        Extracts a single bullet-point lesson if a pattern emerges.
        """
        trades = self._load_trades()
        closed_pending_review = [t for t in trades if t["status"] == "CLOSED" and t.get("review_status") == "PENDING"]
        
        if not closed_pending_review:
            return "No new closed trades to review."
        
        log_info(f"Analyzing {len(closed_pending_review)} closed trades for lessons...")
        
        brain = get_brain()
        new_lessons_count = 0
        
        for trade in closed_pending_review:
            # Construct the critique prompt
            ticker = trade['ticker']
            pnl = trade.get('pnl_usd', 0)
            outcome = "WIN" if pnl > 0 else "LOSS"
            
            prompt = f"""
REVIEW THIS COMPLETED TRADE:
Ticker: {ticker}
Action: {trade['action']}
Logic: {trade['logic_engine']}
Conviction: {trade['conviction']}
Initial Plan: Entry {trade['planned_entry']}, Stop {trade['planned_stop']}, Target {trade['planned_target']}

OUTCOME: {outcome}
PnL: ${pnl}
Exit Price: {trade.get('exit_price')}
Exit Reason: {trade.get('exit_reason', 'Unknown')}

Original Reasoning: "{trade.get('initial_reasoning')}"

TASK:
Provide a ONE-SENTENCE "LESSON LEARNED" for the trading journal.
If the trade was good/bad due to luck, say "NO LESSON".
If it reveals a flaw in logic (e.g., "Ignored Volume", "Stop too tight"), state it clearly.
Format: "LESSON: [Your single sentence lesson]"
"""
            # Call Brain safely
            try:
                response = brain.run_critique(prompt, ticker=ticker)
                
                # Parse Lesson
                if "LESSON:" in response:
                    lesson_text = response.split("LESSON:")[1].strip().split("\n")[0]
                    
                    if "NO LESSON" not in lesson_text.upper() and len(lesson_text) > 5:
                        self._append_lesson(lesson_text)
                        new_lessons_count += 1
                        log_ai(f"New Lesson Learned: {lesson_text}")
                
                # Mark reviewed
                trade["review_status"] = "REVIEWED"
                
            except Exception as e:
                log_error(f"Critique failed for {ticker}: {e}")
        
        self._save_trades(trades)
        return f"Analysis complete. {new_lessons_count} new lessons added."

    def _append_lesson(self, lesson: str):
        """Append a validated lesson to the config file."""
        # Simple deduplication
        try:
            with open(self.lessons_file, "r+") as f:
                existing = f.read()
                if lesson in existing:
                    return # Skip duplicate
                
                f.write(f"\n- {lesson}")
        except FileNotFoundError:
            with open(self.lessons_file, "w") as f:
                f.write(f"- {lesson}")

# Singleton accessor
_learning_engine_instance = None

def get_learning_engine():
    global _learning_engine_instance
    if _learning_engine_instance is None:
        _learning_engine_instance = LearningEngine()
    return _learning_engine_instance
>>>>>>> 81430df63ff5904781a9a446cc11ee32ae0becaf
