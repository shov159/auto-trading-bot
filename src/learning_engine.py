import json
import os
import re
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.logger import log_info, log_error, log_ok, log_warn, log_ai

HISTORY_FILE = "data/trade_history.json"
LESSONS_FILE = "config/lessons_learned.txt"

class LearningEngine:
    def __init__(self):
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Ensure data directory and history file exist."""
        if not os.path.exists("data"):
            try:
                os.makedirs("data")
            except FileExistsError:
                pass
        
        if not os.path.exists(HISTORY_FILE):
            self._save_history([])

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load trade history safely."""
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log_error(f"Failed to load trade history: {e}")
            return []

    def _save_history(self, history: List[Dict[str, Any]]):
        """Save trade history atomically."""
        try:
            # Write to temp file first
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(HISTORY_FILE), text=True)
            with os.fdopen(fd, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Atomic rename
            shutil.move(tmp_path, HISTORY_FILE)
        except Exception as e:
            log_error(f"Failed to save trade history: {e}")
            # Try to cleanup temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def log_trade_entry(self, trade_data: Dict[str, Any], order_id: str):
        """Log a new trade entry to history with extended schema."""
        try:
            history = self._load_history()
            
            # Enrich with detailed status
            analysis = trade_data.get("analysis", {})
            raw_data = analysis.get("raw_data", {})
            validation = analysis.get("validation", {})
            
            entry_record = {
                # Identity
                "order_id": str(order_id),
                "ticker": trade_data.get("ticker"),
                "action": trade_data.get("action"),
                "status": "OPEN",
                
                # Timing
                "entry_time": datetime.now().isoformat(),
                "close_time": None,
                
                # Planned Execution
                "qty": trade_data.get("qty"),
                "planned_entry": trade_data.get("entry"),
                "planned_stop": trade_data.get("stop_loss"),
                "planned_target": trade_data.get("take_profit"),
                "planned_risk": trade_data.get("risk"),
                
                # Actual Execution (to be filled later)
                "filled_entry_price": None,
                "filled_exit_price": None,
                "exit_reason": None,
                "pnl_usd": None,
                "pnl_pct": None,
                
                # Context Snapshot
                "reasoning": analysis.get("reasoning", ""),
                "market_context": {
                    "price": raw_data.get("price"),
                    "rsi": raw_data.get("rsi"),
                    "volume_ratio": raw_data.get("volume_ratio"),
                    "sector": raw_data.get("sector"),
                    "atr": raw_data.get("atr")
                },
                "validation_snapshot": validation,
                
                # Learning
                "critique": None,
                "lesson": None
            }
            
            history.append(entry_record)
            self._save_history(history)
                
            log_info(f"Logged trade entry for {entry_record['ticker']} (Order: {order_id})")
            
        except Exception as e:
            log_error(f"Failed to log trade entry: {e}")

    def update_trade_outcomes(self, alpaca_client):
        """Sync closed trades with Alpaca history and calculate outcomes."""
        if not alpaca_client:
            return 0
            
        try:
            history = self._load_history()
            updated_count = 0
            
            # Filter for OPEN trades
            open_trades = [t for t in history if t["status"] == "OPEN"]
            if not open_trades:
                return 0

            # Fetch recent closed orders (last 50 should cover recent activity)
            try:
                closed_orders = alpaca_client.get_orders(status="closed", limit=50)
                orders_map = {str(o.id): o for o in closed_orders}
            except Exception as e:
                log_warn(f"Could not fetch Alpaca orders: {e}")
                return 0

            for trade in open_trades:
                order_id = str(trade.get("order_id"))
                
                if order_id in orders_map:
                    order = orders_map[order_id]
                    
                    # Only process if actually filled
                    if order.filled_at:
                        trade["status"] = "CLOSED"
                        trade["close_time"] = str(order.filled_at)
                        
                        # Entry Fill
                        fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0
                        trade["filled_entry_price"] = fill_price
                        
                        # Attempt to find exit leg (simplified for now)
                        # Ideally we'd look up the legs, but for now we'll mark as closed
                        # and rely on manual PnL or advanced leg matching later.
                        # For simple learning, we check current price vs entry if sold,
                        # but since order is closed, we need the SELL leg.
                        # TODO: Implement full leg matching.
                        # Current fallback: assume closed means done.
                        
                        # Try to detect if SL or TP hit based on order status or legs
                        exit_reason = "MANUAL_OR_Leg"
                        
                        trade["exit_reason"] = exit_reason
                        updated_count += 1
                        log_ok(f"Marked trade {trade['ticker']} as CLOSED (Fill: ${fill_price})")
            
            if updated_count > 0:
                self._save_history(history)
            
            return updated_count
                
        except Exception as e:
            log_error(f"Failed to update trade outcomes: {e}")
            return 0

    def analyze_past_performance(self):
        """Critique recent closed trades and generate lessons."""
        log_ai("Running Post-Mortem Analysis...")
        
        try:
            history = self._load_history()
            
            # Filter for closed trades without critique
            to_analyze = [t for t in history if t["status"] == "CLOSED" and not t.get("critique")]
            
            if not to_analyze:
                log_info("No new closed trades to analyze.")
                return "No new trades to analyze."
            
            # Import brain here to avoid circular import at module level if possible, 
            # but standard import is fine if structure allows.
            from src.ai_brain import get_brain
            brain = get_brain()
            
            lessons = []
            
            # Analyze last 3 closed trades
            for trade in to_analyze[-3:]:
                log_ai(f"Critiquing trade: {trade['ticker']}...")
                
                prompt = self._build_critique_prompt(trade)
                
                # Use public method if available, else protected
                if hasattr(brain, 'run_critique'):
                    response = brain.run_critique(prompt, ticker=trade["ticker"])
                else:
                    response = brain._call_ai_api(prompt, ticker=trade["ticker"])
                
                # Extract lesson
                lesson = self._extract_lesson(response)
                
                trade["critique"] = response
                if lesson:
                    trade["lesson"] = lesson
                    lessons.append(lesson)
            
            self._save_history(history)
            
            # Append unique lessons to file
            count = 0
            if lessons:
                count = self._append_lessons(lessons)
                return f"Learned {count} new lessons from {len(to_analyze)} trades."
            
            return "Analysis complete. No significant lessons extracted."
            
        except Exception as e:
            log_error(f"Analysis failed: {e}")
            return f"Error: {e}"

    def _build_critique_prompt(self, trade):
        context = trade.get('market_context', {})
        return f"""
# POST-MORTEM TRADING ANALYSIS

I made a trade that is now CLOSED. I need you to critique it and provide a single, actionable LESSON.

## TRADE DETAILS
- Ticker: {trade.get('ticker')}
- Action: {trade.get('action')}
- Date: {trade.get('entry_time')}
- Planned Entry: {trade.get('planned_entry')}
- Filled Entry: {trade.get('filled_entry_price')}
- Reasoning: {trade.get('reasoning')}

## MARKET CONTEXT (At Entry)
- Price: {context.get('price')}
- RSI: {context.get('rsi')}
- Volume Ratio: {context.get('volume_ratio')}
- Sector: {context.get('sector')}
- ATR: {context.get('atr')}

## TASK
1. Critique the reasoning vs context. Was it impulsive? Did I ignore the RSI?
2. Provide a single "LESSON" to avoid this mistake (or reinforce this success).
3. The lesson must be generic enough to apply to future trades (e.g., "Don't buy if RSI > 75").
4. STRICT FORMAT: "LESSON: [Your lesson here]"

Example:
LESSON: Avoid entering long positions when RSI is above 75, even if news is good.
"""

    def _extract_lesson(self, text):
        """Extract lesson string from LLM response."""
        # Look for "LESSON: ..." pattern
        match = re.search(r"LESSON:\s*(.*)", text, re.IGNORECASE)
        if match:
            lesson = match.group(1).strip()
            # Basic validation
            if len(lesson) > 10 and len(lesson) < 200:
                return lesson
        return None

    def _append_lessons(self, new_lessons):
        count = 0
        try:
            existing_lessons = set()
            if os.path.exists(LESSONS_FILE):
                with open(LESSONS_FILE, "r", encoding='utf-8') as f:
                    for line in f:
                        if line.strip() and not line.startswith("#"):
                            existing_lessons.add(line.strip().lstrip("- "))
            
            with open(LESSONS_FILE, "a", encoding='utf-8') as f:
                if os.path.getsize(LESSONS_FILE) > 0:
                    f.write("\n")
                    
                for lesson in new_lessons:
                    clean_lesson = lesson.replace("\n", " ").strip()
                    if clean_lesson not in existing_lessons:
                        f.write(f"- {clean_lesson}\n")
                        log_ok(f"New Lesson Learned: {clean_lesson}")
                        count += 1
                        existing_lessons.add(clean_lesson)
            return count
                        
        except Exception as e:
            log_error(f"Failed to save lessons: {e}")
            return 0

_engine_instance = None
def get_learning_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LearningEngine()
    return _engine_instance
