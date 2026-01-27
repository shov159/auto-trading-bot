import json
import os
import re
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.logger import log_info, log_error, log_ok, log_warn, log_ai
from src.ai_brain import get_brain

HISTORY_FILE = "data/trade_history.json"
LESSONS_FILE = "config/lessons_learned.txt"

class LearningEngine:
    def __init__(self):
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        if not os.path.exists("data"):
            try:
                os.makedirs("data")
            except FileExistsError:
                pass
        
        # Initialize empty history if not exists
        if not os.path.exists(HISTORY_FILE):
            self._atomic_write([])

    def _atomic_write(self, data: List[Dict]):
        """Write data to history file atomically to prevent corruption."""
        try:
            # Create temp file
            fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(HISTORY_FILE), text=True)
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            shutil.move(temp_path, HISTORY_FILE)
        except Exception as e:
            log_error(f"Atomic write failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _load_history(self) -> List[Dict]:
        """Safely load trade history."""
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            log_error("Corrupt history file, starting fresh backup")
            # Backup corrupt file if it exists and is not empty
            if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
                shutil.copy(HISTORY_FILE, f"{HISTORY_FILE}.bak")
            return []

    def log_trade_entry(self, trade_data: Dict[str, Any], order_id: str):
        """
        Log a new trade entry with comprehensive context snapshot.
        """
        try:
            history = self._load_history()
            
            # Create standardized trade record
            entry_record = {
                "order_id": str(order_id),
                "ticker": trade_data.get("ticker"),
                "action": trade_data.get("action"),
                "qty": trade_data.get("qty", 0),
                "entry_time": datetime.now().isoformat(),
                "status": "OPEN",
                
                # Plan details
                "planned_entry": trade_data.get("entry"),
                "planned_stop": trade_data.get("stop_loss"),
                "planned_target": trade_data.get("take_profit"),
                "risk_amount": trade_data.get("risk", 0),
                
                # Context Snapshot
                "reasoning": trade_data.get("analysis", {}).get("reasoning", ""),
                "market_context": trade_data.get("analysis", {}).get("raw_data", {}),
                "analysis_full": trade_data.get("analysis", {}),
                
                # Outcomes (to be filled later)
                "close_time": None,
                "filled_entry_price": None,
                "filled_exit_price": None,
                "pnl_usd": None,
                "pnl_pct": None,
                "exit_reason": None, # TP, SL, MANUAL
                
                # Learning
                "critique": None,
                "lesson": None
            }
            
            history.append(entry_record)
            self._atomic_write(history)
            log_info(f"Logged trade entry: {entry_record['ticker']} (ID: {order_id})")
            
        except Exception as e:
            log_error(f"Failed to log trade entry: {e}")

    def update_trade_outcomes(self, alpaca_client):
        """
        Sync open trades with Alpaca to detect closures and calculate PnL.
        Handles bracket orders by checking child orders (legs).
        """
        if not alpaca_client:
            log_warn("No Alpaca client provided for outcome sync")
            return
            
        try:
            history = self._load_history()
            updated = False
            
            # Get all closed orders recently (limit 50)
            # We fetch closed orders to find exits
            try:
                closed_orders = alpaca_client.get_orders(status="closed", limit=100, nested=True)
                # Map order ID to order object for easy lookup
                orders_map = {str(o.id): o for o in closed_orders}
            except Exception as e:
                log_warn(f"Alpaca sync failed: {e}")
                return

            for trade in history:
                if trade["status"] == "OPEN":
                    parent_id = str(trade.get("order_id"))
                    
                    # 1. Check if parent order is closed/filled
                    if parent_id in orders_map:
                        parent_order = orders_map[parent_id]
                        
                        # Update fill info if not set
                        if parent_order.status == 'filled' and not trade.get("filled_entry_price"):
                            trade["filled_entry_price"] = float(parent_order.filled_avg_price) if parent_order.filled_avg_price else 0
                        
                        # 2. Check legs (TP/SL) if available
                        # Alpaca 'nested=True' returns legs in the order object usually, 
                        # but we might need to search the closed_orders list for orders with this parent_id
                        
                        exit_order = None
                        exit_reason = "UNKNOWN"
                        
                        # Search for child orders in closed_orders
                        for o in closed_orders:
                            if str(o.parent_id) == parent_id and o.status == 'filled':
                                exit_order = o
                                # Determine if SL or TP based on price relative to entry
                                # Simplified logic: usually we can check order type (STOP/LIMIT)
                                if o.order_type == 'stop' or o.order_type == 'stop_limit':
                                    exit_reason = "STOP_LOSS"
                                elif o.order_type == 'limit':
                                    exit_reason = "TAKE_PROFIT"
                                break
                        
                        # If we found an exit
                        if exit_order:
                            trade["status"] = "CLOSED"
                            trade["close_time"] = str(exit_order.filled_at)
                            trade["filled_exit_price"] = float(exit_order.filled_avg_price)
                            trade["exit_reason"] = exit_reason
                            
                            # Calculate PnL
                            if trade["filled_entry_price"] and trade["filled_exit_price"]:
                                entry = trade["filled_entry_price"]
                                exit_px = trade["filled_exit_price"]
                                qty = float(trade["qty"])
                                
                                if trade["action"] == "BUY":
                                    pnl = (exit_px - entry) * qty
                                    pnl_pct = (exit_px - entry) / entry
                                else: # SELL/SHORT
                                    pnl = (entry - exit_px) * qty
                                    pnl_pct = (entry - exit_px) / entry
                                
                                trade["pnl_usd"] = round(pnl, 2)
                                trade["pnl_pct"] = round(pnl_pct * 100, 2)
                                
                                log_ok(f"Trade CLOSED: {trade['ticker']} | PnL: ${pnl:.2f} ({pnl_pct:.1%}) | {exit_reason}")
                            
                            updated = True
                        
                        # Handle manual close or expiration (parent closed but no legs filled yet?)
                        elif parent_order.status in ['canceled', 'expired', 'rejected']:
                             trade["status"] = "CANCELED"
                             updated = True
            
            if updated:
                self._atomic_write(history)
                
        except Exception as e:
            log_error(f"Failed to update outcomes: {e}")

    def analyze_past_performance(self, max_trades=3) -> str:
        """
        Critique recent closed trades using the Brain.
        Returns a summary string of results.
        """
        log_ai("Running Post-Mortem Analysis...")
        
        try:
            history = self._load_history()
            
            # Filter: Closed, PnL calculated, No critique yet
            to_analyze = [
                t for t in history 
                if t["status"] == "CLOSED" 
                and t.get("pnl_usd") is not None
                and not t.get("critique")
            ]
            
            if not to_analyze:
                log_info("No eligible trades for analysis.")
                return "No new closed trades to analyze."
            
            brain = get_brain()
            new_lessons = []
            
            # Process last N trades
            for trade in to_analyze[-max_trades:]:
                log_ai(f"Critiquing {trade['ticker']} (PnL: {trade['pnl_usd']})...")
                
                prompt = self._build_critique_prompt(trade)
                
                # Call Brain public method (safe)
                response = brain.run_critique(prompt, ticker=trade["ticker"])
                
                # Extract clean lesson
                lesson = self._extract_lesson(response)
                
                trade["critique"] = response
                if lesson:
                    trade["lesson"] = lesson
                    new_lessons.append(lesson)
                    log_ok(f"Generated Lesson: {lesson}")
            
            # Save critiques
            self._atomic_write(history)
            
            # Append to lessons file
            if new_lessons:
                count = self._append_lessons(new_lessons)
                return f"Learned {count} new lessons from {len(to_analyze)} trades."
            
            return f"Analyzed {len(to_analyze)} trades but no new significant lessons found."
            
        except Exception as e:
            log_error(f"Analysis failed: {e}")
            return f"Error during analysis: {e}"

    def _build_critique_prompt(self, trade):
        """Construct a detailed prompt for the Teacher Agent."""
        ctx = trade.get('market_context', {})
        pnl_str = f"${trade.get('pnl_usd')} ({trade.get('pnl_pct')}%)"
        
        return f"""
# POST-MORTEM TRADING ANALYSIS

## SCENARIO
I executed a trade based on your analysis. It is now closed.
Your goal is to critique the decision process and extract a generalized LESSON.

## TRADE DATA
- **Ticker:** {trade.get('ticker')}
- **Action:** {trade.get('action')}
- **Entry:** ${trade.get('filled_entry_price')}
- **Exit:** ${trade.get('filled_exit_price')}
- **Result:** {pnl_str}
- **Exit Reason:** {trade.get('exit_reason')}
- **Original Reasoning:** "{trade.get('reasoning')}"

## MARKET CONTEXT (At Entry)
- Price: {ctx.get('price')}
- RSI: {ctx.get('rsi')}
- Vol Ratio: {ctx.get('volume_ratio')}
- Sector: {ctx.get('sector')}

## YOUR TASK
1. Analyze WHY the trade succeeded or failed.
2. Was the original reasoning sound? Did we ignore a red flag (like high RSI, low volume)?
3. Formulate a SINGLE, concise lesson rule.
   - Good: "Avoid buying breakouts when volume ratio is < 1.0"
   - Bad: "Don't buy TSLA next time" (Too specific)
   
## OUTPUT FORMAT
Return ONLY the lesson line starting with "LESSON:".
Example:
LESSON: Always wait for RSI to cool down below 70 before long entries.
"""

    def _extract_lesson(self, text: str) -> Optional[str]:
        """Strict extraction of LESSON: line."""
        # Clean text
        text = text.strip()
        
        # Look for explicit prefix
        match = re.search(r"LESSON:\s*(.*)", text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Validation: Length check (10-150 chars)
            if 10 < len(candidate) < 150:
                return candidate
        
        return None

    def _append_lessons(self, new_lessons: List[str]) -> int:
        """Append unique lessons to file."""
        added_count = 0
        try:
            # Load existing
            existing = set()
            if os.path.exists(LESSONS_FILE):
                with open(LESSONS_FILE, "r", encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            existing.add(line.lstrip("- "))
            
            # Append new
            with open(LESSONS_FILE, "a", encoding='utf-8') as f:
                if os.path.getsize(LESSONS_FILE) > 0:
                    f.write("\n")
                
                for lesson in new_lessons:
                    clean = lesson.replace("\n", " ").strip()
                    if clean not in existing:
                        f.write(f"- {clean}\n")
                        existing.add(clean)
                        added_count += 1
            
            return added_count
        except Exception as e:
            log_error(f"Failed to save lessons: {e}")
            return 0

_instance = None
def get_learning_engine():
    global _instance
    if _instance is None:
        _instance = LearningEngine()
    return _instance
