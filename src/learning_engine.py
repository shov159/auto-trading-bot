import json
import os
import re
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
        
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "w") as f:
                json.dump([], f)

    def log_trade_entry(self, trade_data: Dict[str, Any], order_id: str):
        """Log a new trade entry to history."""
        try:
            # Ensure file exists
            if not os.path.exists(HISTORY_FILE):
                 with open(HISTORY_FILE, "w") as f: json.dump([], f)

            with open(HISTORY_FILE, "r") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
            
            # Enrich with initial status
            entry_record = {
                "order_id": order_id,
                "ticker": trade_data.get("ticker"),
                "action": trade_data.get("action"),
                "entry_time": datetime.now().isoformat(),
                "status": "OPEN",
                "planned_entry": trade_data.get("entry"),
                "planned_stop": trade_data.get("stop_loss"),
                "planned_target": trade_data.get("take_profit"),
                "reasoning": trade_data.get("analysis", {}).get("reasoning", ""),
                "market_context": trade_data.get("analysis", {}).get("raw_data", {}),
                "outcome": None,
                "critique": None,
                "lesson": None
            }
            
            history.append(entry_record)
            
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=2)
                
            log_info(f"Logged trade entry for {entry_record['ticker']} (Order: {order_id})")
            
        except Exception as e:
            log_error(f"Failed to log trade entry: {e}")

    def update_trade_outcomes(self, alpaca_client):
        """Sync closed trades with Alpaca history."""
        if not alpaca_client:
            return
            
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
            
            updated = False
            
            # Get closed orders from Alpaca (last 50)
            # Note: This is a simplified check. A full check would verify filled status.
            try:
                closed_orders = alpaca_client.get_orders(status="closed", limit=50)
                orders_map = {str(o.id): o for o in closed_orders}
            except Exception as e:
                log_warn(f"Could not fetch Alpaca orders: {e}")
                return

            for trade in history:
                if trade["status"] == "OPEN":
                    order_id = str(trade.get("order_id"))
                    if order_id in orders_map:
                        alpaca_order = orders_map[order_id]
                        
                        trade["status"] = "CLOSED"
                        trade["close_time"] = str(alpaca_order.filled_at) if alpaca_order.filled_at else datetime.now().isoformat()
                        
                        # Store fill price as approx outcome for now
                        fill_price = float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else 0
                        trade["filled_entry_price"] = fill_price
                        
                        updated = True
                        log_ok(f"Marked trade {trade['ticker']} as CLOSED")
            
            if updated:
                with open(HISTORY_FILE, "w") as f:
                    json.dump(history, f, indent=2)
                
        except Exception as e:
            log_error(f"Failed to update trade outcomes: {e}")

    def analyze_past_performance(self):
        """Critique recent trades and generate lessons."""
        log_ai("Running Post-Mortem Analysis...")
        
        try:
            if not os.path.exists(HISTORY_FILE):
                return "No trade history found."

            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
            
            # Filter for closed trades without critique
            to_analyze = [t for t in history if t["status"] == "CLOSED" and not t.get("critique")]
            
            if not to_analyze:
                log_info("No new closed trades to analyze.")
                return "No new trades to analyze."
            
            brain = get_brain()
            lessons = []
            
            for trade in to_analyze[-3:]: # Analyze last 3 max
                log_ai(f"Critiquing trade: {trade['ticker']}...")
                prompt = self._build_critique_prompt(trade)
                
                # Use brain to call API
                response = brain._call_ai_api(prompt, ticker=trade["ticker"])
                
                # Extract lesson
                lesson = self._extract_lesson(response)
                
                trade["critique"] = response
                if lesson:
                    trade["lesson"] = lesson
                    lessons.append(lesson)
            
            # Save updated history
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=2)
            
            # Append unique lessons to file
            if lessons:
                count = self._append_lessons(lessons)
                return f"Learned {count} new lessons."
            
            return "Analysis complete. No new significant lessons."
            
        except Exception as e:
            log_error(f"Analysis failed: {e}")
            return f"Error: {e}"

    def _build_critique_prompt(self, trade):
        context = trade.get('market_context', {})
        return f"""
# POST-MORTEM ANALYSIS

I made a trade that is now CLOSED. I need you to critique it and provide a LESSON LEARNED.

## TRADE DETAILS
- Ticker: {trade.get('ticker')}
- Action: {trade.get('action')}
- Date: {trade.get('entry_time')}
- Reasoning: {trade.get('reasoning')}

## MARKET CONTEXT (At Entry)
- Price: {context.get('price')}
- RSI: {context.get('rsi')}
- Volume Ratio: {context.get('volume_ratio')}
- Sector: {context.get('sector')}

## OUTCOME
The trade is closed. 
(Note: Exact PnL is not provided, so assume we need to verify if the ENTRY LOGIC was sound based on the context).

## TASK
1. Critique the reasoning. Was it too aggressive? Did I ignore the RSI?
2. Provide a single, concise "LESSON" that I can add to my system prompt to avoid this mistake (or reinforce this success) in the future.
3. Format: "LESSON: [Your lesson here]"

Example:
LESSON: Avoid entering long positions when RSI is above 75, even if news is good.
"""

    def _extract_lesson(self, text):
        match = re.search(r"LESSON: (.*)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
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
                # Add newline if file is not empty and doesn't end with one
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
