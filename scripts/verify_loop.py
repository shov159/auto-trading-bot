"""
verify_loop.py
Simulates the full "Learning Loop" described in the user manual.
1. Mock AI Analysis & Trade Entry
2. Mock Alpaca Execution & Close
3. Trigger Learning Engine
4. Verify Lesson Injection
"""
import os
import sys
import json
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning_engine import get_learning_engine
from src.ai_brain import get_brain

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Simulation")

def run_simulation():
    print("\n🚀 STARTING SIMULATION: The Learning Loop\n" + "="*40)

    # 1. Setup Managers
    engine = get_learning_engine()
    brain = get_brain()

    # Ensure clean slate for test
    history_file = "data/trade_history.json"
    lessons_file = "config/lessons_learned.txt"

    # Back up existing lessons if any (optional, but good practice)
    # For now we just append, which is fine.

    # 2. Simulate Trade Entry (Bot Action)
    print("\n🔹 Step 1: Simulating Trade Entry (AAPL)")
    trade_id = "SIM_ORDER_001"
    trade_data = {
        "ticker": "AAPL",
        "action": "BUY",
        "qty": 10,
        "entry": 150.00,
        "stop_loss": 145.00,
        "take_profit": 160.00,
        "risk": 50.0,
        "analysis": {
            "reasoning": "Strong momentum but RSI is 85 (Overbought).", # Intentionally bad for learning
            "raw_data": {"rsi": 85, "price": 150}
        }
    }

    engine.log_trade_entry(trade_data, trade_id)

    # Verify Open
    with open(history_file, 'r') as f:
        hist = json.load(f)
    print(f"✅ Trade Logged: {hist[-1]['ticker']} Status: {hist[-1]['status']}")

    # 3. Simulate Trade Closing (Time passes...)
    print("\n🔹 Step 2: Simulating Trade Close (Stop Loss Hit)")

    # We mock the Alpaca client that update_trade_outcomes uses
    mock_alpaca = MagicMock()

    # Mock finding the closed order
    mock_order = MagicMock()
    mock_order.id = trade_id
    mock_order.status = "closed"
    mock_order.filled_at = datetime.now()
    mock_order.filled_avg_price = 145.00 # Hit Stop Loss
    mock_order.filled_qty = 10

    # Mock list_orders to return this order when queried
    # The logic in update_trade_outcomes gets order by ID from memory map
    # It calls alpaca_client.get_orders(status="closed", limit=50)
    mock_alpaca.get_orders.return_value = [mock_order]

    # 4. Trigger Learning (The /learn command)
    print("\n🔹 Step 3: Triggering Learning Engine")

    # Patch the AI brain to return a specific lesson so we don't use real credits
    # and to ensure deterministic output for verification
    with patch.object(brain, '_call_ai_api', return_value="LESSON: Avoid buying when RSI is above 80."):

        # A. Sync Outcomes
        count = engine.update_trade_outcomes(mock_alpaca)
        print(f"🔄 Outcomes Synced: {count} trades updated")

        # B. Analyze Performance
        lessons = engine.analyze_past_performance()

        if lessons:
            print(f"🎓 Lessons Generated:\n{lessons}")
        else:
            print("⚠️ No lessons generated (Simulation failed?)")

    # 5. Verify Lesson Injection
    print("\n🔹 Step 4: Verifying Lesson Injection")
    with open(lessons_file, 'r') as f:
        content = f.read()

    if "Avoid buying when RSI is above 80" in content:
        print("✅ SUCCESS: Lesson saved to config/lessons_learned.txt")
    else:
        print(f"❌ FAILURE: Lesson not found in file.\nContent:\n{content}")

    # 6. Verify Brain Prompt Injection
    print("\n🔹 Step 5: Verifying AI Prompt Injection")
    # Actually call the method that builds the prompt
    dummy_market_data = {
        "ticker": "TSLA",
        "price": 200, "change_pct": 2.5, "volume": 1000000,
        "avg_volume": 1000000, "rsi": 50, "atr": 5,
        "sma_20": 190, "sma_50": 180, "sma_200": 160,
        "52w_high": 250, "52w_low": 100,
        "support": 190, "resistance": 210,
        "short_interest": 5, "float_shares": 10000000, "short_ratio": 1
    }

    prompt = brain._build_user_prompt(dummy_market_data)

    if "Avoid buying when RSI is above 80" in prompt:
        print("✅ SUCCESS: Lesson found in AI Prompt!")
    else:
        print("❌ FAILURE: Lesson NOT found in AI Prompt.")

    print("\n" + "="*40 + "\n✅ SIMULATION COMPLETE")

if __name__ == "__main__":
    run_simulation()
