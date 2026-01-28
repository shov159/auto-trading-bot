"""
Tests for Learning Engine
Run with: python tests/test_learning_engine.py
"""
import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning_engine import LearningEngine

class TestLearningEngine(unittest.TestCase):

    def setUp(self):
        # Use temp files for testing
        self.test_history_file = "tests/test_trade_history.json"
        self.test_lessons_file = "tests/test_lessons.txt"

        # Patch paths
        self.patcher1 = patch('src.learning_engine.HISTORY_FILE', self.test_history_file)
        self.patcher2 = patch('src.learning_engine.LESSONS_FILE', self.test_lessons_file)
        self.patcher1.start()
        self.patcher2.start()

        # Create temp files
        with open(self.test_history_file, 'w') as f:
            json.dump([], f)
        with open(self.test_lessons_file, 'w') as f:
            f.write("")

        self.engine = LearningEngine()

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        if os.path.exists(self.test_history_file):
            os.remove(self.test_history_file)
        if os.path.exists(self.test_lessons_file):
            os.remove(self.test_lessons_file)

    def test_log_trade_entry(self):
        print("\n🧪 Testing Trade Logging...")
        trade_data = {
            "ticker": "AAPL",
            "action": "BUY",
            "qty": 10,
            "entry": 150.00,
            "stop_loss": 145.00,
            "take_profit": 160.00,
            "risk": 50.0,
            "analysis": {
                "reasoning": "Bullish divergence",
                "raw_data": {"rsi": 45, "price": 150}
            }
        }

        self.engine.log_trade_entry(trade_data, "order_123")

        with open(self.test_history_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['ticker'], "AAPL")
        self.assertEqual(data[0]['status'], "OPEN")
        self.assertEqual(data[0]['market_context']['rsi'], 45)
        print("✅ Trade Logging PASSED")

    def test_extract_lesson(self):
        print("\n🧪 Testing Lesson Extraction...")
        llm_response = """
        Analysis: You ignored the RSI.
        LESSON: Do not buy if RSI > 70.
        Hope this helps.
        """
        lesson = self.engine._extract_lesson(llm_response)
        self.assertEqual(lesson, "Do not buy if RSI > 70.")
        print("✅ Lesson Extraction PASSED")

    @patch('src.ai_brain.get_brain')
    def test_analyze_past_performance(self, mock_get_brain):
        print("\n🧪 Testing Critique & Lesson Saving...")

        # Setup finished trade in history
        trade = {
            "order_id": "order_123",
            "ticker": "TSLA",
            "action": "BUY",
            "status": "CLOSED",
            "pnl_pct": -2.5,
            "market_context": {"rsi": 80}
        }
        with open(self.test_history_file, 'w') as f:
            json.dump([trade], f)

        # Mock AI response
        mock_brain = MagicMock()
        mock_brain.run_critique.return_value = "LESSON: Avoid high RSI breakouts."
        mock_get_brain.return_value = mock_brain

        # Run analysis
        result = self.engine.analyze_past_performance()

        # Check if lesson saved
        with open(self.test_lessons_file, 'r') as f:
            content = f.read()

        self.assertIn("Avoid high RSI breakouts.", content)
        print("✅ Critique Agent PASSED")

if __name__ == '__main__':
    unittest.main()
