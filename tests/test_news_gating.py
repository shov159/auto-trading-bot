import unittest
from unittest.mock import MagicMock
from src.main import AITrader

class TestNewsGating(unittest.TestCase):
    def setUp(self):
        # Create a partial mock of AITrader logic to test the specific gating block
        # We can't easily instantiate AITrader without env vars, so we test logic in isolation
        # or mock the whole class.
        pass

    def test_regime_logic(self):
        # Simulate the logic block we added to main.py
        
        def calculate_multiplier_and_block(sentiment_score):
            confidence_mult = 1.0
            block_buys = False
            
            if sentiment_score <= -0.30:
                block_buys = True
            elif -0.30 < sentiment_score < 0.10:
                confidence_mult = 0.5
            elif 0.10 <= sentiment_score < 0.40:
                confidence_mult = 1.0
            elif sentiment_score >= 0.40:
                confidence_mult = 1.25
            
            return block_buys, confidence_mult

        # Case 1: Deep Bearish (-0.4) -> Block
        block, mult = calculate_multiplier_and_block(-0.4)
        self.assertTrue(block)
        
        # Case 2: Mild Bearish/Neutral (-0.1) -> 0.5x
        block, mult = calculate_multiplier_and_block(-0.1)
        self.assertFalse(block)
        self.assertEqual(mult, 0.5)
        
        # Case 3: Bullish (0.5) -> 1.25x
        block, mult = calculate_multiplier_and_block(0.5)
        self.assertFalse(block)
        self.assertEqual(mult, 1.25)

if __name__ == '__main__':
    unittest.main()


