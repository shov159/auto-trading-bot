import unittest
from src.risk_manager import RiskManager

class TestRiskManagerMultiplier(unittest.TestCase):
    def setUp(self):
        self.config = {
            'risk': {
                'max_position_pct': 0.10, # 10% for easy math
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05,
                'max_drawdown_pct': 0.10
            }
        }
        self.rm = RiskManager(self.config)

    def test_multiplier_clamping(self):
        account_value = 100000
        price = 100
        # Base Allocation: 100,000 * 0.10 = 10,000. Qty = 100 shares.

        # Test Neutral (1.0)
        qty = self.rm.calculate_position_size(account_value, price, confidence_mult=1.0)
        self.assertEqual(qty, 100)

        # Test Low Clamp (0.1 -> 0.25)
        # Expected: 10,000 * 0.25 = 2,500. Qty = 25.
        qty = self.rm.calculate_position_size(account_value, price, confidence_mult=0.1)
        self.assertEqual(qty, 25)

        # Test High Clamp (2.0 -> 1.5)
        # Expected: 10,000 * 1.5 = 15,000. Qty = 150.
        qty = self.rm.calculate_position_size(account_value, price, confidence_mult=2.0)
        self.assertEqual(qty, 150)

    def test_invalid_inputs(self):
        self.assertEqual(self.rm.calculate_position_size(1000, -50), 0)
        self.assertEqual(self.rm.calculate_position_size(-1000, 50), 0)

if __name__ == '__main__':
    unittest.main()

