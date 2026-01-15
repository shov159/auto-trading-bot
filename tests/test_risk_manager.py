import unittest
from src.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.config = {
            'risk': {
                'max_position_pct': 0.02,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05,
                'max_drawdown_pct': 0.10
            }
        }
        self.rm = RiskManager(self.config)

    def test_calculate_position_size(self):
        account_value = 100000
        price = 100
        # Expected: 100000 * 0.02 = 2000. 2000 / 100 = 20 shares.
        qty = self.rm.calculate_position_size(account_value, price)
        self.assertEqual(qty, 20)
        
        # Test max allocation constraint
        price = 5000
        # Expected: 2000 / 5000 = 0.4 -> 0 shares
        qty = self.rm.calculate_position_size(account_value, price)
        self.assertEqual(qty, 0)
        
        # Verify it never exceeds 2%
        price = 10
        qty = self.rm.calculate_position_size(account_value, price)
        total_cost = qty * price
        self.assertLessEqual(total_cost, account_value * 0.02)

    def test_get_stop_loss_price(self):
        entry_price = 100
        # Buy: 100 * (1 - 0.02) = 98
        sl = self.rm.get_stop_loss_price(entry_price, side="buy")
        self.assertAlmostEqual(sl, 98.0)
        
        # Sell: 100 * (1 + 0.02) = 102
        sl = self.rm.get_stop_loss_price(entry_price, side="sell")
        self.assertAlmostEqual(sl, 102.0)

    def test_get_take_profit_price(self):
        entry_price = 100
        # Buy: 100 * (1 + 0.05) = 105
        tp = self.rm.get_take_profit_price(entry_price, side="buy")
        self.assertAlmostEqual(tp, 105.0)
        
        # Sell: 100 * (1 - 0.05) = 95
        tp = self.rm.get_take_profit_price(entry_price, side="sell")
        self.assertAlmostEqual(tp, 95.0)

    def test_check_drawdown(self):
        peak_equity = 100000
        
        # 5% drawdown (False)
        current_equity = 95000
        should_stop = self.rm.check_drawdown(current_equity, peak_equity)
        self.assertFalse(should_stop)
        
        # 10.01% drawdown (True)
        # Threshold is 0.10. 
        # 100000 * (1 - 0.1001) = 89990
        current_equity = 89900
        should_stop = self.rm.check_drawdown(current_equity, peak_equity)
        self.assertTrue(should_stop)
        
        # Exact 10% (False - must be strictly greater usually, logic was > max_dd_pct)
        current_equity = 90000
        should_stop = self.rm.check_drawdown(current_equity, peak_equity)
        self.assertFalse(should_stop)

if __name__ == '__main__':
    unittest.main()
