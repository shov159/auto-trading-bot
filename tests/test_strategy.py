"""
Tests for the trading strategy logic using mock data.
"""
import pytest
import backtrader as bt
import pandas as pd
import numpy as np
from src.strategy import RegimeStrategy

def create_scenario_feed(trend='sideways', volatility=0.01, periods=100):
    """
    Creates a mock backtrader feed with specific trend characteristics.

    Args:
        trend (str): 'bull', 'bear', or 'sideways'.
        volatility (float): Daily volatility (standard deviation).
        periods (int): Number of days.
    """
    date_range = pd.date_range(start='2024-01-01', periods=periods, freq='B')

    # Determine drift based on trend
    if trend == 'bull':
        drift = 0.001 # Positive daily drift
    elif trend == 'bear':
        drift = -0.001 # Negative daily drift
    else: # sideways
        drift = 0.0

    # Generate returns
    np.random.seed(42) # Fixed seed for reproducibility
    returns = np.random.normal(drift, volatility, len(date_range))
    price_path = 100 * (1 + returns).cumprod()

    df = pd.DataFrame(index=date_range)
    df['open'] = price_path
    df['high'] = price_path * 1.01
    df['low'] = price_path * 0.99
    df['close'] = price_path
    df['volume'] = np.random.randint(1000, 10000, len(date_range))

    # pylint: disable=unexpected-keyword-arg
    return bt.feeds.PandasData(dataname=df)

@pytest.mark.parametrize("market_condition", [
    ("bull"),
    ("bear"),
    ("sideways")
])
def test_strategy_scenarios(market_condition):
    """
    Tests the strategy against diverse market conditions.

    Scenarios:
    - Bull Market: Expect successful execution, likely positive return (if momentum works).
    - Bear Market: Expect execution, stop losses might trigger.
    - Sideways: Expect execution.
    """
    cerebro = bt.Cerebro()

    # Feed 1: Benchmark/Regime (SPY)
    # We keep this relatively stable/sideways to allow the strategy to function normally
    # or we could also parameterize this to test regime switching.
    # For this test, we use a standard sideways feed for regime to keep it simple.
    regime_feed = create_scenario_feed(trend='sideways', volatility=0.01)

    # Feed 2: Trading Asset (NVDA) - This follows the parameterized trend
    trading_feed = create_scenario_feed(trend=market_condition, volatility=0.02)

    cerebro.adddata(regime_feed)
    cerebro.adddata(trading_feed)

    cerebro.addstrategy(RegimeStrategy)
    cerebro.broker.setcash(100000.0)

    # Run
    results = cerebro.run()
    assert len(results) > 0

    # Check if we still have cash (didn't blow up to negative, though BT usually handles this)
    assert cerebro.broker.getvalue() > 0
