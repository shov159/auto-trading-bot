"""
End-to-end tests for the trading bot.
"""
import os
import pytest
from src.main import run_backtest

# Define test matrix
ASSET_MATRIX = [
    # High Beta / Tech
    ('NVDA', 'SPY'),
    ('AMD', 'SPY'),
    ('MSFT', 'SPY'),
    ('META', 'SPY'),
    ('AVGO', 'SPY'),
    ('SMH', 'SPY'),

    # Defensive
    ('KO', 'SPY'),
    ('XLU', 'SPY'),

    # Index / Factor Exposure
    ('QQQ', 'SPY'),
    ('IWM', 'SPY'),   # Russell 2000
    ('IWO', 'SPY'),   # Russell 2000 Growth (אופציונלי)
    ('VTV', 'SPY'),
    ('VUG', 'SPY'),
    ('SPLV', 'SPY'),

    # Finance (Broad)
    ('JPM', 'SPY'),
    ('XLF', 'SPY'),

    # Energy
    ('XOM', 'SPY'),

    # Healthcare
    ('JNJ', 'SPY'),
    ('XLV', 'SPY'),

    # Industrials
    ('CAT', 'SPY'),
    ('BA', 'SPY'),

    # Materials
    ('LIN', 'SPY'),

    # Real Estate
    ('VNQ', 'SPY'),

    # Global
    ('EZU', 'SPY'),
    ('EEM', 'SPY'),

    # Crypto Proxy
    ('COIN', 'SPY'),

    # Fintech - Established
    ('V', 'SPY'),
    ('MA', 'SPY'),
    ('AXP', 'SPY'),
    ('PYPL', 'SPY'),

    # Fintech - High Beta
    ('SOFI', 'SPY'),
    ('SOFI', 'XLF'),
    ('SOFI', 'IWM'),
    # ('SQ',   'SPY'), # Delisted or API error
    # ('SQ',   'XLF'),
    ('AFRM', 'SPY'),
    ('UPST', 'SPY'),
    ('HOOD', 'SPY'),

    # Fintech ETFs
    ('ARKF', 'SPY'),
    ('FINX', 'SPY'),
]


@pytest.mark.skipif(
    os.environ.get('RUN_E2E') != 'true',
    reason="Skipping E2E test unless RUN_E2E=true"
)
@pytest.mark.parametrize("ticker, benchmark", ASSET_MATRIX)
def test_e2e_multi_asset(ticker, benchmark):
    """
    Runs the full backtest pipeline across different asset classes.
    """
    print(f"\nRunning E2E for {ticker} vs {benchmark}...")
    try:
        run_backtest(trading_ticker=ticker, benchmark_ticker=benchmark)
    except Exception as e: # pylint: disable=broad-exception-caught
        pytest.fail(f"E2E Backtest Failed for {ticker}: {e}")
