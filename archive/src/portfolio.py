"""
Portfolio Backtest Module.
Runs a simultaneous backtest on multiple assets using the RegimeStrategy.
"""
import backtrader as bt
import yfinance as yf
from src.strategy import RegimeStrategy
from src.data import fix_yf_columns

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
START_DATE = '2024-01-01'
END_DATE = '2025-12-31'
INITIAL_CASH = 100000

# Full Portfolio List (from E2E test)
TICKERS = [
    # High Beta / Tech
    'NVDA', 'AMD', 'MSFT', 'META', 'AVGO', 'SMH',
    # Defensive
    'KO', 'XLU',
    # Index / Factor
    'QQQ', 'IWM', 'VTV', 'VUG', 'SPLV',
    # Finance
    'JPM', 'XLF',
    # Energy
    'XOM',
    # Healthcare
    'JNJ', 'XLV',
    # Industrials
    'CAT', 'BA',
    # Materials
    'LIN',
    # Real Estate
    'VNQ',
    # Global
    'EZU', 'EEM',
    # Crypto Proxy
    'COIN',
    # Fintech
    'V', 'MA', 'AXP', 'PYPL',
    'SOFI', 'AFRM', 'UPST', 'HOOD',
    'ARKF', 'FINX'
]

BENCHMARK = 'SPY'

def load_benchmark(cerebro):
    """Loads the benchmark data into Cerebro."""
    print(f"Fetching data for Portfolio (Benchmark: {BENCHMARK})...")
    try:
        spy_df = yf.download(
            BENCHMARK, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False
        )
        spy_df = fix_yf_columns(spy_df)
        if spy_df.empty:
            raise ValueError("Empty Benchmark Data")

        # pylint: disable=unexpected-keyword-arg
        data_spy = bt.feeds.PandasData(dataname=spy_df, name=BENCHMARK)
        cerebro.adddata(data_spy)
        print(f"Loaded Benchmark: {BENCHMARK}")
        return True

    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"Failed to load Benchmark {BENCHMARK}: {e}. Aborting.")
        return False

def load_assets(cerebro):
    """Loads all trading assets into Cerebro."""
    loaded_count = 0
    for ticker in TICKERS:
        try:
            df = yf.download(
                ticker, start=START_DATE, end=END_DATE,
                auto_adjust=True, progress=False
            )
            df = fix_yf_columns(df)

            if df.empty:
                print(f"Skipping {ticker}: Empty Data")
                continue

            # pylint: disable=unexpected-keyword-arg
            data = bt.feeds.PandasData(dataname=df, name=ticker)
            cerebro.adddata(data)
            loaded_count += 1

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Error loading {ticker}: {e}")

    print(f"\nSuccessfully loaded {loaded_count} assets into Portfolio.")

def print_report(start_value, end_value, strat):
    """Prints the final performance report."""
    print("\n" + "="*40)
    print("PORTFOLIO PERFORMANCE REPORT")
    print("="*40)
    print(f"Initial Value:    ${start_value:,.2f}")
    print(f"Final Value:      ${end_value:,.2f}")

    # Return
    total_return_pct = ((end_value - start_value) / start_value) * 100
    print(f"Total Return:     {total_return_pct:.2f}%")

    # Sharpe
    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_val = sharpe.get('sharperatio')
    if sharpe_val:
        print(f"Sharpe Ratio:     {sharpe_val:.4f}")
    else:
        print("Sharpe Ratio:     N/A")

    # Drawdown
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get('max', {}).get('drawdown', 0.0)
    print(f"Max Drawdown:     {max_dd:.2f}%")
    print("="*40 + "\n")


def run_portfolio():
    """Runs the portfolio backtest."""
    cerebro = bt.Cerebro()

    if not load_benchmark(cerebro):
        return

    load_assets(cerebro)

    # 3. Add Strategy
    cerebro.addstrategy(RegimeStrategy)

    # 4. Broker Setup
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)

    # 5. Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # 6. Run
    start_value = cerebro.broker.getvalue()
    print(f'\nStarting Portfolio Value: ${start_value:,.2f}')

    results = cerebro.run()
    strat = results[0]

    end_value = cerebro.broker.getvalue()

    # 7. Reporting
    print_report(start_value, end_value, strat)

if __name__ == "__main__":
    run_portfolio()
