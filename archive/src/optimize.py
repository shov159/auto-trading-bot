"""
Optimization module for the trading strategy.
Allows running optimization for specific tickers via command line.
"""
import argparse
import backtrader as bt
import yfinance as yf
from src.strategy import RegimeStrategy
from src.data import generate_mock_data, fix_yf_columns

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
START_DATE = '2024-01-01'
END_DATE = '2025-12-31'
INITIAL_CASH = 100000

def fetch_data(ticker, benchmark):
    """Fetches and prepares data for backtrader."""
    print(f"Fetching data for Optimization: {ticker}...")
    try:
        spy_df = yf.download(
            benchmark, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False
        )
        ticker_df = yf.download(
            ticker, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False
        )

        if spy_df.empty or ticker_df.empty:
            raise ValueError("Empty data")

        spy_df = fix_yf_columns(spy_df)
        ticker_df = fix_yf_columns(ticker_df)

    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"Error fetching data: {e}")
        spy_df = generate_mock_data(benchmark, START_DATE, END_DATE)
        ticker_df = generate_mock_data(ticker, START_DATE, END_DATE)

    return spy_df, ticker_df

def print_optimization_result(params, sharpe, drawdown):
    """Helper to print a single result row."""
    print(
        f"{params.stop_loss_pct:<10.2f} | "
        f"{params.mom_sma_long:<10} | "
        f"{params.mr_rsi_oversold:<10} | "
        f"{sharpe:<10.4f} | "
        f"{drawdown:<10.2f}%"
    )

def analyze_results(results):
    """Analyzes optimization results to find the best set."""
    best_sharpe = -float('inf')
    best_params = None
    best_metrics = None

    for run in results:
        for strat in run:
            params = strat.params
            metrics = strat.analyzers.sharpe.get_analysis()
            dd = strat.analyzers.drawdown.get_analysis()

            sharpe_val = metrics.get('sharpe ratio') or metrics.get('sharperatio', 0.0)
            if sharpe_val is None:
                sharpe_val = -99.9

            max_dd = dd.get('max', {}).get('drawdown', 99.9)

            print_optimization_result(params, sharpe_val, max_dd)

            # Optimization Logic: Maximize Sharpe, but strictly penalize high drawdown
            # For Crypto/High Beta, we might accept up to 40% DD if return is huge,
            # but aiming for < 25% generally.

            # Simple fitness function: Sharpe
            if sharpe_val > best_sharpe:
                best_sharpe = sharpe_val
                best_params = params
                best_metrics = {'sharpe': sharpe_val, 'drawdown': max_dd}

    return best_params, best_metrics

def run_optimization(ticker, benchmark='SPY', profile='STANDARD'):
    """
    Runs strategy optimization for a given ticker.
    """
    cerebro = bt.Cerebro()

    # 1. Data Loading
    spy_df, ticker_df = fetch_data(ticker, benchmark)

    # 2. Add Data Feeds
    # pylint: disable=unexpected-keyword-arg
    data_spy = bt.feeds.PandasData(dataname=spy_df, name=benchmark)
    data_ticker = bt.feeds.PandasData(dataname=ticker_df, name=ticker)

    cerebro.adddata(data_spy)
    cerebro.adddata(data_ticker)

    # 3. Add Strategy with Optimization Ranges
    # Ranges depend on profile to save time
    if profile.upper() == 'CRYPTO':
        # Crypto needs wider stops and maybe faster/slower SMAs
        stop_loss_range = [0.03, 0.05, 0.07, 0.10]
        sma_long_range = [10, 20, 30, 50]
        rsi_oversold_range = [10, 15, 20, 25, 30]
    elif profile.upper() == 'DEFENSIVE':
        stop_loss_range = [0.01, 0.02, 0.03]
        sma_long_range = [50, 100, 200]
        rsi_oversold_range = [10, 20]
    else:
        # Standard
        stop_loss_range = [0.02, 0.03, 0.04]
        sma_long_range = [30, 50, 70]
        rsi_oversold_range = [10, 20]

    cerebro.optstrategy(
        RegimeStrategy,
        stop_loss_pct=stop_loss_range,
        mom_sma_long=sma_long_range,
        mr_rsi_oversold=rsi_oversold_range
    )

    # 4. Broker & Analyzers
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # 5. Run Optimization
    print(f"Running optimization for {ticker} ({profile})...")
    # maxcpus=1 for safety, increase if env supports it
    results = cerebro.run(maxcpus=1)

    # 6. Process Results
    print(f"\nOptimization Results for {ticker}:")
    header = (
        f"{'Stop Loss':<10} | {'SMA Long':<10} | {'RSI Over':<10} | "
        f"{'Sharpe':<10} | {'Drawdown':<10}"
    )
    print(header)
    print("-" * 65)

    best_params, best_metrics = analyze_results(results)

    print("-" * 65)
    print(f"Best Parameters for {ticker}:")
    print(f"  Stop Loss: {best_params.stop_loss_pct:.2f}")
    print(f"  SMA Long:  {best_params.mom_sma_long}")
    print(f"  RSI Over:  {best_params.mr_rsi_oversold}")
    print(f"  Sharpe:    {best_metrics['sharpe']:.4f}")
    print(f"  Drawdown:  {best_metrics['drawdown']:.2f}%")

    return best_params

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='NVDA', help='Ticker symbol')
    parser.add_argument(
        '--profile', type=str, default='STANDARD',
        help='Profile (CRYPTO, DEFENSIVE, STANDARD)'
    )
    args = parser.parse_args()

    run_optimization(args.ticker, profile=args.profile)
