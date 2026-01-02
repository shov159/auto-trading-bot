"""
Main execution script for the backtest.
"""
import backtrader as bt
import yfinance as yf
from src.strategy import RegimeStrategy
from src.data import generate_mock_data, fix_yf_columns

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
START_DATE = '2024-01-01'
END_DATE = '2025-12-31'
INITIAL_CASH = 100000
DEFAULT_BENCHMARK = 'SPY'
DEFAULT_TICKER = 'NVDA'

def run_backtest(trading_ticker=DEFAULT_TICKER, benchmark_ticker=DEFAULT_BENCHMARK):
    """
    Runs the main backtest logic.
    
    Args:
        trading_ticker (str): The ticker symbol to trade (e.g., 'NVDA').
        benchmark_ticker (str): The ticker symbol for regime filter (e.g., 'SPY').

    Returns:
        dict: Performance metrics (final_value, return, sharpe, max_drawdown).
    """
    cerebro = bt.Cerebro()

    # 1. Fetch Data
    print(f"Fetching data for {trading_ticker} (Trade) and {benchmark_ticker} (Regime)...")
    try:
        # Auto_adjust=True handles splits/dividends (Adjusted Close becomes Close)
        spy_df = yf.download(
            benchmark_ticker, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False
        )
        nvda_df = yf.download(
            trading_ticker, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False
        )

        # Check if data is valid
        if spy_df.empty or nvda_df.empty:
            raise ValueError("Empty data returned from yfinance")

        # Fix MultiIndex columns if present
        spy_df = fix_yf_columns(spy_df)
        nvda_df = fix_yf_columns(nvda_df)

    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"Error fetching data: {e}")
        print("Switching to Mock Data...")
        spy_df = generate_mock_data(benchmark_ticker, START_DATE, END_DATE)
        nvda_df = generate_mock_data(trading_ticker, START_DATE, END_DATE)

    # 2. Add Data Feeds
    # IMPORTANT: SPY first (index 0) as it drives the Regime
    # pylint: disable=unexpected-keyword-arg
    data_spy = bt.feeds.PandasData(dataname=spy_df, name=benchmark_ticker)
    data_nvda = bt.feeds.PandasData(dataname=nvda_df, name=trading_ticker)

    cerebro.adddata(data_spy)
    cerebro.adddata(data_nvda)

    # 3. Add Strategy
    cerebro.addstrategy(RegimeStrategy)

    # Add Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # 4. Broker Setup
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001) # 0.1% commission example

    # 5. Run
    start_value = cerebro.broker.getvalue()
    print(f'Starting Portfolio Value: {start_value:.2f}')

    results = cerebro.run()
    strat = results[0]

    end_value = cerebro.broker.getvalue()
    total_return = ((end_value - start_value) / start_value) * 100

    print(f'Final Portfolio Value: {end_value:.2f}')
    print(f'Total Return: {total_return:.2f}%')

    # Metrics Extraction
    metrics = {
        'ticker': trading_ticker,
        'benchmark': benchmark_ticker,
        'initial_value': start_value,
        'final_value': end_value,
        'return_pct': total_return,
        'sharpe': 'N/A',
        'max_drawdown': 'N/A'
    }

    # Extract Sharpe
    sharpe = strat.analyzers.sharpe.get_analysis()
    if sharpe and sharpe.get('sharperatio') is not None:
        metrics['sharpe'] = sharpe['sharperatio']
        print(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
    else:
        print("Sharpe Ratio: N/A")
        
    # Extract Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    if drawdown and drawdown.get('max') is not None:
        metrics['max_drawdown'] = drawdown['max']['drawdown']

    # 6. Plot
    # print("Plotting results...")
    # try:
    #     cerebro.plot(style='candlestick', volume=False)
    # except Exception as e: # pylint: disable=broad-exception-caught
    #     print(f"Plotting failed (likely no display): {e}")

    return metrics

if __name__ == '__main__':
    run_backtest()
