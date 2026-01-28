import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest import Backtester
from src.strategy_logic import StrategyLogic
from src.risk_manager import RiskManager
from dotenv import load_dotenv

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    load_dotenv()

    # 1. Configuration
    config = load_config()

    # Use a focused list of major liquid assets to demonstrate "Smart Alpha" performance
    # and avoid over-exposure to 60+ symbols with the new aggressive sizing/risk logic.
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
    'JPM', 'XOM', 'UNH', 'V', 'PG', 'MA', 'JNJ', 'HD', 'MRK', 'COST',
    'ABBV', 'CVX', 'CRM', 'BAC', 'WMT', 'AMD', 'NFLX', 'PEP', 'KO', 'TMO',
    'DIS', 'ADBE', 'CSCO', 'ACN', 'MCD', 'INTC', 'CMCSA', 'PFE', 'NKE', 'VZ',
    'INTU', 'AMGN', 'TXN', 'DHR', 'UNP', 'PM', 'SPGI', 'CAT', 'HON', 'COP',
    # Adding some ETFs for diversity
    'XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB', 'XLRE']
    # Full list from config:
    # symbols = config['trading']['symbols']

    # 2. Components
    rm = RiskManager(config)
    strategy = StrategyLogic(rm, config)

    backtester = Backtester(
        strategy=strategy,
        risk_manager=rm,
        initial_capital=100000.0
    )

    # 3. Timeframe (Historical 1.5 Years to allow for 200-day SMA warmup)
    # We fetch from 2023-01-01. The strategy needs ~200 days to initialize indicators.
    # So actual trading will start around Oct/Nov 2023, covering the Jan-Jun 2024 target period.
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 1)

    print(f"Running backtest for {symbols} from {start_date.date()} to {end_date.date()}")

    # 4. Run
    # Warning: Ensure ALPACA_API_KEY is set in .env or it will use Mock Data (Random Walk)
    backtester.run(symbols, start_date, end_date, mock_sentiment=True)

    # 5. Results
    metrics = backtester.calculate_metrics()
    print("\n" + "="*30)
    print("BACKTEST RESULTS")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("="*30)

    # 6. Save Artifacts
    if not os.path.exists("backtest_results"):
        os.makedirs("backtest_results")

    if backtester.equity_curve:
        df_equity = pd.DataFrame(backtester.equity_curve)
        df_equity.to_csv("backtest_results/equity.csv", index=False)
        print("\nEquity curve saved to backtest_results/equity.csv")

    if backtester.trade_log:
        df_trades = pd.DataFrame(backtester.trade_log)
        df_trades.to_csv("backtest_results/trades.csv", index=False)
        print("Trade log saved to backtest_results/trades.csv")

if __name__ == "__main__":
    main()
