"""
Market Scanner Module.
Scans a universe of assets to select the best candidates based on Momentum and Volatility.
"""
import pandas as pd
import yfinance as yf

# MVP Universe: Top Liquid Stocks & ETFs (S&P 500 / Nasdaq 100 leaders)
UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
    'JPM', 'XOM', 'UNH', 'V', 'PG', 'MA', 'JNJ', 'HD', 'MRK', 'COST',
    'ABBV', 'CVX', 'CRM', 'BAC', 'WMT', 'AMD', 'NFLX', 'PEP', 'KO', 'TMO',
    'DIS', 'ADBE', 'CSCO', 'ACN', 'MCD', 'INTC', 'CMCSA', 'PFE', 'NKE', 'VZ',
    'INTU', 'AMGN', 'TXN', 'DHR', 'UNP', 'PM', 'SPGI', 'CAT', 'HON', 'COP',
    # Adding some ETFs for diversity
    'XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB', 'XLRE'
]

class MarketScanner:
    """
    Scans the market to select assets for the portfolio.
    """
    def __init__(self, universe=None, lookback_days=90):
        self.universe = universe if universe else UNIVERSE
        self.lookback_days = lookback_days

    def get_top_picks(self, top_n=10, max_volatility=0.05):
        """
        Selects top assets based on Momentum, filtering for high volatility.

        Args:
            top_n (int): Number of assets to return.
            max_volatility (float): Maximum allowed daily volatility (e.g., 0.05 for 5%).

        Returns:
            list: List of ticker symbols.
        """
        print(f"Scanning {len(self.universe)} assets...")
        metrics = []

        try:
            # We need a bit more than lookback_days to calculate moving averages if needed
            start_date = (
                pd.Timestamp.now() - pd.Timedelta(days=self.lookback_days + 10)
            ).strftime('%Y-%m-%d')

            # Download all tickers
            print("Fetching market data...")
            data = yf.download(
                self.universe, start=start_date, group_by='ticker',
                auto_adjust=True, progress=False, threads=True
            )

            # Process each ticker
            for ticker in self.universe:
                try:
                    # Handle MultiIndex column structure from batch download
                    if isinstance(data.columns, pd.MultiIndex):
                        df = data[ticker].copy()
                    else:
                        # Fallback if single ticker
                        df = data.copy()

                    # Drop NaN
                    df = df.dropna()

                    if len(df) < 60: # Ensure enough data
                        continue

                    # 1. Calculate Volatility (ATR)
                    # Calculate True Range
                    high_low = df['High'] - df['Low']
                    high_close = (df['High'] - df['Close'].shift()).abs()
                    low_close = (df['Low'] - df['Close'].shift()).abs()
                    
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    
                    # 14-day ATR
                    atr = true_range.rolling(window=14).mean().iloc[-1]
                    
                    # Normalized Volatility (ATR as % of Price)
                    # We compare this to the max_volatility threshold (e.g., 0.05 for 5%)
                    current_price = df['Close'].iloc[-1]
                    volatility_pct = atr / current_price if current_price else 0

                    if volatility_pct > max_volatility:
                        continue

                    # 2. Calculate Momentum (3-Month Return)
                    # Simple: Last Close / First Close - 1
                    momentum = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1

                    metrics.append({
                        'ticker': ticker,
                        'momentum': momentum,
                        'volatility': volatility_pct
                    })

                except Exception: # pylint: disable=broad-exception-caught
                    # Skip ticker if calculation fails
                    continue

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Error during scan: {e}")
            return []

        # Convert to DataFrame for sorting
        if not metrics:
            print("No assets met the criteria.")
            return []

        metrics_df = pd.DataFrame(metrics)

        # Sort by Momentum (Descending)
        top_picks_df = metrics_df.sort_values(by='momentum', ascending=False).head(top_n)

        print("\n--- Top Market Picks ---")
        print(f"{'Ticker':<8} | {'Momentum':<10} | {'Avg Daily Vol':<12}")
        print("-" * 35)
        for _, row in top_picks_df.iterrows():
            print(f"{row['ticker']:<8} | {row['momentum']:.2%}     | {row['volatility']:.2%}")
        print("-" * 35)

        return top_picks_df['ticker'].tolist()

if __name__ == "__main__":
    scanner = MarketScanner()
    picks = scanner.get_top_picks()
    print(f"\nFinal Selection: {picks}")
