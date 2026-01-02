import sys
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
    ('IWM', 'SPY'),
    ('IWO', 'SPY'),
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

def generate_report():
    print("Generating Performance Report...\n")
    
    results = []
    
    for ticker, benchmark in ASSET_MATRIX:
        print(f"--- Running Backtest for {ticker} ---")
        try:
            metrics = run_backtest(trading_ticker=ticker, benchmark_ticker=benchmark)
            results.append(metrics)
        except Exception as e:
            print(f"Failed for {ticker}: {e}")
            results.append({
                'ticker': ticker, 
                'return_pct': 0.0, 
                'sharpe': 0.0, 
                'final_value': 0.0,
                'max_drawdown': 0.0
            })
            
    # Print Markdown Table
    print("\n\n### Backtest Results Summary")
    print(f"| {'Ticker':<6} | {'Benchmark':<9} | {'Final Value ($)':<16} | {'Return (%)':<10} | {'Sharpe Ratio':<12} | {'Max Drawdown (%)':<16} |")
    print(f"| {':---':<6} | {':---':<9} | {':---':<16} | {':---':<10} | {':---':<12} | {':---':<16} |")
    
    for r in results:
        ticker = r.get('ticker', 'N/A')
        bench = r.get('benchmark', 'N/A')
        fv = f"${r.get('final_value', 0):,.2f}"
        ret = f"{r.get('return_pct', 0):.2f}%"
        
        sharpe = r.get('sharpe')
        if isinstance(sharpe, (int, float)):
            sharpe_str = f"{sharpe:.2f}"
        else:
            sharpe_str = "N/A"
            
        dd = r.get('max_drawdown')
        if isinstance(dd, (int, float)):
            dd_str = f"{dd:.2f}%"
        else:
            dd_str = "N/A"
            
        print(f"| {ticker:<6} | {bench:<9} | {fv:<16} | {ret:<10} | {sharpe_str:<12} | {dd_str:<16} |")

if __name__ == "__main__":
    generate_report()
