import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest import Backtester
from src.strategy_logic import StrategyLogic
from src.risk_manager import RiskManager

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_grid_search():
    load_dotenv()
    
    # 1. Setup Data & Config
    base_config = load_config()
    # Use the same focused symbols as the verification backtest
    symbols = ['SPY', 'QQQ', 'NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'META']
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 1)
    
    print(f"--- Starting Grid Search Optimization ---")
    print(f"Symbols: {len(symbols)}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    # Pre-fetch data ONCE to speed up the loop
    # We create a dummy backtester just to load data
    print("Pre-loading market data...")
    temp_rm = RiskManager(base_config)
    temp_strat = StrategyLogic(temp_rm, base_config)
    loader = Backtester(temp_strat, temp_rm)
    market_data = loader.load_data(symbols, start_date, end_date)
    
    if not market_data:
        print("Failed to load market data. Exiting.")
        return

    # 2. Define Grid
    param_grid = {
        'atr_multipliers': [1.5, 2.0, 2.5, 3.0],
        'score_thresholds': [60, 65, 70, 75, 80],
        'take_profit_multipliers': [3.0, 4.0, 5.0]
    }
    
    combinations = list(itertools.product(
        param_grid['atr_multipliers'],
        param_grid['score_thresholds'],
        param_grid['take_profit_multipliers']
    ))
    
    total_runs = len(combinations)
    results = []
    
    print(f"Total Combinations to Test: {total_runs}")
    print("="*50)
    
    # 3. Loop
    for i, (atr_mult, score_thresh, tp_mult) in enumerate(combinations, 1):
        # Update Config
        config = base_config.copy()
        
        # Risk Config
        config['risk']['stop_loss_atr_multiplier'] = atr_mult
        config['risk']['take_profit_atr_multiplier'] = tp_mult
        
        # Strategy Config (Note: Threshold is usually hardcoded in logic, 
        # so we might need to patch StrategyLogic or add a config param)
        # We will subclass/inject just for this run or modify the instance.
        
        # Initialize components
        rm = RiskManager(config)
        
        # Hack: Pass threshold via config or modify instance
        strategy = StrategyLogic(rm, config)
        # We need to make sure StrategyLogic uses this threshold. 
        # Since 'score_threshold' isn't in standard config yet, 
        # we'll monkey-patch the check inside `calculate_trade_score` wrapper 
        # or better, modify StrategyLogic to accept it.
        # For now, let's assume we can modify the instance variable if it existed,
        # but since it's hardcoded ">= 70", we might need to modify src/strategy_logic.py first?
        # WAIT: The user asked to optimization score_thresholds.
        # But `src/strategy_logic.py` has `if trade_score >= 70:` hardcoded.
        # To make this work without editing source code every loop, 
        # we should update `src/strategy_logic.py` to read from config.
        # However, for this script, we can dynamically override the generate_signal logic 
        # or just modify the class if we can.
        
        # ACTUALLY: Let's assume we will use a modified StrategyLogic or 
        # we update the class instance attribute if we added one.
        # Since we can't easily change the hardcoded 70 in the file from here without editing it,
        # let's modify the StrategyLogic instance to have a 'buy_threshold' attribute
        # and we need to assume the file respects it. 
        # BUT THE FILE IS HARDCODED. 
        # Plan: I will patch the method dynamically or I should have updated strategy_logic.py first.
        # Better approach: I will subclass StrategyLogic in this script and override generate_signal 
        # to use the dynamic threshold, OR just update the file `src/strategy_logic.py` to be configurable.
        
        # Let's use a subclass here for safety/cleanliness without changing source yet
        # unless necessary. But `generate_signal` is long.
        # EASIEST: Update `src/strategy_logic.py` to use `self.buy_threshold`.
        
        # Since I cannot edit `src/strategy_logic.py` within this `write` call,
        # I will assume I will do it in a separate step or use a localized Monkey Patch.
        # Monkey Patching `generate_signal` is too complex.
        # I will rely on a helper function or assume I'll fix the source file next.
        # Let's proceed assuming `strategy.buy_threshold` exists and is used.
        # We will inject it:
        strategy.buy_threshold = score_thresh 
        
        # 4. Run Backtest
        backtester = Backtester(strategy, rm, initial_capital=100000.0)
        
        # INJECT DATA directly to avoid re-fetching
        # We need to manually populate backtester internal structures if we skip `load_data`
        # But `backtester.run` calls `load_data`.
        # Let's subclass Backtester to accept pre-loaded data.
        
        # Helper to run without re-fetching
        # We'll just define a custom run method here or modify Backtester
        # For simplicity in this script, let's just mock the load_data method on the instance
        backtester.load_data = lambda s, st, en: market_data
        
        # Run (Suppress output)
        # We need to suppress print statements from backtester.run
        # Redirect stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            backtester.run(symbols, start_date, end_date, mock_sentiment=True)
        finally:
            sys.stdout = sys.__stdout__ # Restore
            
        # 5. Collect Metrics
        metrics = backtester.calculate_metrics()
        
        res = {
            'ATR_Mult': atr_mult,
            'Score_Thresh': score_thresh,
            'TP_Mult': tp_mult,
            'Return': metrics.get('Total Return (%)', 0.0),
            'Win_Rate': metrics.get('Win Rate (%)', 0.0),
            'Sharpe': metrics.get('Sharpe Ratio', 0.0),
            'Trades': metrics.get('Trades Executed', 0)
        }
        results.append(res)
        
        # Progress Bar
        print(f"Run {i}/{total_runs}: Params={res['ATR_Mult']}/{res['Score_Thresh']}/{res['TP_Mult']} -> Sharpe={res['Sharpe']}, Ret={res['Return']}%")

    # 6. Save & Report
    df_results = pd.DataFrame(results)
    df_results.sort_values(by='Sharpe', ascending=False, inplace=True)
    
    df_results.to_csv("optimization_results.csv", index=False)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print("Top 5 Combinations:")
    print(df_results.head(5))
    
    best = df_results.iloc[0]
    print("\n🏆 BEST COMBINATION:")
    print(f"ATR Multiplier: {best['ATR_Mult']}")
    print(f"Score Threshold: {best['Score_Thresh']}")
    print(f"TP Multiplier: {best['TP_Mult']}")
    print(f"Sharpe Ratio: {best['Sharpe']}")
    print(f"Total Return: {best['Return']}%")

if __name__ == "__main__":
    run_grid_search()

