import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import FeatureEngineer
from src.rl.trading_env import AdvancedTradingEnv
from src.strategies.ensemble import EnsembleStrategy
from scripts.train_ensemble import fetch_data

def run_stress_test():
    """
    Stress tests the Ensemble Strategy on specific historical crash periods.
    """
    periods = {
        "COVID Crash": ("2020-02-01", "2020-04-01"),
        "2022 Bear": ("2022-01-01", "2022-06-01")
    }
    
    # Try 2008 if possible, usually yfinance has data but features need lookback
    try:
        raw_2008 = fetch_data("SPY", "2007-01-01", "2009-04-01") # Extra lookback for SMA200
        periods["2008 Crash"] = ("2008-09-01", "2009-03-01")
    except:
        print("Skipping 2008 (Data unavailable)")

    # Load Ensemble
    ensemble = EnsembleStrategy(models_dir="models")
    fe = FeatureEngineer()
    
    results = []

    print("\n--- STARTING STRESS TEST ---\n")

    for name, (start_date, end_date) in periods.items():
        print(f"Testing Period: {name} ({start_date} to {end_date})...")
        
        try:
            # Fetch data with buffer for indicators
            buffer_start = pd.to_datetime(start_date) - pd.Timedelta(days=400)
            raw_df = fetch_data("SPY", buffer_start.strftime("%Y-%m-%d"), end_date)
            
            if raw_df.empty:
                print(f"  No data for {name}")
                continue
                
            # Process Features
            processed_df = fe.preprocess(raw_df)
            
            # Slice to test period
            # Ensure index is datetime
            if not isinstance(processed_df.index, pd.DatetimeIndex):
                processed_df.index = pd.to_datetime(processed_df.index)
                
            test_df = processed_df.loc[start_date:end_date]
            
            if test_df.empty:
                print(f"  Empty test slice for {name} after processing")
                continue
                
            # Run Simulation
            env = AdvancedTradingEnv(test_df)
            obs, _ = env.reset()
            done = False
            
            equity_curve = [env.initial_balance]
            
            while not done:
                # Extract features for ensemble inference
                # Env observation is combined, we need to split if ensemble expects specific input
                # Ensemble predict takes (obs, features)
                # obs from env includes state. features is just the market part.
                # In feature engineer, we stacked them.
                # Market features are first N columns.
                n_features = len(fe.feature_cols)
                market_features = obs[:n_features]
                
                target_pct, _ = ensemble.predict(obs, market_features)
                
                # Step environment (Action is target %)
                # Map scalar to array
                action = np.array([target_pct], dtype=np.float32)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                equity_curve.append(env.net_worth)
                
            # Calculate Metrics
            equity_curve = np.array(equity_curve)
            
            # Max Drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            max_dd = drawdown.min() * 100
            
            # Total Return
            total_return = (equity_curve[-1] / equity_curve[0]) - 1
            
            # Sharpe (Daily)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0.0
                
            print(f"  Max Drawdown: {max_dd:.2f}%")
            print(f"  Total Return: {total_return*100:.2f}%")
            print(f"  Sharpe Ratio: {sharpe:.2f}")
            
            results.append({
                "Period": name,
                "Max DD": max_dd,
                "Return": total_return,
                "Sharpe": sharpe
            })
            
            if max_dd < -15.0:
                print("  ⚠️ ALERT: Drawdown Exceeded 15% Limit!")
                
        except Exception as e:
            print(f"  Error testing {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- TEST COMPLETE ---")
    return pd.DataFrame(results)

if __name__ == "__main__":
    run_stress_test()

