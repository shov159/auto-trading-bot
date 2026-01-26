import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.features import FeatureEngineer
from src.rl.trading_env import AdvancedTradingEnv
from src.strategies.ensemble import EnsembleStrategy

class WalkForwardValidator:
    """
    Performs Walk-Forward Validation (Rolling Window Backtesting).
    Train on [t, t+N], Test on [t+N, t+N+M], slide window.
    """
    
    def __init__(self, df: pd.DataFrame, train_window_size: int = 252, test_window_size: int = 63):
        """
        Args:
            df: Preprocessed DataFrame with features.
            train_window_size: Number of days to train (e.g. 252 for 1 year).
            test_window_size: Number of days to test (e.g. 63 for 1 quarter).
        """
        self.df = df
        self.train_size = train_window_size
        self.test_size = test_window_size
        
    def run_validation(self):
        """
        Executes the rolling window validation.
        """
        total_samples = len(self.df)
        start_index = 0
        
        results = []
        
        while start_index + self.train_size + self.test_size <= total_samples:
            # 1. Define Windows
            train_end = start_index + self.train_size
            test_end = train_end + self.test_size
            
            train_data = self.df.iloc[start_index:train_end]
            test_data = self.df.iloc[train_end:test_end]
            
            print(f"Window: Train {train_data.index[0].date()}->{train_data.index[-1].date()} | Test {test_data.index[0].date()}->{test_data.index[-1].date()}")
            
            # 2. Train Ensemble (Simplified: Retrain PPO only for demo speed, or load pre-trained)
            # In production, we'd retrain ALL models or fine-tune them.
            # For this validator, let's assume we fine-tune a PPO agent quickly.
            
            train_env = DummyVecEnv([lambda: AdvancedTradingEnv(train_data)])
            model = PPO("MlpPolicy", train_env, verbose=0)
            model.learn(total_timesteps=10000)
            
            # 3. Test (Out-of-Sample)
            # Run the environment through the test data
            test_env = AdvancedTradingEnv(test_data)
            obs, _ = test_env.reset()
            done = False
            cumulative_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                cumulative_reward += reward
                done = terminated or truncated
                
            # 4. Metrics
            # Final Net Worth from environment info or state
            final_val = test_env.net_worth
            roi = (final_val - test_env.initial_balance) / test_env.initial_balance
            
            results.append({
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                "roi": roi,
                "reward": cumulative_reward
            })
            
            # Slide Window
            start_index += self.test_size
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test Run
    # Mock data loading
    try:
        from scripts.train_ensemble import fetch_data
        raw = fetch_data("SPY", "2018-01-01", "2023-01-01")
        fe = FeatureEngineer()
        processed = fe.preprocess(raw)
        
        validator = WalkForwardValidator(processed)
        res = validator.run_validation()
        
        print("\nWalk-Forward Results:")
        print(res)
        print(f"Average Out-of-Sample ROI per Quarter: {res['roi'].mean()*100:.2f}%")
        
    except Exception as e:
        print(f"Validation failed: {e}")

