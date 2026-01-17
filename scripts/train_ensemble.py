import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import FeatureEngineer
from src.rl.trading_env import AdvancedTradingEnv

def fetch_data(ticker="SPY", start="2020-01-01", end="2023-01-01"):
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data downloaded")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def train_ensemble():
    # 1. Configuration
    MODELS_DIR = "models"
    LOG_DIR = "logs_rl"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    TICKER = "SPY"
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2023-01-01"
    
    # 2. Data & Features
    raw_df = fetch_data(TICKER, TRAIN_START, TRAIN_END)
    fe = FeatureEngineer()
    processed_df = fe.preprocess(raw_df)
    
    print(f"Data shape after preprocessing: {processed_df.shape}")
    
    env = DummyVecEnv([lambda: AdvancedTradingEnv(processed_df)])
    
    # 4. Train PPO (The Anchor) - OPTIMIZED PARAMS
    print("\n--- Training PPO (Trend Follower) [Optimized] ---")
    ppo_model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=0.00010414875393741091,
        ent_coef=0.0,
        n_steps=4096,
        gamma=0.99
    )
    ppo_model.learn(total_timesteps=50000)
    ppo_model.save(f"{MODELS_DIR}/ppo_model")
    
    # 5. Train A2C (The Reactor / Scalper)
    print("\n--- Training A2C (Mean Reversion) ---")
    a2c_model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    a2c_model.learn(total_timesteps=50000)
    a2c_model.save(f"{MODELS_DIR}/a2c_model")
    
    # 6. Train DDPG (The Sniper / Precision)
    print("\n--- Training DDPG (Precision Sizing) ---")
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    ddpg_model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=LOG_DIR)
    ddpg_model.learn(total_timesteps=30000)
    ddpg_model.save(f"{MODELS_DIR}/ddpg_model")
    
    print("\nTraining Complete. Models saved to 'models/'")

if __name__ == "__main__":
    train_ensemble()
