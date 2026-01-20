import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import FeatureEngineer
from src.rl.trading_env import AdvancedTradingEnv

def fetch_crypto_data(symbol="BTC/USD", start="2024-01-01", end="2024-06-01"):
    print(f"Downloading {symbol} data from Alpaca (5Min)...")
    load_dotenv()
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        raise ValueError("Missing Alpaca Credentials")
        
    client = CryptoHistoricalDataClient(api_key, secret_key)
    
    # 5 Minute Data
    tf = TimeFrame(5, TimeFrameUnit.Minute)
    
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=tf,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end)
    )
    
    bars = client.get_crypto_bars(req)
    df = bars.df
    
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol)
        
    return df

def train_crypto_ensemble():
    # 1. Configuration
    MODELS_DIR = "models_crypto" # Sandbox folder
    LOG_DIR = "logs_rl/crypto"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    SYMBOL = "BTC/USD"
    
    # 2. Data & Features
    try:
        # Shorter window for 5min data to avoid huge download/processing time
        raw_df = fetch_crypto_data(SYMBOL, "2024-01-01", "2024-06-01")
        fe = FeatureEngineer()
        processed_df = fe.preprocess(raw_df)
        print(f"Data shape after preprocessing: {processed_df.shape}")
        
        if processed_df.empty:
            raise ValueError("Processed DF is empty")
            
    except Exception as e:
        print(f"Failed to fetch/process data: {e}")
        return

    # 3. Environment
    env = DummyVecEnv([lambda: AdvancedTradingEnv(processed_df)])
    
    # 4. Train PPO
    print("\n--- Training PPO (Crypto) ---")
    ppo_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    ppo_model.learn(total_timesteps=50000)
    ppo_model.save(f"{MODELS_DIR}/ppo_model")
    
    # 5. Train A2C
    print("\n--- Training A2C (Crypto) ---")
    a2c_model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    a2c_model.learn(total_timesteps=50000)
    a2c_model.save(f"{MODELS_DIR}/a2c_model")
    
    # 6. Train DDPG
    print("\n--- Training DDPG (Crypto) ---")
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    ddpg_model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=LOG_DIR)
    ddpg_model.learn(total_timesteps=50000)
    ddpg_model.save(f"{MODELS_DIR}/ddpg_model")
    
    print(f"\nTraining Complete. Crypto Models saved to '{MODELS_DIR}'")

if __name__ == "__main__":
    train_crypto_ensemble()
