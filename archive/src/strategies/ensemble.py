import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO, A2C, DDPG
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List

class EnsembleStrategy:
    """
    Institutional-Grade Ensemble Strategy.
    Combines PPO (Trend), A2C (Scalp), and DDPG (Precision) agents
    weighted by a Market Regime Classifier.
    """

    def __init__(self, models_dir: str):
        """
        Args:
            models_dir: Path to directory containing saved models (ppo.zip, a2c.zip, ddpg.zip)
        """
        self.models_dir = models_dir
        
        # Load Agents (Placeholders - assuming they exist or will be trained)
        # We wrap in try-except to allow initialization before training
        try:
            # Check for specific model names (btc_ppo or ppo_model)
            # We try standard names first, then look for whatever is there
            if os.path.exists(f"{models_dir}/btc_ppo.zip"):
                self.agent_ppo = PPO.load(f"{models_dir}/btc_ppo")
                self.agent_a2c = A2C.load(f"{models_dir}/btc_a2c")
                self.agent_ddpg = DDPG.load(f"{models_dir}/btc_ddpg")
            else:
                self.agent_ppo = PPO.load(f"{models_dir}/ppo_model")
                self.agent_a2c = A2C.load(f"{models_dir}/a2c_model")
                self.agent_ddpg = DDPG.load(f"{models_dir}/ddpg_model")
        except Exception as e:
            print(f"Warning: Could not load models from {models_dir}: {e}. Ensemble will return neutral.")
            self.agent_ppo = None
            self.agent_a2c = None
            self.agent_ddpg = None

        # Regime Classifier (Random Forest)
        # In a real scenario, this should be trained and loaded via joblib/pickle
        # For now, we initialize a dummy or load if available
        self.regime_model = RandomForestClassifier(n_estimators=100)
        self.is_regime_model_trained = False

    def train_regime_model(self, X_train, y_train):
        """Train the regime classifier on labeled historical data."""
        self.regime_model.fit(X_train, y_train)
        self.is_regime_model_trained = True
        print("Regime Model Trained.")

    def detect_regime(self, features: np.ndarray) -> str:
        """
        Detect market regime: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOL
        """
        if not self.is_regime_model_trained:
            return "RANGING" # Default
            
        # Prediction
        regime_code = self.regime_model.predict([features])[0]
        # Map code to string if necessary, assuming y_train was strings
        return regime_code

    def predict(self, observation: np.ndarray, features: np.ndarray) -> (float, Dict[str, float]):
        """
        Generate a trading signal (Target Position %) based on Ensemble Voting.
        
        Args:
            observation: The RL environment observation vector (Features + State).
            features: Just the market features for Regime Detection.
            
        Returns:
            float: Target position size [-1.0, 1.0]
            dict: Breakdown of votes and regime
        """
        if not self.agent_ppo:
            return 0.0, {}
            
        # 1. Get Individual Predictions
        # Deterministic=True ensures consistent output for inference
        action_ppo, _ = self.agent_ppo.predict(observation, deterministic=True)
        action_a2c, _ = self.agent_a2c.predict(observation, deterministic=True)
        action_ddpg, _ = self.agent_ddpg.predict(observation, deterministic=True)
        
        # Actions are typically arrays, extract scalar
        val_ppo = float(action_ppo[0]) if isinstance(action_ppo, np.ndarray) else float(action_ppo)
        val_a2c = float(action_a2c[0]) if isinstance(action_a2c, np.ndarray) else float(action_a2c)
        val_ddpg = float(action_ddpg[0]) if isinstance(action_ddpg, np.ndarray) else float(action_ddpg)
        
        # 2. Detect Regime
        regime = self.detect_regime(features)
        
        # 3. Weighted Voting Logic
        # Weights: [PPO, A2C, DDPG]
        weights = [0.33, 0.33, 0.33] # Default Equal
        
        if regime == "TRENDING_UP" or regime == "TRENDING_DOWN":
            # Favor Trend Following (PPO)
            weights = [0.60, 0.20, 0.20]
        elif regime == "RANGING":
            # Favor Mean Reversion (A2C/Reactor)
            weights = [0.20, 0.60, 0.20]
        elif regime == "HIGH_VOL":
             # Favor Precision/Conservative (DDPG) or Cash
             weights = [0.10, 0.10, 0.80]
             
        # Calculate Weighted Average
        final_signal = (
            (val_ppo * weights[0]) + 
            (val_a2c * weights[1]) + 
            (val_ddpg * weights[2])
        )
        
        # 4. Disagreement Filter (Veto)
        # If models strongly disagree (std dev high), reduce size or hold
        predictions = [val_ppo, val_a2c, val_ddpg]
        disagreement = np.std(predictions)
        
        if disagreement > 0.5: # Threshold for high disagreement
            # Reduce confidence/size
            final_signal = final_signal * 0.5
            # Or force hold if it's chaos
            # return 0.0
            
        info = {
            "PPO": val_ppo,
            "A2C": val_a2c,
            "DDPG": val_ddpg,
            "Regime": regime,
            "Disagreement": disagreement,
            "Weights": weights
        }
            
        return np.clip(final_signal, -1.0, 1.0), info

