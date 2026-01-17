import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer:
    """
    Generates stationary, machine-learning-friendly features from financial time series.
    Uses pandas-ta for indicator calculation.
    """

    def __init__(self):
        self.regime_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_fitted = False
        # Define feature columns explicitly to ensure order matching
        # Must match the order of creation in preprocess + allow for intermediates if env used them
        self.feature_cols = [
            'log_ret', 
            'atr', 'atr_pct', 
            'rsi', 'rsi_norm', 
            'macd_norm', 'macd_hist_norm',
            'bb_pct_b', 'bb_width', 
            'vol_chg', 
            'dist_sma50', 'dist_sma200', 
            'adx'
        ]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline: Cleans data -> Adds Indicators -> Transforms to Stationary -> Handles NaNs.
        """
        # Ensure standard column names
        df.columns = [c.lower() for c in df.columns]
        
        # 1. Log Returns (Stationary)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Volatility (Stationary)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_pct'] = df['atr'] / df['close']
        
        # 3. Relative Strength (Stationary-ish, bounded 0-100)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rsi_norm'] = (df['rsi'] - 50) / 100.0
        
        # 4. MACD (Stationary transformation)
        macd = ta.macd(df['close'])
        df['macd_norm'] = macd['MACD_12_26_9'] / df['close']
        df['macd_hist_norm'] = macd['MACDh_12_26_9'] / df['close']
        
        # 5. Bollinger Bands (Stationary interaction)
        bb = ta.bbands(df['close'], length=20, std=2)
        df['bb_pct_b'] = bb['BBP_20_2.0']
        df['bb_width'] = bb['BBB_20_2.0']
        
        # 6. Volume Oscillator (Stationary)
        df['vol_chg'] = np.log(df['volume'] / df['volume'].shift(1))
        df['vol_chg'] = df['vol_chg'].replace([np.inf, -np.inf], 0)
        
        # 7. Distance from Moving Averages (Stationary)
        sma_50 = ta.sma(df['close'], length=50)
        sma_200 = ta.sma(df['close'], length=200)
        df['dist_sma50'] = (df['close'] - sma_50) / sma_50
        df['dist_sma200'] = (df['close'] - sma_200) / sma_200
        
        # 8. ADX
        adx = ta.adx(df['high'], df['low'], df['close'])
        df['adx'] = adx['ADX_14'] / 100.0
        
        # Clean NaN values
        df.dropna(inplace=True)
        
        return df

    def get_latest_features(self, processed_df: pd.DataFrame) -> np.ndarray:
        """Extract the latest feature vector for inference."""
        return processed_df.iloc[-1][self.feature_cols].values.astype(np.float32)

    def create_observation(self, market_features: np.ndarray, 
                          balance: float, shares_value: float, 
                          net_worth: float, initial_balance: float, 
                          max_net_worth: float) -> np.ndarray:
        """
        Construct the RL observation vector matching the training environment.
        Shape: [Features (11) + Account State (4)] = 15 dims
        """
        state = np.array([
            balance / initial_balance,
            shares_value / initial_balance,
            net_worth / initial_balance,
            (net_worth - max_net_worth) / max_net_worth if max_net_worth > 0 else 0.0
        ], dtype=np.float32)
        
        return np.concatenate((market_features, state))
