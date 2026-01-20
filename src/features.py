import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator

class FeatureEngineer:
    """
    Generates stationary, machine-learning-friendly features from financial time series.
    Uses 'ta' library for indicator calculation.
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
            'adx',
            'sentiment', 'regime_prob'
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
        atr_ind = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr_ind.average_true_range()
        df['atr_pct'] = df['atr'] / df['close']
        
        # 3. Relative Strength (Stationary-ish, bounded 0-100)
        rsi_ind = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi_ind.rsi()
        df['rsi_norm'] = (df['rsi'] - 50) / 100.0
        
        # 4. MACD (Stationary transformation)
        macd_ind = MACD(close=df['close'])
        df['macd_norm'] = macd_ind.macd() / df['close']
        df['macd_hist_norm'] = macd_ind.macd_diff() / df['close']
        
        # 5. Bollinger Bands (Stationary interaction)
        bb_ind = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_pct_b'] = bb_ind.bollinger_pband()
        df['bb_width'] = bb_ind.bollinger_wband()
        
        # 6. Volume Oscillator (Stationary)
        df['vol_chg'] = np.log(df['volume'] / df['volume'].shift(1))
        df['vol_chg'] = df['vol_chg'].replace([np.inf, -np.inf], 0)
        
        # 7. Distance from Moving Averages (Stationary)
        sma_50 = SMAIndicator(close=df['close'], window=50).sma_indicator()
        sma_200 = SMAIndicator(close=df['close'], window=200).sma_indicator()
        df['dist_sma50'] = (df['close'] - sma_50) / sma_50
        df['dist_sma200'] = (df['close'] - sma_200) / sma_200
        
        # 8. ADX
        adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx_ind.adx() / 100.0
        
        # 9. Sentiment & Regime (Placeholders for missing live data)
        # These features are expected by the trained model (Total 19 dims = 15 features + 4 state)
        df['sentiment'] = 0.0
        df['regime_prob'] = 0.0
        
        # Clean NaN values
        df.dropna(inplace=True)
        
        # Filter DF to only include intended features + raw data needed for Env
        # This ensures Training Env gets the exact same features as Inference
        keep_cols = ['open', 'high', 'low', 'close', 'volume'] + self.feature_cols
        # Handle date/index if needed, usually index is preserved.
        
        # Filter existing columns
        existing_cols = [c for c in keep_cols if c in df.columns]
        return df[existing_cols]

    def get_latest_features(self, processed_df: pd.DataFrame) -> np.ndarray:
        """Extract the latest feature vector for inference."""
        return processed_df.iloc[-1][self.feature_cols].values.astype(np.float32)

    def create_observation(self, market_features: np.ndarray, 
                          balance: float, shares_value: float, 
                          net_worth: float, initial_balance: float, 
                          max_net_worth: float) -> np.ndarray:
        """
        Construct the RL observation vector matching the training environment.
        Shape: [Features (15) + Account State (4)] = 19 dims
        """
        state = np.array([
            balance / initial_balance,
            shares_value / initial_balance,
            net_worth / initial_balance,
            (net_worth - max_net_worth) / max_net_worth if max_net_worth > 0 else 0.0
        ], dtype=np.float32)
        
        return np.concatenate((market_features, state))
