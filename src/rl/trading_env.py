import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class AdvancedTradingEnv(gym.Env):
    """
    A custom trading environment for RL agents with advanced reward shaping.
    Inherits from gymnasium.Env.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super(AdvancedTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # State variables
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.prev_net_worth = initial_balance
        self.trades = []
        
        # Features columns (assume the df passed has these 42+ features)
        # We drop date/index if it's not numeric
        self.obs_features = [c for c in df.columns if c not in ['date', 'Date', 'open', 'high', 'low', 'close', 'volume']]
        # Add 'close' back if needed for PnL, but observation usually uses normalized features.
        # Ideally, we pass the price array separately or keep it in df but not in observation if unscaled.
        # For simplicity, we assume df contains SCALED features for observation + raw price for calc.
        
        # Action Space: Continuous Box [-1, 1]
        # -1 = Sell 100%, +1 = Buy 100%, 0 = Hold
        # Or interpreted as target position % (-1 to 1 short/long, or 0 to 1 for long only)
        # Let's assume Long/Cash only for now -> Action [0, 1] = Target % of Portfolio in Asset
        # Or Action [-1, 1] -> Change in position?
        # Standard: Action determines TARGET % of portfolio.
        # -1 = Short 100% (if allowed) or Sell Everything
        # Let's implement: Continuous [-1, 1].
        # > 0: Buy/Hold Long (Magnitude = confidence/sizing)
        # < 0: Sell/Short (if we allow shorting)
        # For this bot (Long Only spot): 
        # Action > 0: Target Position = Action * Balance. 
        # Action <= 0: Sell everything.
        # Actually, let's Stick to "Target Percentage" of portfolio to be Long.
        # Action in [-1, 1]. If < 0 -> 0% (Cash). If > 0 -> X% invested.
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation Space:
        # Features from DF + Account State (Balance, Shares, Net Worth, PnL)
        # 42 features + 4 state features = 46 dimensions
        # We need to define the shape based on actual data
        n_features = len(self.obs_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.trades = []
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Get market features for current step
        frame = self.df.iloc[self.current_step]
        
        # Features vector
        market_obs = frame[self.obs_features].values.astype(np.float32)
        
        # Account state vector
        # Normalize roughly or log-scale for stability? For now, raw or simple ratio.
        # RL prefers normalized. 
        # Let's use: [Balance/Init, SharesValue/Init, NetWorth/Init, Drawdown]
        current_price = self.df.iloc[self.current_step]['close']
        shares_value = self.shares_held * current_price
        
        state = np.array([
            self.balance / self.initial_balance,
            shares_value / self.initial_balance,
            self.net_worth / self.initial_balance,
            (self.net_worth - self.max_net_worth) / self.max_net_worth # Drawdown
        ], dtype=np.float32)
        
        return np.concatenate((market_obs, state))

    def step(self, action):
        # 1. Execute Action
        current_price = self.df.iloc[self.current_step]['close']
        
        # Action determines Target Portfolio Percentage (Long Only)
        # Map [-1, 1] to [0, 1] for long only? Or just clip?
        # Let's say: Action > 0 means Target % = Action. Action <= 0 means 0% (Cash).
        target_pct = np.clip(action[0], 0, 1) 
        
        target_value = self.net_worth * target_pct
        current_value = self.shares_held * current_price
        
        # Rebalance
        if target_value > current_value:
            # Buy
            amount_to_buy = target_value - current_value
            cost = amount_to_buy # Simply
            # Check cash
            if cost > self.balance:
                cost = self.balance # Cap at cash
            
            shares_to_buy = cost / current_price
            self.shares_held += shares_to_buy
            self.balance -= cost
            
        elif target_value < current_value:
            # Sell
            amount_to_sell = current_value - target_value
            shares_to_sell = amount_to_sell / current_price
            self.shares_held -= shares_to_sell
            self.balance += amount_to_sell

        # 2. Advance Step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        if terminated:
            # Calculate final stats
            pass

        # 3. Calculate Reward (Reward Shaping)
        # Update Net Worth
        next_price = self.df.iloc[self.current_step]['close']
        self.net_worth = self.balance + (self.shares_held * next_price)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        
        # Components
        # A. Realized PnL (Step PnL)
        step_pnl = self.net_worth - self.prev_net_worth
        
        # B. Volatility Penalty (Erratic trading or high variance)
        # Can be proxied by large changes in PnL or Action changes?
        # Let's penalize negative PnL more heavily? Or Variance?
        vol_penalty = 0
        if step_pnl < 0:
            vol_penalty = abs(step_pnl) * 0.1 # Extra 10% pain on loss
            
        # C. Time Decay (Holding losing position)
        time_decay = 0
        if self.shares_held > 0 and step_pnl < 0:
            time_decay = 0.001 * self.initial_balance # Small bleed
            
        # D. Sharpe Boost (If rolling sharpe increases)
        # Simplified: Bonus for consistent positive returns
        sharpe_boost = 0
        if step_pnl > 0:
            sharpe_boost = step_pnl * 0.1 # Bonus for green days
            
        # Total Reward
        # Normalize reward to be somewhat scale invariant? 
        # PPO likes rewards around [-1, 1] or [-10, 10]. 
        # step_pnl can be large (dollars). Divide by initial_balance to get % return?
        # Let's use % Return for reward stability
        pct_return = (step_pnl / self.initial_balance) * 100
        
        reward = pct_return - (vol_penalty/self.initial_balance*100) - (time_decay/self.initial_balance*100) + (sharpe_boost/self.initial_balance*100)
        
        # Update state
        self.prev_net_worth = self.net_worth
        
        info = {
            "net_worth": self.net_worth,
            "step_pnl": step_pnl
        }
        
        return self._next_observation(), reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}")

