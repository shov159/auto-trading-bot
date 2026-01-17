import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional, Tuple, List
from src.indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_macd,
    calculate_atr
)
from src.risk_manager import RiskManager

class StrategyLogic:
    """
    Core strategy logic combining technical indicators, fundamental data,
    and sentiment analysis to generate trading signals (Hybrid V1.5).
    """

    def __init__(self, risk_manager: RiskManager, config: Dict[str, Any]):
        """
        Initialize the StrategyLogic.

        Args:
            risk_manager: Instance of RiskManager for sizing and limits.
            config: Dictionary containing strategy parameters.
        """
        self.risk_manager = risk_manager
        self.config = config
        self.strat_config = config.get('strategy', {})
        
        # Load parameters
        self.rsi_period = self.strat_config.get('rsi_period', 14)
        self.rsi_buy_thresh = 60 # Relaxed from 45
        self.rsi_sell_thresh = 75
        self.sma_period = self.strat_config.get('sma_fast', 50)
        self.sma_slow_period = self.strat_config.get('sma_slow', 200)
        
        # Sentiment thresholds
        self.sentiment_buy_thresh = 0.0 # Relaxed from 0.5
        self.sentiment_sell_thresh = -0.2
        
        # Threshold for Buy Signal (can be overridden by config)
        self.buy_threshold = self.strat_config.get('buy_score_threshold', 60)
        
        # Cache for fundamentals
        self._fundamental_cache = {}

    def _get_fundamental_score(self, symbol: str) -> int:
        """
        Calculate a fundamental score (-20 to +20) based on Market Cap, PE, and Growth.
        Cached to avoid API rate limits.
        """
        if symbol in self._fundamental_cache:
            return self._fundamental_cache[symbol]
        
        score = 0
        try:
            # Fetch info (blocking call, use cache!)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 1. Market Cap (Want > $2B for liquidity)
            mkt_cap = info.get('marketCap', 0)
            if mkt_cap < 2_000_000_000:
                score -= 20
            
            # 2. Valuation (Trailing PE < 50 is reasonable for growth/tech)
            pe = info.get('trailingPE')
            rev_growth = info.get('revenueGrowth')
            
            # Allow High PE if Growth is strong (>20%)
            is_growth_stock = rev_growth and rev_growth > 0.20
            
            if pe and (0 < pe < 50 or is_growth_stock):
                score += 10
            
            # 3. Growth (Revenue Growth > 0)
            if rev_growth and rev_growth > 0:
                score += 10
                
        except Exception as e:
            # On error, assume neutral/no bonus/no penalty if data missing, 
            # or maybe penalty for safety. Let's start neutral (0).
            # Print error only if debugging
            pass
            
        self._fundamental_cache[symbol] = score
        return score

    def calculate_trade_score(self, 
                              symbol: str, 
                              price: float, 
                              sma_200: float, 
                              macd: float, 
                              signal_line: float, 
                              macd_hist_slope: float,
                              rsi: float, 
                              sentiment_score: float) -> Tuple[int, List[str]]:
        """
        Calculate a weighted score (0-100) using Hybrid Logic V1.5.
        
        Formula: (0.4 * Tech) + (0.2 * Fund) + (0.2 * Sent) + (0.2 * Risk_Reward)
        
        Returns:
            Tuple[score, list_of_reasons]
        """
        reasons = []
        
        # 1. Technical Score (0-100)
        # Components:
        # - Trend (Price > SMA200): 40 pts
        # - Momentum (MACD > Signal): 20 pts
        # - Momentum Building (Hist Slope > 0): 10 pts
        # - Value (RSI < 40): 30 pts
        tech_score = 0
        if price > sma_200:
            tech_score += 40
            reasons.append("📈 Bullish Trend (Price > SMA200)")
        if macd > signal_line:
            tech_score += 20
            reasons.append("🟢 MACD Bullish Cross")
        if macd_hist_slope > 0:
            tech_score += 10
            reasons.append("🚀 Momentum Building (MACD Hist)")
        if rsi < 40:
            tech_score += 30
            reasons.append(f"📉 Oversold (RSI {rsi:.1f} < 40)")
            
        # 2. Fundamental Score (0-100 mapped from -20..+20 raw)
        fund_raw = self._get_fundamental_score(symbol)
        
        # Add fundamental reasons (re-deriving slightly or checking raw score)
        # Note: Ideally _get_fundamental_score returns reasons too, but for now we infer.
        if fund_raw >= 10:
             reasons.append("💰 Strong Fundamentals (PE/Growth)")
        elif fund_raw <= -10:
             reasons.append("⚠️ Weak Fundamentals (Small Cap/No Growth)")
             
        fund_score = max(0, min(100, 50 + fund_raw * 2.5)) # Scaling to make impact visible

        # 3. Sentiment Score (0-100)
        # input is -1.0 to 1.0. 
        # -1 -> 0, 0 -> 50, 1 -> 100.
        sent_score = max(0, min(100, (sentiment_score + 1) * 50))
        if sentiment_score > 0.2:
            reasons.append(f"📰 Positive News Sentiment ({sentiment_score:.2f})")
        elif sentiment_score < -0.2:
            reasons.append(f"📢 Negative News Sentiment ({sentiment_score:.2f})")
        
        # 4. Risk/Reward Score (0-100)
        if rsi < 30: 
            rr_score = 100
            reasons.append("💎 Great Risk/Reward (RSI < 30)")
        elif rsi > 70: 
            rr_score = 0
            reasons.append("⚠️ Poor Risk/Reward (Overbought)")
        else: 
            rr_score = 100 - rsi # E.g., RSI 50 -> 50. RSI 40 -> 60.
        
        # Final Weighted Calculation
        final_score = (
            (0.4 * tech_score) + 
            (0.2 * fund_score) + 
            (0.2 * sent_score) + 
            (0.2 * rr_score)
        )
        
        return int(final_score), reasons

    def generate_signal(self, symbol: str, data: pd.DataFrame, sentiment_score: float, account_value: float) -> Dict[str, Any]:
        """
        Analyze data and return a trading signal.

        Args:
            symbol: Ticker symbol.
            data: DataFrame with OHLCV data.
            sentiment_score: Sentiment score (-1.0 to 1.0).
            account_value: Total account equity (for position sizing).

        Returns:
            Dictionary containing action, price, SL/TP details, and reasoning.
        """
        if data.empty or len(data) < self.sma_slow_period:
            return {"action": "HOLD", "reason": "Insufficient Data"}

        # 1. Calculate Indicators
        rsi = calculate_rsi(data['close'], period=self.rsi_period)
        sma_50 = calculate_sma(data['close'], window=self.sma_period)
        sma_200 = calculate_sma(data['close'], window=self.sma_slow_period)
        macd_line, signal_line = calculate_macd(data['close'])
        atr = calculate_atr(data['high'], data['low'], data['close']) # Default window 14
        
        # Calculate MACD Histogram and Slope
        macd_hist = macd_line - signal_line
        # Slope: Current Hist - Previous Hist
        if len(macd_hist) > 1:
            hist_slope = macd_hist.iloc[-1] - macd_hist.iloc[-2]
        else:
            hist_slope = 0

        # Get latest values
        current_close = data['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_sma = sma_50.iloc[-1]
        current_sma_200 = sma_200.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_atr = atr.iloc[-1]

        # 2. Logic Check
        
        # SELL Conditions (Priority Check)
        sell_reasons = []
        if sentiment_score < self.sentiment_sell_thresh:
            sell_reasons.append("Negative Sentiment")
        if current_rsi > self.rsi_sell_thresh:
            sell_reasons.append("RSI Overbought")
        if current_close < current_sma * 0.95:
            sell_reasons.append("Trend Broken (Below SMA)")
            
        if sell_reasons:
            return {
                "action": "SELL",
                "price": current_close,
                "reason": " + ".join(sell_reasons)
            }

        # BUY Conditions - Hybrid Scoring V1.5
        trade_score, reasons_list = self.calculate_trade_score(
            symbol,
            current_close, 
            current_sma_200, 
            current_macd, 
            current_signal, 
            hist_slope,
            current_rsi, 
            sentiment_score
        )
        
        buy_reasons = []
        # Threshold: Only generate a "BUY" signal if Total Score >= Threshold
        if trade_score >= self.buy_threshold:
            buy_reasons.append(f"Strong Score ({trade_score})")
            buy_reasons.extend(reasons_list)
            
            # Calculate Risk Parameters
            quantity = self.risk_manager.calculate_position_size(account_value, current_close)
            stop_loss = self.risk_manager.get_stop_loss_price(current_close, "buy", current_atr)
            take_profit = self.risk_manager.get_take_profit_price(current_close, "buy", current_atr)
            
            if quantity > 0:
                return {
                    "action": "BUY",
                    "price": current_close,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reason": " | ".join(buy_reasons),
                    "debug": {
                        "score": trade_score,
                        "rsi": current_rsi,
                        "sma_50": current_sma,
                        "sma_200": current_sma_200,
                        "macd": current_macd,
                        "signal": current_signal,
                        "sentiment": sentiment_score,
                        "atr": current_atr
                    }
                }
            else:
                return {"action": "HOLD", "reason": "Insufficient Capital for Risk Sizing"}

        return {
            "action": "HOLD", 
            "reason": f"Low Score ({trade_score})",
            "debug": {
                "score": trade_score,
                "rsi": current_rsi,
                "sma_50": current_sma,
                "sma_200": current_sma_200,
                "macd": current_macd,
                "signal": current_signal,
                "sentiment": sentiment_score,
                "atr": current_atr
            }
        }

