from typing import Dict, Any, Union

class RiskManager:
    """
    Manages risk parameters for trading operations including position sizing,
    stop-losses, take-profits, and drawdown control.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RiskManager with configuration.

        Args:
            config: Dictionary containing risk parameters under a 'risk' key.
        """
        self.config = config
        self.risk_config = config.get('risk', {})
        
        # Load risk parameters with defaults if missing
        self.max_pos_pct = self.risk_config.get('max_position_pct', 0.02)
        # self.sl_pct = self.risk_config.get('stop_loss_pct', 0.02) # Deprecated for ATR
        # self.tp_pct = self.risk_config.get('take_profit_pct', 0.05) # Deprecated for ATR
        self.max_dd_pct = self.risk_config.get('max_drawdown_pct', 0.10)
        
        # ATR-based Risk Params
        self.sl_atr_mult = self.risk_config.get('stop_loss_atr_multiplier', 2.0)
        self.tp_atr_mult = self.risk_config.get('take_profit_atr_multiplier', 4.0)

    def calculate_position_size(self, account_value: float, price: float, confidence_mult: float = 1.0) -> int:
        """
        Calculate the number of shares to buy based on account risk.

        Args:
            account_value: Total account value (equity).
            price: Current price of the asset.
            confidence_mult: Multiplier for position size (0.25 to 1.5).

        Returns:
            Number of shares as an integer. Returns 0 if calculation fails or price is invalid.
        """
        if price <= 0 or account_value <= 0:
            return 0

        # Clamp confidence multiplier
        mult = max(0.25, min(confidence_mult, 1.5))

        # Calculate max capital to allocate to this trade
        allocation = account_value * self.max_pos_pct * mult
        
        # Calculate quantity
        quantity = int(allocation // price)
        
        return quantity

    def get_stop_loss_price(self, entry_price: float, side: str = "buy", atr_value: float = 0.0) -> float:
        """
        Calculate the stop-loss price using ATR.

        Args:
            entry_price: The price at which the trade was entered.
            side: 'buy' (long) or 'sell' (short). Defaults to 'buy'.
            atr_value: Current ATR value for dynamic sizing.

        Returns:
            The stop-loss price.
        """
        if atr_value <= 0:
            # Fallback to fixed % if ATR invalid
            default_pct = 0.02
            if side.lower() == "buy": return entry_price * (1 - default_pct)
            else: return entry_price * (1 + default_pct)

        if side.lower() == "buy":
            return entry_price - (atr_value * self.sl_atr_mult)
        elif side.lower() == "sell":
            return entry_price + (atr_value * self.sl_atr_mult)
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")

    def get_take_profit_price(self, entry_price: float, side: str = "buy", atr_value: float = 0.0) -> float:
        """
        Calculate the take-profit price using ATR.

        Args:
            entry_price: The price at which the trade was entered.
            side: 'buy' (long) or 'sell' (short). Defaults to 'buy'.
            atr_value: Current ATR value for dynamic sizing.

        Returns:
            The take-profit price.
        """
        if atr_value <= 0:
             # Fallback to fixed % if ATR invalid
            default_pct = 0.05
            if side.lower() == "buy": return entry_price * (1 + default_pct)
            else: return entry_price * (1 - default_pct)

        if side.lower() == "buy":
            return entry_price + (atr_value * self.tp_atr_mult)
        elif side.lower() == "sell":
            return entry_price - (atr_value * self.tp_atr_mult)
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")

    def check_drawdown(self, current_equity: float, peak_equity: float) -> bool:
        """
        Check if the portfolio has exceeded the maximum drawdown.

        Args:
            current_equity: Current total portfolio value.
            peak_equity: Highest recorded portfolio value.

        Returns:
            True if trading should stop (max drawdown exceeded), False otherwise.
        """
        if peak_equity <= 0:
            return False
            
        drawdown = (peak_equity - current_equity) / peak_equity
        
        return drawdown > self.max_dd_pct
