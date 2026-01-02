"""
Strategy module containing the RegimeStrategy class with Adaptive Profiling.
"""
import backtrader as bt

class RegimeStrategy(bt.Strategy):
    """
    Multi-Regime Strategy with Adaptive Profiling and Volatility Filter:
    - Regime Filter (Macro): Based on SPY Volatility (ATR) and SMA200.
    - Volatility Filter (Asset-Specific): Circuit breaker based on asset ATR.
    - Low Vol Regime -> Momentum Strategy (Strategy A)
    - High Vol Regime -> Mean Reversion Strategy (Strategy B)
    - Supports 'High_Beta', 'Defensive', 'Standard' parameter profiles.
    - Supports Multi-Asset Portfolio execution.
    - Dynamic Risk Budgeting: Adjusts position sizing based on Regime.
    """
    # pylint: disable=no-member, too-many-instance-attributes, super-init-not-called

    # Baseline Parameters (Standard Profile)
    params = (
        ('atr_period', 14),
        ('regime_ma_period', 20),
        ('regime_trend_period', 200), # Regime Trend Filter (SMA 200)

        # Strategy A (Momentum)
        ('mom_sma_long', 50),
        ('mom_sma_short', 20),
        ('mom_rsi_period', 14),
        ('mom_rsi_threshold', 50),

        # Strategy B (Mean Reversion)
        ('mr_rsi_period', 2),
        ('mr_rsi_oversold', 10),
        ('mr_sma_period', 5),

        # Risk Management
        ('risk_pct_cash', 0.10), # Default if not using Sizer
        ('stop_loss_pct', 0.02),

        # Volatility Circuit Breaker
        ('vol_filter_enabled', False),
        ('vol_atr_period', 14),
        ('vol_threshold_pct', 0.05), # Close positions if ATR/Price > 5%
        ('vol_cooldown_periods', 5), # Periods to wait after vol exit before re-entry

        # Adaptive Profile
        ('ticker_profile', 'STANDARD'), # STANDARD, HIGH_BETA, DEFENSIVE
    )

    PROFILES = {
        'HIGH_BETA': {
            'stop_loss_pct': 0.05,
            'mom_sma_long': 30,
            'mr_rsi_oversold': 20,
            'vol_filter_enabled': False,
            'vol_threshold_pct': 0.05,
            'vol_cooldown_periods': 0,
        },
        'DEFENSIVE': {
            'stop_loss_pct': 0.015,
            'mom_sma_long': 100,
            'mr_rsi_oversold': 10,
            'vol_filter_enabled': False,
            'vol_threshold_pct': 0.02,
            'vol_cooldown_periods': 0,
        },
        'CRYPTO': {
            'stop_loss_pct': 0.04,
            'mom_sma_long': 20,
            'mr_rsi_oversold': 30,
            'vol_filter_enabled': True,
            'vol_threshold_pct': 0.05,
            'vol_cooldown_periods': 3,
        },
        'STANDARD': {
            'stop_loss_pct': 0.02,
            'mom_sma_long': 50,
            'mr_rsi_oversold': 10,
            'vol_filter_enabled': False,
            'vol_threshold_pct': 0.03,
            'vol_cooldown_periods': 0,
        }
    }

    # Asset to Profile Mapping
    ASSET_MAP = {
        'COIN': 'CRYPTO', 'MSTR': 'CRYPTO', 'HOOD': 'HIGH_BETA',
        'NVDA': 'HIGH_BETA', 'AMD': 'HIGH_BETA', 'TSLA': 'HIGH_BETA', 'AVGO': 'HIGH_BETA',
        'SOFI': 'HIGH_BETA', 'AFRM': 'HIGH_BETA', 'UPST': 'HIGH_BETA',
        'KO': 'DEFENSIVE', 'JNJ': 'DEFENSIVE', 'XLU': 'DEFENSIVE', 'SPLV': 'DEFENSIVE',
        # Default others to STANDARD
    }

    def log(self, txt, dt=None):
        """Logging helper"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        self.spy = self.datas[0]
        self.assets = self.datas[1:]

        self.inds = {}
        self.states = {}
        self.stop_orders = {} # Key: data feed

        # Regime Indicators (on SPY)
        self.spy_atr = bt.ind.ATR(self.spy, period=self.params.atr_period)
        self.spy_atr_ma = bt.ind.SMA(self.spy_atr, period=self.params.regime_ma_period)
        self.spy_sma200 = bt.ind.SMA(self.spy, period=self.params.regime_trend_period)

        # Init Per-Asset Indicators
        for d in self.assets:
            # pylint: disable=protected-access
            name = d._name
            profile_name = self.ASSET_MAP.get(name, 'STANDARD')

            # Allow overriding profile via params if it's a single asset run
            if len(self.assets) == 1 and self.p.ticker_profile != 'STANDARD':
                profile_name = self.p.ticker_profile

            p = self.PROFILES.get(profile_name, self.PROFILES['STANDARD'])
            print(f"Initializing {name} with profile {profile_name}")

            self.inds[d] = {
                'mom_sma_long': bt.ind.SMA(d.close, period=p['mom_sma_long']),
                'mom_sma_short': bt.ind.SMA(d.close, period=self.params.mom_sma_short),
                'mom_rsi': bt.ind.RSI(d.close, period=self.params.mom_rsi_period),
                'mr_rsi': bt.ind.RSI(d.close, period=self.params.mr_rsi_period),
                'mr_sma': bt.ind.SMA(d.close, period=self.params.mr_sma_period),
                'asset_atr': bt.ind.ATR(d, period=self.params.vol_atr_period),
                'profile_name': profile_name,
                'profile': p
            }

            self.states[d] = {
                'vol_cooldown': 0
            }
            self.stop_orders[d] = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            # pylint: disable=protected-access
            name = order.data._name
            if order.isbuy():
                self.log(
                    f'{name} BUY EXECUTED, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}'
                )

                # Set Stop Loss
                d = order.data
                p = self.inds[d]['profile']
                stop_price = order.executed.price * (1.0 - p['stop_loss_pct'])

                # Cancel existing stop if any (shouldn't be, but safety)
                if self.stop_orders.get(d):
                    self.cancel(self.stop_orders[d])

                self.stop_orders[d] = self.sell(
                    data=d, exectype=bt.Order.Stop, price=stop_price
                )
                self.log(f'{name} STOP LOSS PLACED at {stop_price:.2f}')

            elif order.issell():
                self.log(
                    f'{name} SELL EXECUTED, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}'
                )
                # Cancel stop loss
                d = order.data
                if self.stop_orders.get(d):
                    self.cancel(self.stop_orders[d])
                    self.stop_orders[d] = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # self.log(f'{order.data._name} Order Canceled/Margin/Rejected')
            pass

    def check_volatility(self, d):
        """Checks circuit breaker for a specific data feed."""
        inds = self.inds[d]
        p = inds['profile']

        # Default defaults if not in profile
        vol_enabled = p.get('vol_filter_enabled', False)
        if not vol_enabled:
            return False

        threshold = p.get('vol_threshold_pct', 0.05)
        cooldown = p.get('vol_cooldown_periods', 0)

        current_vol_pct = inds['asset_atr'][0] / d.close[0]

        # Trigger
        if current_vol_pct > threshold:
            if self.getposition(d).size > 0:
                # pylint: disable=protected-access
                self.log(
                    f"{d._name} CIRCUIT BREAKER: Volatility {current_vol_pct:.2%} > "
                    f"{threshold:.2%}. Closing."
                )
                self.close(data=d)

            self.states[d]['vol_cooldown'] = cooldown
            return True

        # Cooldown check
        if self.states[d]['vol_cooldown'] > 0:
            self.states[d]['vol_cooldown'] -= 1
            return True

        return False

    def check_entry_signal(self, d, is_low_vol):
        """Checks entry conditions."""
        inds = self.inds[d]
        p = inds['profile']
        should_buy = False

        if is_low_vol:
            # Strategy A: Momentum
            if d.close[0] > inds['mom_sma_long'][0] and \
               inds['mom_rsi'][0] > self.params.mom_rsi_threshold:
                should_buy = True
        else:
            # Strategy B: Mean Reversion
            if inds['mr_rsi'][0] < p['mr_rsi_oversold']:
                should_buy = True

        return should_buy, p

    def check_exit_signal(self, d, is_low_vol):
        """Checks exit conditions."""
        inds = self.inds[d]
        should_sell = False

        if is_low_vol:
            # Exit A
            if d.close[0] < inds['mom_sma_short'][0]:
                should_sell = True
        else:
            # Exit B
            if d.close[0] > inds['mr_sma'][0]:
                should_sell = True

        return should_sell

    def get_dynamic_allocation(self, profile_name, is_bull_regime):
        """
        Calculates dynamic position sizing based on Regime and Profile.
        Base Allocation = 5%
        """
        base_allocation = 0.05

        if is_bull_regime:
            # Bull Market: Overweight Beta, Underweight Defense
            if profile_name in ['HIGH_BETA', 'CRYPTO']:
                return base_allocation * 1.2 # 6%
            if profile_name == 'DEFENSIVE':
                return base_allocation * 0.8 # 4%
        else:
            # Bear Market: Cut Beta, Overweight Defense
            if profile_name in ['HIGH_BETA', 'CRYPTO']:
                return base_allocation * 0.5 # 2.5%
            if profile_name == 'DEFENSIVE':
                return base_allocation * 1.5 # 7.5%

        # Standard / Default
        return base_allocation

    def next(self):
        # Determine Regime (Global)
        is_low_vol = self.spy_atr[0] < self.spy_atr_ma[0]
        # Bull Regime: SPY > SMA200
        is_bull_regime = self.spy.close[0] > self.spy_sma200[0]

        # Iterate through assets
        for d in self.assets:
            if self.getposition(d).size == 0 and self.stop_orders.get(d):
                 # Cleanup zombie stops
                self.cancel(self.stop_orders[d])
                self.stop_orders[d] = None

            if self.check_volatility(d):
                continue

            # Trading Logic
            if self.getposition(d).size == 0:
                # ENTRY
                should_buy, _ = self.check_entry_signal(d, is_low_vol)

                if should_buy:
                    # Dynamic Risk Budgeting
                    profile_name = self.inds[d]['profile_name']
                    target_allocation = self.get_dynamic_allocation(
                        profile_name, is_bull_regime
                    )

                    port_value = self.broker.get_value()
                    cash_required = port_value * target_allocation

                    # Ensure we have cash
                    if self.broker.get_cash() > cash_required:
                        size = int(cash_required / d.close[0])
                        if size > 0:
                            # pylint: disable=protected-access
                            self.log(
                                f"{d._name} BUY SIGNAL "
                                f"(Regime: {'Bull' if is_bull_regime else 'Bear'}, "
                                f"Alloc: {target_allocation:.1%})"
                            )
                            self.buy(data=d, size=size)

            else:
                # EXIT
                if self.check_exit_signal(d, is_low_vol):
                    # pylint: disable=protected-access
                    self.log(f"{d._name} SELL SIGNAL")
                    self.close(data=d)
