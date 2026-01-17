import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import contextlib
import io

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategy_logic import StrategyLogic
from src.risk_manager import RiskManager as RiskManagerV1
from src.strategy_v2 import TradingBot as TradingBotV2

# Suppress prints during simulation
@contextlib.contextmanager
def suppress_stdout():
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        yield

def fetch_data(symbols, start_date, end_date):
    data = {}
    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")
    for symbol in symbols:
        # Download with sufficient warmup
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, multi_level_index=False)
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            data[symbol] = df
    return data

class Portfolio:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {} # {symbol: {'qty': 0, 'entry': 0.0, 'stop': 0.0}}
        self.equity_curve = []
        self.trade_count = 0

    def update_equity(self, current_prices):
        equity = self.cash
        for sym, pos in self.positions.items():
            if sym in current_prices:
                equity += pos['qty'] * current_prices[sym]
        self.equity_curve.append(equity)
        return equity

    def enter_position(self, symbol, qty, price, stop_loss):
        if qty <= 0: return
        cost = qty * price
        if self.cash >= cost:
            self.cash -= cost
            self.positions[symbol] = {'qty': qty, 'entry': price, 'stop': stop_loss}
            self.trade_count += 1

    def exit_position(self, symbol, price, qty=None):
        if symbol in self.positions:
            pos = self.positions[symbol]
            exit_qty = qty if qty else pos['qty']
            
            proceeds = exit_qty * price
            self.cash += proceeds
            
            if exit_qty == pos['qty']:
                del self.positions[symbol]
            else:
                pos['qty'] -= exit_qty

    def get_metrics(self):
        if not self.equity_curve:
            return 0, 0, 0
            
        final_equity = self.equity_curve[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        # Drawdown
        peaks = pd.Series(self.equity_curve).cummax()
        drawdown = (pd.Series(self.equity_curve) - peaks) / peaks
        max_dd = drawdown.min() * 100
        
        return total_return, self.trade_count, max_dd

def run_comparison():
    # 1. Configuration
    symbols = ['SPY', 'QQQ', 'NVDA'] # Added NVDA for volatility
    initial_capital = 10000.0
    
    # Dates: Test on last 1 year (with buffer for indicators)
    end_date = datetime.now()
    # Need at least 200 trading days for SMA200 BEFORE the test starts
    start_date_data = end_date - timedelta(days=730) # 2 years buffer total (1 yr test + 1 yr warmup)
    start_date_test = end_date - timedelta(days=365) # Actual test start
    
    market_data = fetch_data(symbols, start_date_data, end_date)
    
    # 2. Setup V1
    # Mock config for V1
    config_v1 = {
        'risk': {'max_position_pct': 0.10, 'stop_loss_atr_multiplier': 2.0, 'take_profit_atr_multiplier': 5.0, 'max_drawdown_pct': 1.0},
        'strategy': {'rsi_period': 14, 'sma_fast': 50, 'sma_slow': 200, 'buy_score_threshold': 60}
    }
    rm_v1 = RiskManagerV1(config_v1)
    strat_v1 = StrategyLogic(rm_v1, config_v1)
    portfolio_v1 = Portfolio(initial_capital)

    # 3. Setup V2
    # V2 Bot instance handles logic, but we need to manage state externally for the comparison loop 
    # to keep it fair and synchronized, or we instantiate one bot per symbol? 
    # V2 class stores state in `self.positions` but only for one instance.
    # We will use one bot instance per symbol OR just use the logic methods.
    # The prompt says "Initialize TradingBot (New V2)". 
    # Let's use `execute_trade_logic` which is stateless regarding portfolio (it returns action), 
    # and `manage_open_trade` for exits.
    bot_v2 = TradingBotV2(account_size=initial_capital) # Used for logic calls
    portfolio_v2 = Portfolio(initial_capital)
    
    # 4. Simulation Loop
    print(f"Simulating {start_date_test.date()} to {end_date.date()}...")
    
    # Get all trading days
    all_dates = sorted(list(set().union(*[df.index for df in market_data.values()])))
    test_dates = [d for d in all_dates if d >= start_date_test]
    
    total_days = len(test_dates)
    
    for i, current_date in enumerate(test_dates):
        if i % 20 == 0:
            print(f"Processing {i}/{total_days}...", end='\r')
            
        current_prices = {}
        for sym in symbols:
            if current_date in market_data[sym].index:
                current_prices[sym] = market_data[sym].loc[current_date]['close']
        
        # Update Equities
        v1_equity = portfolio_v1.update_equity(current_prices)
        v2_equity = portfolio_v2.update_equity(current_prices)
        
        for sym in symbols:
            if sym not in current_prices: continue
            
            # Slice Data
            # We need history UP TO current_date (inclusive for signal generation usually happens on Close)
            # In live trading, we run AFTER market close or NEAR close. 
            # So we use data including today.
            hist_data = market_data[sym].loc[:current_date]
            if len(hist_data) < 200: continue
            
            current_price = current_prices[sym]
            
            # --- V1 Execution ---
            # Exit Check (Trailing Stop is in Main Loop for V1, logic in RM)
            # We need to manually simulate V1 exit logic if it's not in generate_signal
            # V1 generate_signal returns SELL if logic says sell.
            # But Trailing Stop is separate in main.py. We'll implement basic stop check here.
            
            # Check V1 Exits first
            if sym in portfolio_v1.positions:
                pos = portfolio_v1.positions[sym]
                # Check Stop Loss
                if current_price <= pos['stop']:
                    portfolio_v1.exit_position(sym, pos['stop']) # Slippage assumed 0
                else:
                    # Update Trailing Stop (Using RM logic)
                    # Need ATR. generate_signal calculates it, but returns it in debug.
                    # Re-calc ATR or approximation? 
                    # Let's trust generate_signal for SELLs and basic SL for now.
                    # Or better, run generate_signal.
                    pass
            
            # with suppress_stdout():
            #     # V1 Signal
            #     # Pass equity for sizing
            #     sig_v1 = strat_v1.generate_signal(sym, hist_data, sentiment_score=0.0, account_value=v1_equity)
            
            sig_v1 = strat_v1.generate_signal(sym, hist_data, sentiment_score=0.5, account_value=v1_equity) # Mock sentiment to allow trades
            # if i % 50 == 0: print(f"V1 {sym} {current_date.date()}: {sig_v1['action']} {sig_v1.get('reason')} Score={sig_v1.get('debug', {}).get('score')}")
            
            if sym in portfolio_v1.positions:
                # Check for Sell signal
                if sig_v1['action'] == 'SELL':
                    portfolio_v1.exit_position(sym, current_price)
            elif sig_v1['action'] == 'BUY':
                # Enter
                qty = sig_v1['quantity']
                # Scale down qty for multi-asset portfolio if needed, but RM config handles it (10% max pos)
                if qty > 0:
                    portfolio_v1.enter_position(sym, qty, current_price, sig_v1['stop_loss'])
            
            # --- V2 Execution ---
            # V2 Logic
            
            # Check V2 Exits
            if sym in portfolio_v2.positions:
                pos = portfolio_v2.positions[sym]
                # Prepare open_trade dict for V2 manager
                open_trade = {
                    "entry_price": pos['entry'],
                    "stop_loss": pos['stop']
                }
                exit_sig = bot_v2.manage_open_trade(open_trade, current_price)
                
                if exit_sig['action'] == 'SELL':
                    portfolio_v2.exit_position(sym, current_price)
                elif exit_sig['action'] == 'SELL_HALF':
                    portfolio_v2.exit_position(sym, current_price, qty=pos['qty']//2)
                    # Update trailing stop
                    if 'trail_stop_to' in exit_sig:
                        portfolio_v2.positions[sym]['stop'] = exit_sig['trail_stop_to']
                elif exit_sig['action'] == 'HOLD':
                    pass
            else:
                # Check Entry
                # with suppress_stdout():
                res_v2 = bot_v2.execute_trade_logic(hist_data)
                if i % 20 == 0 and sym == 'NVDA': print(f"V2 {sym} {current_date.date()}: {res_v2['action']} {res_v2.get('reason', '')}")
                
                if res_v2['action'] == 'BUY':
                    # V2 sizing is internal to bot, but we need to use portfolio cash
                    # bot.risk_manager.calculate_position_size uses account_size passed in init.
                    # We should calculate based on current equity? 
                    # Let's use the result qty but ensure we have cash.
                    qty = res_v2['quantity']
                    # Re-calc based on actual equity to be fair
                    # V2 logic uses self.capital (static $10k). 
                    # Let's adjust to dynamic equity
                    qty = bot_v2.risk_manager.calculate_position_size(
                        v2_equity, 0.05, res_v2['entry_price'], res_v2['stop_loss'] # 5% risk per trade for this test? Or 1%?
                        # V2 default is 1%. V1 is configured to 10% pos size.
                        # Let's try to match risk. V1 10% pos size ~ 2% risk if SL is 20%.
                        # Let's stick to V2 defaults or close to it.
                    )
                    
                    if qty > 0:
                        portfolio_v2.enter_position(sym, qty, current_price, res_v2['stop_loss'])

    # 5. Results
    ret_v1, trades_v1, dd_v1 = portfolio_v1.get_metrics()
    ret_v2, trades_v2, dd_v2 = portfolio_v2.get_metrics()
    
    print("\n" + "="*50)
    print("=== STRATEGY SHOWDOWN ===")
    print(f"{'Metric':<16} | {'V1 (Score Based)':<16} | {'V2 (Regime Based)':<16}")
    print("-" * 54)
    print(f"{'Total Return':<16} | {ret_v1:>15.2f}% | {ret_v2:>15.2f}%")
    print(f"{'Num Trades':<16} | {trades_v1:>16} | {trades_v2:>16}")
    print(f"{'Max Drawdown':<16} | {dd_v1:>15.2f}% | {dd_v2:>15.2f}%")
    print("-" * 54)
    
    winner = "V2" if ret_v2 > ret_v1 else "V1"
    print(f"WINNER: {winner}")
    print("="*50)

if __name__ == "__main__":
    run_comparison()

