"""
Backtesting Engine
Simulates trading execution using historical data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

class Backtester:
    def __init__(self, strategy, risk_manager, initial_capital=100000.0):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = {} # {ticker: qty}
        self.trade_log = []
        self.equity_curve = []
        self.positions = {} # {ticker: {entry, qty, stop, target}}

    def run(self, symbols: List[str], start_date: datetime, end_date: datetime, mock_sentiment=True):
        import yfinance as yf

        print(f"Fetching data for {len(symbols)} symbols...")
        data = {}
        for symbol in symbols:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, multi_level_index=False)
            if not df.empty:
                data[symbol] = df

        # Merge into single timeline
        # Simple Loop through dates
        all_dates = sorted(list(set().union(*[d.index for d in data.values()])))

        print(f"Backtesting over {len(all_dates)} days...")

        for date in all_dates:
            # 1. Update Portfolio Value
            portfolio_value = self.cash
            for sym, qty in self.holdings.items():
                if date in data[sym].index:
                    portfolio_value += qty * data[sym].loc[date]['Close']
                else:
                    # Use last known? Simplified: assume tradeable
                    pass

            self.equity_curve.append({"date": date, "equity": portfolio_value})

            # 2. Check Exits (SL/TP)
            for sym in list(self.holdings.keys()):
                if date not in data[sym].index: continue

                candle = data[sym].loc[date]
                pos = self.positions.get(sym)
                if not pos: continue

                # Check specifics
                curr_price = candle['Close'] # Simplified - could use High/Low
                low = candle['Low']
                high = candle['High']

                exit_price = None
                reason = ""

                # Hit Stop?
                if low <= pos['stop']:
                    exit_price = pos['stop'] # Assuming reliable execution at stop
                    reason = "STOP_LOSS"
                # Hit Target?
                elif high >= pos['target']:
                    exit_price = pos['target']
                    reason = "TAKE_PROFIT"
                # Strategy Exit?
                elif self.strategy.should_exit(sym, candle):
                    exit_price = curr_price
                    reason = "STRATEGY_EXIT"

                if exit_price:
                    self._close_position(sym, exit_price, date, reason)

            # 3. Check Entries
            current_equity = self.equity_curve[-1]['equity']

            for sym in symbols:
                if sym in self.holdings: continue
                if date not in data[sym].index: continue

                candle = data[sym].loc[date]

                # Strategy Signal
                signal = self.strategy.generate_signal(sym, candle, current_equity)

                if signal and signal['action'] == "BUY":
                    entry_price = candle['Close']
                    stop = signal.get('stop_loss', entry_price * 0.95)
                    target = signal.get('take_profit', entry_price * 1.10)

                    # Risk Manager Sizing
                    qty = self.risk_manager.calculate_position_size(current_equity, entry_price)

                    if qty > 0 and self.cash >= qty * entry_price:
                        self._open_position(sym, qty, entry_price, stop, target, date)

    def _open_position(self, ticker, qty, price, stop, target, date):
        cost = qty * price
        self.cash -= cost
        self.holdings[ticker] = qty
        self.positions[ticker] = {'entry': price, 'qty': qty, 'stop': stop, 'target': target}

        self.trade_log.append({
            "ticker": ticker,
            "type": "BUY",
            "date": date,
            "price": price,
            "qty": qty,
            "reason": "SIGNAL"
        })

    def _close_position(self, ticker, price, date, reason):
        qty = self.holdings[ticker]
        proceeds = qty * price
        self.cash += proceeds

        entry_price = self.positions[ticker]['entry']
        pnl = proceeds - (qty * entry_price)

        del self.holdings[ticker]
        del self.positions[ticker]

        self.trade_log.append({
            "ticker": ticker,
            "type": "SELL",
            "date": date,
            "price": price,
            "qty": qty,
            "reason": reason,
            "pnl": pnl
        })

    def calculate_metrics(self):
        df = pd.DataFrame(self.equity_curve)
        if df.empty: return {"Error": "No data"}

        start = df.iloc[0]['equity']
        end = df.iloc[-1]['equity']
        ret = (end - start) / start * 100

        trades = pd.DataFrame(self.trade_log)
        win_rate = 0
        if not trades.empty:
            closed = trades[trades['type'] == 'SELL']
            if not closed.empty:
                wins = closed[closed['pnl'] > 0]
                win_rate = len(wins) / len(closed) * 100

        return {
            "Total Return %": round(ret, 2),
            "Final Equity": round(end, 2),
            "Win Rate %": round(win_rate, 2),
            "Total Trades": len(trades) // 2
        }
