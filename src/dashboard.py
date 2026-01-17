import streamlit as st
import pandas as pd
import os
import sys
import time
from dotenv import load_dotenv
import plotly.express as px
from alpaca.trading.client import TradingClient
import requests

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load Env
load_dotenv()

# Setup Alpaca Client
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
IS_PAPER = True # Assume paper for dashboard or load from config if needed

if API_KEY and SECRET_KEY:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=IS_PAPER)
else:
    trading_client = None

# Page Config
st.set_page_config(
    page_title="AI Trading Bot - Mission Control",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Trading Bot - Mission Control")

# --- Helper Functions ---

def get_alpaca_history():
    """Fetch portfolio history from Alpaca using Direct REST API."""
    if not API_KEY or not SECRET_KEY:
        return pd.DataFrame()
    
    try:
        # Construct URL manually to avoid SDK missing method issues
        base_url = "https://paper-api.alpaca.markets" if IS_PAPER else "https://api.alpaca.markets"
        url = f"{base_url}/v2/account/portfolio/history"
        
        headers = {
            "APCA-API-KEY-ID": API_KEY,
            "APCA-API-SECRET-KEY": SECRET_KEY
        }
        params = {
            "period": "1M",
            "timeframe": "1D",
            "extended_hours": "false"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            st.error(f"Alpaca API Error: {response.text}")
            return pd.DataFrame()
            
        data = response.json()
        
        # Data structure: {'timestamp': [...], 'equity': [...], ...}
        if 'timestamp' in data and 'equity' in data:
            # Timestamp is usually unix seconds
            dates = pd.to_datetime(data['timestamp'], unit='s')
            equity = data['equity']
            
            # Filter out None values in equity (can happen)
            df = pd.DataFrame({'date': dates, 'equity': equity})
            df = df.dropna()
            return df
            
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return pd.DataFrame()

def load_data():
    """Load data (prefer Live Alpaca, fallback to CSV)."""
    equity_df = pd.DataFrame()
    trades_df = pd.DataFrame()
    log_lines = []
    
    # 1. Equity: Try Live first
    if trading_client:
        equity_df = get_alpaca_history()
    
    # Fallback to CSV if empty or failed
    if equity_df.empty and os.path.exists("backtest_results/equity.csv"):
        equity_df = pd.read_csv("backtest_results/equity.csv")
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        
    # 2. Trades: Currently only CSV tracks specific bot trades accurately logic-wise
    # (Alpaca has all trades, filtering by bot ID is harder without specific tagging)
    if os.path.exists("backtest_results/trades.csv"):
        trades_df = pd.read_csv("backtest_results/trades.csv")
        
    # 3. Logs
    log_file = "logs/trading_bot.log"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            log_lines = f.readlines()[-50:]
            log_lines = [line.strip() for line in reversed(log_lines)]
            
    return equity_df, trades_df, log_lines

def get_live_metrics():
    """Fetch live account metrics."""
    if not trading_client:
        return 0, 0, 0
        
    try:
        account = trading_client.get_account()
        total_equity = float(account.equity)
        last_equity = float(account.last_equity)
        buying_power = float(account.buying_power)
        
        day_pnl = total_equity - last_equity
        day_pnl_pct = (day_pnl / last_equity) * 100 if last_equity else 0
        
        return total_equity, day_pnl, buying_power
    except Exception:
        return 0, 0, 0

def get_alpaca_positions():
    """Fetch live positions from Alpaca."""
    if not trading_client:
        return None
        
    try:
        positions = trading_client.get_all_positions()
        pos_data = []
        for p in positions:
            pos_data.append({
                "Symbol": p.symbol,
                "Qty": float(p.qty),
                "Entry Price": float(p.avg_entry_price),
                "Current Price": float(p.current_price),
                "P&L ($)": float(p.unrealized_pl),
                "P&L (%)": float(p.unrealized_plpc) * 100
            })
        return pd.DataFrame(pos_data)
    except Exception as e:
        return str(e)

# --- Real-Time Container Loop ---
dashboard_placeholder = st.empty()

while True:
    # Fetch Data
    equity_df, trades_df, logs = load_data()
    current_equity, day_pnl, buying_power = get_live_metrics()
    positions_df = get_alpaca_positions()
    
    with dashboard_placeholder.container():
        # --- Top Row: KPI Metrics ---
        col1, col2, col3, col4 = st.columns(4)

        total_return_pct = 0.0
        trades_count = len(trades_df) if not trades_df.empty else 0
        win_rate = 0.0

        if not trades_df.empty and 'pnl' in trades_df.columns:
            wins = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (wins / trades_count) * 100 if trades_count > 0 else 0

        # If using Live Alpaca metrics
        if trading_client and current_equity > 0:
            display_metric_1_label = "Current Equity"
            display_metric_1_value = f"${current_equity:,.2f}"
            
            display_metric_2_label = "Day P&L"
            display_metric_2_value = f"${day_pnl:,.2f}"
            
            display_metric_3_label = "Buying Power"
            display_metric_3_value = f"${buying_power:,.2f}"
            
            display_metric_4_label = "Open Pos Count"
            pos_count = len(positions_df) if isinstance(positions_df, pd.DataFrame) else 0
            display_metric_4_value = str(pos_count)

        else:
            # Fallback to CSV Backtest metrics
            if not equity_df.empty:
                start_eq = equity_df['equity'].iloc[0]
                end_eq = equity_df['equity'].iloc[-1]
                total_return_pct = ((end_eq - start_eq) / start_eq) * 100
                
            display_metric_1_label = "Total Return (Sim)"
            display_metric_1_value = f"{total_return_pct:.2f}%"
            
            display_metric_2_label = "Total P&L (Sim)"
            pnl_sim = (end_eq - start_eq) if not equity_df.empty else 0
            display_metric_2_value = f"${pnl_sim:.2f}"
            
            display_metric_3_label = "Total Trades"
            display_metric_3_value = str(trades_count)
            
            display_metric_4_label = "Win Rate"
            display_metric_4_value = f"{win_rate:.1f}%"

        with col1:
            st.metric(display_metric_1_label, display_metric_1_value)
        with col2:
            st.metric(display_metric_2_label, display_metric_2_value, delta=day_pnl if trading_client else None)
        with col3:
            st.metric(display_metric_3_label, display_metric_3_value)
        with col4:
            st.metric(display_metric_4_label, display_metric_4_value)

        # --- Row 2: Equity Curve ---
        st.subheader("📈 Equity Curve")
        if not equity_df.empty:
            fig = px.line(equity_df, x='date', y='equity', title='Portfolio Value Over Time')
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
        else:
            st.info("No equity data available yet.")

        # --- Row 3: Active Positions (Live) ---
        st.subheader("💼 Active Positions (Live)")
        if isinstance(positions_df, pd.DataFrame) and not positions_df.empty:
            st.dataframe(positions_df.style.format({
                "Qty": "{:.2f}",
                "Entry Price": "${:.2f}",
                "Current Price": "${:.2f}",
                "P&L ($)": "${:.2f}",
                "P&L (%)": "{:.2f}%"
            }), use_container_width=True)
        elif isinstance(positions_df, str): # Error message
            st.error(f"Could not fetch positions: {positions_df}")
        elif positions_df is None:
            st.warning("Alpaca API Keys not found. Cannot fetch live positions.")
        else:
            st.info("No active positions.")

        # --- Row 4: Recent Trades (History) ---
        st.subheader("📜 Recent Trades (History)")
        if not trades_df.empty:
            trades_df_sorted = trades_df.sort_index(ascending=False).head(10)
            st.dataframe(trades_df_sorted, use_container_width=True)
        else:
            st.info("No trades executed yet.")

        # --- Row 5: Live Logs ---
        st.subheader("📝 Live Logs")
        if logs:
            log_text = "\n".join(logs)
            st.text_area("Latest Logs (Last 50 lines)", log_text, height=300, key=f"logs_{time.time()}")
        else:
            st.info("No logs found.")
            
    # Sleep to prevent high CPU usage, update every 2 seconds
    time.sleep(2)
