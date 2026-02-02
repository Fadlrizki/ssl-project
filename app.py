import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
from datetime import datetime, timedelta

# Import modules baru
from engine import build_probability_table_from_ticker, backtest
from engine_v2 import process_stock, fetch_data, add_indicators
from risk_manager import RiskManager, PositionSizer
# from performance_tracker import PerformanceTracker, TradeRecord
from utils import data_utils, cache_manager, date_utils, format_utils, validation_utils

# ======================================================
# CONFIG
# ======================================================

CACHE_VERSION = "v4"  # Updated version
CACHE_SCREENING = f"screening_cache_{CACHE_VERSION}.pkl"
TRIGGER_CACHE = f"trigger_cache_{CACHE_VERSION}.pkl"
PROB_CACHE = f"prob_cache_{CACHE_VERSION}.pkl"   
BACKTEST_CACHE = f"backtest_cache_{CACHE_VERSION}.pkl"   

TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")

EXCEL_FILE = "daftar_saham.xlsx"
KODE_COLUMN = "Kode"
MAX_WORKERS = 4

REQUIRED_COLS = {
    "Kode", "MajorTrend", "MinorPhase", "SetupState",
    "FinalDecision", "RSI", "VOL_BEHAVIOR"
}

# # Initialize session state for performance tracker
# if "performance_tracker" not in st.session_state:
#     st.session_state.performance_tracker = PerformanceTracker()

# ======================================================
# PAGE
# ======================================================
st.set_page_config(layout="wide")
st.title("📊 IDX Price Action Screener V3")
st.caption("Daily trend • Minor phase • Volume behavior • Risk Management")

# ======================================================
# UPDATED HELPERS (using new modules)
# ======================================================

def color_decision(val):
    """Color code based on decision"""
    if val == "ENTRY_READY":
        return "background-color: #d4edda; color: #155724;"
    elif val == "SETUP_PENDING":
        return "background-color: #fff3cd; color: #856404;"
    elif val == "WAIT":
        return "background-color: #f8d7da; color: #721c24;"
    return ""

def clear_cache():
    """Clear all cache files"""
    cache_files = [
        CACHE_SCREENING,
        TRIGGER_CACHE,
        PROB_CACHE,
        BACKTEST_CACHE
    ]
    
    cleared = 0
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                cleared += 1
            except Exception as e:
                st.error(f"Failed to remove {cache_file}: {e}")
    
    # Also clear cache_manager cache
    cache_manager.clear_old_cache(max_age_days=0)
    
    st.success(f"✅ {cleared} cache files cleared")

def load_cache_safe(path):
    """Safe cache loading with validation"""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
        if df is None or df.empty:
            return pd.DataFrame()
        if not REQUIRED_COLS.issubset(df.columns):
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error loading cache {path}: {e}")
        return pd.DataFrame()

def save_cache(df, path):
    """Save dataframe to cache"""
    try:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        st.error(f"Error saving cache {path}: {e}")

def run_backtest_cached(kode):
    """Run backtest with caching"""
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    key = f"{kode}_{today}"
    
    # Load cache
    cache = load_backtest_cache()
    
    # Check if cached result exists and is from today
    if key in cache:
        st.info(f"Using cached backtest result for {kode}")
        return cache[key]
    
    # Run new backtest
    with st.spinner(f"Running backtest for {kode}..."):
        result = backtest(f"{kode}.JK", mode="decision")
    
    # Save to cache
    cache[key] = result
    save_backtest_cache(cache)
    
    return result

def load_backtest_cache():
    """Load backtest cache"""
    if os.path.exists(BACKTEST_CACHE):
        try:
            return pickle.load(open(BACKTEST_CACHE, "rb"))
        except Exception:
            return {}
    return {}

def save_backtest_cache(cache):
    """Save backtest cache"""
    with open(BACKTEST_CACHE, "wb") as f:
        pickle.dump(cache, f)

def render_technical_chart(df, kode, suffix="detail"):
    """
    Fixed technical chart with proper Plotly properties
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.15, 0.15, 0.25],
        subplot_titles=(
            f"{kode} - Price & EMAs",
            "RSI (14)",
            "MACD",
            "Volume & Volume MA20"
        )
    )

    # =========================
    # 1. PRICE CHART
    # =========================
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # EMA Lines
    ema_configs = [
        ("EMA13", "blue", 1.2),
        ("EMA21", "orange", 1.5),
        ("EMA50", "red", 2.0)
    ]
    
    for ema_name, color, width in ema_configs:
        if ema_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ema_name],
                    mode="lines",
                    name=ema_name,
                    line=dict(color=color, width=width)
                ),
                row=1, col=1
            )

    # =========================
    # 2. RSI CHART
    # =========================
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(width=1.5, color="purple")
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1, row=2, col=1)
        fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="gray", opacity=0.05, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1, row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)

    # =========================
    # 3. MACD CHART
    # =========================
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        # MACD Line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="blue", width=1.5)
            ),
            row=3, col=1
        )
        
        # Signal Line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                mode="lines",
                name="Signal",
                line=dict(color="orange", width=1.5)
            ),
            row=3, col=1
        )
        
        # MACD Histogram
        colors = ['rgba(0, 128, 0, 0.8)' if val >= 0 else 'rgba(255, 0, 0, 0.8)' for val in df["MACD_Hist"]]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Hist"],
                name="Histogram",
                marker_color=colors,
                opacity=0.7,
                width=0.8
            ),
            row=3, col=1
        )
        
        fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)

    # =========================
    # 4. VOLUME CHART
    # =========================
    if "Volume" in df.columns:
        # Hitung volume dalam juta atau ribu
        volume_max = df["Volume"].max()
        if volume_max >= 1_000_000_000:
            volume_divisor = 1_000_000_000
            volume_unit = "B"
        elif volume_max >= 1_000_000:
            volume_divisor = 1_000_000
            volume_unit = "M"
        else:
            volume_divisor = 1_000
            volume_unit = "K"
        
        # Warna volume
        colors = []
        for i in range(len(df)):
            if i == 0:
                colors.append('gray')
            else:
                colors.append('green' if df["Close"].iloc[i] > df["Close"].iloc[i-1] else 'red')
        
        # Volume bars
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"] / volume_divisor,
                name=f"Volume ({volume_unit})",
                marker_color=colors,
                opacity=0.8,
                width=0.7,
                marker_line_width=0
            ),
            row=4, col=1
        )

        # Volume MA20
        if "VOL_MA20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["VOL_MA20"] / volume_divisor,
                    mode="lines",
                    name=f"Vol MA20 ({volume_unit})",
                    line=dict(color="orange", width=2)
                ),
                row=4, col=1
            )

    # =========================
    # LAYOUT & STYLING - FIXED
    # =========================
    fig.update_layout(
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        # xaxis4_rangeslider_visible=True,
        margin=dict(t=60, b=40, l=60, r=60),
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Update axis titles - FIXED: gunakan dict title
    fig.update_yaxes(
        title=dict(text="Price (IDR)", font=dict(size=12)),
        row=1, col=1,
        tickformat=","
    )
    
    fig.update_yaxes(
        title=dict(text="RSI", font=dict(size=12)),
        row=2, col=1,
        range=[0, 100]
    )
    
    fig.update_yaxes(
        title=dict(text="MACD", font=dict(size=12)),
        row=3, col=1
    )
    
    # Volume y-axis
    if "Volume" in df.columns:
        volume_title = f"Volume ({volume_unit})"
        fig.update_yaxes(
            title=dict(text=volume_title, font=dict(size=12)),
            row=4, col=1,
            tickformat=","
        )
    
    # X-axis for volume chart
    fig.update_xaxes(
        title=dict(text="Date", font=dict(size=12)),
        row=4, col=1
    )
    
    # Handle secondary axis for volume ratio dengan cara yang benar
    if "VOL_RATIO" in df.columns:
        # Buat trace terpisah untuk volume ratio
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["VOL_RATIO"],
                mode="lines",
                name="Volume Ratio",
                line=dict(color="purple", width=1, dash="dash"),
                yaxis="y5"
            )
        )
        
        # Update layout untuk secondary axis
        fig.update_layout(
            yaxis5=dict(
                title="Volume Ratio",
                overlaying="y4",
                side="right",
                range=[0, max(3, df["VOL_RATIO"].max() * 1.1)],
                showgrid=False,
                tickfont=dict(color="purple")
            )
        )
        
        # Tambah horizontal lines di axis yang benar
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", 
                     line_width=1, row=4, col=1, yref="y5")
        fig.add_hline(y=1.5, line_dash="dash", line_color="orange", 
                     line_width=1, row=4, col=1, yref="y5")
        fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                     line_width=1, row=4, col=1, yref="y5")

    # Volume statistics annotation
    if "Volume" in df.columns:
        avg_volume = df["Volume"].mean() / volume_divisor
        current_volume = df["Volume"].iloc[-1] / volume_divisor
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"Current: {current_volume:,.1f}{volume_unit} ({volume_ratio:.1f}x avg)",
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )

    st.plotly_chart(fig, use_container_width=True, key=f"chart_{kode}_{suffix}")
    
    # Volume statistics table
    if "Volume" in df.columns and "VOL_MA20" in df.columns:
        with st.expander("📊 Volume Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Today's Volume", 
                         f"{df['Volume'].iloc[-1]/1e6:.1f}M",
                         f"{(df['Volume'].iloc[-1]/df['VOL_MA20'].iloc[-1]-1)*100:.1f}%")
            
            with col2:
                st.metric("20-day Avg Volume", 
                         f"{df['VOL_MA20'].iloc[-1]/1e6:.1f}M",
                         "")
            
            with col3:
                volume_ratio = df['Volume'].iloc[-1] / df['VOL_MA20'].iloc[-1] if df['VOL_MA20'].iloc[-1] > 0 else 0
                st.metric("Volume Ratio", 
                         f"{volume_ratio:.2f}",
                         "High" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.8 else "Low")
            
            with col4:
                volume_trend = "Rising" if df['Volume'].iloc[-1] > df['Volume'].iloc[-5:-1].mean() else "Falling"
                st.metric("Volume Trend", volume_trend, "")

def retry_single_stock(kode):
    """Retry processing a single stock"""
    # Remove from cache if exists
    if os.path.exists(CACHE_SCREENING):
        df = load_cache_safe(CACHE_SCREENING)
        df = df[df["Kode"] != kode]
        save_cache(df, CACHE_SCREENING)
    
    # Process stock with force refresh
    return process_stock(kode, use_cache=False)

# ======================================================
# NEW: RISK MANAGEMENT SECTION
# ======================================================
def render_risk_management_calculator(kode, current_price):
    """Render risk management calculator"""
    st.subheader("🎯 Risk Management Calculator")
    
    with st.expander("Configure Risk Parameters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            capital = st.number_input(
                "Trading Capital (IDR)",
                min_value=1000000,
                value=10000000,
                step=1000000,
                key=f"capital_{kode}"
            )
        
        with col2:
            risk_per_trade = st.slider(
                "Risk per Trade (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key=f"risk_{kode}"
            )
        
        with col3:
            stop_loss_pct = st.slider(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                key=f"sl_{kode}"
            )
        
        with col4:
            risk_reward_ratio = st.slider(
                "Risk:Reward Ratio",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key=f"rr_{kode}"
            )
    
    if st.button("Calculate Position", key=f"calc_pos_{kode}"):
        try:
            # Calculate position size
            position_size = RiskManager.calculate_position_size(
                capital, risk_per_trade, stop_loss_pct
            )
            
            # Calculate number of shares
            shares = int(position_size / current_price)
            investment = shares * current_price
            
            # Calculate stop loss price
            stop_loss_price = RiskManager.calculate_stop_loss_price(
                current_price, stop_loss_pct, "long"
            )
            
            # Calculate target price
            target_price = RiskManager.calculate_target_price(
                current_price, stop_loss_pct, risk_reward_ratio, "long"
            )
            
            # Calculate risk metrics
            potential_loss = investment - (shares * stop_loss_price)
            potential_profit = (shares * target_price) - investment
            rr_actual = RiskManager.calculate_risk_reward_ratio(
                current_price, stop_loss_price, target_price, "long"
            )
            
            # Display results
            st.success("**Position Calculation Results:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Position Size", f"IDR {position_size:,.0f}")
                st.metric("Shares", f"{shares:,}")
                st.metric("Investment", f"IDR {investment:,.0f}")
            
            with col2:
                st.metric("Stop Loss", f"IDR {stop_loss_price:,.0f}")
                st.metric("Target", f"IDR {target_price:,.0f}")
                st.metric("Stop Loss %", f"{stop_loss_pct}%")
            
            with col3:
                st.metric("Potential Loss", f"IDR {potential_loss:,.0f}")
                st.metric("Potential Profit", f"IDR {potential_profit:,.0f}")
                st.metric("Risk:Reward", f"1:{rr_actual:.1f}")
            
            # Risk summary
            st.info(f"""
            **Risk Summary for {kode}:**
            - You're risking **IDR {potential_loss:,.0f}** ({risk_per_trade}% of capital)
            - Potential reward: **IDR {potential_profit:,.0f}**
            - Stop loss hit: {((current_price - stop_loss_price) / current_price * 100):.1f}% drop needed
            - Target hit: {((target_price - current_price) / current_price * 100):.1f}% gain needed
            """)
            
        except Exception as e:
            st.error(f"Error in risk calculation: {e}")

# ======================================================
# NEW: PERFORMANCE TRACKING SECTION
# ======================================================
# def render_performance_tracking():
#     """Render performance tracking dashboard"""
#     st.subheader("📊 Performance Tracking")
    
#     tracker = st.session_state.performance_tracker
#     summary = tracker.get_summary()
    
#     if summary:
#         # Key metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Trades", summary['total_trades'])
#             st.metric("Win Rate", f"{summary['win_rate']:.1f}%")
        
#         with col2:
#             st.metric("Total P&L", format_utils.format_currency(summary['total_net_pnl']))
#             st.metric("Total Return", f"{summary['total_return_pct']:.1f}%")
        
#         with col3:
#             st.metric("Profit Factor", f"{summary['profit_factor']:.2f}")
#             st.metric("Max Drawdown", f"{summary['max_drawdown']:.1f}%")
        
#         with col4:
#             st.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
#             st.metric("Avg Trade", format_utils.format_currency(summary['avg_win']))
        
#         # Equity Curve Chart
#         if tracker.equity_curve and len(tracker.equity_curve) > 1:
#             fig_eq = go.Figure()
#             fig_eq.add_trace(go.Scatter(
#                 x=list(range(len(tracker.equity_curve))),
#                 y=tracker.equity_curve,
#                 mode='lines',
#                 name='Equity Curve',
#                 line=dict(color='green', width=2),
#                 fill='tozeroy',
#                 fillcolor='rgba(0, 255, 0, 0.1)'
#             ))
            
#             # Add drawdown area
#             equity_series = pd.Series(tracker.equity_curve)
#             rolling_max = equity_series.expanding().max()
#             drawdown = (equity_series - rolling_max) / rolling_max * 100
            
#             fig_eq.add_trace(go.Scatter(
#                 x=list(range(len(drawdown))),
#                 y=drawdown,
#                 mode='lines',
#                 name='Drawdown %',
#                 line=dict(color='red', width=1),
#                 yaxis='y2',
#                 fill='tozeroy',
#                 fillcolor='rgba(255, 0, 0, 0.1)'
#             ))
            
#             fig_eq.update_layout(
#                 title="Equity Curve & Drawdown",
#                 xaxis_title="Trade Number",
#                 yaxis_title="Capital (IDR)",
#                 yaxis2=dict(
#                     title="Drawdown %",
#                     titlefont=dict(color="red"),
#                     tickfont=dict(color="red"),
#                     overlaying="y",
#                     side="right"
#                 ),
#                 height=400,
#                 showlegend=True
#             )
            
#             st.plotly_chart(fig_eq, use_container_width=True)
        
#         # Recent Trades Table
#         st.subheader("Recent Trades")
#         trades_df = tracker.get_trades_df()
#         if not trades_df.empty:
#             # Format the dataframe for display
#             display_cols = ['symbol', 'entry_time', 'exit_time', 'entry_price', 
#                           'exit_price', 'quantity', 'net_pnl', 'net_pnl_pct', 'win']
            
#             if all(col in trades_df.columns for col in display_cols):
#                 display_df = trades_df[display_cols].copy()
#                 display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
#                 display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
#                 display_df['net_pnl'] = display_df['net_pnl'].apply(lambda x: format_utils.format_currency(x))
#                 display_df['net_pnl_pct'] = display_df['net_pnl_pct'].apply(lambda x: f"{x:.2f}%")
#                 display_df['win'] = display_df['win'].apply(lambda x: '✅' if x else '❌')
                
#                 st.dataframe(display_df.sort_values('exit_time', ascending=False), 
#                            use_container_width=True)
    
#     # Manual Trade Entry
#     with st.expander("➕ Add Manual Trade"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             symbol_trade = st.text_input("Symbol", key="trade_symbol")
#             entry_price = st.number_input("Entry Price", min_value=0.0, key="entry_price")
#             exit_price = st.number_input("Exit Price", min_value=0.0, key="exit_price")
        
#         with col2:
#             quantity = st.number_input("Quantity", min_value=1, key="trade_qty")
#             direction = st.selectbox("Direction", ["long", "short"], key="trade_dir")
        
#         col3, col4 = st.columns(2)
#         with col3:
#             entry_date = st.date_input("Entry Date", key="entry_date")
#             entry_time = st.time_input("Entry Time", key="entry_time")
        
#         with col4:
#             exit_date = st.date_input("Exit Date", key="exit_date")
#             exit_time = st.time_input("Exit Time", key="exit_time")
        
#         if st.button("Add Trade", key="add_trade"):
#             try:
#                 # Combine date and time
#                 entry_datetime = datetime.combine(entry_date, entry_time)
#                 exit_datetime = datetime.combine(exit_date, exit_time)
                
#                 trade = tracker.add_trade(
#                     symbol=symbol_trade,
#                     entry_price=entry_price,
#                     exit_price=exit_price,
#                     entry_time=entry_datetime,
#                     exit_time=exit_datetime,
#                     quantity=quantity,
#                     direction=direction
#                 )
                
#                 st.success(f"✅ Trade added! P&L: {format_utils.format_currency(trade['net_pnl'])} ({trade['net_pnl_pct']:.2f}%)")
#                 st.rerun()
                
#             except Exception as e:
#                 st.error(f"Error adding trade: {e}")
    
#     # Export options
#     if st.button("📊 Export Performance Report"):
#         tracker.export_report(format='excel')
#         st.success("Performance report exported to performance_data/ folder")

# ======================================================
# HELPER FUNCTIONS FOR BROKER SUMMARY
# ======================================================
def find_latest_cache(trade_date, max_back=7):
    """
    Cari file cache mundur dari trade_date sampai max_back hari.
    """
    dt = datetime.strptime(trade_date, "%Y-%m-%d")
    for i in range(max_back+1):
        check_date = (dt - timedelta(days=i)).strftime("%Y-%m-%d")
        path = f"cache/trigger_result_{check_date}.pkl"
        if os.path.exists(path):
            return path, check_date
    return None, None

def get_trade_date(today=None):
    if today is None:
        today = datetime.today()
    # mundur ke hari kerja terakhir kalau weekend
    while today.weekday() >= 5:  # 5 = Sabtu, 6 = Minggu
        today -= timedelta(days=1)
    return today.strftime("%Y-%m-%d")

def save_trigger_cache(df, trade_date=None):
    if trade_date is None:
        trade_date = get_trade_date()
    # pastikan folder cache ada
    os.makedirs("cache", exist_ok=True)
    path = f"cache/trigger_result_{trade_date}.pkl"
    with open(path, "wb") as f:
        pickle.dump(df, f)
    st.info(f"✅ Cache saved → {path}")

def load_broker_summary(trade_date, max_back=7):
    """Load broker summary CSV file"""
    dt = datetime.strptime(trade_date, "%Y-%m-%d")
    for i in range(max_back + 1):
        check_date = (dt - timedelta(days=i)).strftime("%Y-%m-%d")
        path = f"broksum/broker_summary-{check_date}.csv"
        
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    # Cek kolom yang ada
                    st.info(f"Loaded broker summary from {check_date}")
                    st.write(f"File: {path}")
                    st.write(f"Columns: {df.columns.tolist()}")
                    st.write(f"Rows: {len(df)}")
                    return df, check_date
            except Exception as e:
                st.error(f"Error loading {path}: {e}")
                continue
    
    st.warning(f"Tidak ditemukan broker summary file untuk tanggal {trade_date} (mundur {max_back} hari)")
    return None, None

def load_trigger_cache_pickle(trade_date, max_back=7):
    dt = datetime.strptime(trade_date, "%Y-%m-%d")
    for i in range(max_back + 1):
        check_date = (dt - timedelta(days=i)).strftime("%Y-%m-%d")
        path = f"cache/trigger_result_{check_date}.pkl"
        if os.path.exists(path):
            df = pickle.load(open(path, "rb"))
            return df, check_date
    return None, None

def show_status(name, trade_date, used_date, df):
    """
    Tampilkan status data berdasarkan tanggal yang dipakai.
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.info(f"❌ {name} tidak tersedia untuk hari ini maupun fallback")
        return False
    elif used_date != trade_date:
        st.info(f"ℹ️ {name} {trade_date} belum tersedia, pakai data {used_date}")
        return True
    else:
        st.success(f"✅ {name} {trade_date} sudah update")
        return True

def load_trigger_cache():
    if os.path.exists(TRIGGER_CACHE):
        try:
            return pickle.load(open(TRIGGER_CACHE, "rb"))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ======================================================
# LOAD SAHAM
# ======================================================
saham_df = pd.read_excel(EXCEL_FILE)
codes = saham_df[KODE_COLUMN].dropna().unique().tolist()

cached_df = load_cache_safe(CACHE_SCREENING)

# ======================================================
# SIDEBAR CONFIGURATION
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Cache management
    st.subheader("Cache Management")
    if st.button("🗑️ Clear All Cache", use_container_width=True):
        clear_cache()
        st.rerun()
    
    # Performance tracking
    # st.subheader("Performance Tracking")
    # tracker = st.session_state.performance_tracker
    # st.metric("Current Capital", format_utils.format_currency(tracker.current_capital))
    # st.metric("Total Trades", tracker.get_summary().get('total_trades', 0))
    
    # if st.button("Reset Performance", use_container_width=True):
    #     tracker.reset()
    #     st.success("Performance tracker reset")
    #     st.rerun()
    
    # Risk parameters
    st.subheader("Default Risk Parameters")
    default_capital = st.number_input("Default Capital (IDR)", 
                                     value=10000000,
                                     min_value=1000000,
                                     step=1000000)
    default_risk = st.slider("Default Risk %", 0.5, 5.0, 2.0, 0.5)
    
    # System info
    st.subheader("System Info")
    st.text(f"Version: {CACHE_VERSION}")
    st.text(f"Today: {TODAY}")
    st.text(f"Stocks in list: {len(codes)}")

# ======================================================
# MAIN SCREENING SECTION
# ======================================================
st.header("🚀 Stock Screening")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🗑️ Clear Screening Cache", use_container_width=True):
        if os.path.exists(CACHE_SCREENING):
            os.remove(CACHE_SCREENING)
        st.session_state["scan"] = pd.DataFrame()
        st.success("Screening cache cleared")
        st.rerun()

with col2:
    if st.button("🚀 Run Full Screening", use_container_width=True, type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_stock, k, use_cache=True): k for k in codes}
            done = 0
            total = len(codes)
            
            for f in as_completed(futures):
                kode = futures[f]
                try:
                    r = f.result()
                    if r and "Kode" in r and "Price" in r:
                        r["ProcessTime"] = pd.Timestamp.now()
                        results.append(r)
                except Exception as e:
                    st.error(f"Error {kode}: {e}")
                
                done += 1
                progress.progress(done / total)
                status.text(f"Processed {done}/{total} saham")
        
        df_new = pd.DataFrame(results)
        
        # Combine with existing cache
        if not cached_df.empty:
            df_scan = pd.concat([cached_df, df_new], ignore_index=True)
        else:
            df_scan = df_new.copy()
        
        df_scan = df_scan.drop_duplicates(subset=["Kode"], keep="last").reset_index(drop=True)
        
        save_cache(df_scan, CACHE_SCREENING)
        st.session_state["scan"] = df_scan
        st.success(f"✅ Screening selesai: {len(df_scan)} saham valid")
        st.rerun()

# ======================================================
# GUARD - Check if screening results exist
# ======================================================
if "scan" not in st.session_state or st.session_state["scan"].empty:
    if cached_df.empty:
        st.warning("Belum ada hasil screening. Klik 'Run Full Screening' untuk memulai.")
        st.stop()
    else:
        st.session_state["scan"] = cached_df

df = st.session_state["scan"].copy()

# ======================================================
# FILTER SECTION
# ======================================================
st.markdown("### 🔎 Filter Results")

# Quick filters
col1, col2, col3, col4 = st.columns(4)
with col1:
    show_only_strong = st.checkbox("Show STRONG Only", value=True)
    if show_only_strong:
        df = df[df["MajorTrend"] == "STRONG"]

with col2:
    show_entry_ready = st.checkbox("Show ENTRY_READY Only")
    if show_entry_ready:
        df = df[df["FinalDecision"] == "ENTRY_READY"]

with col3:
    min_confidence = st.slider("Min Confidence %", 0, 100, 50)
    df = df[df["MinorConfidence%"] >= min_confidence]

with col4:
    max_rsi = st.slider("Max RSI", 30, 100, 80)
    df = df[df["RSI"] <= max_rsi]

# Advanced filters
with st.expander("Advanced Filters"):
    cols = st.columns(3)
    
    with cols[0]:
        major_filter = st.multiselect("Major Trend", df["MajorTrend"].unique())
        if major_filter:
            df = df[df["MajorTrend"].isin(major_filter)]
    
    with cols[1]:
        minor_filter = st.multiselect("Minor Phase", df["MinorPhase"].unique())
        if minor_filter:
            df = df[df["MinorPhase"].isin(minor_filter)]
    
    with cols[2]:
        decision_filter = st.multiselect("Final Decision", df["FinalDecision"].unique())
        if decision_filter:
            df = df[df["FinalDecision"].isin(decision_filter)]
    
    cols2 = st.columns(3)
    with cols2[0]:
        kode_filter = st.text_input("Search Kode").upper()
        if kode_filter:
            df = df[df["Kode"].str.contains(kode_filter)]
    
    with cols2[1]:
        vol_filter = st.multiselect("Volume Behavior", df["VOL_BEHAVIOR"].unique())
        if vol_filter:
            df = df[df["VOL_BEHAVIOR"].isin(vol_filter)]
    
    with cols2[2]:
        candle_filter = st.multiselect("Candle Type", df["Latest_Candle"].unique())
        if candle_filter:
            df = df[df["Latest_Candle"].isin(candle_filter)]

# ======================================================
# RESULTS TABLE
# ======================================================
st.subheader(f"📋 Screening Results ({len(df)} saham)")

# Format dataframe for display
display_df = df.copy()
if not display_df.empty:
    # Format numeric columns
    display_df["Price"] = display_df["Price"].apply(lambda x: format_utils.format_currency(x))
    display_df["Volume"] = display_df["Volume"].apply(lambda x: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    display_df["RSI"] = display_df["RSI"].apply(lambda x: f"{x:.1f}")
    
    # Color code based on decision
    def color_decision(val):
        if val == "ENTRY_READY":
            return "background-color: #d4edda; color: #155724;"
        elif val == "SETUP_PENDING":
            return "background-color: #fff3cd; color: #856404;"
        elif val == "WAIT":
            return "background-color: #f8d7da; color: #721c24;"
        return ""
    
    # Select columns to display
    display_cols = ["Kode", "Price", "PriceChange%", "MajorTrend", "MinorPhase", 
                   "MinorConfidence%", "RSI", "VOL_BEHAVIOR","Volume" ,"Latest_Candle", "FinalDecision"]
    
    display_df = display_df[display_cols]

# Display interactive dataframe
try:
    event = st.dataframe(
        display_df.style.applymap(color_decision, subset=['FinalDecision']),
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun"
    )
except Exception as e:
    st.error(f"Error dalam styling DataFrame: {e}")
    # st.write("Kolom yang tersedia:", display_df.columns.tolist())
    
    # Tampilkan DataFrame tanpa styling
    event = st.dataframe(
        display_df,
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun"
    )

# ======================================================
# DETAIL VIEW FOR SELECTED STOCK
# ======================================================
if event.selection.rows:
    selected_idx = event.selection.rows[0]
    row = df.iloc[selected_idx]
    kode = row["Kode"]
    
    st.divider()
    st.header(f"📊 Detailed Analysis: {kode}")
    
    # Create tabs for different sections
    tab1, tab2, tab3= st.tabs(["📈 Chart & Metrics", "🎯 Risk Management", "🤖 Probability Analysis"])
    
    with tab1:
        # Chart and basic metrics
        col_retry1, col_retry2 = st.columns([1, 3])
        
        with col_retry1:
            if st.button(f"🔁 Refresh {kode}", key=f"btn_retry_single_{kode}"):
                with st.spinner(f"Refreshing {kode}..."):
                    r = retry_single_stock(kode)
                    
                    if r:
                        df = pd.concat(
                            [df[df["Kode"] != kode], pd.DataFrame([r])],
                            ignore_index=True
                        )
                        save_cache(df, CACHE_SCREENING)
                        st.session_state["scan"] = df
                        st.success("Refresh berhasil")
                        st.rerun()
                    else:
                        st.error("Refresh gagal")
        
        # Chart
        st.subheader(f"📈 Technical Chart - {kode}")
        
        ticker = f"{kode}.JK"
        try:
            df_daily = fetch_data(
                ticker,
                interval="1d",
                period="12mo",
                force_refresh=False
            )
            
            if df_daily is None or df_daily.empty:
                st.warning("Data chart tidak tersedia")
            else:
                df_daily = add_indicators(df_daily)
                render_technical_chart(df_daily, kode, suffix="detail")
                
        except Exception as e:
            st.error(f"Gagal render chart: {e}")
        
        # Technical Summary
        st.subheader("📊 Technical Summary")
        
        tech_cols = st.columns(3)
        
        with tech_cols[0]:
            st.metric("Price", format_utils.format_currency(row["Price"]))
            st.metric("Price Change", f"{row['PriceChange%']}%")
            st.metric("Volume", f"{row['Volume']/1e6:.2f}M")
            st.metric("Volume Ratio", f"{row['VOL_RATIO']:.2f}")
        
        with tech_cols[1]:
            st.metric("RSI", f"{row['RSI']:.1f}")
            st.metric("Stoch %K", f"{row['Stoch_K']:.1f}")
            st.metric("Distance to SMA50", f"{row['Dist_to_SMA50']:.1f}%" if not pd.isna(row['Dist_to_SMA50']) else "N/A")
            st.metric("Major Trend", row["MajorTrend"])
        
        with tech_cols[2]:
            st.metric("Minor Phase", row["MinorPhase"])
            st.metric("Confidence", f"{row['MinorConfidence']} ({row['MinorConfidence%']}%)")
            st.metric("Setup State", row["SetupState"])
            st.metric("Final Decision", row["FinalDecision"])
        
        # Gap Analysis
        st.subheader("📏 Gap Analysis")
        gap_cols = st.columns(3)
        with gap_cols[0]:
            st.metric("Gap to EMA13", f"{row['Gap_EMA13%']}%")
        with gap_cols[1]:
            st.metric("Gap to EMA21", f"{row['Gap_EMA21%']}%")
        with gap_cols[2]:
            st.metric("Gap to EMA50", f"{row['Gap_EMA50%']}%")
    
    with tab2:
        # Risk Management Calculator
        render_risk_management_calculator(kode, row["Price"])
    
    with tab3:
        # Probability Analysis and Backtest
        st.subheader("🤖 Probability Analysis")
        
        if st.button("Run Backtest Analysis", key=f"run_analysis_{kode}"):
            with st.spinner(f"Running analysis for {kode}..."):
                st.session_state["backtest_result"] = backtest(
                    f"{kode}.JK",
                    mode="decision"
                )
                st.session_state["prob_table"] = build_probability_table_from_ticker(
                    f"{kode}.JK"
                )
        
        # Display backtest results if available
        if "backtest_result" in st.session_state:
            result = st.session_state["backtest_result"]
            
            if result:
                bias = result.get("Bias")
                
                if bias in ["HIJAU", "MERAH"]:
                    st.subheader("🔮 Tomorrow's Bias Prediction")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Bias", bias, 
                              delta="Bullish" if bias == "HIJAU" else "Bearish",
                              delta_color="normal" if bias == "HIJAU" else "inverse")
                    col2.metric("Probability Green", f"{result.get('ProbHijau', 0)}%")
                    col3.metric("Probability Red", f"{result.get('ProbMerah', 0)}%")
                    
                    st.caption(
                        f"Sample size: {result.get('Sample', '-')} historical matches | "
                        f"Confidence: {result.get('Confidence', '-')}"
                    )
                    
                    if "DecisionContext" in result:
                        ctx = result["DecisionContext"]
                        st.subheader("🧠 Decision Context")
                        
                        ctx_data = {
                            "Parameter": [
                                "Major Trend", "Minor Phase", "RSI Bucket", 
                                "Volume Behavior", "Latest Candle", "Avg Volume Ratio", "Match Type"
                            ],
                            "Value": [
                                ctx.get("MajorTrend", "-"),
                                ctx.get("MinorPhase", "-"),
                                ctx.get("RSI_BUCKET", "-"),
                                ctx.get("VOL_BEHAVIOR", "-"),
                                ctx.get("latest_candle", "-"),
                                ctx.get("AvgVolRatio", "-"),
                                ctx.get("MatchType", "-")
                            ]
                        }
                        
                        st.dataframe(pd.DataFrame(ctx_data), use_container_width=True)
                
                elif bias == "NO_MATCH":
                    st.warning("⚠️ No matching historical conditions found")
                elif bias == "NO_MODEL":
                    st.warning("⚠️ Probability model not yet formed")
                elif bias == "NO_SETUP":
                    st.info("ℹ️ No valid setup detected")
        
        # Display probability table if available
        if "prob_table" in st.session_state:
            prob_table = st.session_state["prob_table"]
            
            if prob_table is not None and not prob_table.empty:
                st.subheader("📊 Probability Table (180 Days)")
                
                # Sort by highest green probability
                display_prob = prob_table.sort_values("%Hijau", ascending=False).head(10)
                
                # Color code based on probability
                def color_prob(val):
                    if val >= 70:
                        return "background-color: #d4edda; color: #155724;"
                    elif val >= 60:
                        return "background-color: #fff3cd; color: #856404;"
                    return ""
                
                st.dataframe(
                    display_prob.style.applymap(color_prob, subset=['%Hijau', '%Merah']),
                    use_container_width=True
                )
            else:
                st.info("No probability table available for this stock")
    
    # with tab4:
    #     # Performance tracking for this stock
    #     st.subheader(f"📊 Performance Tracking - {kode}")
        
    #     tracker = st.session_state.performance_tracker
    #     trades_df = tracker.get_trades_df()
        
    #     if not trades_df.empty and kode in trades_df['symbol'].values:
    #         stock_trades = trades_df[trades_df['symbol'] == kode]
            
    #         # Stock-specific metrics
    #         col1, col2, col3, col4 = st.columns(4)
            
    #         with col1:
    #             total_trades = len(stock_trades)
    #             st.metric("Total Trades", total_trades)
            
    #         with col2:
    #             win_rate = (stock_trades['win'].sum() / total_trades * 100) if total_trades > 0 else 0
    #             st.metric("Win Rate", f"{win_rate:.1f}%")
            
    #         with col3:
    #             total_pnl = stock_trades['net_pnl'].sum()
    #             st.metric("Total P&L", format_utils.format_currency(total_pnl))
            
    #         with col4:
    #             avg_pnl = stock_trades['net_pnl'].mean() if total_trades > 0 else 0
    #             st.metric("Avg P&L", format_utils.format_currency(avg_pnl))
            
    #         # Display trades for this stock
    #         st.dataframe(
    #             stock_trades[['entry_time', 'exit_time', 'entry_price', 'exit_price', 
    #                         'quantity', 'net_pnl', 'net_pnl_pct', 'win']].sort_values('exit_time', ascending=False),
    #             use_container_width=True
    #         )
    #     else:
    #         st.info(f"No trades recorded for {kode} yet")

# ======================================================
# TRIGGER SCREENING SECTION
# ======================================================
st.divider()
st.header("🔔 Trigger Screening")

trigger_col1, trigger_col2 = st.columns([1, 3])
with trigger_col1:
    if st.button("Run Trigger Screening", key="btn_trigger_screening", use_container_width=True):
        df_screen = st.session_state.get("scan")
        
        if df_screen is None or df_screen.empty:
            st.warning("Belum ada hasil screening")
            st.stop()
        
        df_strong = df_screen[df_screen["MajorTrend"] == "STRONG"]
        
        if df_strong.empty:
            st.info("Tidak ada emiten dengan MajorTrend STRONG")
            st.stop()
        
        progress = st.progress(0.0)
        status = st.empty()
        hijau_results = []
        total = len(df_strong)
        
        for i, row in enumerate(df_strong.itertuples(), start=1):
            kode = row.Kode
            try:
                result = run_backtest_cached(kode)
                
                if result and result.get("Bias") == "HIJAU":
                    hijau_results.append({
                        "Kode": kode,
                        "MajorTrend": row.MajorTrend,
                        "MinorPhase": row.MinorPhase,
                        "RSI": row.RSI,
                        "VOL_BEHAVIOR": row.VOL_BEHAVIOR,
                        "ProbHijau": result.get("ProbHijau"),
                        "ProbMerah": result.get("ProbMerah"),
                        "Sample": result.get("Sample"),
                        "Confidence": result.get("Confidence"),
                        "MatchType": result.get("DecisionContext", {}).get("MatchType")
                    })
                    
            except Exception as e:
                st.error(f"Error processing {kode}: {e}")
            
            progress.progress(i / total)
            status.text(f"Processed {i}/{total} STRONG stocks")
        
        df_trigger = pd.DataFrame(hijau_results)
        
        if not df_trigger.empty:
            save_trigger_cache(df_trigger, TODAY)
            st.success(f"✅ Trigger screening completed: {len(df_trigger)} stocks with HIJAU bias")
        else:
            st.info("No stocks with HIJAU bias found")
        
        st.rerun()

# Display trigger results
# Display trigger results
TRADE_DATE = TODAY
df_trigger, trigger_used_date = load_trigger_cache_pickle(TRADE_DATE)

if df_trigger is not None and not df_trigger.empty:
    st.subheader(f"🌱 Stocks with STRONG Trend & HIJAU Bias ({len(df_trigger)})")
    
    if show_status("Trigger screening", TRADE_DATE, trigger_used_date, df_trigger):
        # Format for display
        display_trigger = df_trigger.copy()
        
        # Format percentage columns if they exist
        if "ProbHijau" in display_trigger.columns:
            display_trigger["ProbHijau"] = display_trigger["ProbHijau"].apply(
                lambda x: f"{x}%" if pd.notna(x) else "N/A"
            )
        
        if "ProbMerah" in display_trigger.columns:
            display_trigger["ProbMerah"] = display_trigger["ProbMerah"].apply(
                lambda x: f"{x}%" if pd.notna(x) else "N/A"
            )
        
        # Add RSI from original screening data if available
        if "RSI" not in display_trigger.columns and "scan" in st.session_state:
            # Merge with original screening data to get RSI
            scan_df = st.session_state.get("scan", pd.DataFrame())
            if not scan_df.empty and "Kode" in scan_df.columns and "RSI" in scan_df.columns:
                display_trigger = display_trigger.merge(
                    scan_df[["Kode", "RSI"]],
                    on="Kode",
                    how="left"
                )
        
        # Format RSI if it exists now
        if "RSI" in display_trigger.columns:
            display_trigger["RSI"] = display_trigger["RSI"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            )
        
        # Reorder columns for better display
        preferred_order = ["Kode", "MajorTrend", "MinorPhase", "RSI", "VOL_BEHAVIOR", 
                          "ProbHijau", "ProbMerah", "Sample", "Confidence", "MatchType"]
        
        # Only include columns that exist
        display_cols = [col for col in preferred_order if col in display_trigger.columns]
        remaining_cols = [col for col in display_trigger.columns if col not in display_cols]
        
        display_trigger = display_trigger[display_cols + remaining_cols]
        
        st.dataframe(display_trigger, use_container_width=True)

# ======================================================
# BROKER SUMMARY ENRICHMENT
# ======================================================
df_broker, broker_used_date = load_broker_summary(TRADE_DATE)

if df_broker is not None and not df_broker.empty:
    if show_status("Broker summary", TRADE_DATE, broker_used_date, df_broker):
        if df_trigger is not None and not df_trigger.empty and 'Kode' in df_trigger.columns:
            # Merge data
            df_final = df_trigger.merge(
                df_broker,
                left_on="Kode",
                right_on="stock",
                how="left"
            )
            
            if not df_final.empty:
                st.subheader("📊 Trigger + Broker Summary")
                
                # Buat layout dengan dua kolom utama
                main_col, detail_col = st.columns([2, 1])
                
                with main_col:
                    # Tampilkan tabel utama
                    main_cols = ['Kode', 'MajorTrend', 'MinorPhase', 
                               'ProbHijau', 'ProbMerah', 'Confidence', 'net_volume']
                    main_cols = [col for col in main_cols if col in df_final.columns]
                    
                    if main_cols:
                        display_df = df_final[main_cols].copy()
                        
                        # Format kolom
                        for col in ['ProbHijau', 'ProbMerah']:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(
                                    lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
                                )
                        
                        if 'net_volume' in display_df.columns:
                            display_df['net_volume'] = display_df['net_volume'].apply(
                                lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
                            )
                        
                        # Styling
                        def color_prob(val):
                            if isinstance(val, str) and '%' in val:
                                try:
                                    prob = float(val.replace('%', ''))
                                    if prob >= 70:
                                        return 'background-color: #d4edda; font-weight: bold;'
                                    elif prob >= 60:
                                        return 'background-color: #fff3cd;'
                                except:
                                    pass
                            return ''
                        
                        if 'ProbHijau' in display_df.columns:
                            styled_df = display_df.style.applymap(color_prob, subset=['ProbHijau'])
                            selection = st.dataframe(
                                styled_df,
                                use_container_width=True,
                                height=400,
                                on_select="rerun",
                                selection_mode="single-row"
                            )
                        else:
                            selection = st.dataframe(
                                display_df,
                                use_container_width=True,
                                height=400,
                                on_select="rerun",
                                selection_mode="single-row"
                            )
                        
                        # Download button
                        csv = display_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"trigger_broker_{TODAY}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                with detail_col:
                    selected_kode = None
                    
                    # Coba ambil dari table selection pertama
                    if selection.selection.rows:
                        selected_idx = selection.selection.rows[0]
                        selected_kode = display_df.iloc[selected_idx]['Kode']
                    
                    # Fallback ke selectbox
                    if not selected_kode and 'Kode' in display_df.columns:
                        selected_kode = st.selectbox(
                            "Pilih Saham:",
                            display_df['Kode'].tolist(),
                            key="broker_detail_select"
                        )
                    
                    if selected_kode:
                        # Ambil data
                        selected_data = df_final[df_final['Kode'] == selected_kode].iloc[0]
                        
                        # Header
                        st.markdown(f"### {selected_kode}")
                        
                        # Metrics card
                        with st.container(border=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if 'ProbHijau' in selected_data and pd.notna(selected_data['ProbHijau']):
                                    prob = float(selected_data['ProbHijau'])
                                    st.metric("Prob. Hijau", f"{prob:.1f}%")
                            
                            with col2:
                                if 'net_volume' in selected_data and pd.notna(selected_data['net_volume']):
                                    net_vol = float(selected_data['net_volume'])
                                    st.metric("Net Volume", f"{net_vol:,.0f}")
                        
                        # Daily summary
                        if 'daily_summary' in selected_data and pd.notna(selected_data['daily_summary']):
                            st.info(f"📊 {selected_data['daily_summary']}")
                        
                        # Buyer vs Seller comparison
                        st.markdown("### 🤝 Buyer vs Seller")
                        
                        col_buyer, col_seller = st.columns(2)
                        
                        with col_buyer:
                            st.markdown("##### 🟢 **Top Buyers**")
                            if 'top5_buyers' in selected_data and pd.notna(selected_data['top5_buyers']):
                                buyers_text = selected_data['top5_buyers']
                                buyers_lines = buyers_text.split('\n')
                                for line in buyers_lines:
                                    st.markdown(f"<div style='margin-bottom: 5px;'>{line}</div>", 
                                               unsafe_allow_html=True)
                            else:
                                st.info("No buyer data")
                        
                        with col_seller:
                            st.markdown("##### 🔴 **Top Sellers**")
                            if 'top5_sellers' in selected_data and pd.notna(selected_data['top5_sellers']):
                                sellers_text = selected_data['top5_sellers']
                                sellers_lines = sellers_text.split('\n')
                                for line in sellers_lines:
                                    st.markdown(f"<div style='margin-bottom: 5px;'>{line}</div>", 
                                               unsafe_allow_html=True)
                            else:
                                st.info("No seller data")
                        
                        # Mini chart untuk net volume
                        if 'net_volume' in selected_data and pd.notna(selected_data['net_volume']):
                            try:
                                net_vol = float(selected_data['net_volume'])
                                st.markdown("### 📊 Volume Analysis")
                                
                                # Simple bar chart
                                if net_vol != 0:
                                    max_val = max(abs(net_vol), 100000)
                                    percentage = (abs(net_vol) / max_val) * 100
                                    
                                    color = "green" if net_vol > 0 else "red"
                                    label = "Buyer Dominan" if net_vol > 0 else "Seller Dominan"
                                    
                                    st.markdown(f"""
                                    <div style="margin: 10px 0;">
                                        <div style="background-color: #f0f0f0; height: 20px; border-radius: 10px; overflow: hidden;">
                                            <div style="background-color: {color}; height: 100%; width: {min(percentage, 100)}%; 
                                                        display: flex; align-items: center; padding-left: 10px; color: white; font-weight: bold;">
                                                {label}
                                            </div>
                                        </div>
                                        <div style="text-align: center; margin-top: 5px;">
                                            <strong>{net_vol:,.0f} lot</strong>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            except:
                                pass

# # ======================================================
# # PERFORMANCE TRACKING DASHBOARD
# # ======================================================
# st.divider()
# render_performance_tracking()

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption(f"📊 IDX Price Action Screener V3 • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")