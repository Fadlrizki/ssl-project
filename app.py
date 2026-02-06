"""
IDX Price Action Screener V3
Streamlit Dashboard with Value Trx Integration
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
from datetime import datetime, timedelta

# Import modules
from engine import build_probability_table_from_ticker, backtest
from engine_v2 import process_stock, fetch_data, add_indicators
from utils import  cache_manager,  format_utils


# ======================================================
# CONFIG
# ======================================================

CACHE_VERSION = "v4"
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

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    layout="wide",
    page_title="IDX Price Action Screener V3",
    page_icon="📊"
)

st.title("📊 IDX Price Action Screener V3")
st.caption("Daily trend • Minor phase • Volume behavior • Value Trx Analysis")

# ======================================================
# HELPER FUNCTIONS
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
    """Render technical chart with proper layout"""
    num_points = len(df)
    
    # Calculate bar width
    if num_points <= 50:
        bar_padding = 0.8
    elif num_points <= 100:
        bar_padding = 0.7
    elif num_points <= 200:
        bar_padding = 0.6
    else:
        bar_padding = 0.5
    
    if num_points > 1:
        avg_day_gap = (df.index[-1] - df.index[0]).days / (num_points - 1)
        bar_width_ms = avg_day_gap * 24 * 3600 * 1000 * bar_padding
    else:
        bar_width_ms = 24 * 3600 * 1000

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.25, 0.15, 0.15],
        subplot_titles=(
            f"{kode} - Price & EMAs",
            "Volume & Volume MA20",
            "RSI (14)",
            "MACD"
        )
    )

    # 1. PRICE CHART
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

    # 2. VOLUME CHART
    if "Volume" in df.columns:
        # Calculate volume unit
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
        
        # Volume colors
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
                width=bar_width_ms,
                marker_line_width=0,
                offset=0
            ),
            row=2, col=1
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
                row=2, col=1
            )

    # 3. RSI CHART
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(width=1.5, color="purple")
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1, row=3, col=1)
        fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="gray", opacity=0.05, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1, row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)

    # 4. MACD CHART
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
            row=4, col=1
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
            row=4, col=1
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
            row=4, col=1
        )
        
        fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)

    # Layout
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
        margin=dict(t=60, b=40, l=60, r=60),
        template="plotly_white",
        hovermode="x unified",
        bargap=0.1,
        bargroupgap=0.05,
        barmode='overlay'
    )
    
    # Update axes
    fig.update_yaxes(title=dict(text="Price (IDR)", font=dict(size=12)), row=1, col=1, tickformat=",")
    
    if "Volume" in df.columns:
        volume_title = f"Volume ({volume_unit})"
        fig.update_yaxes(
            title=dict(text=volume_title, font=dict(size=12)),
            row=2, col=1,
            tickformat=","
        )
    
    fig.update_yaxes(title=dict(text="RSI", font=dict(size=12)), row=3, col=1, range=[0, 100])
    fig.update_yaxes(title=dict(text="MACD", font=dict(size=12)), row=4, col=1)
    fig.update_xaxes(title=dict(text="Date", font=dict(size=12)), row=4, col=1)

    # Volume ratio if available
    if "VOL_RATIO" in df.columns and "Volume" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["VOL_RATIO"],
                mode="lines",
                name="Volume Ratio",
                line=dict(color="purple", width=1, dash="dash"),
                yaxis="y5"
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            yaxis5=dict(
                title="Volume Ratio",
                overlaying="y2",
                side="right",
                range=[0, max(3, df["VOL_RATIO"].max() * 1.1)],
                showgrid=False,
                tickfont=dict(color="purple")
            )
        )
        
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", line_width=1, row=2, col=1, yref="y5")
        fig.add_hline(y=1.5, line_dash="dash", line_color="orange", line_width=1, row=2, col=1, yref="y5")
        fig.add_hline(y=2.0, line_dash="dash", line_color="red", line_width=1, row=2, col=1, yref="y5")

    # Volume stats annotation
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

def retry_single_stock(kode):
    """Retry processing a single stock"""
    # Remove from cache if exists
    if os.path.exists(CACHE_SCREENING):
        df = load_cache_safe(CACHE_SCREENING)
        df = df[df["Kode"] != kode]
        save_cache(df, CACHE_SCREENING)
    
    # Process stock with force refresh - SELALU include Value Trx
    return process_stock(kode, use_cache=False, include_value_trx=True)

# ======================================================
# BROKER SUMMARY HELPER FUNCTIONS
# ======================================================
def get_trade_date(today=None):
    """Get last trading day"""
    if today is None:
        today = datetime.today()
    while today.weekday() >= 5:
        today -= timedelta(days=1)
    return today.strftime("%Y-%m-%d")

def save_trigger_cache(df, trade_date=None):
    """Save trigger cache"""
    if trade_date is None:
        trade_date = get_trade_date()
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
                    return df, check_date
            except Exception as e:
                continue
    
    return None, None

def load_trigger_cache_pickle(trade_date, max_back=7):
    """Load trigger cache with fallback"""
    dt = datetime.strptime(trade_date, "%Y-%m-%d")
    for i in range(max_back + 1):
        check_date = (dt - timedelta(days=i)).strftime("%Y-%m-%d")
        path = f"cache/trigger_result_{check_date}.pkl"
        if os.path.exists(path):
            df = pickle.load(open(path, "rb"))
            return df, check_date
    return None, None

def show_status(name, trade_date, used_date, df):
    """Show data status"""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.info(f"❌ {name} tidak tersedia untuk hari ini maupun fallback")
        return False
    elif used_date != trade_date:
        st.info(f"ℹ️ {name} {trade_date} belum tersedia, pakai data {used_date}")
        return True
    else:
        st.success(f"✅ {name} {trade_date} sudah update")
        return True

# ======================================================
# LOAD STOCK LIST
# ======================================================
@st.cache_data
def load_stock_list():
    """Load stock list from Excel"""
    try:
        saham_df = pd.read_excel(EXCEL_FILE)
        codes = saham_df[KODE_COLUMN].dropna().unique().tolist()
        return codes
    except Exception as e:
        st.error(f"Error loading stock list: {e}")
        return []

codes = load_stock_list()
cached_df = load_cache_safe(CACHE_SCREENING)

# ======================================================
# TOP CONFIGURATION BAR
# ======================================================
# Configuration bar at the top
st.markdown("---")
config_col1, config_col2 = st.columns([1, 3])

with config_col1:
    if st.button("🗑️ Clear All Cache", use_container_width=True, type="primary"):
        clear_cache()
        st.rerun()

with config_col2:
    # Cache info
    cache_age = "N/A"
    if os.path.exists(CACHE_SCREENING):
        mod_time = os.path.getmtime(CACHE_SCREENING)
        cache_age = (datetime.now() - datetime.fromtimestamp(mod_time)).seconds // 60
    st.caption(f"📅 {TODAY} | 📊 {len(codes)} stocks | 💾 Cache: {cache_age} min")

# ======================================================
# MAIN SCREENING SECTION
# ======================================================
st.header("🚀 Stock Screening")

# Di bagian run screening:
if st.button("🚀 Run Full Screening", use_container_width=True, type="primary"):
    results = []
    progress = st.progress(0)
    status = st.empty()
    error_log = []
    
    with ThreadPoolExecutor(max_workers=2) as ex:  # Kurangi workers untuk stabil
        # SELALU include Value Trx
        futures = {ex.submit(process_stock, k, use_cache=True, include_value_trx=True): k for k in codes}
        done = 0
        total = len(codes)
        
        for f in as_completed(futures):
            kode = futures[f]
            try:
                r = f.result(timeout=30)  # Timeout 30 detik
                if r and "Kode" in r and "Price" in r:
                    r["ProcessTime"] = pd.Timestamp.now()
                    results.append(r)
                else:
                    error_log.append(f"{kode}: No valid result")
            except Exception as e:
                error_log.append(f"{kode}: {str(e)}")
                st.error(f"Error {kode}: {e}")
            
            done += 1
            progress.progress(done / total)
            status.text(f"Processed {done}/{total} saham ({len(error_log)} errors)")
    
    # Tampilkan error log jika ada
    if error_log:
        with st.expander("⚠️ Error Log"):
            for error in error_log:
                st.text(error)
    
    df_new = pd.DataFrame(results)
    
    if not df_new.empty:
        # Combine with existing cache
        if not cached_df.empty:
            df_scan = pd.concat([cached_df, df_new], ignore_index=True)
        else:
            df_scan = df_new.copy()
        
        df_scan = df_scan.drop_duplicates(subset=["Kode"], keep="last").reset_index(drop=True)
        
        save_cache(df_scan, CACHE_SCREENING)
        st.session_state["scan"] = df_scan
        st.success(f"✅ Screening selesai: {len(df_scan)} saham valid")
        st.info(f"📊 Value Trx calculated for {len([r for r in results if 'ValueTrx' in r])} stocks")
        
        # Tampilkan statistik Value Trx
        if "ValueTrx" in df_scan.columns:
            valid_value_trx = df_scan[df_scan["ValueTrx"] > 0]
            st.info(f"📈 Valid Value Trx: {len(valid_value_trx)} stocks")
    else:
        st.warning("Tidak ada hasil screening yang valid")
    
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

# Quick filters - HAPUS RSI FILTER
col1, col2, col3 = st.columns(3)  # Hanya 3 kolom, hapus RSI
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
    
    # Value Trx Filters - FIXED NO ERROR
    st.markdown("---")
    st.subheader("💰 Value Trx Filters")
    
    value_col1, value_col2 = st.columns(2)
    
    with value_col1:
        if "ValueTrx" in df.columns and not df.empty:
            # Cari nilai maksimum yang valid
            valid_values = df["ValueTrx"][df["ValueTrx"] > 0]
            if not valid_values.empty:
                max_val = float(valid_values.max())
                min_val = float(valid_values.min())
                
                if max_val > min_val:  # Pastikan ada range
                    # Convert to billions untuk display
                    max_val_b = max_val / 1_000_000_000
                    min_val_b = min_val / 1_000_000_000
                    
                    min_value_b = st.slider(
                        "Min Value Trx (Billion)", 
                        min_value=0.0,
                        max_value=round(max_val_b, 1),
                        value=0.0,
                        step=0.1
                    )
                    
                    # Convert back to actual value
                    min_value_trx = min_value_b * 1_000_000_000
                    df = df[df["ValueTrx"] >= min_value_trx]
                else:
                    st.info(f"All Value Trx: Rp {max_val:,.0f}")
            else:
                st.info("No Value Trx data available")
    
    with value_col2:
        if "Liquidity_Score" in df.columns:
            liquidity_filter = st.multiselect(
                "Liquidity Score",
                df["Liquidity_Score"].unique() if not df.empty else []
            )
            if liquidity_filter:
                df = df[df["Liquidity_Score"].isin(liquidity_filter)]

# Sorting options
sort_col1, sort_col2 = st.columns([2, 1])

with sort_col1:
    sort_options = ["Kode", "Price", "PriceChange%", "MinorConfidence%", "FinalDecision"]
    if "ValueTrx" in df.columns:
        sort_options.append("ValueTrx")
    
    sort_by = st.selectbox("Sort by:", sort_options, index=0)

with sort_col2:
    sort_order = st.selectbox("Order:", ["Ascending", "Descending"], index=1)

# Apply sorting
if not df.empty and sort_by in df.columns:
    df = df.sort_values(sort_by, ascending=(sort_order == "Ascending"))

# ======================================================
# RESULTS TABLE - SIMPLIFIED
# ======================================================
st.subheader(f"📋 Screening Results ({len(df)} saham)")

# Format dataframe for display
display_df = df.copy()
if not display_df.empty:
    # Format numeric columns
    display_df["Price"] = display_df["Price"].apply(lambda x: format_utils.format_currency(x))
    
    # Format Volume
    if "Volume" in display_df.columns:
        display_df["Volume_Display"] = display_df["Volume"].apply(
            lambda x: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
        )
    
    display_df["RSI"] = display_df["RSI"].apply(lambda x: f"{x:.1f}")
    
    # Format Value Trx - SIMPLE
    if "ValueTrx" in display_df.columns:
        display_df["ValueTrx_Display"] = display_df["ValueTrx"].apply(
            lambda x: f"Rp {x:,.0f}" if pd.notna(x) and x > 0 else "-"
        )
    
    # Select columns to display
    display_cols = ["Kode", "Sector","Industry","Price", "PriceChange%", "MajorTrend", "MinorPhase", 
                   "MinorConfidence%", "RSI", "Volume_Display"]
    
    # Add VOL_BEHAVIOR
    if "VOL_BEHAVIOR" in display_df.columns:
        display_cols.append("VOL_BEHAVIOR")
    
    # Add Value Trx
    if "ValueTrx_Display" in display_df.columns:
        display_cols.append("ValueTrx_Display")
    
    # Add remaining columns
    display_cols.extend(["Latest_Candle", "FinalDecision"])
    
    # Filter only existing columns
    display_cols = [col for col in display_cols if col in display_df.columns]
    display_df = display_df[display_cols]
    
    # Rename columns for better display
    column_rename = {
        "ValueTrx_Display": "Value Trx",
        "PriceChange%": "Chg %",
        "MinorConfidence%": "Conf %",
        "VOL_BEHAVIOR": "Vol Behavior",
        "Volume_Display": "Volume",
        "Latest_Candle": "Candle"
    }
    
    # Apply rename
    rename_dict = {k: v for k, v in column_rename.items() if k in display_df.columns}
    display_df = display_df.rename(columns=rename_dict)

# Display table
try:
    event = st.dataframe(
        display_df.style.applymap(color_decision, subset=['FinalDecision']),
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun"
    )
except Exception as e:
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
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Chart & Metrics", "💰 Value Trx", "🤖 Probability"])
    
    with tab1:
        # Refresh button
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
        
        # Technical Chart
        st.subheader(f"📈 Technical Chart - {kode}")
        
        ticker = f"{kode}.JK"
        try:
            df_daily = fetch_data(ticker, interval="1d", period="12mo", force_refresh=False)
            
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
            if "VOL_RATIO" in row:
                st.metric("Volume Ratio", f"{row['VOL_RATIO']:.2f}")
        
        with tech_cols[1]:
            st.metric("RSI", f"{row['RSI']:.1f}")
            if "Stoch_K" in row:
                st.metric("Stoch %K", f"{row['Stoch_K']:.1f}")
            if "Dist_to_SMA50" in row and not pd.isna(row['Dist_to_SMA50']):
                st.metric("Distance to SMA50", f"{row['Dist_to_SMA50']:.1f}%")
            st.metric("Major Trend", row["MajorTrend"])
        
        with tech_cols[2]:
            st.metric("Minor Phase", row["MinorPhase"])
            st.metric("Confidence", f"{row['MinorConfidence']} ({row['MinorConfidence%']}%)")
            if "SetupState" in row:
                st.metric("Setup State", row["SetupState"])
            st.metric("Final Decision", row["FinalDecision"])
        
        # Gap Analysis
        st.subheader("📏 Gap Analysis")
        gap_cols = st.columns(3)
        with gap_cols[0]:
            if "Gap_EMA13%" in row:
                st.metric("Gap to EMA13", f"{row['Gap_EMA13%']}%")
        with gap_cols[1]:
            if "Gap_EMA21%" in row:
                st.metric("Gap to EMA21", f"{row['Gap_EMA21%']}%")
        with gap_cols[2]:
            if "Gap_EMA50%" in row:
                st.metric("Gap to EMA50", f"{row['Gap_EMA50%']}%")
    
    with tab2:
        # Value Trx Analysis - SELALU DITAMPILKAN
        st.subheader("💰 Value Transaction Analysis")
        
        # Value Trx Metrics
        value_cols = st.columns(4)
        
        with value_cols[0]:
            if "ValueTrx_Rp" in row:
                st.metric("Value Trx Today", row["ValueTrx_Rp"])
            elif "ValueTrx" in row and row["ValueTrx"] > 0:
                st.metric("Value Trx Today", f"Rp {row['ValueTrx']:,.0f}")
            else:
                st.metric("Value Trx Today", "N/A")
        
        with value_cols[1]:
            if "ValueTrx_B" in row and row["ValueTrx_B"] > 0:
                st.metric("Value (Billion)", f"{row['ValueTrx_B']:.2f}B")
            elif "ValueTrx" in row and row["ValueTrx"] > 0:
                st.metric("Value (Billion)", f"{row['ValueTrx']/1e9:.2f}B")
            else:
                st.metric("Value (Billion)", "N/A")
        
        with value_cols[2]:
            if "AvgPrice" in row and row["AvgPrice"] > 0:
                st.metric("Avg Price", f"Rp {row['AvgPrice']:,.0f}")
            if "VWAP" in row and row["VWAP"] > 0:
                st.metric("VWAP", f"Rp {row['VWAP']:,.0f}")
        
        with value_cols[3]:
            if "ValueTrx_Status" in row:
                status_badge = {
                    "SUCCESS": "🟢",
                    "SUCCESS_DAILY": "🟡",
                    "1M_ACCURATE": "✅",
                    "DAILY_APPROXIMATION": "📊",
                    "FALLBACK": "⚠️",
                    "NO_DATA_AVAILABLE": "❌"
                }.get(row["ValueTrx_Status"], "❓")
                
                if "ValueTrx_Method" in row:
                    st.metric("Method", f"{status_badge} {row['ValueTrx_Method']}")
                else:
                    st.metric("Status", f"{status_badge} {row['ValueTrx_Status']}")
        
        # Value Trx Historical Chart
        st.subheader("📈 Value Trx History")
        
        ticker = f"{kode}.JK"
        try:
            df_daily = fetch_data(ticker, interval="1d", period="6mo", force_refresh=False)
            
            if df_daily is not None and not df_daily.empty:
                fig = render_value_trx_chart(kode, df_daily)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada data historis untuk Value Trx chart")
        except Exception as e:
            st.error(f"Gagal render Value Trx chart: {e}")
        
        # Value Trx Details
        with st.expander("📊 Value Trx Details"):
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                if "ValueTrx_Volume" in row and row["ValueTrx_Volume"] > 0:
                    st.write(f"**Volume (1m):** {row['ValueTrx_Volume']:,.0f} lembar")
                
                if "ValueTrx_Bars" in row:
                    st.write(f"**1m Bars:** {row['ValueTrx_Bars']}")
                
                if "ValueTrx_Ratio" in row and not pd.isna(row["ValueTrx_Ratio"]):
                    ratio = row["ValueTrx_Ratio"]
                    st.write(f"**Value Ratio:** {ratio:.2f}")
                    st.progress(min(ratio / 3.0, 1.0))
            
            with col_info2:
                if "AvgPrice" in row and "Price" in row and row["AvgPrice"] > 0:
                    premium = ((row["Price"] - row["AvgPrice"]) / row["AvgPrice"] * 100)
                    st.write(f"**Price vs Avg:** {premium:+.1f}%")
                    st.write("(Close vs Transaction Avg Price)")
                
                if "Liquidity_Score" in row:
                    score_color = {
                        "HIGH": "🟢",
                        "MEDIUM": "🟡",
                        "LOW": "🔴"
                    }.get(row["Liquidity_Score"], "⚪")
                    st.write(f"**Liquidity Score:** {score_color} {row['Liquidity_Score']}")
    
    with tab3:
        # Probability Analysis
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
    
        with tab4:
        # Broker Summary
            st.subheader("📊 Broker Summary Integration")
            
            TRADE_DATE = TODAY
            df_broker, broker_used_date = load_broker_summary(TRADE_DATE)
            
            if df_broker is not None and not df_broker.empty:
                if show_status("Broker summary", TRADE_DATE, broker_used_date, df_broker):
                    # Try to merge with current stock
                    if kode in df_broker['stock'].values:
                        broker_data = df_broker[df_broker['stock'] == kode].iloc[0]
                        
                        # TAMBAH INI: Display Sector and Industry
                        st.markdown("##### 📊 Company Info")
                        info_cols = st.columns(3)
                        
                        with info_cols[0]:
                            if 'Sector' in row:
                                st.metric("Sector", row['Sector'])
                        
                        with info_cols[1]:
                            if 'Industry' in row:
                                st.metric("Industry", row['Industry'])
                        
                        with info_cols[2]:
                            if 'net_volume' in broker_data:
                                net_vol = broker_data['net_volume']
                                color = "🟢" if net_vol > 0 else "🔴"
                                st.metric("Net Volume", f"{color} {abs(net_vol):,.0f}")
                        
                        # Display broker metrics
                        broker_cols = st.columns(3)
                        
                        with broker_cols[0]:
                            if 'avg_buy_price_buyers' in broker_data and broker_data['avg_buy_price_buyers'] > 0:
                                st.metric("Avg Buy Price", f"Rp {broker_data['avg_buy_price_buyers']:,.0f}")
                        
                        with broker_cols[1]:
                            if 'avg_sell_price_sellers' in broker_data and broker_data['avg_sell_price_sellers'] > 0:
                                st.metric("Avg Sell Price", f"Rp {broker_data['avg_sell_price_sellers']:,.0f}")
                        
                        with broker_cols[2]:
                            if 'daily_summary' in broker_data:
                                st.metric("Summary", broker_data['daily_summary'])
                        
                        # Display top buyers/sellers if available
                        col_buyer, col_seller = st.columns(2)
                        
                        with col_buyer:
                            st.markdown("##### 🟢 Top Buyers")
                            if 'top5_buyers' in broker_data and pd.notna(broker_data['top5_buyers']):
                                buyers_text = broker_data['top5_buyers']
                                buyers_lines = buyers_text.split('\n')
                                for line in buyers_lines:
                                    st.write(line)
                        
                        with col_seller:
                            st.markdown("##### 🔴 Top Sellers")
                            if 'top5_sellers' in broker_data and pd.notna(broker_data['top5_sellers']):
                                sellers_text = broker_data['top5_sellers']
                                sellers_lines = sellers_text.split('\n')
                                for line in sellers_lines:
                                    st.write(line)
                    else:
                        st.info(f"Tidak ada data broker summary untuk {kode} pada tanggal {broker_used_date}")
            else:
                st.info("Broker summary data tidak tersedia")
                

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
        preferred_order = ["Kode", 'Sector','Industry',"MajorTrend", "MinorPhase", "RSI", "VOL_BEHAVIOR", 
                          "ProbHijau", "ProbMerah", "Sample", "Confidence", "MatchType"]
        if "scan" in st.session_state:
            scan_df = st.session_state["scan"]
            if not scan_df.empty and 'Kode' in scan_df.columns:
                # Merge to get Sector and Industry
                display_trigger = display_trigger.merge(
                    scan_df[['Kode', 'Sector', 'Industry']].drop_duplicates(subset=['Kode']),
                    on='Kode',
                    how='left'
                )
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
            if "scan" in st.session_state:
                scan_df = st.session_state["scan"]
                if not scan_df.empty and 'Kode' in scan_df.columns:
                    df_trigger_enriched = df_trigger.merge(
                        scan_df[['Kode', 'Sector', 'Industry']].drop_duplicates(subset=['Kode']),
                        on='Kode',
                        how='left'
                    )
                else:
                    df_trigger_enriched = df_trigger.copy()
            else:
                df_trigger_enriched = df_trigger.copy()
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
                    # Tampilkan tabel utama - TAMBAHKAN AVG PRICE COLUMNS
                    main_cols = ['Kode', 'Sector', 'Industry', 'MajorTrend', 'MinorPhase', 
                               'ProbHijau', 'ProbMerah', 'Confidence', 
                               'net_volume', 'avg_buy_price_buyers', 'avg_sell_price_sellers']
                    
                    # Filter hanya kolom yang ada di df_final
                    available_cols = [col for col in main_cols if col in df_final.columns]
                    
                    if available_cols:
                        display_df = df_final[available_cols].copy()
                        
                        # Format kolom
                        for col in ['ProbHijau', 'ProbMerah']:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(
                                    lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
                                )
                        
                        # Format net_volume
                        if 'net_volume' in display_df.columns:
                            display_df['net_volume'] = display_df['net_volume'].apply(
                                lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
                            )
                        
                        # Format avg prices
                        if 'avg_buy_price_buyers' in display_df.columns:
                            display_df['avg_buy_price_buyers'] = display_df['avg_buy_price_buyers'].apply(
                                lambda x: f"{x:,.0f}" if pd.notna(x) and x != 0 else "N/A"
                            )
                        
                        if 'avg_sell_price_sellers' in display_df.columns:
                            display_df['avg_sell_price_sellers'] = display_df['avg_sell_price_sellers'].apply(
                                lambda x: f"{x:,.0f}" if pd.notna(x) and x != 0 else "N/A"
                            )
                        
                        # Styling untuk probabilitas dan avg prices
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
                        
                        # Warna untuk avg buy price (hijau untuk harga tinggi)
                        def color_avg_buy(val):
                            if isinstance(val, str) and val != 'N/A':
                                try:
                                    price = float(val.replace(',', ''))
                                    # Anda bisa menyesuaikan threshold sesuai kebutuhan
                                    if price > 10000:
                                        return 'background-color: #e8f5e9; color: #2e7d32;'
                                    elif price > 5000:
                                        return 'background-color: #f1f8e9;'
                                except:
                                    pass
                            return ''
                        
                        # Warna untuk avg sell price (merah untuk harga rendah)
                        def color_avg_sell(val):
                            if isinstance(val, str) and val != 'N/A':
                                try:
                                    price = float(val.replace(',', ''))
                                    if price < 1000:
                                        return 'background-color: #ffebee; color: #c62828;'
                                except:
                                    pass
                            return ''
                        
                        # Apply styling
                        styled_df = display_df.copy()
                        
                        if 'ProbHijau' in styled_df.columns:
                            styled_df = styled_df.style.applymap(color_prob, subset=['ProbHijau'])
                        
                        if 'avg_buy_price_buyers' in styled_df.columns:
                            styled_df = styled_df.applymap(color_avg_buy, subset=['avg_buy_price_buyers'])
                        
                        if 'avg_sell_price_sellers' in styled_df.columns:
                            styled_df = styled_df.applymap(color_avg_sell, subset=['avg_sell_price_sellers'])
                        
                        # Warna untuk net volume
                        if 'net_volume' in styled_df.columns:
                            def color_net_volume(val):
                                if isinstance(val, str) and val != 'N/A':
                                    try:
                                        # Hapus koma dan konversi ke int
                                        clean_val = val.replace(',', '').replace('N/A', '0')
                                        volume = int(float(clean_val)) if clean_val else 0
                                        
                                        if volume > 0:
                                            return 'background-color: #e8f5e9; color: #2e7d32; font-weight: bold;'
                                        elif volume < 0:
                                            return 'background-color: #ffebee; color: #c62828; font-weight: bold;'
                                    except:
                                        pass
                                return ''
                            
                            styled_df = styled_df.applymap(color_net_volume, subset=['net_volume'])
                        
                        # Tampilkan tabel
                        selection = st.dataframe(
                            styled_df,
                            use_container_width=True,
                            height=400,
                            on_select="rerun",
                            selection_mode="single-row"
                        )
                        
                        # Download button
                        st.markdown("---")
                        
                        # Download FULL DATA
                        full_csv = df_final.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 Download CSV (Full Data)",
                            full_csv,
                            f"trigger_broker_{TODAY}.csv",
                            "text/csv",
                            use_container_width=True,
                            type="primary",
                            help="Download semua data termasuk broker summary dan avg prices"
                        )
                        
                        # Info kecil tentang apa yang di-download
                        st.caption(f"📋 File akan berisi {len(df_final)} baris dan {len(df_final.columns)} kolom termasuk broker data")
                        
                        # Tampilkan stats ringkasan
                        with st.expander("📊 Statistics Summary"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if 'ProbHijau' in df_final.columns:
                                    avg_prob = df_final['ProbHijau'].mean()
                                    st.metric("Avg Prob HIJAU", f"{avg_prob:.1f}%")
                            
                            with col2:
                                if 'net_volume' in df_final.columns:
                                    total_volume = df_final['net_volume'].sum()
                                    st.metric("Total Net Volume", f"{total_volume:,.0f}")
                            
                            with col3:
                                if 'avg_buy_price_buyers' in df_final.columns:
                                    avg_buy_price = df_final['avg_buy_price_buyers'].mean()
                                    st.metric("Avg Buy Price", f"{avg_buy_price:,.0f}" if not pd.isna(avg_buy_price) else "N/A")
                
                with detail_col:
                    selected_kode = None
                    
                    # Coba ambil dari table selection
                    try:
                        if hasattr(selection, 'selection') and selection.selection.rows:
                            selected_idx = selection.selection.rows[0]
                            selected_kode = display_df.iloc[selected_idx]['Kode']
                    except:
                        pass
                    
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
                        
                        # Header dengan lebih banyak info
                        st.markdown(f"### {selected_kode}")
                        
                        # Metrics card dengan avg prices
                        with st.container(border=True):
                            # Gunakan layout yang lebih lebar
                            st.markdown("""
                            <style>
                            .metrics-container {
                                padding: 10px;
                            }
                            .metric-value {
                                font-size: 20px !important;
                                font-weight: bold;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns([1, 1, 1])
                            
                            with col1:
                                if 'ProbHijau' in selected_data and pd.notna(selected_data['ProbHijau']):
                                    prob = float(selected_data['ProbHijau'])
                                    # Gunakan container dalam col untuk lebih banyak kontrol
                                    with st.container():
                                        st.markdown("**Prob. Hijau**")
                                        st.markdown(f"<div class='metric-value'>{prob:.1f}%</div>", unsafe_allow_html=True)
                            
                            with col2:
                                if 'net_volume' in selected_data and pd.notna(selected_data['net_volume']):
                                    net_vol = float(selected_data['net_volume'])
                                    color = "🟢" if net_vol > 0 else "🔴"
                                    with st.container():
                                        st.markdown("**Net Volume**")
                                        st.markdown(f"<div class='metric-value'>{color} {abs(net_vol):,.0f}</div>", unsafe_allow_html=True)
                            
                            with col3:
                                if 'Confidence' in selected_data:
                                    confidence = selected_data['Confidence']
                                    # Warna berdasarkan confidence
                                    conf_color = {
                                        'VERY_HIGH': '🟢',
                                        'HIGH': '🟡',
                                        'MEDIUM': '🟠',
                                        'LOW': '🔴',
                                        'VERY_LOW': '⚫'
                                    }.get(confidence, '⚪')
                                    
                                    with st.container():
                                        st.markdown("**Confidence**")
                                        st.markdown(f"<div class='metric-value'>{conf_color} {confidence}</div>", unsafe_allow_html=True)
                        
                        # Price metrics card
                        with st.container(border=True):
                            st.markdown("**💰 Price Information**")
                            col_price1, col_price2 = st.columns(2)
                            
                            with col_price1:
                                if 'avg_buy_price_buyers' in selected_data and pd.notna(selected_data['avg_buy_price_buyers']):
                                    avg_buy = float(selected_data['avg_buy_price_buyers'])
                                    st.metric("Avg Buy Price", f"{avg_buy:,.0f}")
                            
                            with col_price2:
                                if 'avg_sell_price_sellers' in selected_data and pd.notna(selected_data['avg_sell_price_sellers']):
                                    avg_sell = float(selected_data['avg_sell_price_sellers'])
                                    st.metric("Avg Sell Price", f"{avg_sell:,.0f}")
                            
                            # Calculate spread jika ada kedua data
                            if ('avg_buy_price_buyers' in selected_data and 'avg_sell_price_sellers' in selected_data and
                                pd.notna(selected_data['avg_buy_price_buyers']) and 
                                pd.notna(selected_data['avg_sell_price_sellers'])):
                                
                                avg_buy = float(selected_data['avg_buy_price_buyers'])
                                avg_sell = float(selected_data['avg_sell_price_sellers'])
                                spread = avg_buy - avg_sell
                                spread_pct = (spread / avg_sell * 100) if avg_sell > 0 else 0
                                
                                st.caption(f"**Spread:** {spread:,.0f} ({spread_pct:+.1f}%)")
                        
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
                                    # Highlight avg prices dalam text
                                    line_display = line
                                    if 'Avg Buy:' in line or 'Avg Sell:' in line:
                                        line_display = f"<div style='background-color: #f8f9fa; padding: 5px; border-radius: 4px; margin: 2px 0;'>{line}</div>"
                                    else:
                                        line_display = f"<div style='margin-bottom: 5px;'>{line}</div>"
                                    
                                    st.markdown(line_display, unsafe_allow_html=True)
                            else:
                                st.info("No buyer data")
                        
                        with col_seller:
                            st.markdown("##### 🔴 **Top Sellers**")
                            if 'top5_sellers' in selected_data and pd.notna(selected_data['top5_sellers']):
                                sellers_text = selected_data['top5_sellers']
                                sellers_lines = sellers_text.split('\n')
                                for line in sellers_lines:
                                    # Highlight avg prices dalam text
                                    line_display = line
                                    if 'Avg Buy:' in line or 'Avg Sell:' in line:
                                        line_display = f"<div style='background-color: #f8f9fa; padding: 5px; border-radius: 4px; margin: 2px 0;'>{line}</div>"
                                    else:
                                        line_display = f"<div style='margin-bottom: 5px;'>{line}</div>"
                                    
                                    st.markdown(line_display, unsafe_allow_html=True)
                            else:
                                st.info("No seller data")
                        
                        # Volume analysis dengan lebih banyak info
                        if 'net_volume' in selected_data and pd.notna(selected_data['net_volume']):
                            try:
                                net_vol = float(selected_data['net_volume'])
                                
                                # Tambahkan info price jika ada
                                price_info = ""
                                if ('avg_buy_price_buyers' in selected_data and 
                                    'avg_sell_price_sellers' in selected_data and
                                    pd.notna(selected_data['avg_buy_price_buyers']) and 
                                    pd.notna(selected_data['avg_sell_price_sellers'])):
                                    
                                    avg_buy = float(selected_data['avg_buy_price_buyers'])
                                    avg_sell = float(selected_data['avg_sell_price_sellers'])
                                    price_info = f" | Buy: {avg_buy:,.0f} | Sell: {avg_sell:,.0f}"
                                
                                st.markdown(f"### 📊 Volume Analysis {price_info}")
                                
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
                                            <strong>{abs(net_vol):,.0f} lot</strong>
                                            <div style="font-size: 12px; color: #666;">
                                                Net: {net_vol:,.0f} lot
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            except:
                                pass
                
                # Tampilkan data lengkap dalam expander
                with st.expander("🔍 View Full Data Structure"):
                    st.write(f"**Total Columns:** {len(df_final.columns)}")
                    st.write(f"**Total Rows:** {len(df_final)}")
                    
                    # Tampilkan semua kolom
                    all_columns = df_final.columns.tolist()
                    st.write("**All Available Columns:**")
                    
                    # Group columns by category
                    price_cols = [col for col in all_columns if 'price' in col.lower() or 'avg' in col.lower()]
                    broker_cols = [col for col in all_columns if 'buyer' in col.lower() or 'seller' in col.lower() or 'broker' in col.lower()]
                    trigger_cols = [col for col in all_columns if col not in price_cols + broker_cols]
                    
                    col_cat1, col_cat2, col_cat3 = st.columns(3)
                    
                    with col_cat1:
                        st.write("**Trigger Data:**")
                        for col in sorted(trigger_cols):
                            st.code(col)
                    
                    with col_cat2:
                        st.write("**Price Data:**")
                        for col in sorted(price_cols):
                            st.code(col)
                    
                    with col_cat3:
                        st.write("**Broker Data:**")
                        for col in sorted(broker_cols):
                            st.code(col)
                    
                    # Tampilkan preview data
                    st.write("**Data Preview (first 5 rows):**")
                    st.dataframe(df_final.head(), use_container_width=True)
            else:
                st.warning("Tidak ada overlap antara saham trigger dan broker summary.")
        else:
            st.info("Menunggu hasil trigger screening...")
# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption(f"📊 IDX Price Action Screener V3 • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")