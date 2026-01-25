import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle

from engine import backtest_score
from engine_v2 import process_stock, fetch_data, add_indicators

# ======================================================
# CONFIG
# ======================================================
EXCEL_FILE = "daftar_saham.xlsx"
KODE_COLUMN = "Kode"
MAX_WORKERS = 8

CACHE_VERSION = "v3"
CACHE_SCREENING = f"screening_cache_{CACHE_VERSION}.pkl"

REQUIRED_COLS = {
    "Kode", "MajorTrend", "MinorPhase", "SetupState",
    "FinalDecision", "RSI", "VOL_BEHAVIOR"
}

# ======================================================
# PAGE
# ======================================================
st.set_page_config(layout="wide")
st.title("📊 IDX Price Action Screener V2")
st.caption("Daily trend • Minor phase • Volume behavior")
# dfc = fetch_data("GOLF.JK", "1d", "5d")
# st.write(dfc.tail())
# st.write("LAST DATE:", dfc.index[-1])


# ======================================================
# HELPERS
# ======================================================
def load_cache_safe(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
        if not REQUIRED_COLS.issubset(df.columns):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def save_cache(df, path):
    with open(path, "wb") as f:
        pickle.dump(df, f)

# ======================================================
# LOAD SAHAM
# ======================================================
saham_df = pd.read_excel(EXCEL_FILE)
codes = saham_df[KODE_COLUMN].dropna().unique().tolist()

cached_df = load_cache_safe(CACHE_SCREENING)

# ======================================================
# RUN SCREENING
# ======================================================
if st.button("🚀 Run Screening"):
    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_stock, k): k
            for k in codes
            if cached_df.empty or k not in cached_df["Kode"].values
        }

        # pakai cache dulu
        if not cached_df.empty:
            results.extend(cached_df.to_dict("records"))

        done = len(results)
        total = len(codes)

        for f in as_completed(futures):
            kode = futures[f]
            try:
                r = f.result()
                if r:
                    results.append(r)
                    done += 1
            except Exception:
                pass

            progress.progress(done / total)
            status.text(f"Processed {done}/{total}")

    df_scan = pd.DataFrame(results)
    save_cache(df_scan, CACHE_SCREENING)
    st.session_state["scan"] = df_scan
    st.success(f"Selesai: {len(df_scan)} saham valid")

# ======================================================
# GUARD
# ======================================================
if "scan" not in st.session_state or st.session_state["scan"].empty:
    st.warning("Belum ada hasil screening")
    st.stop()

df = st.session_state["scan"].copy()

# ======================================================
# HORIZONTAL FILTER BAR
# ======================================================
st.markdown("### 🔎 Filter")

cols = st.columns(3)
filters = {
    "MajorTrend": cols[0].multiselect("MajorTrend", sorted(df["MajorTrend"].unique())),
    "MinorPhase": cols[1].multiselect("MinorPhase", sorted(df["MinorPhase"].unique())),
    "SetupStLatest_Candleate": cols[2].multiselect("Latest_Candle", sorted(df["Latest_Candle"].unique())),
}
for col, val in filters.items():
    if val:
        df = df[df[col].isin(val)]

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    kode_filter = st.text_input("Kode").upper()
    if kode_filter:
        df = df[df["Kode"].str.contains(kode_filter)]

with c2:
    rsi_mode = st.selectbox("RSI", ["All", "> 70", "< 30", "40–70"])
    if rsi_mode == "> 70":
        df = df[df["RSI"] > 70]
    elif rsi_mode == "< 30":
        df = df[df["RSI"] < 30]
    elif rsi_mode == "40–70":
        df = df[(df["RSI"] >= 40) & (df["RSI"] <= 70)]

with c3:
    candle = st.multiselect(
        "Candle",
        sorted(df["Latest_Candle"].dropna().unique())
    )
    if candle:
        df = df[df["Latest_Candle"].isin(candle)]

with c4:
    vol_beh = st.multiselect(
        "Volume",
        sorted(df["VOL_BEHAVIOR"].dropna().unique())
    )
    if vol_beh:
        df = df[df["VOL_BEHAVIOR"].isin(vol_beh)]

with c5:
    if st.checkbox("⚡ Scalping >5%"):
        df = df[(df["RSI"] > 70) & (df["Latest_Candle"] == "Hijau Kuat (Impulse)")]

# ======================================================
# RESULT TABLE
# ======================================================
st.subheader("📋 Screening Result")
event = st.dataframe(
    df,
    width="stretch",
    selection_mode="single-row",
    on_select="rerun"
)

# ======================================================
# DETAIL VIEW
# ======================================================
if event.selection.rows:
    row = df.iloc[event.selection.rows[0]]
    kode = row["Kode"]

    dfc = fetch_data(f"{kode}.JK", "1d", "12mo")
    if dfc is None or dfc.empty:
        st.warning("Data chart tidak tersedia")
        st.stop()

    dfc = add_indicators(dfc)

    fig = go.Figure()
    fig.add_candlestick(
        x=dfc.index,
        open=dfc.Open,
        high=dfc.High,
        low=dfc.Low,
        close=dfc.Close
    )
    fig.add_scatter(x=dfc.index, y=dfc.EMA13, name="EMA13")
    fig.add_scatter(x=dfc.index, y=dfc.EMA21, name="EMA21")
    fig.add_scatter(x=dfc.index, y=dfc.EMA50, name="EMA50")

    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")

    st.table(pd.DataFrame({
        "Metric": [
            "Minor Phase", "Confidence",
            "RSI", "Stoch %K",
            "Volume Behavior", "Volume Ratio",
            "Final Decision"
        ],
        "Value": [
            row["MinorPhase"],
            f'{row["MinorConfidence"]} ({row["MinorConfidence%"]}%)',
            row["RSI"],
            row["Stoch_K"],
            row["VOL_BEHAVIOR"],
            row["VOL_RATIO"],
            row["FinalDecision"]
        ]
    }))

# ======================================================
# BACKTEST
# ======================================================
st.subheader("📊 Backtest")
if event.selection.rows and st.button("Run Backtest"):
    kode = df.iloc[event.selection.rows[0]]["Kode"]
    df_score, prob = backtest_score(f"{kode}.JK")
    if df_score is not None:
        st.dataframe(prob)
