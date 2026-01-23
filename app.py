import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle

from engine import process_stock, fetch_data, add_indicators

# =========================
# CONFIG
# =========================
EXCEL_FILE = "daftar_saham.xlsx"
KODE_COLUMN = "Kode"
MAX_WORKERS = 8
CACHE_SCREENING = "screening_cache.pkl"

st.set_page_config(layout="wide")
st.title("📊 IDX Price Action Screener (Complete Version)")
st.caption("Daily trend • Pullback • Momentum • Volume")

# =========================
# LOAD DAFTAR SAHAM
# =========================
saham_df = pd.read_excel(EXCEL_FILE)
codes = saham_df[KODE_COLUMN].dropna().unique().tolist()

# selected_codes = st.multiselect("Pilih Saham untuk Screening", codes)

# =========================
# LOAD CACHE SCREENING
# =========================
if os.path.exists(CACHE_SCREENING):
    with open(CACHE_SCREENING, "rb") as f:
        cached_df = pickle.load(f)
else:
    cached_df = pd.DataFrame()

# =========================
# RUN SCREENING
# =========================
# =========================
# RUN SCREENING
# =========================
if st.button("🚀 Run Screening"):
    # langsung semua kode saham
    total = len(codes)
    progress = st.progress(0)
    status = st.empty()
    results = []
    success = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for k in codes:
            # jika sudah ada cache, skip proses
            if not cached_df.empty and k in cached_df["Kode"].values:
                row = cached_df[cached_df["Kode"]==k].iloc[0].to_dict()
                results.append(row)
                success += 1
                progress.progress(success/total)
                status.text(f"Processed {success}/{total} ({success/total:.1%}) | Success: {success}")
            else:
                futures[ex.submit(process_stock, k)] = k

        for f in as_completed(futures):
            try:
                r = f.result()
                if r:
                    results.append(r)
                    success += 1
                progress.progress(success/total)
                status.text(f"Processed {success}/{total} ({success/total:.1%}) | Success: {success}")
            except Exception as e:
                print(f"Thread error: {futures[f]} | {e}")

    # Simpan ke session_state
    df_scan = pd.DataFrame(results)
    st.session_state["scan"] = df_scan

    # Update cache disk
    with open(CACHE_SCREENING, "wb") as f:
        pickle.dump(df_scan, f)

    st.success(f"Screening selesai: {success}/{total} saham berhasil diproses")


# =========================
# GUARD
# =========================
if "scan" not in st.session_state or st.session_state["scan"].empty:
    st.warning("Tidak ada saham yang berhasil diproses.")
    st.stop()

df = st.session_state["scan"]

# Pastikan semua kolom yang dibutuhkan ada
required_cols = [
    "Total_Score", "Score_Dasar", "Candle_Effect",
    "RSI", "Stoch_K", "Dist_to_SMA50",
    "Latest_Candle",
    "MajorTrend", "MinorPhase", "SetupState",
    "VOL_STATE", "FinalDecision"
]

for col in required_cols:
    if col not in df.columns:
        df[col] = 0

# Sort aman
df = df.sort_values("Total_Score", ascending=False).reset_index(drop=True)
# st.write("Jumlah saham yang berhasil diproses:", len(df))
# st.write(df.head())


# =========================
# FILTER
# =========================
st.subheader("Filter Hasil Screening")
cols = st.columns(5)
filters = {
    "MajorTrend": cols[0].multiselect("MajorTrend", sorted(df["MajorTrend"].unique())),
    "MinorPhase": cols[1].multiselect("MinorPhase", sorted(df["MinorPhase"].unique())),
    "SetupState": cols[2].multiselect("SetupState", sorted(df["SetupState"].unique())),
    "VOL_STATE": cols[3].multiselect("VOL_STATE", sorted(df["VOL_STATE"].unique())),
    "FinalDecision": cols[4].multiselect("FinalDecision", sorted(df["FinalDecision"].unique())),
}

for col, val in filters.items():
    if val:
        df = df[df[col].isin(val)]

st.subheader("📋 Screening Result")
event = st.dataframe(df, use_container_width=True, selection_mode="single-row", on_select="rerun")

# =========================
# AUTO-UPDATE CHART & METRICS
# =========================
if event.selection.rows:
    row = df.iloc[event.selection.rows[0]]
    kode = row["Kode"]

    # ambil data dengan cache engine
    dfc = fetch_data(f"{kode}.JK", "1d", "12mo")
    if dfc is not None and not dfc.empty:
        dfc = add_indicators(dfc)
        st.subheader(f"📈 {kode} | Close: {dfc['Close'].iloc[-1]:.0f}")

        fig = go.Figure()
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=dfc.index,
            open=dfc['Open'],
            high=dfc['High'],
            low=dfc['Low'],
            close=dfc['Close'],
            name="Candle"
        ))
        # EMA lines
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc["EMA13"], mode='lines', line=dict(color='blue'), name='EMA13'))
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc["EMA21"], mode='lines', line=dict(color='orange'), name='EMA21'))
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc["EMA50"], mode='lines', line=dict(color='red'), name='EMA50'))
        # Volume
        fig.add_trace(go.Bar(x=dfc.index, y=dfc["Volume"], name="Volume", marker_color='grey', yaxis='y2'))

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False, position=1.0),
            legend=dict(x=0, y=1.1, orientation="h")
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # TEKNIKAL METRICS
        # =========================
        metrics = {
            "RSI": row["RSI"],
            "Stochastic %K": row["Stoch_K"],
            "Distance to SMA50 (%)": row["Dist_to_SMA50"],
            "SMA50": row["SMA50"],
            "Latest Candle": row["Latest_Candle"],
            "Score Dasar": row["Score_Dasar"],
            "Candle Effect": row["Candle_Effect"],
            "Total Score": row["Total_Score"]
        }
        st.table(pd.DataFrame(metrics.items(), columns=["Metric","Value"]))
