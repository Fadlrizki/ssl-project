import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
from engine import build_probability_table_from_ticker,backtest
from engine_v2 import process_stock, fetch_data, add_indicators

# ======================================================
# CONFIG
# ======================================================

CACHE_VERSION = "v3"
CACHE_SCREENING = f"screening_cache_{CACHE_VERSION}.pkl"
TRIGGER_CACHE = f"trigger_cache_{CACHE_VERSION}.pkl"
PROB_CACHE = f"prob_cache_{CACHE_VERSION}.pkl"   


EXCEL_FILE = "daftar_saham.xlsx"
KODE_COLUMN = "Kode"
MAX_WORKERS = 8

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
# dfc = fetch_data("CTTH.JK", "1d", "12mo", force_refresh=True)
# dfc2 = fetch_data("CTTH.JK", "4h", "6mo", force_refresh=True)
# st.write("CTTH(DEBUG)")
# st.write(dfc.tail(1).sort_index(ascending=False))
# st.write("LAST DATE:", dfc.index[-1])
# st.write(dfc2.tail(1).sort_index(ascending=False))
# st.write("LAST DATE:", dfc2.index[-1])


# ======================================================
# HELPERS
# ======================================================

def clear_cache():
    # hapus file cache screening
    if os.path.exists(CACHE_SCREENING):
        os.remove(CACHE_SCREENING)
    # hapus file cache trigger
    if os.path.exists(TRIGGER_CACHE):
        os.remove(TRIGGER_CACHE)
    # hapus file cache probability
    if os.path.exists(PROB_CACHE):
        os.remove(PROB_CACHE)

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

def save_trigger_cache(df):
    with open(TRIGGER_CACHE, "wb") as f:
        pickle.dump(df, f)

def run_trigger_screening(codes, use_cache=True):
    if use_cache:
        cached = load_trigger_cache()
        if not cached.empty:
            return cached

    hijau_results = []
    for kode in codes:
        try:
            result = backtest(f"{kode}.JK", mode="decision")
            if result and result.get("Bias") == "HIJAU":
                hijau_results.append({
                    "Kode": kode,
                    "ProbHijau": result.get("ProbHijau"),
                    "ProbMerah": result.get("ProbMerah"),
                    "Sample": result.get("Sample"),
                    "Confidence": result.get("Confidence"),
                    "MatchType": result.get("DecisionContext", {}).get("MatchType")
                })
        except Exception:
            continue

    df_hijau = pd.DataFrame(hijau_results)
    save_trigger_cache(df_hijau)
    return df_hijau

# ======================================================
# LOAD SAHAM
# ======================================================
saham_df = pd.read_excel(EXCEL_FILE)
codes = saham_df[KODE_COLUMN].dropna().unique().tolist()

cached_df = load_cache_safe(CACHE_SCREENING)

def load_trigger_cache():
    if os.path.exists(TRIGGER_CACHE):
        try:
            return pickle.load(open(TRIGGER_CACHE, "rb"))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_trigger_cache(df):
    with open(TRIGGER_CACHE, "wb") as f:
        pickle.dump(df, f)

# ======================================================
# RUN SCREENING
# ======================================================
if st.button("🗑️ Clear All Cache"):
    clear_cache()
    st.success("Cache berhasil dihapus. Silakan jalankan screening ulang.")

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
                # ✅ Validasi hasil sebelum append
                if r and "Kode" in r and "Price" in r:
                    results.append(r)
                    done += 1
            except Exception as e:
                print(f"Error {kode}: {e}")

            progress.progress(done / total)
            status.text(f"Processed {done}/{total}")

    # ✅ Gabungkan cache + hasil baru dengan aman
    df_new = pd.DataFrame(results)
    df_scan = pd.concat([cached_df, df_new], ignore_index=True)

    # ✅ Drop duplikat berdasarkan Kode, ambil yang paling baru
    df_scan = df_scan.drop_duplicates(subset=["Kode"], keep="last").reset_index(drop=True)

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
        st.caption(f"📅 Data terakhir: {dfc.index[-1].date()} | 💰 Close: {dfc['Close'].iloc[-1]}")
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
    st.session_state["backtest_result"] = backtest(f"{kode}.JK", mode="decision")
    # st.session_state["backtest_strategy"] = backtest(f"{kode}.JK", mode="strategy")
    st.session_state["prob_table"] = build_probability_table_from_ticker(f"{kode}.JK")

# tampilkan hasil backtest jika ada
if "backtest_result" in st.session_state:
    result = st.session_state["backtest_result"]
    st.subheader("🔮 Prediksi Market Besok")
    if result:
        bias = result.get("Bias")
        if bias in ["HIJAU", "MERAH"]:
            col1, col2, col3 = st.columns(3)
            col1.metric("Bias", bias)
            col2.metric("Prob Hijau", f"{result.get('ProbHijau', 0)}%")
            col3.metric("Prob Merah", f"{result.get('ProbMerah', 0)}%")
            st.caption(
                f"Sample historis: {result.get('Sample', '-')}"
                f" | Confidence: {result.get('Confidence', '-')}"
            )
        elif bias == "NO_MATCH":
            st.warning("⚠️ Tidak ditemukan kondisi historis relevan")
        elif bias == "NO_MODEL":
            st.warning("⚠️ Model probabilitas belum terbentuk")
        elif bias == "NO_SETUP":
            st.info("ℹ️ Tidak ada setup valid")
        else:
            st.json(result)

    if result and "DecisionContext" in result:
        st.subheader("🧠 Konteks Market")
        ctx = result["DecisionContext"]
        ctx_df = pd.DataFrame(
            {"Value": [
                ctx.get("MajorTrend"),
                ctx.get("MinorPhase"),
                ctx.get("RSI_BUCKET"),
                ctx.get("VOL_BEHAVIOR"),
                ctx.get("latest_candle"),
                ctx.get("AvgVolRatio"),
                ctx.get("MatchType"),
            ]},
            index=[
                "Major Trend",
                "Minor Phase",
                "RSI Bucket",
                "Volume Behavior",
                "Candle Terakhir",
                "Avg Volume Ratio",
                "Match Type",
            ]
        )
        st.table(ctx_df)

if "prob_table" in st.session_state:
    prob_table = st.session_state["prob_table"]
    st.subheader("📊 Probability Table (180 Hari)")
    if prob_table is None or prob_table.empty:
        st.warning("Tidak ada data STRONG MajorTrend dalam 180 hari")
    else:
        st.dataframe(prob_table, use_container_width=True)

# if "backtest_strategy" in st.session_state and st.session_state["backtest_strategy"] is not None:
#     st.subheader("📈 Strategy Backtest (Historical)")
#     df_strategy = st.session_state["backtest_strategy"]
#     st.dataframe(df_strategy.sort_index(ascending=False), use_container_width=True)


# ======================================================
# TRIGGER SCREENING
# ======================================================
if st.button("🔔 Run Trigger Screening(WARNING!!!)"):
    df_hijau = run_trigger_screening(codes, use_cache=False)  # paksa refresh
    st.session_state["trigger_result"] = df_hijau

if "trigger_result" in st.session_state:
    df_hijau = st.session_state["trigger_result"]
    st.subheader("🌱 Emiten dengan Bias HIJAU untuk Besok")
    if not df_hijau.empty:
        st.dataframe(df_hijau, use_container_width=True)
    else:
        st.info("Tidak ada emiten dengan Bias HIJAU untuk besok.")
