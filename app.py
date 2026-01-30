import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
from engine import build_probability_table_from_ticker,backtest
from engine_v2 import process_stock, fetch_data, add_indicators
from datetime import datetime, timedelta

# ======================================================
# CONFIG
# ======================================================

CACHE_VERSION = "v3"
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
# PAGE
# ======================================================
st.set_page_config(layout="wide")
st.title("📊 IDX Price Action Screener V2")
st.caption("Daily trend • Minor phase • Volume behavior")
# dfc = fetch_data("UFOE.JK", "1d", "12mo", force_refresh=True)
# dfc2 = fetch_intraday_safe("UFOE.JK", "1h", "6mo")
# # dfc2 = fetch_intraday_safe("UFOE.JK")
# st.write("UFOE(DEBUG)")
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

def load_backtest_cache():
    if os.path.exists(BACKTEST_CACHE):
        try:
            return pickle.load(open(BACKTEST_CACHE, "rb"))
        except Exception:
            return {}
    return {}

def save_backtest_cache(cache):
    with open(BACKTEST_CACHE, "wb") as f:
        pickle.dump(cache, f)

def run_backtest_cached(kode):
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    key = f"{kode}_{today}"

    cache = load_backtest_cache()

    if key in cache:
        return cache[key]

    result = backtest(f"{kode}.JK", mode="decision")

    cache[key] = result
    save_backtest_cache(cache)

    return result

def render_technical_chart(df, kode):
    """
    Panel 1 : Candlestick + EMA
    Panel 2 : RSI
    Panel 3 : Volume + Volume MA20
    """

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.2, 0.25],
        subplot_titles=(
            f"{kode} - Price",
            "RSI (14)",
            "Volume"
        )
    )


    # =========================
    # PRICE - CANDLESTICK
    # =========================
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ),
        row=1, col=1
    )

    # EMA
    for ema in ["EMA13", "EMA21", "EMA50"]:
        if ema in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ema],
                    mode="lines",
                    name=ema
                ),
                row=1, col=1
            )

    # =========================
    # RSI
    # =========================
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(width=1.5)
            ),
            row=2, col=1
        )

        # RSI level
        fig.add_hline(y=70, line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", row=2, col=1)

    # =========================
    # VOLUME
    # =========================
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume"
        ),
        row=3, col=1
    )

    # Volume MA20
    if "VOL_MA20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["VOL_MA20"],
                mode="lines",
                name="Vol MA20"
            ),
            row=3, col=1
        )

    # =========================
    # LAYOUT
    # =========================
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def retry_single_stock(kode):
    if os.path.exists(CACHE_SCREENING):
        df = load_cache_safe(CACHE_SCREENING)
        df = df[df["Kode"] != kode]
        save_cache(df, CACHE_SCREENING)

    return process_stock(kode)

# ======================================================
# HELPER BROKSUM
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
    dt = datetime.strptime(trade_date, "%Y-%m-%d")
    for i in range(max_back + 1):
        check_date = (dt - timedelta(days=i)).strftime("%Y-%m-%d")
        path = f"broksum/broker_summary-{check_date}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, check_date
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
        st.stop()
    elif used_date != trade_date:
        st.info(f"ℹ️ {name} {trade_date} belum tersedia, pakai data {used_date}")
    else:
        st.success(f"✅ {name} {trade_date} sudah update")


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

# ======================================================
# RUN SCREENING
# ======================================================
if st.button("🗑️ Clear All Cache", key="btn_clear_cache"):
    clear_cache()
    st.success("Cache berhasil dihapus. Silakan jalankan screening ulang.")

if st.button("🚀 Run Screening", key="btn_run_screening"):
    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_stock, k): k for k in codes}
        done = 0
        total = len(codes)

        for f in as_completed(futures):
            kode = futures[f]
            try:
                r = f.result()
                if r and "Kode" in r and "Price" in r:
                    # tambahkan timestamp supaya unik
                    r["ProcessTime"] = pd.Timestamp.now()
                    results.append(r)
            except Exception as e:
                print(f"Error {kode}: {e}")

            done += 1
            progress.progress(done / total)
            status.text(f"Processed {done}/{total}")

    df_new = pd.DataFrame(results)

    # gabungkan cache + hasil baru
    if not cached_df.empty:
        df_scan = pd.concat([cached_df, df_new], ignore_index=True)
    else:
        df_scan = df_new.copy()

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

    # =========================
    # RETRY BUTTON
    # =========================
    col_retry1, col_retry2 = st.columns([1, 3])

    with col_retry1:
        if st.button(f"🔁 Retry Fetch {kode}", key=f"btn_retry_single_{kode}"):
            with st.spinner(f"Retry {kode}..."):
                r = retry_single_stock(kode)

                if r:
                    df = pd.concat(
                        [df[df["Kode"] != kode], pd.DataFrame([r])],
                        ignore_index=True
                    )
                    save_cache(df, CACHE_SCREENING)
                    st.session_state["scan"] = df
                    st.success("Retry berhasil")
                    st.rerun()
                else:
                    st.error("Retry gagal (data masih kosong)")

    # =========================
    # CHART
    # =========================
    st.subheader(f"📈 Chart {kode}")

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
            render_technical_chart(df_daily, kode)

    except Exception as e:
        st.error(f"Gagal render chart: {e}")

    # =========================
    # METRIC TABLE
    # =========================
    st.subheader("📊 Technical Summary")

    st.table(pd.DataFrame({
        "Metric": [
            "Minor Phase",
            "Confidence",
            "RSI",
            "Stoch %K",
            "Volume Behavior",
            "Volume Ratio",
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

if event.selection.rows:
    kode = df.iloc[event.selection.rows[0]]["Kode"]

    if st.button(
        "Run Backtest",
        key=f"run_backtest_{kode}"
    ):
        with st.spinner(f"Running backtest {kode}..."):
            st.session_state["backtest_result"] = backtest(
                f"{kode}.JK",
                mode="decision"
            )
            st.session_state["prob_table"] = build_probability_table_from_ticker(
                f"{kode}.JK"
            )

else:
    st.info("Pilih 1 emiten untuk menjalankan backtest")

# tampilkan hasil backtest jika ada
if "backtest_result" in st.session_state:
    result = st.session_state["backtest_result"]

    st.subheader("🔮 Prediksi Market Besok")

    if not result:
        st.warning("⚠️ Tidak ada hasil backtest")
    else:
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
        ctx = result["DecisionContext"]

        st.subheader("🧠 Konteks Market")

        ctx_df = pd.DataFrame(
            {
                "Value": [
                    ctx.get("MajorTrend", "-"),
                    ctx.get("MinorPhase", "-"),
                    ctx.get("RSI_BUCKET", "-"),
                    ctx.get("VOL_BEHAVIOR", "-"),
                    ctx.get("latest_candle", "-"),
                    ctx.get("AvgVolRatio", "-"),
                    ctx.get("MatchType", "-"),
                ]
            },
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
    st.subheader("📊 Probability Table (180 Hari)")

    prob_table = st.session_state["prob_table"]

    if prob_table is None or prob_table.empty:
        st.warning("Tidak ada data STRONG MajorTrend dalam 180 hari")
    else:
        st.dataframe(prob_table, use_container_width=True)


# if "backtest_strategy" in st.session_state and st.session_state["backtest_strategy"] is not None:
#     st.subheader("📈 Strategy Backtest (Historical)")
#     df_strategy = st.session_state["backtest_strategy"]
#     st.dataframe(df_strategy.sort_index(ascending=False), use_container_width=True)


# ======================================================
# TRIGGER SCREENING (COMPUTE & SAVE)
# ======================================================
if st.button("🔔 Run Trigger Screening", key="btn_trigger_screening"):
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
                    "ProbHijau": result.get("ProbHijau"),
                    "ProbMerah": result.get("ProbMerah"),
                    "Sample": result.get("Sample"),
                    "Confidence": result.get("Confidence"),
                    "MatchType": result.get("DecisionContext", {}).get("MatchType")
                })

        except Exception:
            pass

        progress.progress(i / total)
        status.text(f"Trigger screening {i}/{total} emiten STRONG")

    df_trigger = pd.DataFrame(hijau_results)

    save_trigger_cache(df_trigger, TODAY)

    st.success(f"Trigger screening selesai ({len(df_trigger)} emiten)")
    st.rerun()

# ======================================================
# TRIGGER RESULT VIEW (READ FROM CACHE)
# ======================================================
TRADE_DATE = TODAY

df_trigger, trigger_used_date = load_trigger_cache_pickle(TRADE_DATE)
st.subheader("🌱 Emiten STRONG dengan Bias HIJAU")
show_status("Trigger screening", TRADE_DATE, trigger_used_date, df_trigger)

st.dataframe(df_trigger, use_container_width=True)

# ======================================================
# BROKER SUMMARY ENRICHMENT
# ======================================================
df_broker, broker_used_date = load_broker_summary(TRADE_DATE)
show_status("Broker summary", TRADE_DATE, broker_used_date, df_broker)

df_final = df_trigger.merge(
    df_broker,
    left_on="Kode",
    right_on="stock",
    how="left"
)

# st.subheader("📊 Trigger + Broker Summary")
# st.dataframe(df_final, use_container_width=True)


st.subheader("📊 Trigger + Broker Summary")

event_trigger = st.dataframe(
    df_final,
    use_container_width=True,
    selection_mode="single-row",
    on_select="rerun"
)

# kalau ada baris dipilih
if event_trigger.selection.rows:
    row = df_final.iloc[event_trigger.selection.rows[0]]
    kode = row["Kode"]

    st.subheader(f"📈 Chart {kode}")

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
            render_technical_chart(df_daily, kode)
    except Exception as e:
        st.error(f"Gagal render chart: {e}")

    # detail summary table
    st.subheader("📊 Technical Summary")
    st.table(pd.DataFrame({
        "Metric": [
            "Major Trend",
            "Minor Phase",
            "Prob Hijau",
            "Prob Merah",
            "Sample",
            "Confidence",
            "MatchType"
        ],
        "Value": [
            row["MajorTrend"],
            row["MinorPhase"],
            row["ProbHijau"],
            row["ProbMerah"],
            row["Sample"],
            row["Confidence"],
            row["MatchType"]
        ]
    }))
