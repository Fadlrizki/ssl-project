import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="ARA & Weekly Scanner",
    layout="wide"
)

# ======================================================
# UTILITIES
# ======================================================
def scalar(x):
    return float(x.item()) if hasattr(x, "item") else float(x)

def f(x):
    try:
        return float(x)
    except:
        return np.nan

@st.cache_data(ttl=3600)
def download_data(ticker, period, interval="1d"):
    return yf.download(ticker, period=period, interval=interval, progress=False)

@st.cache_data(ttl=3600)
def load_tickers():
    df = pd.read_excel("daftar_saham.xlsx")
    df.columns = df.columns.str.strip()

    if "Kode" not in df.columns:
        st.error("Kolom 'Kode' tidak ditemukan di daftar_saham.xlsx")
        st.stop()

    df["yf_ticker"] = df["Kode"].astype(str).str.upper() + ".JK"
    return df

# ======================================================
# ===================== ARA LOGIC =======================
# ======================================================
def find_ara_candidates(ticker, lookback=60):
    df = download_data(ticker, f"{lookback}d")

    if df.empty or len(df) < 25:
        return None

    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    today = df.iloc[-1]

    open_p  = scalar(today["Open"])
    close_p = scalar(today["Close"])
    high_p  = scalar(today["High"])
    vol     = scalar(today["Volume"])
    avgvol  = scalar(today["AvgVol20"])

    if avgvol == 0 or pd.isna(avgvol):
        return None

    ret_1d = (close_p - open_p) / open_p

    if (
        0.05 <= ret_1d <= 0.09 and
        close_p >= 0.95 * high_p and
        vol >= 1.5 * avgvol and
        close_p > open_p
    ):
        return {
            "Ticker": ticker,
            "Return_%": round(ret_1d * 100, 2),
            "Volume_Ratio": round(vol / avgvol, 2),
            "Close_vs_High_%": round(close_p / high_p * 100, 2),
            "Close": round(close_p, 2)
        }
    return None


def continuation_rate(ticker, lookback=120):
    df = download_data(ticker, f"{lookback}d")

    if df.empty or len(df) < 30:
        return None

    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    results = []

    for i in range(20, len(df) - 1):
        d = df.iloc[i]
        n = df.iloc[i + 1]

        open_p  = scalar(d["Open"])
        close_p = scalar(d["Close"])
        high_p  = scalar(d["High"])
        vol     = scalar(d["Volume"])
        avgvol  = scalar(d["AvgVol20"])

        if avgvol == 0 or pd.isna(avgvol):
            continue

        ret_1d = (close_p - open_p) / open_p

        if (
            0.05 <= ret_1d <= 0.09 and
            close_p >= 0.95 * high_p and
            vol >= 1.5 * avgvol
        ):
            next_ret = (
                scalar(n["Close"]) - scalar(n["Open"])
            ) / scalar(n["Open"])

            results.append(next_ret >= 0.03)

    if not results:
        return None

    return {
        "Ticker": ticker,
        "Sample": len(results),
        "Continuation_%": round(sum(results) / len(results) * 100, 1)
    }


def show_raw_data(ticker, days=15):
    df = download_data(ticker, f"{days}d")

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Return_%"] = (df["Close"] - df["Open"]) / df["Open"] * 100
    df["Close_vs_High_%"] = df["Close"] / df["High"] * 100
    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["AvgVol20"].replace(0, pd.NA)

    df = df.sort_index(ascending=False)
    return df.round(2)

# ======================================================
# ================= WEEKLY SCREENER ====================
# ======================================================
EMA_FAST = 13
EMA_MID = 21
EMA_SLOW = 50
RSI_MIN = 30
RSI_MAX = 45
SWING_LOOKBACK = 30

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def fibonacci_levels(df, lookback=30):
    recent = df.tail(lookback)
    swing_low = recent["Low"].min()
    swing_high = recent["High"].max()

    buy_low = swing_high - 0.618 * (swing_high - swing_low)
    buy_high = swing_high - 0.382 * (swing_high - swing_low)

    tp1 = swing_high + 0.272 * (swing_high - swing_low)
    tp2 = swing_high + 0.618 * (swing_high - swing_low)

    return swing_low, swing_high, buy_low, buy_high, tp1, tp2


def weekly_summary(stocks):
    results = []

    for stock in stocks:
        df = download_data(f"{stock}.JK", "6mo")

        if df.empty or len(df) < 60:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        df["EMA13"] = ema(df["Close"], EMA_FAST)
        df["EMA21"] = ema(df["Close"], EMA_MID)
        df["EMA50"] = ema(df["Close"], EMA_SLOW)
        df["RSI"] = rsi(df["Close"])

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = f(latest["Close"])
        ema13 = f(latest["EMA13"])
        ema21 = f(latest["EMA21"])
        ema50 = f(latest["EMA50"])
        rsi_v = f(latest["RSI"])

        trend_ok = close > ema13 > ema21 > ema50
        rsi_ok = RSI_MIN <= rsi_v <= RSI_MAX and rsi_v > f(prev["RSI"])

        score = sum([trend_ok, rsi_ok]) * 50
        if score < 50:
            continue

        swing_low, swing_high, buy_low, buy_high, tp1, tp2 = fibonacci_levels(df)

        results.append({
            "Stock": stock,
            "Price": round(close, 2),
            "RSI": round(rsi_v, 1),
            "Score": score,
            "Buy_Low": round(buy_low, 2),
            "Buy_High": round(buy_high, 2),
            "TP1": round(tp1, 2),
            "TP2": round(tp2, 2)
        })

    return pd.DataFrame(results).sort_values("Score", ascending=False)

# ======================================================
# ====================== UI =============================
# ======================================================
menu = st.sidebar.radio(
    "📊 Menu",
    ["ARA Scanner", "Weekly Summary"]
)

# ====================== ARA ============================
if menu == "ARA Scanner":
    st.title("📈 ARA Scanner Indonesia")

    df_tickers = load_tickers()

    selected_code = st.selectbox(
        "Pilih Kode Saham",
        df_tickers["Kode"]
    )

    ticker = df_tickers.loc[
        df_tickers["Kode"] == selected_code,
        "yf_ticker"
    ].iloc[0]

    if st.button("🔍 Analyze"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ARA Candidate")
            ara = find_ara_candidates(ticker)
            if ara:
                st.dataframe(pd.DataFrame([ara]))
            else:
                st.info("Tidak memenuhi")


        with col2:
            st.subheader("Continuation Rate")
            cont = continuation_rate(ticker)
            if cont:
                st.dataframe(pd.DataFrame([cont]))
            else:
                st.info("Data tidak cukup")

        st.subheader("Raw Data (15 Hari)")
        raw = show_raw_data(ticker)
        if raw is not None:
            st.dataframe(raw)

# =================== WEEKLY ============================
if menu == "Weekly Summary":
    st.title("📊 Weekly Rally Summary")

    STOCKS = [
        "YUPI","ISAT","BREN","SMDM","SILO","STAR","AVIA","INTP","CNMA","TLKM",
        "BOGA","ADMF","TALF","KLBF","BDMN","BNII","BNGA","MMLP","JPFA","PWON",
        "BELL","BBHI","KPIG","SMAR","BBSI","SRAJ","ASII","MTEL","BPTR","NOBU",
        "INDF","UNTR","SONA","ERTX","BACA","TRJA","PTRO","FORU","BMRI","PZZA",
        "SDPC","YPAS","TARA","JSMR","GTSI","INKP","BABP","MBAP","PGEO","NASA",
        "POLA","BRMS","KKGI","IPTV","MAIN","MBMA","RIGS","BUVA","HRME","CITY",
        "BBCA","ESTI","ABMM","CTRA","IBFN","TPIA","BBNI","SMGR","MITI","BBRI",
        "AISA","MARK","ENAK","PNSE","MAXI","IPOL","PANI","INDY","MGLV","BBYB",
        "SULI","JKON","TOOL","MTFN","IPCM","PKPK","DRMA","MPMX","APLN","JGLE",
        "CDIA","BHIT"
    ]

    if st.button("🚀 Run Weekly Scan"):
        with st.spinner("Scanning market..."):
            df_weekly = weekly_summary(STOCKS)

        st.dataframe(df_weekly, use_container_width=True)
