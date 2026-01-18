import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(
    page_title="ARA Scanner",
    layout="wide"
)

# =========================
# Utilities
# =========================
def scalar(x):
    return float(x.item()) if hasattr(x, "item") else float(x)

@st.cache_data(ttl=3600)
def load_tickers():
    df = pd.read_excel("daftar_saham.xlsx")
    df.columns = df.columns.str.strip()

    if "Kode" not in df.columns:
        st.error("Kolom 'Kode' tidak ditemukan di daftar_saham.xlsx")
        st.stop()

    # pastikan format Yahoo Finance
    df["yf_ticker"] = df["Kode"].astype(str).str.upper() + ".JK"

    return df

@st.cache_data(ttl=3600)
def download_data(ticker, period):
    return yf.download(ticker, period=period, progress=False)

# =========================
# Core Logic
# =========================
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
            "Close": close_p
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
    df["Range_%"] = (df["High"] - df["Low"]) / df["Open"] * 100
    df["Close_vs_High_%"] = df["Close"] / df["High"] * 100
    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["AvgVol20"].replace(0, pd.NA)

    df = df.sort_index(ascending=False)
    return df.round(2)

# =========================
# UI
# =========================
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

st.caption(f"📡 Fetching data: {ticker}")

if st.button("🔍 Analyze"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ARA Candidate (Last Day)")
        ara = find_ara_candidates(ticker)
        if ara:
            st.dataframe(pd.DataFrame([ara]))
        else:
            st.info("Tidak memenuhi kriteria ARA")

    with col2:
        st.subheader("Continuation Rate")
        cont = continuation_rate(ticker)
        if cont:
            st.dataframe(pd.DataFrame([cont]))
        else:
            st.info("Data tidak cukup")

    st.subheader("Raw Price Data (15 Hari)")
    raw = show_raw_data(ticker)
    if raw is not None:
        cols = [
            "Open", "High", "Low", "Close",
            "Volume", "Return_%", "Close_vs_High_%", "Volume_Ratio"
        ]
        st.dataframe(raw[cols])
