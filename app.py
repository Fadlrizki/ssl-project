import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config("IDX Scanner", layout="wide")

# ======================
# DATA
# ======================
@st.cache_data(ttl=3600)
def fetch_data(ticker, period):
    df = yf.download(f"{ticker}.JK", period=period, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# ======================
# INDICATORS
# ======================
ema = lambda s, n: s.ewm(span=n, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    g, l = d.clip(lower=0), -d.clip(upper=0)
    rs = g.rolling(p).mean() / (l.rolling(p).mean() + 1e-9)
    return 100 - 100 / (1 + rs)

def fibo(df, n=30):
    r = df.tail(n)
    lo, hi = r.Low.min(), r.High.max()
    return {
        "Buy_Low": hi - 0.618 * (hi - lo),
        "Buy_High": hi - 0.382 * (hi - lo),
        "TP1": hi + 0.272 * (hi - lo),
        "TP2": hi + 0.618 * (hi - lo),
    }

# ======================
# PAGE: RALLY FIBO (HOME)
# ======================
def page_rally_fibo():
    st.title("📈 Broker Summary")

    
    STOCKS = ["YUPI","ISAT","BREN","SMDM","SILO","STAR","AVIA","INTP","CNMA","TLKM",
    "BOGA","ADMF","TALF","KLBF","BDMN","BNII","BNGA","MMLP","JPFA","PWON",
    "BELL","BBHI","KPIG","SMAR","BBSI","SRAJ","ASII","MTEL","BPTR","NOBU",
    "INDF","UNTR","SONA","ERTX","BACA","TRJA","PTRO","FORU","BMRI","PZZA",
    "SDPC","YPAS","TARA","JSMR","GTSI","INKP","BABP","MBAP","PGEO","NASA",
    "POLA","BRMS","KKGI","IPTV","MAIN","MBMA","RIGS","BUVA","HRME","CITY",
    "BBCA","ESTI","ABMM","CTRA","IBFN","TPIA","BBNI","SMGR","MITI","BBRI",
    "AISA","MARK","ENAK","PNSE","MAXI","IPOL","PANI","INDY","MGLV","BBYB",
    "SULI","JKON","TOOL","MTFN","IPCM","PKPK","DRMA","MPMX","APLN","JGLE",
    "CDIA","BHIT"]
    out = []

    if st.button("🔍 Scan Rally Fibo"):
        for s in STOCKS:
            df = fetch_data(s, "6mo")
            if df is None or len(df) < 60:
                continue

            df["EMA13"], df["EMA21"], df["EMA50"] = ema(df.Close,13), ema(df.Close,21), ema(df.Close,50)
            df["RSI"] = rsi(df.Close)

            c, p = df.iloc[-1], df.iloc[-2]

            if (
                c.Close > c.EMA13 > c.EMA21 > c.EMA50 and
                35 <= c.RSI <= 55 and c.RSI > p.RSI
            ):
                out.append({
                    "Stock": s,
                    "Close": round(c.Close,2),
                    "RSI": round(c.RSI,1),
                    **{k: round(v,2) for k,v in fibo(df).items()}
                })

        if out:
            st.dataframe(pd.DataFrame(out), use_container_width=True)
        else:
            st.warning("Tidak ada kandidat")

# ======================
# PAGE: ARA SCANNER
# ======================
def page_ara():
    st.title("⚡ ARA Scanner")

    ticker = st.selectbox("Kode Saham", ["GTSI","BUVA","BACA","BRMS","BBCA"])
    df = fetch_data(ticker, "60d")

    if df is None or len(df) < 25:
        st.warning("Data tidak cukup")
        return

    df["AvgVol20"] = df.Volume.rolling(20).mean()
    t = df.iloc[-1]

    ret = (t.Close - t.Open) / t.Open

    if (
        0.05 <= ret <= 0.09 and
        t.Close >= 0.95 * t.High and
        t.Volume >= 1.5 * t.AvgVol20
    ):
        st.success("✅ Kandidat ARA")
        st.metric("Return %", round(ret*100,2))
        st.metric("Volume Ratio", round(t.Volume / t.AvgVol20,2))
    else:
        st.info("❌ Tidak memenuhi ARA")

    st.subheader("Raw Data (15 hari)")
    st.dataframe(df.tail(15).sort_index(ascending=False))

# ======================
# SIDEBAR ROUTER
# ======================
st.sidebar.title("📊 Menu")

if "page" not in st.session_state:
    st.session_state.page = "Broker Summary"

if st.sidebar.button("📈 Broker Summary"):
    st.session_state.page = "Broker Summary"

if st.sidebar.button("⚡ ARA Scanner"):
    st.session_state.page = "ARA"

# ======================
# PAGE ROUTING
# ======================
if st.session_state.page == "Broker Summary":
    page_rally_fibo()
else:
    page_ara()
