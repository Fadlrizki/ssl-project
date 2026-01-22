import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
EXCEL_FILE = "daftar_saham.xlsx"
KODE_COLUMN = "Kode"

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

EMA_FAST = 13
EMA_MID  = 21
EMA_SLOW = 50

LOOKBACK_HL_4H = 720
SLOPE_WINDOW = 10
EMA_COMPRESS_TH = 0.003

MAX_WORKERS = 8

# ======================================================
# PAGE
# ======================================================
st.set_page_config(layout="wide")
st.title("📊 IDX Price Action Screener")
st.caption("Daily trend • 4H pullback • Volume confirmation")

# ======================================================
# CACHE
# ======================================================
def cache_path(ticker, interval):
    return os.path.join(CACHE_DIR, f"{ticker}_{interval}.parquet")

def load_cache(ticker, interval):
    p = cache_path(ticker, interval)
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None

def save_cache(df, ticker, interval):
    df.to_parquet(cache_path(ticker, interval))

# ======================================================
# FETCH DATA
# ======================================================
def fetch_data(ticker, interval, period):
    cached = load_cache(ticker, interval)
    if cached is not None and len(cached) > 50:
        return cached.copy()

    try:
        df = yf.download(
            ticker,
            interval=interval,
            period=period,
            progress=False,
            threads=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open","High","Low","Close","Volume"]].dropna()
        if len(df) < 50 or df["Volume"].tail(5).sum() == 0:
            return None

        save_cache(df, ticker, interval)
        return df
    except:
        return None

# ======================================================
# INDICATORS
# ======================================================
def add_indicators(df):
    df = df.copy()
    df["EMA13"] = df["Close"].ewm(span=EMA_FAST).mean()
    df["EMA21"] = df["Close"].ewm(span=EMA_MID).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_SLOW).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    return df

def ema_slope(s):
    return s.iloc[-1] - s.iloc[-SLOPE_WINDOW] if len(s) > SLOPE_WINDOW else 0

# ======================================================
# ENGINE
# ======================================================
def major_trend_daily(df):
    if df.iloc[-1]["EMA21"] <= df.iloc[-1]["EMA50"]:
        return "INVALID"
    if ema_slope(df["EMA21"]) <= 0:
        return "INVALID"
    return "STRONG"

def minor_phase_4h(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    compress = abs(last["EMA13"] - last["EMA21"]) / last["EMA21"] < EMA_COMPRESS_TH

    if last["Close"] < df["Low"].tail(LOOKBACK_HL_4H).min():
        return "BREAKDOWN"
    if compress:
        return "EMA_COMPRESS_PULLBACK"
    if last["Close"] > last["EMA21"] and prev["Close"] < prev["EMA21"]:
        return "PULLBACK_RECOVERED"
    if last["Close"] > last["EMA21"]:
        return "TREND_CONTINUE"
    return "NEUTRAL"

def setup_state(minor):
    if minor in ["PULLBACK_RECOVERED","TREND_CONTINUE"]:
        return "STAGE2_READY"
    if minor == "EMA_COMPRESS_PULLBACK":
        return "SETUP_PENDING"
    return "WAIT"

def stage2_trigger(df, setup):
    if setup != "STAGE2_READY":
        return False
    return df.iloc[-1]["Close"] > df.iloc[-1]["EMA13"]

def volume_state(df):
    last = df.iloc[-1]
    ma20 = df["VOL_MA20"].iloc[-1]
    r = round(last["Volume"] / ma20, 2) if ma20 > 0 else 0
    state = "DRYING" if r < 0.7 else "EXPANSION" if r > 1.2 else "NORMAL"
    return r, state

def final_decision(major, minor, setup, stage2, vol):
    if major == "INVALID" or minor == "BREAKDOWN":
        return "SKIP"
    if setup == "SETUP_PENDING":
        return "SETUP_PENDING"
    if major == "STRONG" and setup == "STAGE2_READY" and stage2 and vol != "DRYING":
        return "ENTRY_READY"
    return "WAIT"

# ======================================================
# WORKER
# ======================================================
def process_stock(kode):
    ticker = f"{kode}.JK"

    d1 = fetch_data(ticker, "1d", "12mo")
    h4 = fetch_data(ticker, "4h", "12mo")
    if d1 is None or h4 is None:
        return None

    d1 = add_indicators(d1)
    h4 = add_indicators(h4)

    # =========================
    # PRICE
    # =========================
    price_today = d1["Close"].iloc[-1]
    price_yesterday = d1["Close"].iloc[-2]
    price_change_pct = round((price_today / price_yesterday - 1) * 100, 2)

    # =========================
    # GAP EMA
    # =========================
    ema21 = d1["EMA21"].iloc[-1]
    ema50 = d1["EMA50"].iloc[-1]

    gap_ema21 = round((price_today / ema21 - 1) * 100, 2)
    gap_ema50 = round((price_today / ema50 - 1) * 100, 2)

    # =========================
    # VOLUME
    # =========================
    vol_today = d1["Volume"].iloc[-1]
    vol_yesterday = d1["Volume"].iloc[-2]
    vol_ma20 = d1["VOL_MA20"].iloc[-1]

    vol_ratio_ma20 = round(vol_today / vol_ma20, 2) if vol_ma20 > 0 else 0
    vol_change_d1 = round((vol_today / vol_yesterday - 1) * 100, 2)

    if vol_ratio_ma20 < 0.7:
        vol_state = "DRYING"
    elif vol_ratio_ma20 > 1.2:
        vol_state = "EXPANSION"
    else:
        vol_state = "NORMAL"

    # =========================
    # TREND ENGINE
    # =========================
    major = major_trend_daily(d1)
    minor = minor_phase_4h(h4)
    setup = setup_state(minor)
    stage2 = stage2_trigger(h4, setup)

    final = final_decision(
        major, minor, setup, stage2, vol_state
    )

    # =========================
    # RETURN (SEMUA KOLUMN)
    # =========================
    return {
        "Kode": kode,
        "Price": price_today,
        "PriceChange%": price_change_pct,
        "Gap_EMA21%": gap_ema21,
        "Gap_EMA50%": gap_ema50,
        "MajorTrend": major,
        "MinorPhase": minor,
        "SetupState": setup,
        "Stage2Valid": stage2,
        "VOL_TODAY": vol_today,
        "VOL_YESTERDAY": vol_yesterday,
        "VOL_MA20": vol_ma20,
        "VOL_RATIO_MA20": vol_ratio_ma20,
        "VOL_CHANGE_D1": vol_change_d1,
        "VOL_STATE": vol_state,
        "FinalDecision": final
    }


# ======================================================
# RUN
# ======================================================
if st.button("🚀 Run Screening"):
    saham = pd.read_excel(EXCEL_FILE)
    res = []
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        for f in as_completed([ex.submit(process_stock,k) for k in saham[KODE_COLUMN]]):
            if f.result():
                res.append(f.result())
    st.session_state["scan"] = pd.DataFrame(res)

if "scan" not in st.session_state:
    st.stop()

df = st.session_state["scan"]

# ======================================================
# FILTER UI
# ======================================================
st.subheader("🎛️ Filter")

r1 = st.columns(3)
r2 = st.columns(3)

f_major = r1[0].multiselect("MajorTrend", sorted(df["MajorTrend"].unique()))
f_minor = r1[1].multiselect("MinorPhase", sorted(df["MinorPhase"].unique()))
f_final = r1[2].multiselect("FinalDecision", sorted(df["FinalDecision"].unique()))

f_setup = r2[0].multiselect("SetupState", sorted(df["SetupState"].unique()))
f_vol   = r2[1].multiselect("VOL_STATE", sorted(df["VOL_STATE"].unique()))
f_stage = r2[2].multiselect("Stage2Valid", sorted(df["Stage2Valid"].unique()))

for col, val in {
    "MajorTrend": f_major,
    "MinorPhase": f_minor,
    "FinalDecision": f_final,
    "SetupState": f_setup,
    "VOL_STATE": f_vol,
    "Stage2Valid": f_stage
}.items():
    if val:
        df = df[df[col].isin(val)]

# ======================================================
# TABLE
# ======================================================
st.subheader("📋 Screening Result")
st.info("👉 Klik salah satu baris untuk menampilkan chart")

event = st.dataframe(
    df,
    use_container_width=True,
    selection_mode="single-row",
    on_select="rerun"
)

# ======================================================
# CHART
# ======================================================
if event.selection.rows:
    kode = df.iloc[event.selection.rows[0]]["Kode"]
    dfc = add_indicators(fetch_data(f"{kode}.JK","1d","6mo"))

    st.subheader(f"📈 {kode} | Close: {dfc['Close'].iloc[-1]:.0f}")

    ap = [
        mpf.make_addplot(dfc["EMA13"], color="blue"),
        mpf.make_addplot(dfc["EMA21"], color="orange"),
        mpf.make_addplot(dfc["EMA50"], color="red")
    ]

    fig,_ = mpf.plot(
        dfc, type="candle", volume=True,
        addplot=ap, style="yahoo",
        returnfig=True, figsize=(12,6)
    )
    st.pyplot(fig)
