import pandas as pd
import numpy as np
import yfinance as yf
import os

# =========================
# CONFIG
# =========================
EMA_FAST = 13
EMA_MID = 21
EMA_SLOW = 50
SMA50_PERIOD = 50
VOL_MA_PERIOD = 20
RSI_PERIOD = 14
STOCH_PERIOD = 14
SLOPE_WINDOW = 10
EMA_COMPRESS_TH = 0.003
BODY_THRESHOLD = 0.02
CANDLE_DIST_TH = 5
TOLERANCE = 0.02  # untuk gap EMA
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# CACHE FUNCTIONS
# =========================
def cache_path(ticker, interval):
    return os.path.join(CACHE_DIR, f"{ticker}_{interval}.parquet")

def load_cache(ticker, interval):
    p = cache_path(ticker, interval)
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None

def save_cache(df, ticker, interval):
    df.to_parquet(cache_path(ticker, interval))

# =========================
# FETCH DATA
# =========================
def fetch_data(ticker, interval="1d", period="12mo"):
    try:
        cached = load_cache(ticker, interval)
        if cached is not None and len(cached) > 50:
            return cached.copy()

        df = yf.download(ticker, interval=interval, period=period, progress=False, threads=False, auto_adjust=False)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open","High","Low","Close","Volume"]].dropna()
        if len(df) < 10 or df["Volume"].tail(5).sum() == 0:
            return None

        save_cache(df, ticker, interval)
        return df
    except Exception as e:
        print(f"Failed download: {ticker} | {e}")
        return None

# =========================
# INDICATORS
# =========================
def add_indicators(df):
    df = df.copy()
    df["EMA13"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=EMA_MID, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(SMA50_PERIOD, min_periods=1).mean()
    df["VOL_MA20"] = df["Volume"].rolling(VOL_MA_PERIOD, min_periods=1).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(RSI_PERIOD, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Stochastic %K
    low_min = df["Low"].rolling(STOCH_PERIOD, min_periods=1).min()
    high_max = df["High"].rolling(STOCH_PERIOD, min_periods=1).max()
    df["STOCH"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    return df

def ema_slope(series):
    if len(series) <= SLOPE_WINDOW:
        return 0
    return series.iloc[-1] - series.iloc[-SLOPE_WINDOW]

# =========================
# SCORING LOGIC
# =========================
def score_day(d1, h4):
    score = 0
    if d1["Close"] > d1["SMA50"]:
        score += 1
    if d1["Close"] >= d1["EMA13"] >= d1["EMA21"] >= d1["EMA50"]:
        gap_ok = abs(d1["Close"]/d1["EMA13"] - 1) <= TOLERANCE
        if gap_ok:
            score += 1
    if h4["Close"] >= h4["EMA13"] >= h4["EMA21"] >= h4["EMA50"]:
        score += 1
    return score

# =========================
# BACKTEST
# =========================
def backtest_score(ticker):
    d1 = fetch_data(ticker, "1d", "12mo")
    h4 = fetch_data(ticker, "4h", "12mo")
    if d1 is None or h4 is None:
        return None

    d1 = add_indicators(d1)
    h4 = add_indicators(h4)

    h4_daily = h4.resample("1D").last()
    combined = pd.DataFrame(index=d1.index)
    combined["Close"] = d1["Close"]
    combined["SMA50"] = d1["SMA50"]
    combined["EMA13"] = d1["EMA13"]
    combined["EMA21"] = d1["EMA21"]
    combined["EMA50"] = d1["EMA50"]
    combined["H4_Close"] = h4_daily["Close"]

    scores = []
    for date in combined.index:
        d1_row = combined.loc[date]
        try:
            h4_row = combined.loc[date]  # gunakan 4h daily aggregated
        except:
            continue
        # pastikan ada next day
        if date == combined.index[-1]:
            next_return = np.nan  # H-1 tidak ada next day
        else:
            next_return = (combined["Close"].shift(-1).loc[date]/d1_row["Close"] - 1)*100
        s = score_day(d1_row, h4_row)
        scores.append({
            "Date": date,
            "Score": s,
            "NextDayReturn": next_return
        })

    df_score = pd.DataFrame(scores)
    prob_table = df_score.groupby("Score").agg(
        Count=("Score","size"),
        Prob_up=("NextDayReturn", lambda x: (x>0).sum()/len(x)*100),
        Avg_next_day_return=("NextDayReturn","mean")
    ).reset_index()
    return df_score, prob_table

# =========================
# CANDLE INFO
# =========================
def latest_candle_info(df):
    if len(df) < 2:
        return "N/A", False, False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body_pct = abs(last["Close"] - last["Open"]) / max(last["Open"], 1)
    is_red = last["Close"] < last["Open"]
    is_green = last["Close"] > last["Open"]
    breakdown = is_red and body_pct > BODY_THRESHOLD and last["Close"] < last["SMA50"] and prev["Close"] >= last["SMA50"]
    approaching = is_green and abs((last["Close"] - last["SMA50"]) / max(last["SMA50"],1) * 100) <= CANDLE_DIST_TH
    if breakdown:
        return "Merah Kuat & Breakdown", -1, False
    if approaching:
        return "Hijau & Mendekati SMA50", False, 1
    if is_red:
        return "Merah Biasa", 0, False
    if is_green:
        return "Hijau Biasa", False, 0
    return "Doji/Netral", 0, 0

# =========================
# SCORE DASAR
# =========================
def score_dasar(rsi, stoch, dist_sma50):
    score = 0
    if not np.isnan(rsi) and rsi < 30:
        score += 1
    if not np.isnan(stoch) and stoch < 20:
        score += 1
    if not np.isnan(dist_sma50) and -10 <= dist_sma50 <= 5:
        score += 1
    return score

def compute_dist_sma50(df):
    last = df.iloc[-1]
    sma50 = last["SMA50"]
    if pd.isna(sma50) or sma50 == 0:
        return np.nan
    return (last["Close"] - sma50) / sma50 * 100

# =========================
# ENGINE LOGIC
# =========================
def major_trend_daily(df):
    last = df.iloc[-1]
    if last["EMA21"] <= last["EMA50"]:
        return "INVALID"
    if ema_slope(df["EMA21"]) <= 0:
        return "INVALID"
    return "STRONG"

def minor_phase_4h(df):
    if len(df) < 2:
        return "NEUTRAL"
    last, prev = df.iloc[-1], df.iloc[-2]
    low50 = df["Low"].tail(50)
    low50_min = low50.min() if not low50.empty else np.nan
    compress = abs(last["EMA13"] - last["EMA21"]) / max(last["EMA21"],1) < EMA_COMPRESS_TH
    if not np.isnan(low50_min) and last["Close"] < low50_min:
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
    if len(df) < 1:
        return False
    return df["Close"].iloc[-1] > df["EMA13"].iloc[-1] or ema_slope(df["EMA13"]) > 0

def volume_state(df):
    if len(df) < 1:
        return 0, 0, 0, 0, 0, "DRYING"
    last, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    vol_ma20 = df["VOL_MA20"].iloc[-1] if not np.isnan(df["VOL_MA20"].iloc[-1]) else 1
    vol_ratio = round(last["Volume"] / vol_ma20,2) if vol_ma20>0 else 0
    vol_state = "DRYING" if vol_ratio<0.7 else "EXPANSION" if vol_ratio>1.2 else "NORMAL"
    vol_change = round((last["Volume"]/prev["Volume"] - 1)*100,2) if prev["Volume"]>0 else 0
    return last["Volume"], prev["Volume"], vol_ma20, vol_ratio, vol_change, vol_state

def final_decision(major, minor, setup, stage2, vol_state):
    if major=="INVALID" or minor=="BREAKDOWN":
        return "SKIP"
    if setup=="SETUP_PENDING":
        return "SETUP_PENDING"
    if major=="STRONG" and setup=="STAGE2_READY" and stage2 and vol_state!="DRYING":
        return "ENTRY_READY"
    return "WAIT"

# =========================
# PROCESS STOCK
# =========================
def process_stock(kode):
    ticker = f"{kode}.JK"
    try:
        d1 = fetch_data(ticker, "1d", "12mo")
        h4 = fetch_data(ticker, "4h", "12mo")
        if d1 is None or h4 is None or len(d1)<10 or len(h4)<10:
            print(f"Skip {kode}: not enough data")
            return None

        d1 = add_indicators(d1)
        h4 = add_indicators(h4)

        major = major_trend_daily(d1)
        minor = minor_phase_4h(h4)
        setup = setup_state(minor)
        stage2 = stage2_trigger(h4, setup)

        vol_today, vol_yesterday, vol_ma20, vol_ratio_ma20, vol_change_d1, vol_state = volume_state(d1)
        price_today = d1["Close"].iloc[-1]
        price_yesterday = d1["Close"].iloc[-2] if len(d1) > 1 else d1["Close"].iloc[-1]
        price_change_pct = round((price_today/price_yesterday - 1)*100,2)
        gap_ema21 = round((price_today/d1["EMA21"].iloc[-1]-1)*100,2)
        gap_ema50 = round((price_today/d1["EMA50"].iloc[-1]-1)*100,2)

        rsi = d1["RSI"].iloc[-1]
        stoch = d1["STOCH"].iloc[-1]
        sma50 = d1["SMA50"].iloc[-1]
        dist_to_sma50 = compute_dist_sma50(d1)

        candle_label, candle_red_breakdown, candle_green_approach = latest_candle_info(d1)
        score = score_dasar(rsi, stoch, dist_to_sma50)
        candle_effect = 1 if candle_green_approach else -1 if candle_red_breakdown else 0
        total_score = score + candle_effect + (1 if stage2 else 0) + (1 if vol_state=="EXPANSION" else 0)
        final_dec = final_decision(major, minor, setup, stage2, vol_state)

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
            "FinalDecision": final_dec,
            "RSI": round(rsi,2),
            "SMA50": round(sma50,2),
            "Dist_to_SMA50": round(dist_to_sma50,2) if not np.isnan(dist_to_sma50) else np.nan,
            "Stoch_K": round(stoch,2),
            "Latest_Candle": candle_label,
            "Score_Dasar": score,
            "Candle_Effect": candle_effect,
            "Total_Score": total_score
        }

    except Exception as e:
        print(f"Failed process_stock: {kode} | {e}")
        return None
