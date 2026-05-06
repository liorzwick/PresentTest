# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. הגדרות כלליות
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_GROUP")
CUSTOM_TICKERS_FILE = "mystock.csv"

MIN_MARKET_CAP = 2_000_000_000
MIN_DOLLAR_VOL_50 = 20_000_000
MIN_PRICE = 12.0
COOLDOWN_DAYS = 5
TOP_RESULTS = 10
SCAN_PERIOD = "1y"

market_cap_cache = {}

def load_brain():
    brain = {
        "max_base_depth": 0.65,               # הורחב כדי לתפוס גם התאוששויות עמוקות (Turnarounds כמו HUN)
        "max_tightness_depth": 0.15,          # הורחב כדי לאפשר כיווץ הגיוני לשוק הנוכחי
        "min_breakout_close_strength": 0.55,
        "min_rs_65": 0.03,
        "max_dist_from_52w_high_normal": 0.18,
        "max_dist_from_52w_high_below_150": 0.35,
        "max_gap_above_pivot": 0.02,
        "max_entry_extension": 0.04,          
        "breakout_volume_ratio": 1.4,
        "watchlist_volume_ratio": 0.75,
        "min_contractions": 2,
        "max_contractions": 4,
        "pivot_tolerance": 0.03,
        "min_base_length": 40,                # 🚨 תוקן! בסיס חייב להיות לפחות חודשיים (פסילת מיקרו-תבניות כמו FTS)
        "max_base_length": 200,
        "max_dry_up_ratio": 0.78,
        "atr_contraction_ratio": 0.95,
        "watchlist_max_dist": 0.06,           
        "min_touch_count": 2,
        "max_risk_pct": 12.0,
        "allow_unknown_market_cap": True,
        "swing_window": 4,
    }

    try:
        if os.path.exists("brain.json"):
            with open("brain.json", "r", encoding="utf-8") as f:
                brain.update(json.load(f))
    except Exception:
        print("⚠️ אזהרה: לא ניתן לטעון את brain.json, משתמש בברירות מחדל.")

    return brain


BRAIN = load_brain()


# ==========================================
# 2. עזרי קבצים / טלגרם / אנטי-ספאם
# ==========================================
def append_dataframe(df, file_path):
    try:
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(file_path, mode="a", header=False, index=False, encoding="utf-8")
    except Exception:
        pass


def send_telegram(message):
    print("\n=== תוכן ההודעה המלאה ===")
    print(message)
    print("=========================\n")

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ מצב סימולציה - הטוקן או ה-ID חסרים ב-GitHub Secrets")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        response = requests.post(url, json=payload, timeout=12)
        if response.status_code != 200:
            print(f"❌ טלגרם חסם את ההודעה לקבוצה: {response.text}")
        else:
            print("✅ ההודעה נשלחה בהצלחה לקבוצת הטלגרם!")
    except Exception as e:
        print(f"❌ שגיאת תקשורת עם טלגרם: {e}")


def log_signal(ticker, price, status):
    log_file = "trading_log.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_data = pd.DataFrame([{
        "Date": now,
        "Ticker": ticker,
        "Price": round(float(price), 2),
        "Status": status
    }])

    append_dataframe(new_data, log_file)


def save_to_smart_memory(
    ticker, price, stop_loss, risk_pct, vol_ratio, pivot, close_strength,
    rs_65, tightness, pattern_type, status, setup_score, dry_up_ratio, touches
):
    memory_file = "smart_memory.csv"
    now = datetime.now().strftime("%Y-%m-%d")

    new_record = pd.DataFrame([{
        "Date": now,
        "Ticker": ticker,
        "Price": round(float(price), 2),
        "Pivot": round(float(pivot), 2),
        "Stop_Loss": round(float(stop_loss), 2),
        "Risk_Pct": round(float(risk_pct), 2),
        "Volume_Ratio": round(float(vol_ratio), 2),
        "Close_Strength": round(float(close_strength), 2),
        "RS_65": round(float(rs_65), 4),
        "Tightness_Pct": round(float(tightness) * 100, 2),
        "Pattern_Type": pattern_type,
        "Status": status,
        "Setup_Score": round(float(setup_score), 1),
        "DryUp_Ratio": round(float(dry_up_ratio), 2),
        "Touches": int(touches)
    }])

    append_dataframe(new_record, memory_file)


def should_skip_spam(ticker, is_breakout):
    log_file = "trading_log.csv"
    if not os.path.isfile(log_file):
        return False

    try:
        df = pd.read_csv(log_file, encoding="utf-8-sig")
        if df.empty:
            return False

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        ticker_history = df[df["Ticker"] == ticker].sort_values(by="Date", ascending=False)
        if ticker_history.empty:
            return False

        last_sent = ticker_history.iloc[0]["Date"]

        if is_breakout:
            return last_sent.date() == datetime.now().date()

        days_passed = (datetime.now() - last_sent).days
        return days_passed < COOLDOWN_DAYS
    except Exception:
        return False


def load_tickers():
    if os.path.exists(CUSTOM_TICKERS_FILE):
        try:
            df = pd.read_csv(CUSTOM_TICKERS_FILE, encoding="utf-8-sig")
            col_name = next((c for c in df.columns if c.strip().lower() in ["ticker", "symbol"]), None)

            if col_name:
                tickers = (
                    df[col_name]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
                tickers = [t for t in tickers if t and (t.replace("-", "").isalnum())]
                tickers = sorted(list(set(tickers)))
                print(f"✅ נטענו {len(tickers)} מניות מתוך {CUSTOM_TICKERS_FILE}")
                return tickers
        except Exception:
            pass

    return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "PLTR"]


# ==========================================
# 3. אינדיקטורים
# ==========================================
def normalize_ohlcv_columns(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df[~df.index.duplicated(keep="first")]
    return df


def add_indicators(df):
    df = normalize_ohlcv_columns(df)

    # הוספת min_periods פותרת את בעיית הקריסה בשלב 1
    df["SMA_21"] = df["Close"].rolling(21, min_periods=10).mean()
    df["SMA_50"] = df["Close"].rolling(50, min_periods=25).mean()
    df["SMA_150"] = df["Close"].rolling(150, min_periods=75).mean()
    df["SMA_200"] = df["Close"].rolling(200, min_periods=100).mean()

    df["Vol_10"] = df["Volume"].rolling(10, min_periods=5).mean()
    df["Vol_20"] = df["Volume"].rolling(20, min_periods=10).mean()
    df["Vol_50"] = df["Volume"].rolling(50, min_periods=25).mean()
    df["DollarVol_50"] = df["Close"].rolling(50, min_periods=25).mean() * df["Vol_50"]

    df["Prev_Close"] = df["Close"].shift(1)
    df["ROC_65"] = df["Close"].pct_change(65)
    df["High_252"] = df["High"].rolling(252, min_periods=120).max()

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Prev_Close"]).abs(),
            (df["Low"] - df["Prev_Close"]).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR_14"] = tr.rolling(14, min_periods=7).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"]
    df["Range_Pct"] = (df["High"] - df["Low"]) / df["Close"]

    return df


def get_spy_data():
    try:
        spy = yf.download("SPY", period=SCAN_PERIOD, auto_adjust=True, progress=False)
        spy = normalize_ohlcv_columns(spy)

        if not spy.empty and len(spy) > 200:
            spy["SMA_50"] = spy["Close"].rolling(50).mean()
            spy["SMA_150"] = spy["Close"].rolling(150).mean()
            spy["SMA_200"] = spy["Close"].rolling(200).mean()
            spy["ROC_65"] = spy["Close"].pct_change(65)
            spy["ATR_14"] = pd.concat(
                [
                    spy["High"] - spy["Low"],
                    (spy["High"] - spy["Close"].shift(1)).abs(),
                    (spy["Low"] - spy["Close"].shift(1)).abs()
                ],
                axis=1
            ).max(axis=1).rolling(14).mean()
            spy["ATR_Pct"] = spy["ATR_14"] / spy["Close"]
            return spy
    except Exception:
        pass

    return pd.DataFrame()


def market_filter_ok(spy_df):
    if spy_df.empty or len(spy_df) < 220:
        return False

    today = spy_df.iloc[-1]
    sma200_old = spy_df["SMA_200"].iloc[-20]
    atr_pct = float(today["ATR_Pct"]) if pd.notna(today["ATR_Pct"]) else 0.0

    if pd.isna(today["SMA_200"]) or pd.isna(sma200_old):
        return False

    trend_ok = (
        float(today["Close"]) > float(today["SMA_50"]) > float(today["SMA_150"]) > float(today["SMA_200"])
        and float(today["SMA_200"]) > float(sma200_old)
    )

    volatility_ok = atr_pct < 0.03
    return trend_ok and volatility_ok


# ==========================================
# 4. בדיקות איכות יקום
# ==========================================
def check_market_cap(ticker):
    if ticker in market_cap_cache:
        return market_cap_cache[ticker]

    market_cap = None

    try:
        t = yf.Ticker(ticker)

        try:
            fi = t.fast_info
            if fi:
                market_cap = fi.get("marketCap", None) or fi.get("market_cap", None)
        except Exception:
            pass

        if not market_cap:
            try:
                info = t.info
                market_cap = info.get("marketCap", None)
            except Exception:
                pass
    except Exception:
        pass

    market_cap_cache[ticker] = market_cap
    return market_cap


# ==========================================
# 5. Swing points ו-detectors מותאמים חומרה 🚨
# ==========================================
def find_swing_highs(arr, window=4):
    arr = np.asarray(arr, dtype=float)
    peaks = []

    for i in range(window, len(arr) - window):
        segment = arr[i - window:i + window + 1]
        if not np.all(np.isfinite(segment)):
            continue

        if arr[i] == np.max(segment) and arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            peaks.append(i)

    return peaks


def find_swing_lows(arr, window=4):
    arr = np.asarray(arr, dtype=float)
    lows = []

    for i in range(window, len(arr) - window):
        segment = arr[i - window:i + window + 1]
        if not np.all(np.isfinite(segment)):
            continue

        if arr[i] == np.min(segment) and arr[i] < arr[i - 1] and arr[i] <= arr[i + 1]:
            lows.append(i)

    return lows


def dedupe_indices(indices, values, min_sep=25, keep_higher=True):
    if not indices:
        return []

    indices = sorted(indices)
    kept = [indices[0]]

    for idx in indices[1:]:
        last = kept[-1]
        # 🚨 חוק מרחק: חייבים להיות במרחק של לפחות חודש אחד מהשני
        if idx - last >= min_sep:
            kept.append(idx)
        else:
            if keep_higher and values[idx] > values[last]:
                kept[-1] = idx
            elif (not keep_higher) and values[idx] < values[last]:
                kept[-1] = idx

    return kept


def get_vcp_signal(hist):
    recent = hist.tail(250).copy() # מסתכל 250 ימים (שנה) אחורה כדי לתפוס תבניות כמו FTNT
    if len(recent) < 80:
        return None

    highs = recent["High"].astype(float).values
    lows = recent["Low"].astype(float).values
    vols = recent["Volume"].astype(float).values
    closes = recent["Close"].astype(float).values
    n = len(recent)

    pivot = float(np.max(highs[:-5]))
    if not np.isfinite(pivot) or pivot <= 0:
        return None

    swing_window = int(BRAIN["swing_window"])
    swing_highs = find_swing_highs(highs, window=swing_window)

    touch_candidates = [
        i for i in swing_highs
        if i < n - 3 and highs[i] >= pivot * (1 - BRAIN["pivot_tolerance"])
    ]

    if len(touch_candidates) < 2:
        raw_hits = np.where(highs[:-3] >= pivot * (1 - BRAIN["pivot_tolerance"]))[0].tolist()
        touch_candidates = raw_hits

    # 🚨 חוק מרחק בזמן: מינימום 25 ימי מסחר (יותר מחודש) בין נגיעה לנגיעה! (מחסל מניות רועשות)
    touches = dedupe_indices(touch_candidates, highs, min_sep=25, keep_higher=True)

    if len(touches) < BRAIN["min_touch_count"]:
        return None

    max_touches = int(BRAIN["max_contractions"]) + 1
    if len(touches) > max_touches:
        touches = touches[-max_touches:]

    base_start = max(0, touches[0] - 10)
    base_end = n - 1
    base_len = base_end - base_start + 1

    # 🚨 חוק אורך בסיס: תבנית חייבת להתבשל לפחות חודשיים-שלושה.
    if base_len < BRAIN["min_base_length"]:
        return None

    depths = []
    pullback_lows = []

    for a, b in zip(touches[:-1], touches[1:]):
        seg_low_idx_rel = int(np.argmin(lows[a:b + 1]))
        seg_low_idx = a + seg_low_idx_rel
        seg_low_price = float(lows[seg_low_idx])

        depth = (pivot - seg_low_price) / pivot
        depths.append(float(depth))
        pullback_lows.append((seg_low_idx, seg_low_price))

    if len(depths) < BRAIN["min_contractions"]:
        return None

    # 🚨 חוק עומק מינימלי: מסנן ירידות זעירות של 4%-6% כמו שראינו ב-FTS. חייב לרדת לפחות 10% לניעור.
    if max(depths) < 0.10:
        return None

    if max(depths) > BRAIN["max_base_depth"]:
        return None

    if depths[-1] > BRAIN["max_tightness_depth"]:
        return None

    # 🚨 התכווצות: התיקון האחרון (ידית) חייב להיות קטן בחצי מההתרסקות המקסימלית (מקסימום 55% מעומק הבסיס)
    if depths[-1] > max(depths) * 0.55:
        return None

    # שפלים עולים (בסלחנות כדי לאפשר Shakeout קל)
    low_prices = [x[1] for x in pullback_lows]
    if len(low_prices) >= 2 and low_prices[-1] < low_prices[0] * 0.95:
        return None

    base_df = recent.iloc[base_start:base_end + 1].copy()
    if len(base_df) < 20:
        return None

    early_slice = base_df.iloc[:max(10, len(base_df) // 3)]
    late_slice = base_df.iloc[-10:]

    early_vol = early_slice["Volume"].mean()
    late_vol = late_slice["Volume"].mean()
    dry_up_ratio = float(late_vol / early_vol) if early_vol and np.isfinite(early_vol) else 1.0

    if not np.isfinite(dry_up_ratio):
        dry_up_ratio = 1.0

    if dry_up_ratio > BRAIN["max_dry_up_ratio"]:
        return None

    early_atr = early_slice["ATR_Pct"].mean()
    late_atr = late_slice["ATR_Pct"].mean()
    atr_ratio = float(late_atr / early_atr) if early_atr and np.isfinite(early_atr) else 1.0

    if np.isfinite(atr_ratio) and atr_ratio > 1.05:
        return None

    last_touch = touches[-1]
    tight_zone_start = max(0, last_touch - 12)
    tight_low = float(np.min(lows[tight_zone_start:]))
    last_pullback_low = float(pullback_lows[-1][1])

    pullback_text = " > ".join(f"{d*100:.1f}%" for d in depths)

    return {
        "pivot_price": pivot,
        "tight_low": min(tight_low, last_pullback_low),
        "last_pullback_low": last_pullback_low,
        "tightness": depths[-1],
        "base_depth": max(depths),
        "dry_up_ratio": dry_up_ratio,
        "atr_ratio": atr_ratio,
        "touches": len(touches),
        "contractions": len(depths),
        "contraction_text": pullback_text,
        "base_length": base_len,
        "type": "VCP"
    }


def get_flat_base_signal(hist):
    recent = hist.tail(120).copy() # מסתכל חצי שנה אחורה
    if len(recent) < 45: # 🚨 בסיס שטוח חייב להיות לפחות 45 ימי מסחר
        return None

    highs = recent["High"].astype(float)
    lows = recent["Low"].astype(float)
    closes = recent["Close"].astype(float)

    pivot = float(highs.iloc[:-3].max())
    low = float(lows.min())
    depth = (pivot - low) / pivot if pivot > 0 else 999

    # 🚨 עומק בסיס שטוח חייב להיות לפחות 10% כדי להעיף מניות רועשות
    if not (0.10 <= depth <= 0.25):
        return None

    last_15 = recent.tail(15)
    close_std_pct = float(last_15["Close"].std() / last_15["Close"].mean())
    if not np.isfinite(close_std_pct) or close_std_pct > 0.03:
        return None

    early_vol = float(recent["Volume"].head(20).mean())
    late_vol = float(recent["Volume"].tail(10).mean())
    dry_up_ratio = late_vol / early_vol if early_vol > 0 else 1.0

    if not np.isfinite(dry_up_ratio) or dry_up_ratio > 0.80:
        return None

    tight_low = float(last_15["Low"].min())
    atr_ratio = float(last_15["ATR_Pct"].mean() / recent["ATR_Pct"].head(20).mean()) if recent["ATR_Pct"].head(20).mean() > 0 else 1.0

    return {
        "pivot_price": pivot,
        "tight_low": tight_low,
        "last_pullback_low": tight_low,
        "tightness": (pivot - tight_low) / pivot,
        "base_depth": depth,
        "dry_up_ratio": dry_up_ratio,
        "atr_ratio": atr_ratio,
        "touches": 2,
        "contractions": 1,
        "contraction_text": f"{depth*100:.1f}%",
        "base_length": len(recent),
        "type": "Flat Base"
    }


# ==========================================
# 6. דירוג setup
# ==========================================
def calc_setup_score(alert):
    score = 0.0

    rs_score = min(max(alert["rs_65"], 0) * 250, 25)
    tight_score = max(0, (1 - min(alert["tightness"], 0.10) / 0.10) * 20)
    dryup_score = max(0, (1 - min(alert["dry_up_ratio"], 1.0)) * 20)
    pivot_score = max(0, (1 - min(abs(alert["dist_to_pivot"]), 0.03) / 0.03) * 15)
    close_score = min(max(alert["close_strength"], 0), 1) * 10
    volume_score = min(alert["vol_ratio"] / 2.0, 1.0) * 5
    touch_score = min(alert["touches"], 4) * 2.5
    bonus = 5 if not alert["is_below_150"] else 0

    score = rs_score + tight_score + dryup_score + pivot_score + close_score + volume_score + touch_score + bonus
    return round(score, 1)


# ==========================================
# 7. סריקת שוק ראשית
# ==========================================
def scan_market():
    tickers = load_tickers()
    if not tickers:
        return

    print("📥 בודק את מגמת השוק (SPY)...")
    spy = get_spy_data()

    market_warning = ""
    spy_rs = 0.0

    if spy.empty:
        market_warning = "🔴 <b>שגיאה: לא ניתן למשוך נתוני שוק (SPY).</b> הסריקה ממשיכה ללא פילטר מגמה.\n\n"
    else:
        spy_rs = float(spy.iloc[-1]["ROC_65"]) if pd.notna(spy.iloc[-1]["ROC_65"]) else 0.0
        if not market_filter_ok(spy):
            market_warning = "⚠️ <b>שים לב: השוק הכללי לא במצב אידיאלי לפריצות.</b> הסריקה ממשיכה, אך הסיכון לפריצות שווא גבוה.\n\n"

    all_potentials = []
    waiting_for_pivot_tickers = [] 

    stats = {
        "total_scanned": 0,
        "pass_basic_data": 0,
        "pass_price_vol": 0,
        "pass_sma": 0,
        "pass_52w": 0,
        "pass_rs": 0,
        "pass_market_cap": 0,
        "pass_pattern": 0,
        "pass_pivot_dist": 0,
        "final_approved": 0
    }

    for ticker in tickers:
        stats["total_scanned"] += 1
        print(f"סורק את {ticker}...", end="\r")

        try:
            df = yf.download(ticker, period=SCAN_PERIOD, auto_adjust=True, progress=False)
            df = normalize_ohlcv_columns(df)

            if df.empty or len(df) < 200:
                continue

            df = add_indicators(df)
            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            past_data = df.iloc[:-1].copy()

            required_cols = ["SMA_50", "SMA_150", "SMA_200", "ATR_14", "Vol_50", "High_252", "ROC_65"]
            if any(pd.isna(today[c]) for c in required_cols):
                continue
            
            stats["pass_basic_data"] += 1

            close = float(today["Close"])
            open_price = float(today["Open"])
            high_252 = float(today["High_252"])
            dollar_vol_50 = float(today["DollarVol_50"])

            if close < MIN_PRICE or dollar_vol_50 < MIN_DOLLAR_VOL_50:
                continue
                
            stats["pass_price_vol"] += 1

            if close <= float(today["SMA_50"]):
                continue

            is_below_150 = close < float(today["SMA_150"])

            if not is_below_150:
                if not (close > float(today["SMA_150"]) > float(today["SMA_200"])):
                    continue
            else:
                if float(today["SMA_50"]) <= float(yesterday["SMA_50"]):
                    continue
                    
            stats["pass_sma"] += 1

            dist_52w = (close / high_252) - 1.0
            max_dist = (
                BRAIN["max_dist_from_52w_high_below_150"]
                if is_below_150
                else BRAIN["max_dist_from_52w_high_normal"]
            )
            if dist_52w < -max_dist:
                continue
                
            stats["pass_52w"] += 1

            stock_rs = float(today["ROC_65"]) - float(spy_rs)
            required_rs = BRAIN["min_rs_65"] * (2 if is_below_150 else 1)
            if stock_rs < required_rs:
                continue
                
            stats["pass_rs"] += 1

            market_cap = check_market_cap(ticker)
            if market_cap is not None and market_cap < MIN_MARKET_CAP:
                continue
            if market_cap is None and not BRAIN["allow_unknown_market_cap"]:
                continue
                
            stats["pass_market_cap"] += 1

            pattern = get_vcp_signal(past_data)
            if not pattern:
                pattern = get_flat_base_signal(past_data)
            if not pattern:
                continue
                
            stats["pass_pattern"] += 1

            pivot = float(pattern["pivot_price"])
            dist_to_pivot = (close / pivot) - 1.0

            is_breakout = float(yesterday["Close"]) <= pivot and close > pivot
            is_near_breakout = (-BRAIN["watchlist_max_dist"] <= dist_to_pivot <= 0.0)

            if not (is_breakout or is_near_breakout):
                waiting_for_pivot_tickers.append(f"{ticker} ({dist_to_pivot*100:.1f}%)")
                continue
                
            stats["pass_pivot_dist"] += 1

            if should_skip_spam(ticker, is_breakout):
                continue

            day_range = max(float(today["High"]) - float(today["Low"]), 1e-9)
            close_strength = (close - float(today["Low"])) / day_range
            vol_ratio = float(today["Volume"]) / float(today["Vol_50"]) if float(today["Vol_50"]) > 0 else 0.0
            gap_from_pivot = (open_price / pivot) - 1.0

            if is_breakout:
                req_vol = 1.8 if is_below_150 else BRAIN["breakout_volume_ratio"]
                req_close = 0.60 if is_below_150 else BRAIN["min_breakout_close_strength"]

                if close_strength < req_close:
                    continue
                if vol_ratio < req_vol:
                    continue
                if gap_from_pivot > BRAIN["max_gap_above_pivot"]:
                    continue

                status = "🔥 פריצה פעילה!"
            else:
                if close_strength < 0.45:
                    continue
                if vol_ratio < BRAIN["watchlist_volume_ratio"]:
                    continue

                status = "👀 מתבשלת (Watchlist)"

            if close > pivot * (1 + BRAIN["max_entry_extension"]):
                continue

            atr = float(today["ATR_14"])
            stop_price = min(float(pattern["tight_low"]), float(pattern["last_pullback_low"])) - (0.5 * atr)

            if stop_price >= close:
                continue

            risk_pct = (close - stop_price) / close * 100
            if risk_pct > BRAIN["max_risk_pct"]:
                continue

            alert_data = {
                "ticker": ticker,
                "close": close,
                "pivot": pivot,
                "stop_loss": stop_price,
                "risk_pct": risk_pct,
                "vol_ratio": vol_ratio,
                "type": pattern["type"],
                "rs_65": stock_rs,
                "close_strength": close_strength,
                "status": status,
                "dist_to_pivot": dist_to_pivot,
                "tightness": float(pattern["tightness"]),
                "is_below_150": is_below_150,
                "dry_up_ratio": float(pattern["dry_up_ratio"]),
                "touches": int(pattern["touches"]),
                "contractions": int(pattern["contractions"]),
                "contraction_text": pattern["contraction_text"],
                "base_depth": float(pattern["base_depth"]),
                "base_length": int(pattern["base_length"]),
                "market_cap": market_cap
            }

            alert_data["setup_score"] = calc_setup_score(alert_data)
            all_potentials.append(alert_data)
            
            stats["final_approved"] += 1

        except Exception:
            pass

        time.sleep(0.15)

    print("\n" + "=" * 50)
    print("📊 סטטיסטיקת משפך סינון (Funnel Stats):")
    print(f"סה\"כ נסרקו: {stats['total_scanned']}")
    print(f"עברו נתונים בסיסיים: {stats['pass_basic_data']}")
    print(f"עברו מחיר וווליום דולרי: {stats['pass_price_vol']}")
    print(f"עברו ממוצעים נעים (SMA): {stats['pass_sma']}")
    print(f"עברו מרחק משיא שנתי (52w): {stats['pass_52w']}")
    print(f"עברו כוח יחסי (RS): {stats['pass_rs']}")
    print(f"עברו שווי שוק: {stats['pass_market_cap']}")
    print(f"✅ עברו זיהוי תבנית VCP/Flat Base: {stats['pass_pattern']}")
    print(f"🎯 קרובים לפיבוט (בחלון כניסה): {stats['pass_pivot_dist']}")
    print(f"🏆 אושרו סופית (לאחר ספאם, סיכון וכו'): {stats['final_approved']}")
    print("=" * 50)
    
    print("\n👀 מניות שעברו זיהוי תבנית תקינה אך עדיין רחוקות מהפיבוט (שלב לפני אחרון):")
    if waiting_for_pivot_tickers:
        print(", ".join(waiting_for_pivot_tickers))
    else:
        print("אין מניות כאלו כרגע.")
    print("=" * 50)

    all_potentials_sorted = sorted(
        all_potentials,
        key=lambda x: (-x["setup_score"], abs(x["dist_to_pivot"]))
    )

    final_selection = []
    below_150_count = 0

    for stock in all_potentials_sorted:
        if len(final_selection) >= TOP_RESULTS:
            break

        if stock["is_below_150"]:
            if below_150_count < 3:
                final_selection.append(stock)
                below_150_count += 1
        else:
            final_selection.append(stock)

    final_bo = [s for s in final_selection if "פריצה פעילה" in s["status"]]
    final_wl = [s for s in final_selection if "מתבשלת" in s["status"]]
    total_sent = len(final_bo) + len(final_wl)

    if total_sent > 0:
        print(f"🔥 הסריקה הסתיימה! נמצאו {total_sent} מניות לשליחה.")

        msg = "🎯 <b>סריקת VCP יומית הסתיימה!</b>\n"
        if market_warning:
            msg += market_warning

        msg += f"<i>(מציג {total_sent} מניות מדורגות לפי איכות setup. מקסימום 3 מתחת לממוצע 150)</i>\n\n"

        if final_bo:
            msg += f"🔥 <b>פריצות אקטיביות ({len(final_bo)}):</b>\n\n"

            for a in final_bo:
                tv_link = f"https://il.tradingview.com/chart/?symbol={a['ticker']}"
                warning_150 = " ⚠️ (מתחת ל-150)" if a["is_below_150"] else ""

                msg += f"🚀 <b>{a['ticker']}</b> | {a['type']}{warning_150}\n"
                msg += f"⭐ <b>ציון:</b> {a['setup_score']:.1f} | 📈 <b>RS:</b> {a['rs_65'] * 100:.1f}% | 📊 <b>ווליום:</b> {a['vol_ratio']:.1f}x\n"
                msg += f"📐 <b>כיווץ:</b> {a['tightness'] * 100:.1f}% | 🫗 <b>Dry-Up:</b> {a['dry_up_ratio']:.2f} | 🔋 <b>סגירה:</b> {a['close_strength'] * 100:.0f}%\n"
                msg += f"🔁 <b>Contractions:</b> {a['contraction_text']} | 🎯 <b>פיבוט:</b> ${a['pivot']:.2f}\n"
                msg += f"💵 <b>מחיר:</b> ${a['close']:.2f} | 🛡️ <b>סטופ:</b> ${a['stop_loss']:.2f} (סיכון: {a['risk_pct']:.1f}%-)\n"
                msg += f"🔗 <a href='{tv_link}'>גרף ב-TradingView</a>\n"
                msg += "────────────────\n"

                log_signal(a["ticker"], a["close"], a["status"])
                save_to_smart_memory(
                    a["ticker"], a["close"], a["stop_loss"], a["risk_pct"],
                    a["vol_ratio"], a["pivot"], a["close_strength"], a["rs_65"],
                    a["tightness"], a["type"], "Sent", a["setup_score"],
                    a["dry_up_ratio"], a["touches"]
                )

        if final_wl:
            msg += f"👀 <b>מתבשלות למעקב ({len(final_wl)}):</b>\n\n"

            for a in final_wl:
                tv_link = f"https://il.tradingview.com/chart/?symbol={a['ticker']}"
                warning_150 = " ⚠️ (מתחת ל-150)" if a["is_below_150"] else ""

                msg += f"⏳ <b>{a['ticker']}</b> | {a['type']}{warning_150}\n"
                msg += f"⭐ <b>ציון:</b> {a['setup_score']:.1f} | 📈 <b>RS:</b> {a['rs_65'] * 100:.1f}% | 📊 <b>ווליום:</b> {a['vol_ratio']:.1f}x\n"
                msg += f"📐 <b>כיווץ:</b> {a['tightness'] * 100:.1f}% | 🫗 <b>Dry-Up:</b> {a['dry_up_ratio']:.2f} | 🔋 <b>סגירה:</b> {a['close_strength'] * 100:.0f}%\n"
                msg += f"🔁 <b>Contractions:</b> {a['contraction_text']} | 🎯 <b>פיבוט:</b> ${a['pivot']:.2f} (מרחק: {a['dist_to_pivot'] * 100:.1f}%)\n"
                msg += f"💵 <b>מחיר:</b> ${a['close']:.2f} | 🛡️ <b>סטופ משוער:</b> ${a['stop_loss']:.2f} (סיכון: {a['risk_pct']:.1f}%-)\n"
                msg += f"🔗 <a href='{tv_link}'>גרף ב-TradingView</a>\n"
                msg += "────────────────\n"

                log_signal(a["ticker"], a["close"], a["status"])
                save_to_smart_memory(
                    a["ticker"], a["close"], a["stop_loss"], a["risk_pct"],
                    a["vol_ratio"], a["pivot"], a["close_strength"], a["rs_65"],
                    a["tightness"], a["type"], "Sent", a["setup_score"],
                    a["dry_up_ratio"], a["touches"]
                )

        send_telegram(msg)

    else:
        print("💤 הסריקה הסתיימה. לא נמצאו מניות חדשות לשליחה בסיבוב זה.")
        send_telegram(
            f"✅ הסריקה הסתיימה.\n\n{market_warning}"
            f"אין פריצות או מניות חדשות במעקב שלא נשלחו כבר היום."
        )

    print("=" * 50)


if __name__ == "__main__":
    scan_market()
