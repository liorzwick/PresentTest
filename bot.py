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
        "max_base_depth": 0.65,               
        "max_tightness_depth": 0.15,          
        "min_breakout_close_strength": 0.55,
        "min_rs_65": 0.03,
        "max_dist_from_52w_high_normal": 0.18,
        "max_dist_from_52w_high_below_150": 0.35,
        "max_gap_above_pivot": 0.02,
        "max_entry_extension": 0.04,          
        "breakout_volume_ratio": 1.3,         
        "watchlist_volume_ratio": 0.75,
        "pivot_tolerance": 0.035,             
        "min_base_length": 20,                
        "max_base_length": 200,
        "max_dry_up_ratio": 0.85,             
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
    
    # מינימום תקופות שמונע קריסה ובאגים במשפך הסינון
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

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Prev_Close"]).abs(),
        (df["Low"] - df["Prev_Close"]).abs()
    ], axis=1).max(axis=1)

    df["ATR_14"] = tr.rolling(14, min_periods=7).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"]
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
            spy["ATR_14"] = pd.concat([
                spy["High"] - spy["Low"],
                (spy["High"] - spy["Close"].shift(1)).abs(),
                (spy["Low"] - spy["Close"].shift(1)).abs()
            ], axis=1).max(axis=1).rolling(14).mean()
            spy["ATR_Pct"] = spy["ATR_14"] / spy["Close"]
            return spy
    except Exception:
        pass
    return pd.DataFrame()

def market_filter_ok(spy_df):
    if spy_df.empty or len(spy_df) < 220: return False
    today = spy_df.iloc[-1]
    sma200_old = spy_df["SMA_200"].iloc[-20]
    atr_pct = float(today["ATR_Pct"]) if pd.notna(today["ATR_Pct"]) else 0.0
    if pd.isna(today["SMA_200"]) or pd.isna(sma200_old): return False
    trend_ok = (float(today["Close"]) > float(today["SMA_50"]) > float(today["SMA_150"]) > float(today["SMA_200"])
                and float(today["SMA_200"]) > float(sma200_old))
    return trend_ok and (atr_pct < 0.03)

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
            if fi: market_cap = fi.get("marketCap", None) or fi.get("market_cap", None)
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
# 5. מנוע זיהוי תבניות חכם (Cluster Pivot)
# ==========================================
def find_swing_highs(arr, window=4):
    arr = np.asarray(arr, dtype=float)
    peaks = []
    for i in range(window, len(arr) - window):
        segment = arr[i - window:i + window + 1]
        if not np.all(np.isfinite(segment)): continue
        if arr[i] == np.max(segment) and arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            peaks.append(i)
    return peaks

def dedupe_indices(indices, values, min_sep=10, keep_higher=True):
    if not indices: return []
    indices = sorted(indices)
    kept = [indices[0]]
    for idx in indices[1:]:
        last = kept[-1]
        if idx - last >= min_sep:
            kept.append(idx)
        else:
            if keep_higher and values[idx] > values[last]:
                kept[-1] = idx
            elif (not keep_higher) and values[idx] < values[last]:
                kept[-1] = idx
    return kept

def get_vcp_signal(hist):
    recent = hist.tail(250).copy() 
    if len(recent) < 80: return None

    highs = recent["High"].astype(float).values
    lows = recent["Low"].astype(float).values
    vols = recent["Volume"].astype(float).values
    n = len(recent)

    swing_highs = find_swing_highs(highs, window=4)
    
    best_touches = []
    best_pivot = 0.0

    for p_idx in swing_highs:
        if p_idx > n - 3: continue
        p_val = float(highs[p_idx])
        
        group = [i for i in swing_highs if i < n - 3 and abs(highs[i] - p_val) / max(p_val, 1e-9) <= BRAIN["pivot_tolerance"]]
        group = dedupe_indices(group, highs, min_sep=10, keep_higher=True)
        
        if len(group) >= BRAIN["min_touch_count"]:
            group_pivot = float(np.max([highs[i] for i in group]))
            if len(group) > len(best_touches) or (len(group) == len(best_touches) and group_pivot > best_pivot):
                best_touches = group
                best_pivot = group_pivot

    if len(best_touches) < BRAIN["min_touch_count"]: 
        return None

    pivot = best_pivot
    touches = best_touches

    first_touch = touches[0]
    last_touch = touches[-1]

    base_len = last_touch - first_touch
    if base_len < BRAIN["min_base_length"] or base_len > BRAIN["max_base_length"]:
        return None

    base_low = float(np.min(lows[first_touch:last_touch+1]))
    base_depth = (pivot - base_low) / pivot

    if base_depth < 0.10 or base_depth > BRAIN["max_base_depth"]:
        return None

    handle_data_len = n - last_touch
    if handle_data_len < 3: return None 

    handle_low = float(np.min(lows[last_touch:]))
    handle_depth = (pivot - handle_low) / pivot

    if handle_depth > BRAIN["max_tightness_depth"]: return None
    if handle_depth > base_depth * 0.65: return None

    base_vol = np.mean(vols[first_touch:last_touch]) if base_len > 0 else 1
    handle_vol = np.mean(vols[last_touch:])
    dry_up_ratio = float(handle_vol / base_vol) if base_vol > 0 else 1.0

    if dry_up_ratio > BRAIN["max_dry_up_ratio"]: return None

    return {
        "pivot_price": pivot,
        "tight_low": handle_low,
        "last_pullback_low": handle_low,
        "tightness": handle_depth,
        "base_depth": base_depth,
        "dry_up_ratio": dry_up_ratio,
        "touches": len(touches),
        "base_length": base_len,
        "type": "VCP"
    }

# ==========================================
# מנוע זיהוי ריטסטים נפרד (Pullbacks) 🔄 
# ==========================================
def get_retest_signal(hist):
    recent = hist.tail(300).copy()
    if len(recent) < 100: return None

    highs = recent["High"].astype(float).values
    lows = recent["Low"].astype(float).values
    closes = recent["Close"].astype(float).values
    vols = recent["Volume"].astype(float).values
    n = len(recent)

    search_highs = highs[-60:-5]
    if len(search_highs) == 0: return None
    
    markup_peak_val = float(np.max(search_highs))
    markup_peak_idx = int(n - 60 + np.argmax(search_highs))

    pre_breakout_highs = highs[:markup_peak_idx]
    if len(pre_breakout_highs) < 40: return None

    swing_highs = find_swing_highs(pre_breakout_highs, window=4)
    best_touches = []
    best_pivot = 0.0

    for p_idx in swing_highs:
        if p_idx > len(pre_breakout_highs) - 3: continue
        p_val = float(pre_breakout_highs[p_idx])
        group = [i for i in swing_highs if i < len(pre_breakout_highs) - 3 and abs(pre_breakout_highs[i] - p_val) / max(p_val, 1e-9) <= BRAIN["pivot_tolerance"]]
        group = dedupe_indices(group, pre_breakout_highs, min_sep=10, keep_higher=True)
        
        if len(group) >= BRAIN["min_touch_count"]:
            group_pivot = float(np.max([pre_breakout_highs[i] for i in group]))
            if len(group) > len(best_touches) or (len(group) == len(best_touches) and group_pivot > best_pivot):
                best_touches = group
                best_pivot = group_pivot

    if len(best_touches) < BRAIN["min_touch_count"]: return None
    pivot = best_pivot

    if markup_peak_val < pivot * 1.08: return None
    
    base_len = best_touches[-1] - best_touches[0]
    if base_len < 20: return None
    
    base_low = float(np.min(lows[best_touches[0]:markup_peak_idx]))
    base_depth = (pivot - base_low) / pivot

    # 🚨 התיקון הקריטי: חוק רצפת הבטון (מחסל מניות שבורות כמו NATL)
    pullback_zone_lows = lows[markup_peak_idx:]
    if len(pullback_zone_lows) == 0: return None
    lowest_since_peak = float(np.min(pullback_zone_lows))
    
    if lowest_since_peak < pivot * 0.965:
        return None

    current_close = closes[-1]
    dist_to_pivot = (current_close / pivot) - 1.0

    if dist_to_pivot < -0.025 or dist_to_pivot > 0.045: return None

    breakout_vol = np.mean(vols[max(0, markup_peak_idx-20):markup_peak_idx+1])
    pullback_vol = np.mean(vols[-5:])
    if breakout_vol == 0: return None
    dry_up_ratio = float(pullback_vol / breakout_vol)
    
    if dry_up_ratio > 0.85: return None

    current_low = float(np.min(lows[-5:]))
    pullback_depth = (markup_peak_val - current_low) / markup_peak_val

    return {
        "pivot_price": pivot,
        "tight_low": current_low,
        "last_pullback_low": current_low,
        "tightness": pullback_depth, 
        "base_depth": base_depth,
        "dry_up_ratio": dry_up_ratio,
        "touches": len(best_touches),
        "base_length": base_len,
        "type": "Retest"
    }

# ==========================================
# 6. דירוג setup
# ==========================================
def calc_setup_score(alert):
    rs_score = min(max(alert["rs_65"], 0) * 250, 25)
    tight_score = max(0, (1 - min(alert["tightness"], 0.10) / 0.10) * 20)
    dryup_score = max(0, (1 - min(alert["dry_up_ratio"], 1.0)) * 20)
    pivot_score = max(0, (1 - min(abs(alert["dist_to_pivot"]), 0.03) / 0.03) * 15)
    close_score = min(max(alert["close_strength"], 0), 1) * 10
    volume_score = min(alert["vol_ratio"] / 2.0, 1.0) * 5
    touch_score = min(alert.get("touches", 2), 4) * 2.5
    bonus = 5 if not alert["is_below_150"] else 0

    return round(rs_score + tight_score + dryup_score + pivot_score + close_score + volume_score + touch_score + bonus, 1)

# ==========================================
# 7. סריקת שוק ראשית
# ==========================================
def scan_market():
    tickers = load_tickers()
    if not tickers: return

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
        "total_scanned": 0, "pass_basic_data": 0, "pass_price_vol": 0, 
        "pass_sma": 0, "pass_52w": 0, "pass_rs": 0, "pass_market_cap": 0,
        "pass_pattern": 0, "pass_pivot_dist": 0, "final_approved": 0
    }

    for ticker in tickers:
        stats["total_scanned"] += 1
        print(f"סורק את {ticker}...", end="\r")

        try:
            df = yf.download(ticker, period=SCAN_PERIOD, auto_adjust=True, progress=False)
            df = normalize_ohlcv_columns(df)
            if df.empty or len(df) < 200: continue

            df = add_indicators(df)
            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            past_data = df.iloc[:-1].copy()

            if any(pd.isna(today[c]) for c in ["SMA_50", "SMA_150", "SMA_200", "ATR_14", "Vol_50", "High_252", "ROC_65"]): continue
            stats["pass_basic_data"] += 1

            close = float(today["Close"])
            open_price = float(today["Open"])
            if close < MIN_PRICE or float(today["DollarVol_50"]) < MIN_DOLLAR_VOL_50: continue
            stats["pass_price_vol"] += 1

            if close <= float(today["SMA_50"]): continue
            is_below_150 = close < float(today["SMA_150"])
            if not is_below_150:
                if not (close > float(today["SMA_150"]) > float(today["SMA_200"])): continue
            else:
                if float(today["SMA_50"]) <= float(yesterday["SMA_50"]): continue
            stats["pass_sma"] += 1

            max_dist = BRAIN["max_dist_from_52w_high_below_150"] if is_below_150 else BRAIN["max_dist_from_52w_high_normal"]
            if (close / float(today["High_252"])) - 1.0 < -max_dist: continue
            stats["pass_52w"] += 1

            stock_rs = float(today["ROC_65"]) - float(spy_rs)
            if stock_rs < BRAIN["min_rs_65"] * (2 if is_below_150 else 1): continue
            stats["pass_rs"] += 1

            market_cap = check_market_cap(ticker)
            if market_cap is not None and market_cap < MIN_MARKET_CAP: continue
            if market_cap is None and not BRAIN["allow_unknown_market_cap"]: continue
            stats["pass_market_cap"] += 1

            is_retest = False
            pattern = get_vcp_signal(past_data)
            if not pattern:
                pattern = get_retest_signal(past_data)
                if pattern: is_retest = True
            if not pattern: continue
            stats["pass_pattern"] += 1

            pivot = float(pattern["pivot_price"])
            dist_to_pivot = (close / pivot) - 1.0
            day_range = max(float(today["High"]) - float(today["Low"]), 1e-9)
            close_strength = (close - float(today["Low"])) / day_range
            vol_ratio = float(today["Volume"]) / float(today["Vol_50"]) if float(today["Vol_50"]) > 0 else 0.0

            if is_retest:
                status = "🔄 ריטסט לקו הפריצה"
                stats["pass_pivot_dist"] += 1
                if should_skip_spam(ticker, False): continue
            else:
                is_breakout = float(yesterday["Close"]) <= pivot and close > pivot
                is_near_breakout = (-BRAIN["watchlist_max_dist"] <= dist_to_pivot <= 0.0)

                if not (is_breakout or is_near_breakout):
                    waiting_for_pivot_tickers.append(f"{ticker} ({dist_to_pivot*100:.1f}%)")
                    continue
                stats["pass_pivot_dist"] += 1

                if should_skip_spam(ticker, is_breakout): continue

                if is_breakout:
                    if close_strength < (0.60 if is_below_150 else BRAIN["min_breakout_close_strength"]): continue
                    if vol_ratio < (1.8 if is_below_150 else BRAIN["breakout_volume_ratio"]): continue
                    if ((open_price / pivot) - 1.0) > BRAIN["max_gap_above_pivot"]: continue
                    status = "🔥 פריצה פעילה!"
                else:
                    if close_strength < 0.45 or vol_ratio < BRAIN["watchlist_volume_ratio"]: continue
                    status = "👀 מתבשלת (Watchlist)"

                if close > pivot * (1 + BRAIN["max_entry_extension"]): continue

            stop_price = min(float(pattern["tight_low"]), float(pattern["last_pullback_low"])) - (0.5 * float(today["ATR_14"]))
            if stop_price >= close: continue

            risk_pct = (close - stop_price) / close * 100
            if risk_pct > BRAIN["max_risk_pct"]: continue

            alert_data = {
                "ticker": ticker, "close": close, "pivot": pivot, "stop_loss": stop_price,
                "risk_pct": risk_pct, "vol_ratio": vol_ratio, "type": pattern["type"],
                "rs_65": stock_rs, "close_strength": close_strength, "status": status,
                "dist_to_pivot": dist_to_pivot, "tightness": float(pattern["tightness"]),
                "is_below_150": is_below_150, "dry_up_ratio": float(pattern["dry_up_ratio"]),
                "touches": int(pattern.get("touches", 2)), 
                "base_depth": float(pattern["base_depth"]),
                "base_length": int(pattern["base_length"]), "market_cap": market_cap
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
    print(f"✅ עברו זיהוי תבנית VCP או ריטסט: {stats['pass_pattern']}")
    print(f"🎯 קרובים לפיבוט (בחלון כניסה): {stats['pass_pivot_dist']}")
    print(f"🏆 אושרו סופית (לאחר ספאם, סיכון וכו'): {stats['final_approved']}")
    print("=" * 50)
    
    print("\n👀 ספסל - מניות שעברו תבנית אך רחוקות מהפיבוט:")
    if waiting_for_pivot_tickers:
        print(", ".join(waiting_for_pivot_tickers))
    else:
        print("אין מניות כאלו כרגע.")
    print("=" * 50)

    all_potentials_sorted = sorted(all_potentials, key=lambda x: (-x["setup_score"], abs(x["dist_to_pivot"])))
    final_selection = []
    below_150_count = 0

    for stock in all_potentials_sorted:
        if len(final_selection) >= TOP_RESULTS: break
        if stock["is_below_150"]:
            if below_150_count < 3:
                final_selection.append(stock)
                below_150_count += 1
        else:
            final_selection.append(stock)

    final_bo = [s for s in final_selection if "פריצה פעילה" in s["status"]]
    final_wl = [s for s in final_selection if "מתבשלת" in s["status"]]
    final_rt = [s for s in final_selection if "ריטסט" in s["status"]]

    if final_selection:
        print(f"🔥 הסריקה הסתיימה! נמצאו {len(final_selection)} מניות לשליחה.")
        msg = "🎯 <b>סריקת VCP וריטסטים יומית הסתיימה!</b>\n"
        if market_warning: msg += market_warning

        if final_bo:
            msg += f"🔥 <b>פריצות אקטיביות ({len(final_bo)}):</b>\n\n"
            for a in final_bo:
                tv_link = f"https://il.tradingview.com/chart/?symbol={a['ticker']}"
                warn = " ⚠️ (שיקום)" if a["is_below_150"] else ""
                msg += f"🚀 <b>{a['ticker']}</b> | VCP {warn}\n⭐ <b>ציון:</b> {a['setup_score']:.1f} | 📈 <b>RS:</b> {a['rs_65'] * 100:.1f}%\n📐 <b>כיווץ:</b> {a['tightness'] * 100:.1f}% | 📊 <b>ווליום:</b> {a['vol_ratio']:.1f}x\n🎯 <b>פיבוט:</b> ${a['pivot']:.2f} | 💵 <b>מחיר:</b> ${a['close']:.2f}\n🛡️ <b>סטופ:</b> ${a['stop_loss']:.2f}\n🔗 <a href='{tv_link}'>גרף ב-TradingView</a>\n────────────────\n"
                log_signal(a["ticker"], a["close"], a["status"])

        if final_rt:
            msg += f"🔄 <b>ריטסט לקו פריצה ישן ({len(final_rt)}):</b>\n\n"
            for a in final_rt:
                tv_link = f"https://il.tradingview.com/chart/?symbol={a['ticker']}"
                warn = " ⚠️ (שיקום)" if a["is_below_150"] else ""
                msg += f"🔄 <b>{a['ticker']}</b> | Pullback {warn}\n⭐ <b>ציון:</b> {a['setup_score']:.1f} | 📉 <b>עומק תיקון:</b> {a['tightness'] * 100:.1f}%\n🫗 <b>יובש בווליום:</b> {a['dry_up_ratio']:.2f}\n🎯 <b>פיבוט נבדק:</b> ${a['pivot']:.2f} | 💵 <b>מחיר:</b> ${a['close']:.2f}\n🛡️ <b>סטופ טכני:</b> ${a['stop_loss']:.2f}\n🔗 <a href='{tv_link}'>גרף ב-TradingView</a>\n────────────────\n"
                log_signal(a["ticker"], a["close"], a["status"])

        if final_wl:
            msg += f"👀 <b>מתבשלות לפריצה ({len(final_wl)}):</b>\n\n"
            for a in final_wl:
                tv_link = f"https://il.tradingview.com/chart/?symbol={a['ticker']}"
                warn = " ⚠️ (שיקום)" if a["is_below_150"] else ""
                msg += f"⏳ <b>{a['ticker']}</b> | VCP {warn}\n⭐ <b>ציון:</b> {a['setup_score']:.1f} | 📐 <b>כיווץ:</b> {a['tightness'] * 100:.1f}%\n🎯 <b>פיבוט יעד:</b> ${a['pivot']:.2f} (מרחק: {a['dist_to_pivot'] * 100:.1f}%)\n💵 <b>מחיר:</b> ${a['close']:.2f} | 🛡️ <b>סטופ:</b> ${a['stop_loss']:.2f}\n🔗 <a href='{tv_link}'>גרף ב-TradingView</a>\n────────────────\n"
                log_signal(a["ticker"], a["close"], a["status"])

        send_telegram(msg)
    else:
        print("💤 הסריקה הסתיימה. לא נמצאו מניות חדשות לשליחה בסיבוב זה.")
        send_telegram(f"✅ הסריקה הסתיימה.\n\n{market_warning}אין פריצות או ריטסטים חדשים שלא נשלחו כבר היום.")
    print("=" * 50)

if __name__ == "__main__":
    scan_market()
