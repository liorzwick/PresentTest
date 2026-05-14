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
MIN_DOLLAR_VOL_50 = 15_000_000  # 15 מיליון דולר מחזור דולרי מינימלי
MIN_PRICE = 8.0                 # 8 דולר מחיר מינימלי
COOLDOWN_DAYS = 5
TOP_RESULTS = 15 
SCAN_PERIOD = "2y"              # שנתיים כדי לבסס ממוצעים ארוכים כראוי

market_cap_cache = {}

def load_brain():
    brain = {
        "min_breakout_close_strength": 0.55,
        "min_rs_65": 0.03,
        "max_dist_from_52w_high_normal": 0.45,   
        "max_dist_from_52w_high_below_150": 0.50, 
        "max_gap_above_pivot": 0.02,
        "max_entry_extension": 0.04,          
        "breakout_volume_ratio": 1.3,         
        "watchlist_volume_ratio": 0.75,
        "pivot_tolerance": 0.055,             
        "max_risk_pct": 12.0,
        "allow_unknown_market_cap": True,
        "min_atr_pct": 0.02,
        "min_touch_count": 2,
        "watchlist_max_dist": 0.07,
    }
    try:
        if os.path.exists("brain.json"):
            with open("brain.json", "r", encoding="utf-8") as f:
                brain.update(json.load(f))
    except Exception:
        pass
    return brain

BRAIN = load_brain()

# ==========================================
# 2. עזרי קבצים, טלגרם ואנטי-ספאם
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
    print("\n" + "="*25)
    print("שולח הודעה מסכמת לטלגרם...")
    print("="*25 + "\n")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=12)
    except Exception as e:
        print(f"⚠️ שגיאה בשליחה לטלגרם: {e}")

def save_to_smart_memory(ticker, price, stop_loss, risk_pct, vol_ratio, pivot, close_strength, rs_65, tightness, pattern_type, status, setup_score, dry_up_ratio, touches):
    memory_file = "smart_memory.csv"
    now = datetime.now().strftime("%Y-%m-%d")
    new_record = pd.DataFrame([{
        "Date": now, "Ticker": ticker, "Price": round(float(price), 2),
        "Pivot": round(float(pivot), 2), "Stop_Loss": round(float(stop_loss), 2),
        "Risk_Pct": round(float(risk_pct), 2), "Volume_Ratio": round(float(vol_ratio), 2),
        "Close_Strength": round(float(close_strength), 2), "RS_65": round(float(rs_65), 4),
        "Tightness_Pct": round(float(tightness) * 100, 2), "Pattern_Type": pattern_type,
        "Status": status, "Setup_Score": round(float(setup_score), 1),
        "DryUp_Ratio": round(float(dry_up_ratio), 2), "Touches": int(touches)
    }])
    append_dataframe(new_record, memory_file)

def should_skip_spam(ticker, current_status):
    memory_file = "smart_memory.csv"
    if not os.path.isfile(memory_file): return False
    try:
        df = pd.read_csv(memory_file, encoding="utf-8-sig")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        ticker_history = df[df["Ticker"] == ticker].sort_values(by="Date", ascending=False)
        if ticker_history.empty: return False

        last_record = ticker_history.iloc[0]
        last_date = last_record["Date"]
        last_status = str(last_record.get("Status", ""))
        days_passed = (datetime.now().date() - last_date.date()).days

        if days_passed == 0 and last_status == current_status: return True
        if "פריצה" in current_status:
            if "פריצה" in last_status and days_passed < 2: return True
            return False 
        if "מתבשלת" in current_status:
            if ("מתבשלת" in last_status or "פריצה" in last_status) and days_passed < COOLDOWN_DAYS: return True
            return False 
        return days_passed < COOLDOWN_DAYS
    except Exception:
        return False

def load_tickers():
    if os.path.exists(CUSTOM_TICKERS_FILE):
        try:
            df = pd.read_csv(CUSTOM_TICKERS_FILE, encoding="utf-8-sig")
            col_name = next((c for c in df.columns if c.strip().lower() in ["ticker", "symbol"]), None)
            if col_name:
                tickers = df[col_name].dropna().astype(str).str.strip().str.upper().str.replace(".", "-", regex=False).tolist()
                return sorted(list(set([t for t in tickers if t.replace("-", "").isalnum()])))
        except Exception:
            pass
    return ["AAPL", "MSFT", "NVDA"]

# ==========================================
# 3. אינדיקטורים
# ==========================================
def normalize_ohlcv_columns(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if getattr(df.index, "tz", None) is not None: df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    return df

def add_indicators(df):
    df = normalize_ohlcv_columns(df)
    df["SMA_50"] = df["Close"].rolling(50, min_periods=25).mean()
    df["SMA_150"] = df["Close"].rolling(150, min_periods=75).mean()
    df["SMA_200"] = df["Close"].rolling(200, min_periods=100).mean()
    df["Vol_50"] = df["Volume"].rolling(50, min_periods=25).mean()
    df["DollarVol_50"] = df["Close"].rolling(50, min_periods=25).mean() * df["Vol_50"]
    df["Prev_Close"] = df["Close"].shift(1)
    df["ROC_65"] = df["Close"].pct_change(65)
    df["High_252"] = df["High"].rolling(252, min_periods=120).max()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Prev_Close"]).abs(), (df["Low"] - df["Prev_Close"]).abs()], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14, min_periods=7).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"] 
    return df

def get_spy_data():
    try:
        spy = yf.download("SPY", period=SCAN_PERIOD, auto_adjust=True, progress=False)
        spy = normalize_ohlcv_columns(spy)
        if not spy.empty and len(spy) > 200:
            spy["ROC_65"] = spy["Close"].pct_change(65)
            return spy
    except Exception: pass
    return pd.DataFrame()

# ==========================================
# 4. מנוע זיהוי תבניות קלאסיות (מבוסס מבנה)
# ==========================================
def calculate_dry_up(vols, base_start, base_end):
    base_vol = np.mean(vols[base_start:base_end]) if base_end > base_start else 1
    handle_vol = np.mean(vols[-5:])
    return float(handle_vol / base_vol) if base_vol > 0 else 1.0

def get_cup_and_handle(highs, lows, vols, closes, n):
    if n < 60: return None
    
    # 1. השפה השמאלית
    recent_highs = highs[-100:-10]
    if len(recent_highs) == 0: return None
    left_lip_idx = int(np.argmax(recent_highs))
    left_lip_val = float(recent_highs[left_lip_idx])
    
    # 2. תחתית הכוס
    if left_lip_idx > len(recent_highs) - 15: return None 
    cup_low_idx = left_lip_idx + int(np.argmin(lows[left_lip_idx: -5]))
    cup_low_val = float(lows[cup_low_idx])
    
    cup_depth = (left_lip_val - cup_low_val) / left_lip_val
    if cup_depth < 0.12 or cup_depth > 0.45: return None 
    
    # 3. השפה הימנית
    right_side_highs = highs[cup_low_idx : -2]
    if len(right_side_highs) == 0: return None
    right_lip_idx = cup_low_idx + int(np.argmax(right_side_highs))
    right_lip_val = float(highs[right_lip_idx])
    
    if abs(left_lip_val - right_lip_val) / left_lip_val > 0.08: return None
    pivot = max(left_lip_val, right_lip_val)
    
    # 4. הידית
    handle_len = (len(highs) - 1) - right_lip_idx
    if handle_len < 3 or handle_len > 25: return None 
    
    handle_low = float(np.min(lows[right_lip_idx:]))
    handle_depth = (right_lip_val - handle_low) / right_lip_val
    
    if handle_depth > cup_depth * 0.6: return None 
    if handle_low < cup_low_val + (pivot - cup_low_val) * 0.4: return None 
    
    return {
        "type": "☕ Cup & Handle", "pivot_price": pivot, "tight_low": handle_low,
        "last_pullback_low": handle_low, "tightness": handle_depth, "base_depth": cup_depth,
        "dry_up_ratio": calculate_dry_up(vols, left_lip_idx, right_lip_idx), "touches": 2, "base_length": len(highs) - left_lip_idx
    }

def get_bull_flag(highs, lows, vols, closes, n):
    if n < 30: return None
    
    recent_30_highs = highs[-30:-2]
    if len(recent_30_highs) == 0: return None
    pole_peak_idx = int(np.argmax(recent_30_highs))
    
    if pole_peak_idx < 5 or pole_peak_idx > 27: return None 
    
    pole_peak_val = float(recent_30_highs[pole_peak_idx])
    pole_start_val = float(np.min(lows[-30:-2][:pole_peak_idx+1]))
    
    if (pole_peak_val - pole_start_val) / pole_start_val < 0.12: return None 
    
    flag_zone_lows = lows[-30:][pole_peak_idx+1:]
    if len(flag_zone_lows) == 0: return None
    flag_low = float(np.min(flag_zone_lows))
    
    flag_depth = (pole_peak_val - flag_low) / pole_peak_val
    if flag_depth > 0.15: return None 
    
    current_close = float(closes[-1])
    if current_close < flag_low + (pole_peak_val - flag_low) * 0.4: return None 
    
    return {
        "type": "🚩 Bull Flag", "pivot_price": pole_peak_val, "tight_low": flag_low,
        "last_pullback_low": flag_low, "tightness": flag_depth, "base_depth": flag_depth,
        "dry_up_ratio": 1.0, "touches": 1, "base_length": (len(highs) - 1) - pole_peak_idx
    }

def get_darvas_box(highs, lows, vols, closes, n):
    box_length = 30 
    if n < box_length + 40: return None

    window_highs = highs[-box_length:-2]
    window_lows = lows[-box_length:-2]
    window_closes = closes[-box_length:-2]
    
    box_top = float(np.max(window_highs))
    box_bottom = float(np.min(window_lows))

    box_depth = (box_top - box_bottom) / box_top
    if box_depth < 0.04 or box_depth > 0.15: return None 

    days_in_box = np.sum((window_closes >= box_bottom * 0.98) & (window_closes <= box_top * 1.02))
    if (days_in_box / len(window_closes)) < 0.70: return None 
        
    top_touches = np.where(window_highs >= box_top * 0.975)[0]
    if len(top_touches) < 2: return None
    if top_touches[-1] - top_touches[0] < 10: return None 

    pre_box_close = float(closes[-(box_length + 20)])
    if pre_box_close >= box_bottom * 0.95: return None 

    current_close = float(closes[-1])
    if current_close < box_bottom + (box_top - box_bottom) * 0.65: return None 

    box_vol = np.mean(vols[-box_length:-5])
    recent_vol = np.mean(vols[-5:])
    dry_up = recent_vol / box_vol if box_vol > 0 else 1.0

    return {
        "type": "📦 Darvas Box", "pivot_price": box_top, "tight_low": box_bottom,
        "last_pullback_low": box_bottom, "tightness": box_depth, "base_depth": box_depth,
        "dry_up_ratio": dry_up, "touches": len(top_touches), "base_length": box_length
    }

def get_double_bottom(highs, lows, vols, n):
    if n < 80: return None
    
    recent_lows = lows[-60:-10]
    if len(recent_lows) == 0: return None
    left_bottom_idx = int(np.argmin(recent_lows))
    left_bottom_val = float(recent_lows[left_bottom_idx])
    
    mid_section = highs[left_bottom_idx : -5]
    if len(mid_section) < 5: return None
    mid_peak_idx = left_bottom_idx + int(np.argmax(mid_section))
    mid_peak_val = float(highs[mid_peak_idx])
    
    right_section = lows[mid_peak_idx : -1]
    if len(right_section) < 3: return None
    right_bottom_idx = mid_peak_idx + int(np.argmin(right_section))
    right_bottom_val = float(lows[right_bottom_idx])
    
    if abs(left_bottom_val - right_bottom_val) / left_bottom_val > 0.05: return None
    
    base_depth = (mid_peak_val - min(left_bottom_val, right_bottom_val)) / mid_peak_val
    if base_depth < 0.10: return None
    
    pivot = mid_peak_val
    handle_low = float(np.min(lows[right_bottom_idx:]))
    handle_depth = (pivot - handle_low) / pivot
    
    return {
        "type": "🧲 Double Bottom", "pivot_price": pivot, "tight_low": handle_low,
        "last_pullback_low": handle_low, "tightness": handle_depth, "base_depth": base_depth,
        "dry_up_ratio": 1.0, "touches": 2, "base_length": right_bottom_idx - left_bottom_idx
    }

def check_classical_patterns(hist):
    hist_filtered = hist.dropna(subset=['High', 'Low', 'Volume', 'Close'])
    if len(hist_filtered) < 60: return None

    highs = hist_filtered["High"].astype(float).values
    lows = hist_filtered["Low"].astype(float).values
    vols = hist_filtered["Volume"].astype(float).values
    closes = hist_filtered["Close"].astype(float).values
    n = len(hist_filtered)

    pattern = get_bull_flag(highs, lows, vols, closes, n)
    if pattern: return pattern

    pattern = get_cup_and_handle(highs, lows, vols, closes, n)
    if pattern: return pattern

    pattern = get_double_bottom(highs, lows, vols, n)
    if pattern: return pattern

    pattern = get_darvas_box(highs, lows, vols, closes, n)
    if pattern: return pattern

    return None

# ==========================================
# 5. דירוג וסריקה
# ==========================================
def calc_setup_score(alert):
    rs_score = min(max(alert["rs_65"], 0) * 250, 25)
    tight_score = max(0, (1 - min(alert["tightness"], 0.10) / 0.10) * 20)
    dryup_score = max(0, (1 - min(alert["dry_up_ratio"], 1.0)) * 20)
    pivot_score = max(0, (1 - min(abs(alert["dist_to_pivot"]), 0.03) / 0.03) * 15)
    close_score = min(max(alert["close_strength"], 0), 1) * 10
    volume_score = min(alert["vol_ratio"] / 2.0, 1.0) * 5
    bonus = 5 if not alert["is_below_150"] else 0
    return round(rs_score + tight_score + dryup_score + pivot_score + close_score + volume_score + bonus, 1)

def scan_market():
    tickers = load_tickers()
    print(f"✅ נטענו {len(tickers)} מניות לסריקה מתוך הקובץ.")
    if not tickers: return

    spy = get_spy_data()
    spy_rs = float(spy.iloc[-1]["ROC_65"]) if not spy.empty and pd.notna(spy.iloc[-1]["ROC_65"]) else 0.0

    all_potentials, waiting_for_pivot_tickers = [], []
    stats = {"total_scanned": 0, "pass_basic": 0, "pass_pattern": 0, "pass_pivot_dist": 0, "final_approved": 0}

    for ticker in tickers:
        stats["total_scanned"] += 1
        print(f"סורק את {ticker}...", end="\r")

        try:
            df = yf.download(ticker, period=SCAN_PERIOD, auto_adjust=True, progress=False)
            if df.empty or len(df) < 200: continue
            df = add_indicators(df)
            today, yesterday, past_data = df.iloc[-1], df.iloc[-2], df.iloc[:-1].copy()

            if any(pd.isna(today[c]) for c in ["SMA_50", "SMA_150", "SMA_200", "ATR_14", "ATR_Pct"]): continue

            close, open_price = float(today["Close"]), float(today["Open"])
            if close < MIN_PRICE or float(today["DollarVol_50"]) < MIN_DOLLAR_VOL_50: continue
            if close <= float(today["SMA_50"]): continue

            if float(today["ATR_Pct"]) < BRAIN["min_atr_pct"]: continue

            stats["pass_basic"] += 1

            is_below_150 = close < float(today["SMA_150"])
            max_dist = BRAIN["max_dist_from_52w_high_below_150"] if is_below_150 else BRAIN["max_dist_from_52w_high_normal"]
            if (close / float(today["High_252"])) - 1.0 < -max_dist: continue

            stock_rs = float(today["ROC_65"]) - float(spy_rs)
            if stock_rs < BRAIN["min_rs_65"]: continue

            pattern = check_classical_patterns(past_data)
            if not pattern: continue
            stats["pass_pattern"] += 1

            pivot = float(pattern["pivot_price"])
            dist_to_pivot = (close / pivot) - 1.0

            if dist_to_pivot < -0.15 or dist_to_pivot > 0.05: continue
            stats["pass_pivot_dist"] += 1 

            vol_ratio = float(today["Volume"]) / float(today["Vol_50"]) if float(today["Vol_50"]) > 0 else 0.0
            close_strength = (close - float(today["Low"])) / max(float(today["High"]) - float(today["Low"]), 1e-9)

            is_breakout = float(yesterday["Close"]) <= pivot and close > pivot
            is_near_breakout = (-BRAIN["watchlist_max_dist"] <= dist_to_pivot <= 0.0)

            if is_breakout:
                status = "🔥 פריצה פעילה!" if close_strength >= 0.55 and vol_ratio >= 1.3 else "🪑 ספסל"
            elif is_near_breakout:
                status = "👀 מתבשלת (Watchlist)"
            else:
                status = "🪑 ספסל"

            stop_price = min(float(pattern["tight_low"]), float(pattern["last_pullback_low"])) - (0.5 * float(today["ATR_14"]))
            risk_pct = (close - stop_price) / close * 100

            if status != "🪑 ספסל":
                if stop_price >= close or risk_pct > 12.0: status = "🪑 ספסל"

            if should_skip_spam(ticker, status): continue

            if status == "🪑 ספסל": waiting_for_pivot_tickers.append(f"{ticker} ({dist_to_pivot*100:.1f}%)")

            alert_data = {
                "ticker": ticker, "close": close, "pivot": pivot, "stop_loss": stop_price,
                "risk_pct": risk_pct, "vol_ratio": vol_ratio, "type": pattern["type"],
                "rs_65": stock_rs, "close_strength": close_strength, "status": status,
                "dist_to_pivot": dist_to_pivot, "tightness": float(pattern["tightness"]),
                "is_below_150": is_below_150, "dry_up_ratio": float(pattern["dry_up_ratio"]),
                "touches": int(pattern["touches"]), "base_depth": float(pattern["base_depth"]),
                "base_length": int(pattern["base_length"])
            }
            alert_data["setup_score"] = calc_setup_score(alert_data)
            all_potentials.append(alert_data)
            stats["final_approved"] += 1

        except Exception as e: 
            print(f"\n⚠️ שגיאה בסימול {ticker}: {e}")

    # הדפסות של מערכת הסינון מועברות לקונסול בלבד:
    print("\n\n--- 📊 דוח סינון סריקה יומית 📊 ---")
    for key, val in stats.items():
        print(f"{key}: {val}")
    print("----------------------------------\n")

    prime = sorted([s for s in all_potentials if s["status"] != "🪑 ספסל"], key=lambda x: -x["setup_score"])
    bench = sorted([s for s in all_potentials if s["status"] == "🪑 ספסל"], key=lambda x: abs(x["dist_to_pivot"]))

    final_selection = (prime + bench)[:TOP_RESULTS]
    if not final_selection:
        send_telegram("✅ הסריקה הקלאסית הסתיימה. אין פריצות או תבניות קלאסיות חדשות.")
        return

    msg = "🎯 <b>סריקת תבניות קלאסיות יומית!</b>\n"
    msg += f"<i>(מציג עד {TOP_RESULTS} מניות מובחרות, מסודרות לפי תבניות טכניות)</i>\n\n"

    pattern_groups = {}
    for s in final_selection:
        ptype = s["type"]
        if ptype not in pattern_groups: pattern_groups[ptype] = []
        pattern_groups[ptype].append(s)

    for ptype, stocks in pattern_groups.items():
        icon = ptype.split()[0]
        pattern_name = ptype.replace(icon, "").strip()
        msg += f"────────────────\n"
        msg += f"{icon} <b>תבנית {pattern_name} ({len(stocks)} מניות):</b>\n\n"

        for a in stocks:
            status_icon = "🔥" if "פריצה" in a["status"] else "⏳" if "מתבשלת" in a["status"] else "🪑"
            clean_status = a['status'].replace(' (Watchlist)', '')

            msg += f"{status_icon} <b>{a['ticker']}</b> | סטטוס: {clean_status}\n"
            msg += f"⭐ <b>ציון:</b> {a['setup_score']:.1f} | 📐 <b>כיווץ:</b> {a['tightness'] * 100:.1f}%\n"
            msg += f"🎯 <b>פיבוט:</b> ${a['pivot']:.2f} | 💵 <b>מחיר:</b> ${a['close']:.2f}\n"
            msg += f"🛡️ <b>סטופ:</b> ${a['stop_loss']:.2f}\n"
            msg += f"🔗 <a href='https://il.tradingview.com/chart/?symbol={a['ticker']}'>TradingView</a>\n\n"

            save_to_smart_memory(a["ticker"], a["close"], a["stop_loss"], a["risk_pct"], a["vol_ratio"], a["pivot"], a["close_strength"], a["rs_65"], a["tightness"], a["type"], a["status"], a["setup_score"], a["dry_up_ratio"], a["touches"])

    send_telegram(msg)

if __name__ == "__main__":
    scan_market()
