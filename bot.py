import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime
import warnings

# ייבוא ספריית הראייה הממוחשבת 
from fastdtw import fastdtw

warnings.filterwarnings("ignore")

# ==========================================
# 1. הגדרות כלליות
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "") 
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_GROUP", "") 

CUSTOM_TICKERS_FILE = "my_stocks.csv"

MIN_MARKET_CAP = 2_000_000_000
MIN_DOLLAR_VOL_50 = 15_000_000  
MIN_PRICE = 8.0                 
COOLDOWN_DAYS = 5
TOP_RESULTS = 15 
SCAN_PERIOD = "2y"              

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
    print("שולח הודעה לטלגרם...")
    print("="*25 + "\n")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ פרטי טלגרם חסרים, מדלג על שליחה.")
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
    return ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "PLTR", "AMD"]

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
# 4. מנוע זיהוי תבניות מבוסס ראייה (DTW) - צלף!
# ==========================================
def normalize_series(series):
    series_array = np.array(series)
    min_val = np.min(series_array)
    max_val = np.max(series_array)
    if max_val == min_val:
        return np.zeros(len(series_array))
    return (series_array - min_val) / (max_val - min_val)

def get_dtw_templates():
    templates = {}
    
    # 📉 ירדנו ל-0.075! (רק 7.5% שגיאה בממוצע ליום)
    strict_threshold = 0.075

    pole = np.linspace(0, 1.0, 10)
    x_flag = np.linspace(0, 4*np.pi, 20)
    flag_trend = np.linspace(1.0, 0.7, 20)
    flag_waves = flag_trend + np.sin(x_flag) * 0.05 
    templates["🚩 דגל שורי"] = {"data": np.concatenate((pole, flag_waves)), "windows": [20, 30, 45, 60], "threshold": strict_threshold}

    rise = np.linspace(0, 1.0, 10)
    initial_pullback = np.linspace(1.0, 0.8, 5) 
    x_box = np.linspace(0, 6*np.pi, 35)
    box = np.ones(35) * 0.9 + np.sin(x_box) * 0.05
    templates["📦 מלבן דרווס"] = {"data": np.concatenate((rise, initial_pullback, box)), "windows": [40, 60, 90, 120], "threshold": strict_threshold}

    l_drop = np.linspace(1.0, 0.1, 15)  
    m_up = np.linspace(0.1, 0.6, 15)    
    m_down = np.linspace(0.6, 0.0, 15)  
    r_up = np.linspace(0.0, 1.0, 15)    
    templates["🧲 תחתית כפולה"] = {"data": np.concatenate((l_drop, m_up, m_down, r_up)), "windows": [50, 80, 120, 180], "threshold": strict_threshold}

    left_cup = np.linspace(-1, 0, 45)**2
    right_cup = np.linspace(0, 1, 20)**2
    x_handle = np.linspace(0, 2*np.pi, 15)
    handle_trend = np.linspace(1.0, 0.7, 15)
    handle_waves = handle_trend + np.cos(x_handle) * 0.03
    templates["☕ ספל וידית"] = {"data": np.concatenate((left_cup, right_cup, handle_waves)), "windows": [60, 90, 150, 250], "threshold": strict_threshold}

    return templates

def check_all_dtw_patterns(closes):
    templates = get_dtw_templates()
    best_pattern = None
    best_score = float('inf')

    for name, config in templates.items():
        for window in config["windows"]:
            if len(closes) < window:
                continue
            
            current_closes = closes[-window:]
            norm_current = normalize_series(current_closes)
            norm_template = normalize_series(config["data"])
            
            distance, path = fastdtw(norm_current, norm_template, dist=lambda x, y: abs(x - y))
            avg_distance = distance / window
            
            if avg_distance < config["threshold"] and avg_distance < best_score:
                
                # --- Sanity Checks (סינוני היגיון אנושי) ---
                # מונע מהמתמטיקה לאשר צורות מעוותות מדי
                if "דגל" in name and current_closes[-1] <= current_closes[0] * 1.05:
                    continue # סוף הדגל חייב להיות גבוה ב-5% לפחות מתחילת התורן
                    
                if "דרווס" in name:
                    # מלבן דרווס דורש התכנסות שקטה לקראת הסוף. נפסול אם יש תנודתיות של מעל 15% בקצה.
                    if (np.max(current_closes[-15:]) - np.min(current_closes[-15:])) / np.max(current_closes[-15:]) > 0.15:
                        continue
                        
                if ("ספל" in name or "תחתית" in name) and current_closes[-1] <= np.min(current_closes) * 1.05:
                    continue # לא יכול להיות שסיימנו בקרקעית המוחלטת, חייב התרוממות מהשפל

                best_score = avg_distance
                w = window
                
                if "דגל" in name:
                    pivot = float(np.max(current_closes[:int(w*0.5)]))
                    low = float(np.min(current_closes[-int(w*0.5):]))
                elif "דרווס" in name:
                    pivot = float(np.max(current_closes[-int(w*0.7):]))
                    low = float(np.min(current_closes[-int(w*0.7):]))
                elif "תחתית" in name:
                    pivot = float(np.max(current_closes[int(w*0.3):int(w*0.7)]))
                    low = float(np.min(current_closes[-int(w*0.4):]))
                else: 
                    pivot = float(np.max(current_closes[int(w*0.6):int(w*0.9)]))
                    low = float(np.min(current_closes[-int(w*0.3):]))
                
                tightness = (pivot - low) / pivot if pivot > 0 else 0
                
                best_pattern = {
                    "type": f"{name} (DTW)",
                    "dtw_distance": round(avg_distance, 3),
                    "threshold": config["threshold"],
                    "pivot_price": pivot,
                    "tight_low": low,
                    "last_pullback_low": low,
                    "tightness": tightness,
                    "base_depth": 0.20,
                    "dry_up_ratio": 1.0, 
                    "touches": 2,
                    "base_length": window
                }
                
    return best_pattern

# ==========================================
# 5. דירוג וסריקה
# ==========================================
def calc_setup_score(alert):
    rs_score = min(max(alert["rs_65"], 0) * 250, 25)
    tight_score = max(0, (1 - min(alert["tightness"], 0.10) / 0.10) * 20)
    pivot_score = max(0, (1 - min(abs(alert["dist_to_pivot"]), 0.03) / 0.03) * 15)
    close_score = min(max(alert["close_strength"], 0), 1) * 10
    volume_score = min(alert["vol_ratio"] / 2.0, 1.0) * 5
    
    dtw_score = max(0, (alert["threshold"] - alert["dtw_distance"]) * 150) # הגדלתי השפעה בגלל המספרים הקטנים
    bonus = 5 if not alert["is_below_150"] else 0
    
    return round(rs_score + tight_score + pivot_score + close_score + volume_score + dtw_score + bonus, 1)

def scan_market():
    tickers = load_tickers()
    print(f"✅ נטענו {len(tickers)} מניות לסריקה.")
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

            close = float(today["Close"])
            if close < MIN_PRICE or float(today["DollarVol_50"]) < MIN_DOLLAR_VOL_50: continue
            if close <= float(today["SMA_50"]): continue
            if float(today["ATR_Pct"]) < BRAIN["min_atr_pct"]: continue

            stats["pass_basic"] += 1

            is_below_150 = close < float(today["SMA_150"])
            max_dist = BRAIN["max_dist_from_52w_high_below_150"] if is_below_150 else BRAIN["max_dist_from_52w_high_normal"]
            if (close / float(today["High_252"])) - 1.0 < -max_dist: continue

            stock_rs = float(today["ROC_65"]) - float(spy_rs)
            if stock_rs < BRAIN["min_rs_65"]: continue

            past_filtered = past_data.dropna(subset=['Close'])
            if len(past_filtered) < 30: continue
            
            closes = past_filtered["Close"].astype(float).values

            pattern = check_all_dtw_patterns(closes)

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
                "base_length": int(pattern["base_length"]),
                "dtw_distance": float(pattern["dtw_distance"]),
                "threshold": float(pattern["threshold"])
            }
                
            alert_data["setup_score"] = calc_setup_score(alert_data)
            all_potentials.append(alert_data)
            stats["final_approved"] += 1

        except Exception as e: 
            pass

    print("\n--- 📊 דוח סינון מולטי-טיים פריים (DTW - מחמיר) 📊 ---")
    for key, val in stats.items():
        print(f"{key}: {val}")
    print("----------------------------------------------------\n")

    if all_potentials:
        print("--- 📋 רשימת המניות שעברו את הסינון 📋 ---")
        sorted_all = sorted(all_potentials, key=lambda x: -x["setup_score"])
        for idx, s in enumerate(sorted_all, 1):
            clean_type = s['type'].replace(' (DTW)', '')
            print(f"{idx}. {s['ticker']:<5} | תבנית: {clean_type:<15} | ציון: {s['setup_score']:<4.1f} | טווח: {s['base_length']} ימים | שגיאה: {s['dtw_distance']:.3f} | פיבוט: ${s['pivot']:.2f} | מחיר: ${s['close']:.2f}")
        print("----------------------------------------------\n")
    else:
        print("לא נמצאו מניות העונות לקריטריונים.")

    prime = sorted([s for s in all_potentials if s["status"] != "🪑 ספסל"], key=lambda x: -x["setup_score"])
    bench = sorted([s for s in all_potentials if s["status"] == "🪑 ספסל"], key=lambda x: abs(x["dist_to_pivot"]))

    final_selection = (prime + bench)[:TOP_RESULTS]
    if not final_selection:
        send_telegram("✅ הסריקה הסתיימה. הבוט הפך לקפדני, לא נמצאו תבניות מתאימות הפעם.")
        return

    msg = "🎯 <b>סריקת ראייה ממוחשבת מרובת זמנים (DTW)!</b>\n"
    msg += f"<i>(מציג עד {TOP_RESULTS} מניות שזוהו באמצעות התאמת צורה)</i>\n\n"

    pattern_groups = {}
    for s in final_selection:
        ptype = s["type"]
        if ptype not in pattern_groups: pattern_groups[ptype] = []
        pattern_groups[ptype].append(s)

    for ptype, stocks in pattern_groups.items():
        icon = ptype.split()[0] if " " in ptype else "📈"
        pattern_name = ptype.replace(icon, "").replace("(DTW)", "").strip()
        msg += f"────────────────\n"
        msg += f"{icon} <b>תבנית {pattern_name} ({len(stocks)} מניות):</b>\n\n"

        for a in stocks:
            status_icon = "🔥" if "פריצה" in a["status"] else "⏳" if "מתבשלת" in a["status"] else "🪑"
            clean_status = a['status'].replace(' (Watchlist)', '')

            msg += f"{status_icon} <b>{a['ticker']}</b> | סטטוס: {clean_status}\n"
            msg += f"⭐ <b>ציון:</b> {a['setup_score']:.1f} | 📅 <b>טווח:</b> {a['base_length']} ימים\n"
            msg += f"🎯 <b>פיבוט:</b> ${a['pivot']:.2f} | 💵 <b>מחיר:</b> ${a['close']:.2f}\n"
            msg += f"🛡️ <b>סטופ:</b> ${a['stop_loss']:.2f}\n"
            msg += f"🔗 <a href='https://il.tradingview.com/chart/?symbol={a['ticker']}'>TradingView</a>\n\n"

            save_to_smart_memory(a["ticker"], a["close"], a["stop_loss"], a["risk_pct"], a["vol_ratio"], a["pivot"], a["close_strength"], a["rs_65"], a["tightness"], a["type"], a["status"], a["setup_score"], a["dry_up_ratio"], a["touches"])

    send_telegram(msg)

if __name__ == "__main__":
    scan_market()
