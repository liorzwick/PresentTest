import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime
import warnings
from fastdtw import fastdtw

warnings.filterwarnings("ignore")

# ==========================================
# 1. הגדרות כלליות
# ==========================================
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_GROUP", "")

CUSTOM_TICKERS_FILE = "mystock.csv"

MIN_DOLLAR_VOL_50 = 15_000_000
MIN_PRICE         = 8.0
COOLDOWN_DAYS     = 5
TOP_RESULTS       = 15
SCAN_PERIOD       = "2y"

def load_brain():
    brain = {
        "min_rs_65":                       0.03,
        "max_dist_from_52w_high_normal":   0.45,
        "max_dist_from_52w_high_below_150": 0.50,
        "max_gap_above_pivot":             0.02,
        "breakout_volume_ratio":           1.3,
        "max_risk_pct":                    12.0,
        "min_atr_pct":                     0.02,
        # watchlist: עד -7% | ספסל: עד -10%
        "watchlist_max_dist":              0.07,
        # VCP — בונוס בלבד, לא פילטר חובה
        "vcp_bonus_threshold":             0.85,
        "rs_periods":                      [63, 126, 189],
        "rs_weights":                      [0.4, 0.35, 0.25],
        # DTW — min_corr מוקל לנתונים אמיתיים
        "min_corr_flag":                   0.80,
        "min_corr_darvas":                 0.76,
        "min_corr_cup":                    0.75,
        # שיני מסור — הורם ל-0.45
        "max_std_noise":                   0.45,
        # ציון מינימום לפני שליחה
        "min_setup_score":                 45.0,
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
    print("\n" + "="*30)
    print("שולח הודעה לטלגרם...")
    print("="*30 + "\n")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ פרטי טלגרם חסרים, מדלג על שליחה.")
        return
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, "text": message,
        "parse_mode": "HTML", "disable_web_page_preview": True
    }
    try:
        requests.post(url, json=payload, timeout=12)
    except Exception as e:
        print(f"⚠️ שגיאה בשליחה לטלגרם: {e}")

def save_to_smart_memory(ticker, price, stop_loss, risk_pct, vol_ratio, pivot,
                          close_strength, rs_weighted, tightness, pattern_type,
                          status, setup_score, dry_up_ratio, touches, contraction_ratio):
    memory_file = "smart_memory.csv"
    now         = datetime.now().strftime("%Y-%m-%d")
    new_record  = pd.DataFrame([{
        "Date": now, "Ticker": ticker,
        "Price": round(float(price), 2),
        "Pivot": round(float(pivot), 2),
        "Stop_Loss": round(float(stop_loss), 2),
        "Risk_Pct": round(float(risk_pct), 2),
        "Volume_Ratio": round(float(vol_ratio), 2),
        "Close_Strength": round(float(close_strength), 2),
        "RS_Weighted": round(float(rs_weighted), 4),
        "Tightness_Pct": round(float(tightness) * 100, 2),
        "Pattern_Type": pattern_type,
        "Status": status,
        "Setup_Score": round(float(setup_score), 1),
        "DryUp_Ratio": round(float(dry_up_ratio), 2),
        "Touches": int(touches),
        "Contraction_Ratio": round(float(contraction_ratio), 3)
    }])
    append_dataframe(new_record, memory_file)

def should_skip_spam(ticker, current_status):
    memory_file = "smart_memory.csv"
    if not os.path.isfile(memory_file): return False
    try:
        df = pd.read_csv(memory_file, encoding="utf-8-sig")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        ticker_history = df[df["Ticker"] == ticker].sort_values("Date", ascending=False)
        if ticker_history.empty: return False

        last       = ticker_history.iloc[0]
        last_date  = last["Date"]
        last_status = str(last.get("Status", ""))
        days_passed = (datetime.now().date() - last_date.date()).days

        if days_passed == 0 and last_status == current_status: return True
        if "פריצה" in current_status:
            return "פריצה" in last_status and days_passed < 2
        if "מתבשלת" in current_status:
            return ("מתבשלת" in last_status or "פריצה" in last_status) and days_passed < COOLDOWN_DAYS
        return days_passed < COOLDOWN_DAYS
    except Exception:
        return False

def load_tickers():
    if os.path.exists(CUSTOM_TICKERS_FILE):
        try:
            df       = pd.read_csv(CUSTOM_TICKERS_FILE, encoding="utf-8-sig")
            col_name = next((c for c in df.columns if c.strip().lower() in ["ticker", "symbol"]), None)
            if col_name:
                tickers = (df[col_name].dropna().astype(str).str.strip()
                           .str.upper().str.replace(".", "-", regex=False).tolist())
                return sorted(list(set([t for t in tickers if t.replace("-", "").isalnum()])))
        except Exception:
            pass
    return ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "PLTR", "AMD"]

# ==========================================
# 3. אינדיקטורים
# ==========================================
def normalize_ohlcv_columns(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df[~df.index.duplicated(keep="first")]

def add_indicators(df):
    df = normalize_ohlcv_columns(df)
    df["SMA_50"]       = df["Close"].rolling(50,  min_periods=25).mean()
    df["SMA_150"]      = df["Close"].rolling(150, min_periods=75).mean()
    df["SMA_200"]      = df["Close"].rolling(200, min_periods=100).mean()
    df["Vol_50"]       = df["Volume"].rolling(50, min_periods=25).mean()
    df["DollarVol_50"] = df["Close"].rolling(50, min_periods=25).mean() * df["Vol_50"]
    df["Prev_Close"]   = df["Close"].shift(1)
    for p in BRAIN["rs_periods"]:
        df[f"ROC_{p}"] = df["Close"].pct_change(p)
    df["High_252"] = df["High"].rolling(252, min_periods=120).max()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Prev_Close"]).abs(),
        (df["Low"]  - df["Prev_Close"]).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"]  = tr.rolling(14, min_periods=7).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"]
    return df

def get_spy_data():
    try:
        spy = yf.download("SPY", period=SCAN_PERIOD, auto_adjust=True, progress=False)
        spy = normalize_ohlcv_columns(spy)
        if not spy.empty and len(spy) > 200:
            for p in BRAIN["rs_periods"]:
                spy[f"ROC_{p}"] = spy["Close"].pct_change(p)
            return spy
    except Exception:
        pass
    return pd.DataFrame()

def calc_weighted_rs(stock_row, spy_row):
    rs = 0.0
    for period, weight in zip(BRAIN["rs_periods"], BRAIN["rs_weights"]):
        col   = f"ROC_{period}"
        s_val = float(stock_row[col]) if col in stock_row.index and pd.notna(stock_row[col]) else 0.0
        b_val = float(spy_row[col])   if col in spy_row.index  and pd.notna(spy_row[col])  else 0.0
        rs   += (s_val - b_val) * weight
    return rs

# ==========================================
# 4. מנוע זיהוי תבניות (DTW)
# ==========================================
def normalize_series(series):
    arr    = np.array(series, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn: return np.zeros(len(arr))
    return (arr - mn) / (mx - mn)

def get_dtw_templates():
    """
    שלוש תבניות. min_corr מוקל לנתונים יומיים אמיתיים.

    שינויים מ-v4:
    - min_corr: דגל 0.88→0.80 | דרווס 0.85→0.76 | ספל 0.82→0.75
    - VCP הפך לבונוס בלבד (הוסר מהתבנית, נשאר בציון)
    - std noise: 0.35→0.45
    """
    templates = {}

    # -------------------------------------------------------
    # דגל שורי
    # תורן (עלייה חדה) + דגל (ירידה אלכסונית גלית)
    # -------------------------------------------------------
    pole       = np.linspace(0, 1.0, 10)
    flag_trend = np.linspace(1.0, 0.7, 20)
    flag_waves = flag_trend + np.sin(np.linspace(0, 4*np.pi, 20)) * 0.05
    templates["🚩 דגל שורי"] = {
        "data":      np.concatenate((pole, flag_waves)),
        "windows":   [30, 45, 60],
        "threshold": 0.12,
        "min_corr":  BRAIN["min_corr_flag"],   # 0.80
        "comp":      0
    }

    # -------------------------------------------------------
    # מלבן דרווס
    # עלייה + פולבק קל + קופסה אופקית
    # -------------------------------------------------------
    rise             = np.linspace(0, 1.0, 10)
    initial_pullback = np.linspace(1.0, 0.8, 5)
    box              = np.ones(35) * 0.9 + np.sin(np.linspace(0, 6*np.pi, 35)) * 0.05
    templates["📦 מלבן דרווס"] = {
        "data":      np.concatenate((rise, initial_pullback, box)),
        "windows":   [40, 60, 90],
        "threshold": 0.12,
        "min_corr":  BRAIN["min_corr_darvas"],  # 0.76
        "comp":      1
    }

    # -------------------------------------------------------
    # ספל וידית
    # כוס עגולה + ידית: יורדת ~20% ועולה חזרה ל-0.93
    # (7% מתחת לשפה = עדיין לא פרצה — הפריצה מזוהה ע"י dist_to_pivot)
    # -------------------------------------------------------
    left_cup    = np.linspace(-1, 0, 45)**2
    right_cup   = np.linspace(0, 1, 20)**2
    handle_down = np.linspace(1.0, 0.80, 8)
    handle_up   = np.linspace(0.80, 0.93, 7)
    templates["☕ ספל וידית"] = {
        "data":      np.concatenate((left_cup, right_cup, handle_down, handle_up)),
        "windows":   [60, 90, 150, 250],
        "threshold": 0.15,
        "min_corr":  BRAIN["min_corr_cup"],     # 0.75
        "comp":      2
    }

    return templates

def check_all_dtw_patterns(closes, volumes, atr_values):
    templates    = get_dtw_templates()
    best_pattern = None
    best_score   = float("inf")

    for name, config in templates.items():
        for window in config["windows"]:
            if len(closes) < window:
                continue

            current_closes  = closes[-window:]
            current_volumes = volumes[-window:]    if len(volumes)    >= window else np.ones(window)
            current_atr     = atr_values[-window:] if len(atr_values) >= window else np.ones(window)

            # מנעול: תנועה אמיתית לפחות 10%
            raw_min, raw_max = np.min(current_closes), np.max(current_closes)
            if raw_min == 0 or (raw_max - raw_min) / raw_min < 0.10:
                continue

            norm_price     = normalize_series(current_closes)
            norm_vol       = normalize_series(current_volumes)
            norm_composite = norm_price * 0.75 + norm_vol * 0.25

            # מנעול: שיני מסור (std מוקל ל-0.45)
            if np.std(norm_price) > BRAIN["max_std_noise"]:
                continue

            x_orig                = np.linspace(0, 1, len(config["data"]))
            x_new                 = np.linspace(0, 1, window)
            resized_template      = np.interp(x_new, x_orig, config["data"])
            norm_template_resized = normalize_series(resized_template)

            corr = np.corrcoef(norm_price, norm_template_resized)[0, 1]
            if pd.isna(corr) or corr < config["min_corr"]:
                continue

            # מנעולי היגיון לפי סוג תבנית
            w        = window
            pole_vol = 1.0
            flag_vol = 1.0

            if "דגל" in name:
                if current_closes[-1] <= current_closes[0] * 1.05:
                    continue
                pole_vol = np.mean(current_volumes[:int(w * 0.35)])
                flag_vol = np.mean(current_volumes[int(w * 0.35):])
                # נפח הדגל צריך לרדת מהתורן (dry-up)
                if pole_vol > 0 and (flag_vol / pole_vol) > 0.90:
                    continue

            elif "ספל" in name:
                cup_bottom    = np.min(current_closes[:int(w * 0.8)])
                handle_bottom = np.min(current_closes[-int(w * 0.2):])
                # הידית לא שוברת תחתית הכוס
                if handle_bottom < cup_bottom:
                    continue
                # הידית מסתיימת במרחק סביר מהשפה (עד -10%)
                cup_rim    = float(np.percentile(current_closes[int(w * 0.6):int(w * 0.9)], 90))
                handle_end = float(current_closes[-1])
                if handle_end < cup_rim * 0.90:
                    continue

            # VCP — חישוב contraction לציון, לא לפילטר
            atr_early         = np.mean(current_atr[:int(w * 0.5)])
            atr_late          = np.mean(current_atr[int(w * 0.5):])
            contraction_ratio = (atr_late / atr_early) if atr_early > 0 else 1.0
            # הוסר: if contraction_ratio > 0.85: continue

            distance, _ = fastdtw(norm_composite, norm_template_resized,
                                   dist=lambda x, y: abs(x - y))
            avg_distance = distance / window

            if avg_distance < config["threshold"] and avg_distance < best_score:
                best_score = avg_distance

                if "דגל" in name:
                    pivot = float(np.percentile(current_closes[:int(w * 0.5)], 95))
                    low   = float(np.min(current_closes[-int(w * 0.5):]))
                elif "דרווס" in name:
                    pivot = float(np.percentile(current_closes[-int(w * 0.7):], 95))
                    low   = float(np.min(current_closes[-int(w * 0.7):]))
                else:  # ספל
                    pivot = float(np.percentile(current_closes[int(w * 0.6):int(w * 0.9)], 95))
                    low   = float(np.min(current_closes[-int(w * 0.3):]))

                tightness = (pivot - low) / pivot if pivot > 0 else 0

                touches = max(2, int(sum(
                    1 for p in current_closes
                    if pivot > 0 and abs(p - pivot) / pivot < 0.02
                )))

                dry_up = round(flag_vol / pole_vol, 2) if "דגל" in name and pole_vol > 0 else 1.0

                best_pattern = {
                    "type":              f"{name} (DTW)",
                    "dtw_distance":      round(avg_distance, 3),
                    "correlation":       round(corr * 100, 1),
                    "complexity_bonus":  config["comp"],
                    "threshold":         config["threshold"],
                    "pivot_price":       pivot,
                    "tight_low":         low,
                    "last_pullback_low": low,
                    "tightness":         tightness,
                    "base_depth":        0.20,
                    "dry_up_ratio":      dry_up,
                    "touches":           touches,
                    "base_length":       window,
                    "contraction_ratio": round(contraction_ratio, 3)
                }

    return best_pattern

# ==========================================
# 5. דירוג וסריקה
# ==========================================
def calc_setup_score(alert):
    rs_score = min(max(alert["rs_weighted"], 0) * 250, 25)

    # tightness — סף גמיש לפי סוג תבנית
    tight_ceiling = 0.10 if "דגל" in alert["type"] else 0.22
    tight_score   = max(0, (1 - min(alert["tightness"], tight_ceiling) / tight_ceiling) * 20)

    # קרבה לפיבוט
    if alert["dist_to_pivot"] > BRAIN["max_gap_above_pivot"]:
        pivot_score = -10
    else:
        pivot_score = max(0, (1 - min(abs(alert["dist_to_pivot"]), 0.03) / 0.03) * 15)

    close_score  = min(max(alert["close_strength"], 0), 1) * 10
    volume_score = min(alert["vol_ratio"] / 2.0, 1.0) * 5

    dtw_score  = max(0, (alert["threshold"] - alert["dtw_distance"]) * 100)
    corr_score = max(0, (alert["correlation"] - 70)) * 2.0   # הורד ל-70 כבסיס

    comp_bonus = alert.get("complexity_bonus", 0) * 15

    # VCP — בונוס בלבד (לא פילטר)
    vcp_bonus = max(0, (BRAIN["vcp_bonus_threshold"] - alert.get("contraction_ratio", 1.0)) * 30)

    touches_bonus = max(0, (alert.get("touches", 2) - 2)) * 3
    trend_bonus   = 5 if not alert["is_below_150"] else 0

    return round(rs_score + tight_score + pivot_score + close_score + volume_score +
                 dtw_score + corr_score + comp_bonus + vcp_bonus + touches_bonus + trend_bonus, 1)

def scan_market():
    tickers = load_tickers()
    print(f"✅ נטענו {len(tickers)} מניות לסריקה.")
    if not tickers: return

    spy      = get_spy_data()
    spy_last = spy.iloc[-1] if not spy.empty else pd.Series(dtype=float)

    all_potentials = []
    stats = {
        "total_scanned": 0, "pass_basic": 0, "pass_trend": 0,
        "pass_rs": 0, "pass_pattern": 0, "pass_pivot_dist": 0, "final_approved": 0
    }

    for ticker in tickers:
        stats["total_scanned"] += 1
        print(f"סורק את {ticker}...", end="\r")

        try:
            df = yf.download(ticker, period=SCAN_PERIOD, auto_adjust=True, progress=False)
            if df.empty or len(df) < 200: continue
            df = add_indicators(df)
            today, yesterday, past_data = df.iloc[-1], df.iloc[-2], df.iloc[:-1].copy()

            if any(pd.isna(today[c]) for c in ["SMA_50", "SMA_150", "SMA_200", "ATR_14", "ATR_Pct"]):
                continue

            close = float(today["Close"])
            if close < MIN_PRICE or float(today["DollarVol_50"]) < MIN_DOLLAR_VOL_50: continue
            if close <= float(today["SMA_50"]): continue
            if float(today["ATR_Pct"]) < BRAIN["min_atr_pct"]: continue
            stats["pass_basic"] += 1

            is_below_150    = close < float(today["SMA_150"])
            weekly_trend_ok = (
                float(today["SMA_50"]) > float(today["SMA_150"]) > float(today["SMA_200"])
            )
            stats["pass_trend"] += 1

            max_dist = (BRAIN["max_dist_from_52w_high_below_150"] if is_below_150
                        else BRAIN["max_dist_from_52w_high_normal"])
            if (close / float(today["High_252"])) - 1.0 < -max_dist: continue

            stock_rs = calc_weighted_rs(today, spy_last)
            if stock_rs < BRAIN["min_rs_65"]: continue
            stats["pass_rs"] += 1

            past_filtered = past_data.dropna(subset=["Close"])
            if len(past_filtered) < 30: continue

            closes     = past_filtered["Close"].astype(float).values
            volumes    = past_filtered["Volume"].astype(float).values
            atr_values = past_filtered["ATR_14"].fillna(method="bfill").astype(float).values

            pattern = check_all_dtw_patterns(closes, volumes, atr_values)
            if not pattern: continue
            stats["pass_pattern"] += 1

            pivot         = float(pattern["pivot_price"])
            dist_to_pivot = (close / pivot) - 1.0

            # -10% עד +2% — watchlist + פריצות טריות בלבד
            if dist_to_pivot < -0.10 or dist_to_pivot > 0.02: continue
            stats["pass_pivot_dist"] += 1

            vol_ratio = (float(today["Volume"]) / float(today["Vol_50"])
                         if float(today["Vol_50"]) > 0 else 0.0)
            close_strength = ((close - float(today["Low"])) /
                              max(float(today["High"]) - float(today["Low"]), 1e-9))

            is_breakout      = float(yesterday["Close"]) <= pivot and close > pivot
            is_near_breakout = (-BRAIN["watchlist_max_dist"] <= dist_to_pivot <= 0.0)

            if is_breakout:
                status = ("🔥 פריצה פעילה!"
                          if close_strength >= 0.55 and vol_ratio >= 1.3
                          else "🪑 ספסל")
            elif is_near_breakout:
                status = "👀 מתבשלת (Watchlist)"
            else:
                status = "🪑 ספסל"

            stop_price = (min(float(pattern["tight_low"]), float(pattern["last_pullback_low"]))
                          - 0.5 * float(today["ATR_14"]))
            risk_pct   = (close - stop_price) / close * 100

            if status != "🪑 ספסל":
                if stop_price >= close or risk_pct > 12.0:
                    status = "🪑 ספסל"

            if should_skip_spam(ticker, status): continue

            alert_data = {
                "ticker": ticker, "close": close, "pivot": pivot,
                "stop_loss": stop_price, "risk_pct": risk_pct,
                "vol_ratio": vol_ratio, "type": pattern["type"],
                "rs_weighted": stock_rs, "close_strength": close_strength,
                "status": status, "dist_to_pivot": dist_to_pivot,
                "tightness": float(pattern["tightness"]),
                "is_below_150": is_below_150, "weekly_trend_ok": weekly_trend_ok,
                "dry_up_ratio": float(pattern["dry_up_ratio"]),
                "touches": int(pattern["touches"]),
                "base_depth": float(pattern["base_depth"]),
                "base_length": int(pattern["base_length"]),
                "dtw_distance": float(pattern["dtw_distance"]),
                "correlation": float(pattern["correlation"]),
                "threshold": float(pattern["threshold"]),
                "complexity_bonus": int(pattern.get("complexity_bonus", 0)),
                "contraction_ratio": float(pattern.get("contraction_ratio", 1.0))
            }

            alert_data["setup_score"] = calc_setup_score(alert_data)

            if alert_data["setup_score"] >= BRAIN["min_setup_score"]:
                all_potentials.append(alert_data)
                stats["final_approved"] += 1

        except Exception:
            pass

    # ---- דוח סינון ----
    print("\n--- 📊 דוח סינון DTW v5 📊 ---")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print("-------------------------------\n")

    if all_potentials:
        print("--- 📋 מניות שעברו את הסינון 📋 ---")
        sorted_all = sorted(all_potentials, key=lambda x: (-x["correlation"], -x["setup_score"]))
        for idx, s in enumerate(sorted_all, 1):
            clean_type = s["type"].replace(" (DTW)", "")
            trend_icon = "📈" if s["weekly_trend_ok"] else "〰️"
            print(f"  {idx:>2}. {s['ticker']:<6} | {clean_type:<16} | "
                  f"ציון: {s['setup_score']:<5.1f} | "
                  f"קורלציה: {s['correlation']}% | "
                  f"VCP: {s['contraction_ratio']:.2f} | "
                  f"נגיעות: {s['touches']} | {trend_icon} | "
                  f"פיבוט: ${s['pivot']:.2f} | "
                  f"מרחק: {s['dist_to_pivot']*100:+.1f}%")
        print("------------------------------------\n")
    else:
        print("לא נמצאו מניות העונות לקריטריונים.")

    prime = sorted([s for s in all_potentials if s["status"] != "🪑 ספסל"],
                   key=lambda x: (-x["correlation"], -x["setup_score"]))
    bench = sorted([s for s in all_potentials if s["status"] == "🪑 ספסל"],
                   key=lambda x: (-x["correlation"], abs(x["dist_to_pivot"])))

    final_selection = (prime + bench)[:TOP_RESULTS]
    if not final_selection:
        send_telegram("✅ הסריקה הסתיימה. הבוט קפדני (רמת צלף), לא נמצאו תבניות מדויקות היום.")
        return

    msg  = "🎯 <b>סריקת ראייה ממוחשבת (DTW v5)</b>\n"
    msg += f"<i>ממוין לפי קורלציה | עד {TOP_RESULTS} מניות</i>\n\n"

    pattern_groups = {}
    for s in final_selection:
        pt = s["type"]
        if pt not in pattern_groups: pattern_groups[pt] = []
        pattern_groups[pt].append(s)

    for ptype, stocks in pattern_groups.items():
        icon         = ptype.split()[0] if " " in ptype else "📈"
        pattern_name = ptype.replace(icon, "").replace("(DTW)", "").strip()
        msg += "────────────────\n"
        msg += f"{icon} <b>תבנית {pattern_name} ({len(stocks)} מניות):</b>\n\n"

        for a in stocks:
            status_icon  = "🔥" if "פריצה" in a["status"] else "⏳" if "מתבשלת" in a["status"] else "🪑"
            clean_status = a["status"].replace(" (Watchlist)", "")
            trend_tag    = "✅ מגמה" if a["weekly_trend_ok"] else "〰️ ציר"
            dist_str     = f"{a['dist_to_pivot']*100:+.1f}%"
            vcp_tag      = "🔄 VCP ✅" if a["contraction_ratio"] < 0.85 else f"VCP: {a['contraction_ratio']:.2f}"

            msg += f"{status_icon} <b>{a['ticker']}</b> | {clean_status} | {trend_tag}\n"
            msg += f"📏 <b>דמיון:</b> {a['correlation']}% | ⭐ <b>ציון:</b> {a['setup_score']:.1f}\n"
            msg += f"🎯 <b>פיבוט:</b> ${a['pivot']:.2f} | 💵 <b>מחיר:</b> ${a['close']:.2f} ({dist_str})\n"
            msg += f"🛡️ <b>סטופ:</b> ${a['stop_loss']:.2f} | ⚠️ <b>סיכון:</b> {a['risk_pct']:.1f}%\n"
            msg += f"📅 <b>טווח:</b> {a['base_length']} ימים | {vcp_tag} | 👆 {a['touches']} נגיעות\n"
            msg += f"🔗 <a href='https://il.tradingview.com/chart/?symbol={a['ticker']}'>TradingView</a>\n\n"

            # שמירה לזיכרון — פריצות ו-watchlist בלבד (לא ספסל)
            if a["status"] != "🪑 ספסל":
                save_to_smart_memory(
                    a["ticker"], a["close"], a["stop_loss"], a["risk_pct"],
                    a["vol_ratio"], a["pivot"], a["close_strength"],
                    a["rs_weighted"], a["tightness"], a["type"], a["status"],
                    a["setup_score"], a["dry_up_ratio"], a["touches"],
                    a["contraction_ratio"]
                )

    send_telegram(msg)

if __name__ == "__main__":
    scan_market()