import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==========================================
# 1. הגדרות בסיסיות
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_GROUP")
CUSTOM_TICKERS_FILE = "mystock.csv"

# סינוני בסיס
MIN_PRICE = 10.0
MIN_MARKET_CAP = 2_000_000_000   # 1 מיליארד דולר
MIN_DOLLAR_VOL = 20_000_000      # 10 מיליון דולר ביום
COOLDOWN_DAYS = 5                # מניעת ספאם

# ==========================================
# 2. מנוע ה-VCP (זיהוי התנגדות אופקית מבוססת זמן)
# ==========================================
def detect_time_spaced_vcp(df, is_turnaround=False):
    """
    מנוע חכם שמחפש בסיסים ארוכים ושטוחים.
    חובה למצוא לפחות 2 שיאים באותו מחיר, המרוחקים לפחות 3 שבועות זה מזה.
    """
    window = df.tail(250) # בודק שנה אחורה
    if len(window) < 100:
        return None

    highs = window['High'].values
    lows = window['Low'].values
    closes = window['Close'].values

    # 1. מציאת כל השיאים המקומיים
    peaks = []
    for i in range(5, len(highs) - 5):
        if highs[i] == np.max(highs[i-5:i+6]):
            peaks.append((i, highs[i]))

    if len(peaks) < 2:
        return None

    best_pattern = None
    max_touches = 0

    # 2. חיפוש תקרת בטון (התנגדות)
    for i, (p_idx, p_val) in enumerate(peaks):
        # מתחילים קבוצת נגיעות עם השיא הנוכחי
        valid_touches = [(p_idx, p_val)]
        
        for j, (other_idx, other_val) in enumerate(peaks):
            if i == j: continue
            
            # אם השיא האחר נמצא באזור של 3% מנקודת הפיבוט שלנו
            if abs(other_val - p_val) / p_val <= 0.03:
                # 🚨 חוק המרחק (Time Spacing): מוודא שהשיא האחר רחוק לפחות 15 ימי מסחר מכל שיא אחר בקבוצה
                is_spaced = all(abs(other_idx - v_idx) >= 15 for v_idx, v_val in valid_touches)
                if is_spaced:
                    valid_touches.append((other_idx, other_val))

        valid_touches.sort(key=lambda x: x[0]) # מסדר כרונולוגית

        # חייבים לפחות 2 נגיעות מרוחקות בזמן (מחסל מניות כמו CW)
        if len(valid_touches) >= 2:
            first_touch_idx = valid_touches[0][0]
            last_touch_idx = valid_touches[-1][0]

            # 🚨 חוק אורך הבסיס: המרחק בין הנגיעה הראשונה לאחרונה חייב להיות לפחות חודשיים (40 ימי מסחר)
            base_duration = last_touch_idx - first_touch_idx
            if base_duration < 40:
                continue

            # הפיבוט האמיתי הוא המחיר הגבוה ביותר מבין הנגיעות שנבחרו
            pivot = np.max([val for idx, val in valid_touches])

            # עומק הבסיס הכולל (ההתרסקות הגדולה ביותר מאז הנגיעה הראשונה)
            base_low = np.min(lows[first_touch_idx:])
            base_depth = (pivot - base_low) / pivot

            max_allowed_depth = 0.70 if is_turnaround else 0.45
            if base_depth < 0.10 or base_depth > max_allowed_depth:
                continue

            # אם לא עברו לפחות 3 ימים מהנגיעה האחרונה, אין לידית זמן להיווצר
            if len(lows) - last_touch_idx < 3:
                continue

            # מדידת הכיווץ (הידית) האחרון
            handle_low = np.min(lows[last_touch_idx:])
            tightness = (pivot - handle_low) / pivot

            # 🚨 חוק התכווצות: הידית חייבת להיות קטנה בחצי מעומק הבסיס המקורי
            if tightness > base_depth * 0.55:
                continue
            
            # עומק מקסימלי לידית 12% (או 15% לשיקום)
            max_handle = 0.15 if is_turnaround else 0.12
            if tightness > max_handle:
                continue

            # קרבה לפיבוט (סטטוס כניסה)
            current_price = closes[-1]
            dist_to_pivot = (current_price / pivot) - 1.0

            # מאפשר קנייה גם אם ברח עד 5.5% מעל הפיבוט (כדי לא לפספס את FTNT)
            if dist_to_pivot < -0.06 or dist_to_pivot > 0.055:
                continue

            # מעדכן את התבנית הטובה ביותר (זו עם הכי הרבה נגיעות)
            if len(valid_touches) > max_touches:
                max_touches = len(valid_touches)
                best_pattern = {
                    "pivot_price": pivot,
                    "tight_low": handle_low,
                    "tightness_pct": tightness * 100,
                    "base_depth_pct": base_depth * 100,
                    "dist_to_pivot": dist_to_pivot * 100,
                    "touches": len(valid_touches),
                    "duration_days": base_duration
                }

    return best_pattern

# ==========================================
# 3. מתנדים, נתונים ופילטר שוק
# ==========================================
def add_indicators(df):
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_150"] = df["Close"].rolling(150).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Vol_50"] = df["Volume"].rolling(50).mean()
    df["DollarVol_50"] = df["Close"].rolling(50).mean() * df["Vol_50"]
    df["ROC_65"] = df["Close"].pct_change(65)
    df["High_252"] = df["High"].rolling(252).max()
    
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    return df

def get_spy_trend():
    try:
        spy = yf.download("SPY", period="2y", auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        
        spy["SMA_200"] = spy["Close"].rolling(200).mean()
        spy["ROC_65"] = spy["Close"].pct_change(65)
        
        today = spy.iloc[-1]
        sma_200_old = spy.iloc[-20]["SMA_200"]
        
        is_uptrend = (today["Close"] > today["SMA_200"]) and (today["SMA_200"] > sma_200_old)
        return is_uptrend, float(today["ROC_65"])
    except:
        return True, 0.0 

def get_market_cap(ticker):
    try:
        t = yf.Ticker(ticker)
        fast = getattr(t, 'fast_info', None)
        if fast and getattr(fast, 'market_cap', None):
            return float(fast.market_cap)
        return float(t.info.get('marketCap', 0))
    except:
        return 0

def load_tickers():
    if os.path.exists(CUSTOM_TICKERS_FILE):
        try:
            df = pd.read_csv(CUSTOM_TICKERS_FILE)
            col_name = next((c for c in df.columns if c.strip().lower() in ['ticker', 'symbol']), None)
            if col_name:
                tickers = df[col_name].dropna().astype(str).str.strip().str.upper().tolist()
                tickers = [t.replace('.', '-') for t in tickers if t.isalpha() or '-' in t or '.' in t]
                return sorted(list(set(tickers)))
        except: pass
    return ['AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'FTNT', 'HUN'] 

# ==========================================
# 4. מערכת תקשורת וזיכרון (למניעת ספאם)
# ==========================================
def should_skip_spam(ticker):
    log_file = 'trading_log.csv'
    if not os.path.isfile(log_file): return False
    try:
        df = pd.read_csv(log_file)
        df['Date'] = pd.to_datetime(df['Date'])
        history = df[df['Ticker'] == ticker].sort_values(by='Date', ascending=False)
        if history.empty: return False
        
        days_passed = (datetime.now() - history.iloc[0]['Date']).days
        return days_passed < COOLDOWN_DAYS
    except:
        return False

def log_sent_signal(ticker):
    log_file = 'trading_log.csv'
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{'Date': now, 'Ticker': ticker}])
    try:
        if not os.path.isfile(log_file): new_data.to_csv(log_file, index=False)
        else: new_data.to_csv(log_file, mode='a', header=False, index=False)
    except: pass

def send_telegram(message):
    print("\n=== דוח סריקה ===")
    print(message)
    print("=================\n")
    
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ לא הוגדרו טוקנים של טלגרם. הסריקה בוצעה במצב סימולציה (הדפסה למסך בלבד).")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=10)
    except: pass

# ==========================================
# 5. המוח: תהליך הסריקה הראשי
# ==========================================
def scan_market():
    tickers = load_tickers()
    if not tickers: return
    
    print("📥 מתחיל סריקה עמוקה... (מחפש בסיסים ארוכים עם נגיעות מרוחקות)")
    market_ok, spy_rs = get_spy_trend()
    
    market_warning = "" if market_ok else "⚠️ <b>השוק חלש (SPY מתחת ל-200). סוחר בזהירות.</b>\n\n"
    results = []

    for ticker in tickers:
        print(f"בודק את {ticker}...", end="\r")
        try:
            df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
            if len(df) < 200: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df = add_indicators(df)
            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            
            if pd.isna(today["SMA_200"]) or pd.isna(today["ATR_14"]): continue
            if float(today["Close"]) < MIN_PRICE: continue
            if float(today["DollarVol_50"]) < MIN_DOLLAR_VOL: continue
            
            if float(today["Close"]) < float(today["SMA_50"]): continue
            
            is_turnaround = float(today["Close"]) < float(today["SMA_150"])
            
            if is_turnaround:
                if float(today["SMA_50"]) <= float(yesterday["SMA_50"]): continue
                if (float(today["ROC_65"]) - spy_rs) < 0.04: continue
            else:
                if (float(today["ROC_65"]) - spy_rs) < 0.02: continue
            
            # --- הפעלת ה-VCP Engine החכם ---
            vcp = detect_time_spaced_vcp(df, is_turnaround)
            if not vcp: continue
            
            if should_skip_spam(ticker): continue
            
            mc = get_market_cap(ticker)
            if mc > 0 and mc < MIN_MARKET_CAP: continue
            
            pivot = vcp["pivot_price"]
            close_price = float(today["Close"])
            is_breakout = (float(yesterday["Close"]) <= pivot) and (close_price > pivot)
            
            vol_ratio = float(today["Volume"]) / float(today["Vol_50"])
            day_range = max(float(today["High"]) - float(today["Low"]), 1e-9)
            close_strength = (close_price - float(today["Low"])) / day_range
            
            if is_breakout:
                if vol_ratio < 1.1 or close_strength < 0.30: continue
                status = "🔥 פריצה אקטיבית!"
            else:
                status = "👀 מתבשלת (Watchlist)"
                
            atr = float(today["ATR_14"])
            stop_loss = vcp["tight_low"] - (0.2 * atr)
            risk_pct = ((close_price - stop_loss) / close_price) * 100
            
            results.append({
                "ticker": ticker,
                "status": status,
                "close": close_price,
                "pivot": pivot,
                "dist": vcp["dist_to_pivot"],
                "base_depth": vcp["base_depth_pct"],
                "tightness": vcp["tightness_pct"],
                "touches": vcp["touches"],
                "duration": vcp["duration_days"],
                "vol_ratio": vol_ratio,
                "rs": (float(today["ROC_65"]) - spy_rs) * 100,
                "stop_loss": stop_loss,
                "risk_pct": risk_pct,
                "is_turnaround": is_turnaround,
                "mc_billions": mc / 1e9 if mc > 0 else 0
            })
            
        except Exception as e:
            pass

    # ==========================================
    # 6. יצירת הדוח לטלגרם
    # ==========================================
    if not results:
        send_telegram("✅ הסריקה הסתיימה. נחסמו תעלות עולות. לא נמצאו מניות עם בסיס אופקי ארוך כרגע.")
        return

    results.sort(key=lambda x: abs(x["dist"]))
    
    msg = f"🎯 <b>סריקת VCP (בסיסים ארוכים ושטוחים) הסתיימה!</b>\n{market_warning}נמצאו {len(results)} סטאפים מובחרים:\n\n"
    
    for r in results[:10]:
        icon = "🚀" if "פריצה" in r["status"] else "⏳"
        warn = " ⚠️ (שיקום מתחת ל-150)" if r["is_turnaround"] else ""
        tv_link = f"https://il.tradingview.com/chart/?symbol={r['ticker']}"
        
        msg += f"{icon} <b>{r['ticker']}</b> | {r['status']}{warn}\n"
        msg += f"📐 <b>מבנה:</b> {r['touches']} נגיעות בתקרה | אורך בסיס: {r['duration']} ימי מסחר\n"
        msg += f"📊 <b>עומק:</b> בסיס {r['base_depth']:.1f}% ⬅️ כיווץ ימני {r['tightness']:.1f}%\n"
        msg += f"📈 <b>עוצמה:</b> {r['rs']:.1f}% | 📊 <b>ווליום:</b> {r['vol_ratio']:.1f}x\n"
        
        if "פריצה" in r["status"]:
            msg += f"🎯 <b>נפרץ קו התנגדות:</b> ${r['pivot']:.2f} | 💵 <b>מחיר:</b> ${r['close']:.2f}\n"
        else:
            msg += f"🎯 <b>פיבוט התנגדות:</b> ${r['pivot']:.2f} (מרחק: {r['dist']:.1f}%) | 💵 <b>מחיר:</b> ${r['close']:.2f}\n"
            
        msg += f"🛡️ <b>סטופ טכני:</b> ${r['stop_loss']:.2f} (סיכון {r['risk_pct']:.1f}%-)\n"
        msg += f"🔗 <a href='{tv_link}'>TradingView</a>\n"
        msg += "────────────────\n"
        
        log_sent_signal(r['ticker'])

    send_telegram(msg)

if __name__ == "__main__":
    scan_market()
