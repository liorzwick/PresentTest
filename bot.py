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
MIN_DOLLAR_VOL = 30_000_000      # 10 מיליון דולר ביום
COOLDOWN_DAYS = 5                # מניעת ספאם

# ==========================================
# 2. מנוע VCP (התנגדות אופקית מרובת נגיעות)
# ==========================================
def detect_horizontal_vcp(df, is_turnaround=False):
    """
    סורק כשנה וחצי אחורה (350 ימים) כדי למצוא קו התנגדות אופקי וברור 
    שנבדק לפחות פעמיים (עדיף יותר), ולאחריו נוצרה התכווצות (ידית).
    """
    window = df.tail(350)
    if len(window) < 150:
        return None

    highs = window['High'].values
    lows = window['Low'].values
    closes = window['Close'].values

    # 1. מציאת כל השיאים המקומיים (חלון של 10 ימים לכל כיוון)
    peaks = []
    for i in range(10, len(highs) - 10):
        if highs[i] == np.max(highs[i-10:i+11]):
            peaks.append((i, highs[i]))

    if len(peaks) < 2:
        return None

    # 2. איתור קו ההתנגדות החזק ביותר (חייב להכיל לפחות 2 נגיעות מקבילות)
    best_group = []
    best_pivot = 0.0

    for i, (p_idx, p_val) in enumerate(peaks):
        # מקבץ את כל השיאים שנמצאים ברצועה צרה של 3% מנקודה זו (Flat Top)
        group = [idx for idx, val in peaks if abs(val - p_val) / p_val <= 0.03]
        
        # מחפש את הרצועה עם הכי הרבה נגיעות, שובר שוויון לפי המחיר הגבוה
        if len(group) >= 2: # 🚨 חוק ברזל: חייבים לפחות 2 שיאים מקבילים!
            if len(group) > len(best_group) or (len(group) == len(best_group) and p_val > best_pivot):
                best_group = group
                best_pivot = np.max([highs[idx] for idx in group])

    # אם לא מצאנו קו התנגדות אופקי עם לפחות 2 שיאים - אין כאן תבנית.
    if len(best_group) < 2:
        return None

    first_touch_idx = best_group[0]
    last_touch_idx = best_group[-1]

    # זמן התבססות: חייב להיות לפחות 30 ימי מסחר בין הנגיעה הראשונה לאחרונה
    if last_touch_idx - first_touch_idx < 30:
        return None

    # 3. מדידת עומק הבסיס (השפל הכי נמוך מאז הנגיעה הראשונה)
    base_low = np.min(lows[first_touch_idx:])
    base_depth = (best_pivot - base_low) / best_pivot

    max_allowed_depth = 0.70 if is_turnaround else 0.45
    if base_depth < 0.10 or base_depth > max_allowed_depth:
        return None

    # 4. מדידת התכווצות (הידית) - מחושבת מהשפל מאז הנגיעה *האחרונה* בתקרה
    # נוודא קודם שעברו לפחות 3 ימים מהנגיעה האחרונה, כדי שהידית תספיק להיווצר
    if len(lows) - last_touch_idx < 3:
        return None

    handle_low = np.min(lows[last_touch_idx:])
    handle_depth = (best_pivot - handle_low) / best_pivot

    # 🚨 חוקי הברזל של מינרוויני לכיווץ ימני 🚨
    # א. שפלים עולים: הירידה האחרונה חייבת להיות קטנה משמעותית מההתרסקות הגדולה (חצי לפחות)
    if handle_depth > (base_depth * 0.55):
        return None
    
    # ב. תקרת זכוכית לכיווץ: הידית ממש לקראת פריצה לא יכולה להיות עמוקה מ-12% (או 15% בשיקום)
    max_handle = 0.15 if is_turnaround else 0.12
    if handle_depth > max_handle:
        return None

    # 5. סטטוס נוכחי (האם אנחנו קרובים לקו ההתנגדות הזה עכשיו?)
    current_price = closes[-1]
    dist_to_pivot = (current_price / best_pivot) - 1.0

    # אם אנחנו רחוקים מ-6% למטה, זה עוד לא הזמן. אם פרצנו מעל 3.5%, כבר ברח.
    if dist_to_pivot < -0.06 or dist_to_pivot > 0.035:
        return None

    return {
        "pivot_price": best_pivot,
        "tight_low": handle_low,
        "tightness_pct": handle_depth * 100,
        "base_depth_pct": base_depth * 100,
        "dist_to_pivot": dist_to_pivot * 100,
        "touches": len(best_group)
    }

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
    return ['AAPL', 'MSFT', 'NVDA', 'META', 'AMZN'] 

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
    
    print("📥 מתחיל סריקה... (מחפש התנגדויות אופקיות של שנתיים אחורה)")
    market_ok, spy_rs = get_spy_trend()
    
    market_warning = "" if market_ok else "⚠️ <b>השוק חלש (SPY מתחת ל-200). סוחר בזהירות.</b>\n\n"
    results = []

    for ticker in tickers:
        print(f"בודק את {ticker}...", end="\r")
        try:
            # 🚨 מוריד עכשיו שנתיים שלמות כדי לתפוס בסיסים ענקיים
            df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
            if len(df) < 250: continue
            
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
            
            # --- המבחן הגדול: ה-VCP מרובה הנגיעות ---
            vcp = detect_horizontal_vcp(df, is_turnaround)
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
        send_telegram("✅ הסריקה הסתיימה. הוסרו תעלות עולות! לא נמצאו מניות עם בסיס אופקי מוצק כרגע.")
        return

    results.sort(key=lambda x: abs(x["dist"]))
    
    msg = f"🎯 <b>סריקת VCP (בסיסים אופקיים בלבד) הסתיימה!</b>\n{market_warning}נמצאו {len(results)} סטאפים מובחרים:\n\n"
    
    for r in results[:10]:
        icon = "🚀" if "פריצה" in r["status"] else "⏳"
        warn = " ⚠️ (שיקום מתחת ל-150)" if r["is_turnaround"] else ""
        tv_link = f"https://il.tradingview.com/chart/?symbol={r['ticker']}"
        
        msg += f"{icon} <b>{r['ticker']}</b> | {r['status']}{warn}\n"
        msg += f"📐 <b>מבנה ({r['touches']} נגיעות בתקרה):</b> בסיס {r['base_depth']:.1f}% ⬅️ התכווצות {r['tightness']:.1f}%\n"
        msg += f"📈 <b>עוצמה:</b> {r['rs']:.1f}% | 📊 <b>ווליום:</b> {r['vol_ratio']:.1f}x\n"
        
        if "פריצה" in r["status"]:
            msg += f"🎯 <b>נפרץ קו ההתנגדות:</b> ${r['pivot']:.2f} | 💵 <b>מחיר:</b> ${r['close']:.2f}\n"
        else:
            msg += f"🎯 <b>פיבוט התנגדות:</b> ${r['pivot']:.2f} (מרחק: {r['dist']:.1f}%) | 💵 <b>מחיר:</b> ${r['close']:.2f}\n"
            
        msg += f"🛡️ <b>סטופ טכני:</b> ${r['stop_loss']:.2f} (סיכון {r['risk_pct']:.1f}%-)\n"
        msg += f"🔗 <a href='{tv_link}'>TradingView</a>\n"
        msg += "────────────────\n"
        
        log_sent_signal(r['ticker'])

    send_telegram(msg)

if __name__ == "__main__":
    scan_market()
