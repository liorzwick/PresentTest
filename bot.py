import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. הגדרות סריקה עצמאיות (ללא קבצים חיצוניים)
# ==========================================
CUSTOM_TICKERS_FILE = "mystock.csv" # <--- שינוי לקריאה מתוך הקובץ

MIN_MARKET_CAP = 2_000_000_000  
MIN_DOLLAR_VOL_50 = 20_000_000  
MIN_PRICE = 12.0

# "המוח" צרוב ישירות לתוך הקוד לבדיקה
BRAIN = {
    "max_base_depth": 0.35, 
    "max_tightness_depth": 0.08, 
    "min_cup_depth": 0.10,
    "max_cup_depth": 0.35, 
    "min_breakout_close_strength": 0.30, 
    "min_rs_65": 0.02,
    "max_dist_from_52w_high": 0.15, 
    "max_gap_above_pivot": 0.02, 
    "max_entry_extension": 0.03,
    "breakout_volume_ratio": 1.1
}

def load_tickers():
    """קורא את רשימת המניות מתוך mystock.csv"""
    if os.path.exists(CUSTOM_TICKERS_FILE):
        try:
            df = pd.read_csv(CUSTOM_TICKERS_FILE)
            col_name = next((c for c in df.columns if c.strip().lower() in ['ticker', 'symbol']), None)
            if col_name:
                tickers = df[col_name].dropna().astype(str).str.strip().str.upper().tolist()
                tickers = [t.replace('.', '-') for t in tickers if t.isalpha() or '-' in t or '.' in t]
                print(f"✅ נטענו {len(set(tickers))} מניות מתוך {CUSTOM_TICKERS_FILE}")
                return sorted(list(set(tickers)))
            else:
                print(f"❌ שגיאה: לא נמצאה עמודת 'Symbol' או 'Ticker' בקובץ {CUSTOM_TICKERS_FILE}")
        except Exception as e:
            print(f"❌ שגיאה בקריאת הקובץ {CUSTOM_TICKERS_FILE}: {e}")
    else:
        print(f"⚠️ הקובץ {CUSTOM_TICKERS_FILE} לא נמצא בתיקייה.")
        
    print("משתמש ברשימת גיבוי...")
    return ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'PLTR']

# ==========================================
# 2. מנוע טכני
# ==========================================
def add_indicators(df):
    df = df.copy()
    if getattr(df.index, "tz", None) is not None: df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='first')]

    df["SMA_21"], df["SMA_50"] = df["Close"].rolling(21).mean(), df["Close"].rolling(50).mean()
    df["SMA_150"], df["SMA_200"] = df["Close"].rolling(150).mean(), df["Close"].rolling(200).mean()
    df["Vol_50"] = df["Volume"].rolling(50).mean()
    df["DollarVol_50"] = df["Close"].rolling(50).mean() * df["Vol_50"]
    df["Prev_Close"] = df["Close"].shift(1)
    df["ROC_65"] = df["Close"].pct_change(65)
    df["High_252"] = df["High"].rolling(252).max()
    
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Prev_Close"]).abs(), (df["Low"] - df["Prev_Close"]).abs()], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    return df

def get_spy_data():
    try:
        spy = yf.Ticker("SPY").history(period="1y")
        if not spy.empty and len(spy) > 200:
            spy["SMA_50"], spy["SMA_150"], spy["SMA_200"] = spy["Close"].rolling(50).mean(), spy["Close"].rolling(150).mean(), spy["Close"].rolling(200).mean()
            spy["ROC_65"] = spy["Close"].pct_change(65)
            return spy
    except: pass
    return pd.DataFrame()

def market_filter_ok(spy_df):
    if spy_df.empty: return False
    today = spy_df.iloc[-1]
    sma200_old = spy_df["SMA_200"].iloc[-20]
    if pd.isna(today["SMA_200"]) or pd.isna(sma200_old): return False
    return (today["Close"] > today["SMA_50"] > today["SMA_150"] > today["SMA_200"]) and (today["SMA_200"] > sma200_old)

def get_ascending_triangle_signal(hist):
    recent = hist.tail(150)
    if len(recent) < 60: return None
    highs, lows = recent["High"].values, recent["Low"].values
    slice1 = highs[:-10]
    if len(slice1) == 0: return None
    peak1_pos = int(np.argmax(slice1))
    peak1_price = float(slice1[peak1_pos])
    
    post_peak1_lows = lows[peak1_pos+1 : -5]
    if len(post_peak1_lows) < 10: return None
    valley1_pos_rel = int(np.argmin(post_peak1_lows))
    valley1_price = float(post_peak1_lows[valley1_pos_rel])
    
    if (peak1_price - valley1_price) / peak1_price > BRAIN['max_base_depth']: return None

    post_valley1_highs = highs[valley1_pos_rel + peak1_pos + 2 : -2]
    if len(post_valley1_highs) < 5: return None
    peak2_pos_rel = int(np.argmax(post_valley1_highs))
    peak2_price = float(post_valley1_highs[peak2_pos_rel])
    if peak2_price < peak1_price * 0.96 or peak2_price > peak1_price * 1.02: return None

    post_peak2_lows = lows[peak2_pos_rel + valley1_pos_rel + peak1_pos + 3 :]
    if len(post_peak2_lows) < 3: return None
    valley2_price = float(np.min(post_peak2_lows))
    
    if valley2_price <= valley1_price * 1.015: return None
    tightness = (peak2_price - valley2_price) / peak2_price
    if tightness > BRAIN['max_tightness_depth']: return None

    return {"pivot_price": max(peak1_price, peak2_price), "tight_low": valley2_price, "tightness": tightness, "type": "VCP"}

def get_cup_handle_signal(hist):
    recent = hist.tail(250)
    if len(recent) < 60: return None
    highs, lows = recent["High"].values, recent["Low"].values
    rim_idx = np.argmax(highs)
    rim_price = float(highs[rim_idx])
    cup_low = float(np.min(lows[rim_idx:]))
    depth = (rim_price - cup_low) / rim_price

    if not (BRAIN['min_cup_depth'] <= depth <= BRAIN['max_cup_depth']): return None
    cup_to_now = recent.iloc[rim_idx:]
    if len(cup_to_now) < 15: return None 
    
    handle_low = float(cup_to_now["Low"].min())
    if handle_low < (cup_low + 0.5 * (rim_price - cup_low)): return None 
    
    tightness = (rim_price - handle_low) / rim_price
    return {"pivot_price": rim_price, "tight_low": handle_low, "tightness": tightness, "type": "Cup & Handle"}

# ==========================================
# 3. סריקת שוק ראשית לבדיקה
# ==========================================
def scan_market_test():
    print("\n" + "="*50)
    print("🚀 מתחיל סריקת מעבדה עצמאית (Test Mode)")
    print("="*50)
    
    tickers = load_tickers() # <--- הפעלת הפונקציה לקריאה מהקובץ
    if not tickers:
        print("לא נמצאו מניות לסריקה.")
        return
        
    print(f"📥 בודק מגמת שוק (SPY)...")
    spy = get_spy_data()
    
    if not market_filter_ok(spy):
        print("🔴 השוק הכללי אינו במגמת עלייה חזקה (Market Filter Failed).")
        print("במצב אמיתי הבוט היה עוצר כאן. לצורך הבדיקה - אנו נמשיך לסרוק בכל זאת...\n")
        spy_rs = 0.0 # איפוס כדי לאפשר סריקה
    else:
        spy_rs = float(spy.iloc[-1]["ROC_65"])
        print("🟢 השוק הכללי חיובי ומתאים למסחר.\n")
    
    alerts = []
    
    for ticker in tickers:
        print(f"סורק את {ticker}...", end="\r")
        try:
            df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
            if df.empty or len(df) < 200: continue
            df = add_indicators(df)
            today, yesterday, past_data = df.iloc[-1], df.iloc[-2], df.iloc[:-1]
            
            # פילטרים בסיסיים
            if any(pd.isna(today[c]) for c in ["SMA_21", "SMA_50", "SMA_150", "SMA_200", "High_252", "ATR_14"]): continue
            if float(today["Close"]) < MIN_PRICE or float(today["DollarVol_50"]) < MIN_DOLLAR_VOL_50: continue
            if not (float(today["Close"]) > float(today["SMA_21"]) > float(today["SMA_50"]) > float(today["SMA_150"]) > float(today["SMA_200"])): continue
            if (float(today["Close"]) / float(today["High_252"]) - 1.0) < -BRAIN['max_dist_from_52w_high']: continue
            
            stock_rs = float(today["ROC_65"]) - spy_rs
            if stock_rs < BRAIN['min_rs_65']: continue
            
            # בדיקת תבניות: משולש או כוס
            pattern = get_ascending_triangle_signal(past_data) or get_cup_handle_signal(past_data)
            if not pattern: continue
                
            pivot = pattern["pivot_price"]
            
            # בדיקת פריצה היום
            if float(yesterday["Close"]) <= pivot and float(today["Close"]) > pivot:
                day_range = max(float(today["High"]) - float(today["Low"]), 1e-9)
                close_strength = (float(today["Close"]) - float(today["Low"])) / day_range
                vol_ratio = float(today["Volume"]) / float(today["Vol_50"])
                gap_from_pivot = (float(today["Open"]) / pivot) - 1.0
                
                if close_strength >= BRAIN['min_breakout_close_strength'] and vol_ratio >= BRAIN['breakout_volume_ratio'] and gap_from_pivot <= BRAIN['max_gap_above_pivot']:
                    if float(today["Close"]) <= pivot * (1 + BRAIN['max_entry_extension']):
                        
                        atr = float(today["ATR_14"])
                        stop_price = float(pattern["tight_low"]) - (0.5 * atr)
                        risk_pct = (float(today["Close"]) - stop_price) / float(today["Close"]) * 100
                        
                        alerts.append({
                            "ticker": ticker, "close": float(today["Close"]), "pivot": pivot,
                            "stop_loss": stop_price, "risk_pct": risk_pct, "vol_ratio": vol_ratio,
                            "type": pattern["type"], "rs_65": stock_rs, "close_strength": close_strength
                        })
                        print(f"✅ התפוצצות אותרה: {ticker}! (תבנית: {pattern['type']})")
                        
        except Exception: pass
        time.sleep(0.1)

    print("\n" + "="*50)
    if alerts:
        print(f"🔥 סריקת המעבדה הסתיימה! נמצאו {len(alerts)} איתותים:\n")
        for a in alerts:
            tv_link = f"https://il.tradingview.com/chart/?symbol={a['ticker']}"
            print(f"💎 {a['ticker']} ({a['type']})")
            print(f"💰 מחיר פריצה: ${a['close']:.2f} (פיבוט: ${a['pivot']:.2f})")
            print(f"🛡️ סטופ לוס: ${a['stop_loss']:.2f} (-{a['risk_pct']:.1f}%)")
            print(f"📊 ווליום: {a['vol_ratio']:.1f}x | RS: {a['rs_65']*100:.1f}%")
            print(f"📈 גרף: {tv_link}")
            print("-" * 30)
    else:
        print("💤 הסריקה הסתיימה. לא נמצאו פריצות איכותיות היום ברשימת הבדיקה.")
    print("="*50)

if __name__ == "__main__":
    scan_market_test()
