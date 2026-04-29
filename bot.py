import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. הגדרות סריקה (Standalone Mode)
# ==========================================
CUSTOM_TICKERS_FILE = "mystock.csv" 

MIN_MARKET_CAP = 2_000_000_000  
MIN_DOLLAR_VOL_50 = 20_000_000  
MIN_PRICE = 12.0

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
    if os.path.exists(CUSTOM_TICKERS_FILE):
        try:
            df = pd.read_csv(CUSTOM_TICKERS_FILE)
            col_name = next((c for c in df.columns if c.strip().lower() in ['ticker', 'symbol']), None)
            if col_name:
                tickers = df[col_name].dropna().astype(str).str.strip().str.upper().tolist()
                tickers = [t.replace('.', '-') for t in tickers if t.replace('-', '').replace('.', '').isalnum()]
                print(f"✅ נטענו {len(set(tickers))} מניות מהקובץ {CUSTOM_TICKERS_FILE}")
                return sorted(list(set(tickers)))
        except Exception as e:
            print(f"⚠️ שגיאה בקריאת הקובץ: {e}")
    return ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'PLTR', 'AMD', 'NFLX']

# ==========================================
# 2. פונקציות עזר טכניות - תיקון ה-MultiIndex
# ==========================================
def add_indicators(df):
    df = df.copy()
    
    # תיקון קריטי: אם הנתונים הגיעו עם MultiIndex (כותרות כפולות), משטחים אותם
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if getattr(df.index, "tz", None) is not None: 
        df.index = df.index.tz_localize(None)
        
    df = df[~df.index.duplicated(keep='first')]
    
    # חישוב אינדיקטורים
    df["SMA_21"] = df["Close"].rolling(21).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_150"] = df["Close"].rolling(150).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Vol_50"] = df["Volume"].rolling(50).mean()
    
    # עכשיו החישוב הזה יעבוד חלק
    df["DollarVol_50"] = df["Close"].rolling(50).mean() * df["Volume"].rolling(50).mean()
    
    df["Prev_Close"] = df["Close"].shift(1)
    df["ROC_65"] = df["Close"].pct_change(65)
    df["High_252"] = df["High"].rolling(252).max()
    
    tr = pd.concat([
        df["High"] - df["Low"], 
        (df["High"] - df["Prev_Close"]).abs(), 
        (df["Low"] - df["Prev_Close"]).abs()
    ], axis=1).max(axis=1)
    
    df["ATR_14"] = tr.rolling(14).mean()
    return df

def get_ascending_triangle_signal(hist):
    recent = hist.tail(150)
    if len(recent) < 60: return None
    highs, lows = recent["High"].values, recent["Low"].values
    slice1 = highs[:-10]
    if len(slice1) == 0: return None
    p1 = int(np.argmax(slice1))
    peak1_price = float(slice1[p1])
    post_p1_lows = lows[p1+1 : -5]
    if len(post_p1_lows) < 10: return None
    v1_price = float(np.min(post_p1_lows))
    v1_pos = p1 + 1 + int(np.argmin(post_p1_lows))
    if (peak1_price - v1_price) / peak1_price > BRAIN['max_base_depth']: return None
    post_v1_highs = highs[v1_pos+1 : -2]
    if len(post_v1_highs) < 5: return None
    peak2_price = float(np.max(post_v1_highs))
    peak2_pos = v1_pos + 1 + int(np.argmax(post_v1_highs))
    if not (peak1_price * 0.96 <= peak2_price <= peak1_price * 1.02): return None
    post_p2_lows = lows[peak2_pos+1 :]
    if len(post_p2_lows) < 3: return None
    v2_price = float(np.min(post_p2_lows))
    if v2_price <= v1_price * 1.015: return None
    tightness = (peak2_price - v2_price) / peak2_price
    if tightness > BRAIN['max_tightness_depth']: return None
    return {"pivot": max(peak1_price, peak2_price), "low": v2_price, "type": "VCP"}

def get_cup_handle_signal(hist):
    recent = hist.tail(250)
    if len(recent) < 60: return None
    highs, lows = recent["High"].values, recent["Low"].values
    rim_idx = np.argmax(highs)
    rim_price = float(highs[rim_idx])
    cup_low = float(np.min(lows[rim_idx:]))
    depth = (rim_price - cup_low) / rim_price
    if not (BRAIN['min_cup_depth'] <= depth <= BRAIN['max_cup_depth']): return None
    if len(recent.iloc[rim_idx:]) < 15: return None 
    handle_low = float(recent.iloc[rim_idx:]["Low"].min())
    if handle_low < (cup_low + 0.5 * (rim_price - cup_low)): return None 
    return {"pivot": rim_price, "low": handle_low, "type": "Cup & Handle"}

# ==========================================
# 3. סורק המעבדה
# ==========================================
def run_standalone_test():
    print("\n" + "="*60)
    print(f"🚀 סריקת מעבדה: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    tickers = load_tickers()
    try:
        spy = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
        spy = add_indicators(spy)
        spy_today = spy.iloc[-1]
        spy_rs = float(spy_today["ROC_65"])
        market_ok = float(spy_today["Close"]) > float(spy_today["SMA_200"])
        if not market_ok:
            print("🔴 אזהרה: SPY מתחת לממוצע 200.")
        print("🟢 השוק נטען בהצלחה.\n")
    except Exception as e:
        print(f"❌ שגיאה בטעינת SPY: {e}")
        return

    alerts = []

    for ticker in tickers:
        print(f"🔍 בודק {ticker}...", end="\r")
        try:
            df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
            if df.empty or len(df) < 200: continue
            df = add_indicators(df)
            today, yesterday, past_data = df.iloc[-1], df.iloc[-2], df.iloc[:-1]
            
            if float(today["Close"]) < MIN_PRICE or float(today["DollarVol_50"]) < MIN_DOLLAR_VOL_50: continue
            if not (float(today["Close"]) > float(today["SMA_50"]) > float(today["SMA_150"]) > float(today["SMA_200"])): continue
            
            stock_rs = float(today["ROC_65"]) - spy_rs
            if stock_rs < BRAIN['min_rs_65']: continue

            pattern = get_ascending_triangle_signal(past_data) or get_cup_handle_signal(past_data)
            if not pattern: continue
            
            pivot = pattern["pivot"]
            dist = (float(today["Close"]) / pivot) - 1.0
            
            is_breakout = (float(yesterday["Close"]) <= pivot and float(today["Close"]) > pivot)
            is_near = (-0.04 <= dist <= 0.0)
            
            if is_breakout or is_near:
                vol_ratio = float(today["Volume"]) / float(today["Vol_50"])
                if is_breakout and vol_ratio < BRAIN['breakout_volume_ratio']: continue
                
                status = "🔥 פריצה אקטיבית" if is_breakout else "👀 ברשימת מעקב"
                atr = float(today["ATR_14"])
                stop = float(pattern["low"]) - (0.5 * atr)
                risk = (float(today["Close"]) - stop) / float(today["Close"]) * 100
                
                alerts.append({
                    "ticker": ticker, "status": status, "type": pattern["type"],
                    "close": float(today["Close"]), "pivot": pivot, "dist": dist*100,
                    "stop": stop, "risk": risk, "vol": vol_ratio
                })
        except: pass

    print("\n" + "="*60)
    if alerts:
        alerts = sorted(alerts, key=lambda x: x["dist"], reverse=True)
        for a in alerts:
            print(f"💎 {a['ticker']} | {a['status']}")
            print(f"📐 תבנית: {a['type']} | מרחק: {a['dist']:.1f}%")
            print(f"💰 מחיר: ${a['close']:.2f} | פיבוט: ${a['pivot']:.2f}")
            print(f"🛡️ סטופ: ${a['stop']:.2f} ({a['risk']:.1f}%) | ווליום: {a['vol']:.1f}x")
            print("-" * 30)
    else:
        print("💤 לא נמצאו מניות מתאימות היום.")
    print("="*60)

if __name__ == "__main__":
    run_standalone_test()
