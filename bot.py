import yfinance as yf 
import pandas as pd 
import numpy as np 
import requests 
import time 
import os 
import json 
from datetime import datetime, timedelta 
import warnings 

warnings.filterwarnings("ignore") 

# ========================================== 
# 1. הגדרות וטוקנים 
# ========================================== 
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") 
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_GROUP") 
CUSTOM_TICKERS_FILE = "mystock.csv" 
MIN_MARKET_CAP = 2_000_000_000  # 1 מיליארד דולר מינימום לחברות בשיקום
MIN_DOLLAR_VOL_50 = 20_000_000  
MIN_PRICE = 10.0 
COOLDOWN_DAYS = 5 

def load_brain(): 
    brain = { 
        "max_base_depth": 0.85,         # פתוח להתרסקויות של עד 85% כדי לתפוס בסיסים ארוכים 
        "max_tightness_depth": 0.12, 
        "min_cup_depth": 0.10, 
        "max_cup_depth": 0.85,          # הוגדל בהתאמה
        "min_breakout_close_strength": 0.25, 
        "min_rs_65": 0.02, 
        "max_dist_from_52w_high_normal": 0.25, 
        "max_dist_from_52w_high_below_150": 0.85, # מניות בשיקום יכולות להיות רחוקות עד 85% מהשיא השנתי
        "max_gap_above_pivot": 0.03, 
        "max_entry_extension": 0.04, 
        "breakout_volume_ratio": 1.1 
    } 
    try: 
        if os.path.exists('brain.json'): 
            with open('brain.json', 'r') as f: 
                brain.update(json.load(f)) 
    except Exception as e: 
        print(f"⚠️ אזהרה: לא ניתן לטעון את brain.json, משתמש בברירות מחדל.") 
    return brain 

BRAIN = load_brain() 

# ========================================== 
# 2. פונקציות זיכרון, טלגרם ואנטי-ספאם 
# ========================================== 
def send_telegram(message): 
    print("\n=== תוכן ההודעה המלאה ===") 
    print(message) 
    print("=========================\n") 
    
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: 
        print("⚠️ [מצב סימולציה - הטוקן או ה-ID חסרים ב-GitHub Secrets]") 
        return 
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage" 
    payload = { 
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message, 
        "parse_mode": "HTML", 
        "disable_web_page_preview": True 
    } 
    try: 
        response = requests.post(url, json=payload, timeout=10) 
        if response.status_code != 200: 
            print(f"❌ טלגרם חסם את ההודעה לקבוצה! התשובה של טלגרם: {response.text}") 
        else: 
            print("✅ ההודעה נשלחה בהצלחה לקבוצת הטלגרם!") 
    except Exception as e: 
        print(f"❌ שגיאת תקשורת עם טלגרם: {e}") 

def log_signal(ticker, price): 
    log_file = 'trading_log.csv' 
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    new_data = pd.DataFrame([{'Date': now, 'Ticker': ticker, 'Price': price}]) 
    try: 
        if not os.path.isfile(log_file): 
            new_data.to_csv(log_file, index=False) 
        else: 
            new_data.to_csv(log_file, mode='a', header=False, index=False) 
    except: 
        pass 

def save_to_smart_memory(ticker, price, stop_loss, risk_pct, vol_ratio, pivot, close_strength, rs_65, tightness, pattern_type, status): 
    memory_file = "smart_memory.csv" 
    now = datetime.now().strftime("%Y-%m-%d") 
    new_record = pd.DataFrame([{ 
        "Date": now, "Ticker": ticker, "Price": round(price, 2), "Pivot": round(pivot, 2), 
        "Stop_Loss": round(stop_loss, 2), "Risk_Pct": round(risk_pct, 2), "Volume_Ratio": round(vol_ratio, 2), 
        "Close_Strength": round(close_strength, 2), "RS_65": round(rs_65, 4), "Tightness_Pct": round(tightness * 100, 2), 
        "Pattern_Type": pattern_type, "Status": status 
    }]) 
    try: 
        if not os.path.isfile(memory_file): 
            new_record.to_csv(memory_file, index=False) 
        else: 
            new_record.to_csv(memory_file, mode='a', header=False, index=False) 
    except: 
        pass 

def should_skip_spam(ticker): 
    log_file = 'trading_log.csv' 
    if not os.path.isfile(log_file): return False 
    try: 
        df = pd.read_csv(log_file) 
        df['Date'] = pd.to_datetime(df['Date']) 
        ticker_history = df[df['Ticker'] == ticker].sort_values(by='Date', ascending=False) 
        if ticker_history.empty: return False 
        last_sent = ticker_history.iloc[0]['Date'] 
        days_passed = (datetime.now() - last_sent).days 
        return days_passed < COOLDOWN_DAYS 
    except: 
        return False 

def load_tickers(): 
    if os.path.exists(CUSTOM_TICKERS_FILE): 
        try: 
            df = pd.read_csv(CUSTOM_TICKERS_FILE) 
            col_name = next( (c for c in df.columns if c.strip().lower() in ['ticker', 'symbol']), None ) 
            if col_name: 
                tickers = ( df[col_name].dropna().astype(str).str.strip().str.upper().tolist() ) 
                tickers = [ t.replace('.', '-') for t in tickers if t.isalpha() or '-' in t or '.' in t ] 
                print(f"✅ נטענו {len(set(tickers))} מניות מתוך {CUSTOM_TICKERS_FILE}") 
                return sorted(list(set(tickers))) 
        except: pass 
    return ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'PLTR'] 

# ========================================== 
# 3. פונקציות עזר מורחבות 
# ========================================== 
def check_market_cap(ticker): 
    try: 
        t = yf.Ticker(ticker) 
        fast = getattr(t, 'fast_info', None) 
        if fast is not None: 
            mc = getattr(fast, 'market_cap', None) 
            if mc and mc > 0: return float(mc) 
        info = t.info 
        mc = info.get('marketCap') or info.get('market_cap') 
        if mc and mc > 0: return float(mc) 
    except: 
        pass 
    return None 

def find_swing_highs(arr, window=5): 
    peaks = [] 
    for i in range(window, len(arr) - window): 
        local_max = max(arr[i - window : i + window + 1]) 
        if arr[i] == local_max and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]: 
            peaks.append((i, float(arr[i]))) 
    return peaks 

def find_swing_lows(arr, window=5): 
    troughs = [] 
    for i in range(window, len(arr) - window): 
        local_min = min(arr[i - window : i + window + 1]) 
        if arr[i] == local_min and arr[i] < arr[i - 1] and arr[i] < arr[i + 1]: 
            troughs.append((i, float(arr[i]))) 
    return troughs 

# ========================================== 
# 4. מנוע טכני 
# ========================================== 
def add_indicators(df): 
    df = df.copy() 
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0) 
    if getattr(df.index, "tz", None) is not None: 
        df.index = df.index.tz_localize(None) 
    df = df[~df.index.duplicated(keep='first')] 
    df["SMA_21"] = df["Close"].rolling(21).mean() 
    df["SMA_50"] = df["Close"].rolling(50).mean() 
    df["SMA_150"] = df["Close"].rolling(150).mean() 
    df["SMA_200"] = df["Close"].rolling(200).mean() 
    df["Vol_50"] = df["Volume"].rolling(50).mean() 
    df["DollarVol_50"] = df["Close"].rolling(50).mean() * df["Vol_50"] 
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

def get_spy_data(): 
    try: 
        spy = yf.download("SPY", period="1y", auto_adjust=True, progress=False) 
        if isinstance(spy.columns, pd.MultiIndex): 
            spy.columns = spy.columns.get_level_values(0) 
        if not spy.empty and len(spy) > 200: 
            spy["SMA_50"] = spy["Close"].rolling(50).mean() 
            spy["SMA_150"] = spy["Close"].rolling(150).mean() 
            spy["SMA_200"] = spy["Close"].rolling(200).mean() 
            spy["ROC_65"] = spy["Close"].pct_change(65) 
            return spy 
    except: 
        pass 
    return pd.DataFrame() 

def market_filter_ok(spy_df): 
    if spy_df.empty: return False 
    today = spy_df.iloc[-1] 
    sma200_old = spy_df["SMA_200"].iloc[-20] 
    if pd.isna(today["SMA_200"]) or pd.isna(sma200_old): return False 
    return ( (today["Close"] > today["SMA_50"] > today["SMA_150"] > today["SMA_200"]) and (today["SMA_200"] > sma200_old) ) 

# ========================================== 
# תבניות (מסוננות מפני תעלות עולות ומלכודות V)
# ========================================== 
def get_ascending_triangle_signal(hist): 
    recent = hist.tail(150) 
    if len(recent) < 60: return None 
    highs = recent["High"].values 
    lows = recent["Low"].values 
    
    swing_highs = find_swing_highs(highs[:-21], window=5) 
    if len(swing_highs) < 2: return None 
    peak1_pos, peak1_price = max(swing_highs, key=lambda x: x[1]) 
    
    post_peak1_lows = find_swing_lows(lows[peak1_pos + 1 : -5], window=3) 
    if not post_peak1_lows: return None 
    valley1_pos_rel, valley1_price = min(post_peak1_lows, key=lambda x: x[1]) 
    valley1_abs_pos = valley1_pos_rel + peak1_pos + 1 
    
    base_depth = (peak1_price - valley1_price) / max(peak1_price, 1e-9)
    if base_depth < 0.08 or base_depth > BRAIN['max_base_depth']: return None 
    
    post_valley1_highs = find_swing_highs(highs[valley1_abs_pos + 1 : -2], window=4) 
    if not post_valley1_highs: return None 
    peak2_pos_rel, peak2_price = max(post_valley1_highs, key=lambda x: x[1]) 
    
    if peak2_price < peak1_price * 0.85 or peak2_price > peak1_price * 1.02: return None 
    peak2_abs_pos = peak2_pos_rel + valley1_abs_pos + 1 
    
    if len(lows) - peak2_abs_pos < 3: return None

    post_peak2_lows = find_swing_lows(lows[peak2_abs_pos + 1 :], window=2) 
    if not post_peak2_lows: 
        raw = lows[peak2_abs_pos + 1 :] 
        if len(raw) < 3: return None 
        valley2_price = float(np.min(raw)) 
    else: 
        _, valley2_price = min(post_peak2_lows, key=lambda x: x[1]) 
        
    if valley2_price <= valley1_price * 1.015: return None 
    
    tightness = (peak2_price - valley2_price) / max(peak2_price, 1e-9)
    if tightness > BRAIN['max_tightness_depth']: return None 
    
    if tightness >= base_depth * 0.75: return None

    return { 
        "pivot_price": max(peak1_price, peak2_price), 
        "tight_low": valley2_price, 
        "tightness": tightness, 
        "type": "VCP" 
    } 

def get_cup_handle_signal(hist): 
    recent = hist.tail(250) 
    if len(recent) < 60: return None 
    highs = recent["High"].values 
    lows = recent["Low"].values 
    
    swing_highs = find_swing_highs(highs[:-21], window=5) 
    if not swing_highs: return None 
    left_rim_idx, left_rim_price = max(swing_highs, key=lambda x: x[1]) 
    
    cup_section = lows[left_rim_idx:] 
    if len(cup_section) < 21: return None 
    cup_low_rel_idx = int(np.argmin(cup_section))
    cup_low_abs_idx = left_rim_idx + cup_low_rel_idx
    cup_low_price = float(cup_section[cup_low_rel_idx]) 
    
    depth = (left_rim_price - cup_low_price) / max(left_rim_price, 1e-9)
    if not (0.10 <= depth <= BRAIN['max_cup_depth']): return None 
    
    recovery_highs = find_swing_highs(highs[cup_low_abs_idx:], window=3)
    if not recovery_highs: return None
    right_rim_rel_idx, right_rim_price = max(recovery_highs, key=lambda x: x[1])
    right_rim_abs_idx = cup_low_abs_idx + right_rim_rel_idx

    if right_rim_price < left_rim_price * 0.85 or right_rim_price > left_rim_price * 1.05: return None

    handle_section = lows[right_rim_abs_idx:]
    
    if len(handle_section) < 3: return None 
    
    handle_low = float(np.min(handle_section)) 
    if handle_low < (cup_low_price + 0.35 * (left_rim_price - cup_low_price)): return None 
    
    tightness = (right_rim_price - handle_low) / max(right_rim_price, 1e-9)
    if tightness >= depth * 0.75: return None
    if tightness > BRAIN['max_tightness_depth']: return None 

    return { 
        "pivot_price": right_rim_price, 
        "tight_low": handle_low, 
        "tightness": tightness, 
        "type": "Cup & Handle" 
    } 

def get_multi_touch_vcp_signal(hist): 
    recent = hist.tail(250) 
    if len(recent) < 60: return None 
    highs = recent["High"].values 
    lows = recent["Low"].values 
    
    peaks = [] 
    for i in range(10, len(highs) - 10): 
        if highs[i] == max(highs[i - 10 : i + 11]): 
            peaks.append((i, float(highs[i]))) 
            
    if len(peaks) < 3: return None 
    
    best_group = [] 
    best_pivot = 0.0 
    for p_idx, p_val in peaks: 
        group = [ idx for idx, val in peaks if abs(val - p_val) / max(p_val, 1e-9) <= 0.04 ] 
        if len(group) > len(best_group) or ( len(group) == len(best_group) and p_val > best_pivot ): 
            best_group = group 
            best_pivot = float(np.max([highs[i] for i in group])) 
            
    if len(best_group) < 3: return None 
    if best_group[-1] - best_group[0] < 30: return None 
    
    first_touch_idx = best_group[0] 
    base_low = float(np.min(lows[first_touch_idx:])) 
    depth = (best_pivot - base_low) / max(best_pivot, 1e-9) 
    
    if depth > BRAIN['max_base_depth'] or depth < 0.10: return None 
    
    last_touch_idx = best_group[-1]
    if len(highs) - last_touch_idx < 3: return None

    recent_low = float(np.min(lows[-15:])) 
    tightness = (best_pivot - recent_low) / max(best_pivot, 1e-9) 
    if tightness > BRAIN['max_tightness_depth']: return None 
    
    if tightness >= depth * 0.60: return None

    return { 
        "pivot_price": best_pivot, 
        "tight_low": recent_low, 
        "tightness": tightness, 
        "type": f"Flat Base ({len(best_group)} Touches)" 
    } 

# ========================================== 
# 5. סריקת שוק ראשית 
# ========================================== 
def scan_market(): 
    tickers = load_tickers() 
    if not tickers: return 
    print(f"📥 בודק את מגמת השוק (SPY)...") 
    spy = get_spy_data() 
    market_warning = "" 
    spy_rs = 0.0 
    if spy.empty: 
        market_warning = ( "🔴 <b>שגיאה: לא ניתן למשוך נתוני שוק (SPY).</b> " "הסריקה ממשיכה ללא פילטר מגמה.\n\n" ) 
    else: 
        spy_rs = float(spy.iloc[-1]["ROC_65"]) 
        if not market_filter_ok(spy): 
            market_warning = ( "⚠️ <b>שים לב: השוק הכללי חלש (SPY מתחת לממוצעים).</b> " "הסריקה ממשיכה לבקשתך, אך הסיכון לפריצות שווא גבוה.\n\n" ) 
            
    all_potentials = [] 
    market_cap_cache = {} 
    
    for ticker in tickers: 
        print(f"סורק את {ticker}...", end="\r") 
        try: 
            df = yf.download(ticker, period="1y", auto_adjust=True, progress=False) 
            if df.empty or len(df) < 200: continue 
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0) 
            df = add_indicators(df) 
            today = df.iloc[-1] 
            yesterday = df.iloc[-2] 
            past_data = df.iloc[:-1] 
            
            if any(pd.isna(today[c]) for c in ["SMA_50", "SMA_150", "SMA_200", "ATR_14"]): continue 
            if float(today["Close"]) < MIN_PRICE: continue 
            if float(today["DollarVol_50"]) < MIN_DOLLAR_VOL_50: continue 
            
            if ticker not in market_cap_cache: 
                mc = check_market_cap(ticker) 
                market_cap_cache[ticker] = mc 
            else: 
                mc = market_cap_cache[ticker] 
            if mc is not None and mc < MIN_MARKET_CAP: 
                print(f" ↳ {ticker} נפסל: market cap ${mc/1e9:.1f}B < מינימום") 
                continue 
                
            if not (float(today["Close"]) > float(today["SMA_50"])): continue 
            is_below_150 = float(today["Close"]) < float(today["SMA_150"]) 
            if is_below_150 and float(today["SMA_50"]) <= float(yesterday["SMA_50"]): continue 
            
            high_252 = float(today["High_252"]) if not pd.isna(today["High_252"]) else None 
            if high_252 is not None and high_252 > 0: 
                dist_52w = (float(today["Close"]) / high_252) - 1.0 
                max_dist = ( BRAIN['max_dist_from_52w_high_below_150'] if is_below_150 else BRAIN['max_dist_from_52w_high_normal'] ) 
                if dist_52w < -max_dist: continue 
                
            required_rs = BRAIN['min_rs_65'] * 2 if is_below_150 else BRAIN['min_rs_65'] 
            stock_rs = float(today["ROC_65"]) - spy_rs 
            if stock_rs < required_rs: continue 
            
            pattern = ( get_ascending_triangle_signal(past_data) or get_cup_handle_signal(past_data) or get_multi_touch_vcp_signal(past_data) ) 
            if not pattern: continue 
            if is_below_150 and pattern["tightness"] > 0.05: continue 
            
            pivot = pattern["pivot_price"] 
            dist_to_pivot = (float(today["Close"]) / pivot) - 1.0 
            is_breakout = (float(yesterday["Close"]) <= pivot and float(today["Close"]) > pivot) 
            is_near_breakout = (-0.05 <= dist_to_pivot <= 0.0) 
            
            if is_breakout or is_near_breakout: 
                if should_skip_spam(ticker): continue 
                day_range = max(float(today["High"]) - float(today["Low"]), 1e-9) 
                close_strength = (float(today["Close"]) - float(today["Low"])) / day_range 
                vol_ratio = float(today["Volume"]) / float(today["Vol_50"]) 
                gap_from_pivot = (float(today["Open"]) / pivot) - 1.0 
                if is_breakout: 
                    req_vol = 1.5 if is_below_150 else BRAIN['breakout_volume_ratio'] 
                    req_close = 0.5 if is_below_150 else BRAIN['min_breakout_close_strength'] 
                    if (close_strength < req_close or vol_ratio < req_vol or gap_from_pivot > BRAIN['max_gap_above_pivot']): continue 
                    status = "🔥 פריצה פעילה!" 
                else: 
                    status = "👀 מתבשלת (Watchlist)" 
                    
                if float(today["Close"]) <= pivot * (1 + BRAIN['max_entry_extension']): 
                    atr = float(today["ATR_14"]) 
                    stop_price = float(pattern["tight_low"]) - (0.2 * atr) 
                    risk_pct = (float(today["Close"]) - stop_price) / float(today["Close"]) * 100 
                    mc_str = f"${mc/1e9:.1f}B" if mc else "N/A" 
                    alert_data = { 
                        "ticker": ticker, "close": float(today["Close"]), "pivot": pivot, 
                        "stop_loss": stop_price, "risk_pct": risk_pct, "vol_ratio": vol_ratio, 
                        "type": pattern["type"], "rs_65": stock_rs, "close_strength":close_strength, 
                        "status": status, "dist_to_pivot": dist_to_pivot * 100, 
                        "tightness": pattern["tightness"], "is_below_150": is_below_150, 
                        "market_cap": mc_str, "dist_52w": dist_52w * 100 if high_252 else None 
                    } 
                    all_potentials.append(alert_data) 
        except Exception as e: 
            pass 
        time.sleep(0.15) 

# ========================================== 
# 6. שיגור האיתותים לטלגרם 
# ========================================== 
    print("\n" + "=" * 50) 
    all_potentials_sorted = sorted(all_potentials, key=lambda x: abs(x["dist_to_pivot"])) 
    final_selection = [] 
    below_150_count = 0 
    for stock in all_potentials_sorted: 
        if len(final_selection) >= 10: break 
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
        print(f"🔥 הסריקה הסתיימה! מציג {total_sent} מניות (מתוכן {below_150_count} מתחת לממוצע 150) בטלגרם.") 
        msg = "🎯 <b>סריקת VCP יומית הסתיימה!</b>\n" 
        if market_warning: 
            msg += f"{market_warning}\n" 
        msg += ( f"<i>(מציג {total_sent} מניות קרובות לפיבוט. " f"מקסימום 3 מתחת לממוצע 150)</i>\n\n" ) 
        
        def format_stock_block(a, icon): 
            tv_link = f"https://il.tradingview.com/chart/?symbol={a['ticker']}" 
            warning_150 = " ⚠️ (מתחת ל-150)" if a["is_below_150"] else "" 
            dist_52w_str = ( f" | 📉 <b>מ-52W High:</b> {a['dist_52w']:.1f}%" if a["dist_52w"] is not None else "" ) 
            line = f"{icon} <b>{a['ticker']}</b> | תבנית: {a['type']}{warning_150}\n" 
            line += f"💼 <b>שווי שוק:</b> {a['market_cap']}{dist_52w_str}\n" 
            line += f"📈 <b>עוצמה (RS):</b> {a['rs_65']*100:.1f}% | 📊 <b>ווליום:</b> {a['vol_ratio']:.1f}x\n" 
            line += ( f"📐 <b>כיווץ (Tightness):</b> {a['tightness']*100:.1f}% | " f"🔋 <b>עוצמת סגירה:</b> {a['close_strength']*100:.0f}%\n" ) 
            line += f"🎯 <b>פיבוט:</b> ${a['pivot']:.2f} | 💵 <b>מחיר:</b> ${a['close']:.2f}\n" 
            line += f"🛡️ <b>סטופ לוס:</b> ${a['stop_loss']:.2f} (סיכון: {a['risk_pct']:.1f}%-)\n" 
            line += f"🔗 <a href='{tv_link}'>גרף ב-TradingView</a>\n" 
            line += "────────────────\n" 
            return line 
            
        if final_bo: 
            msg += f"🔥 <b>פריצות אקטיביות ({len(final_bo)}):</b>\n\n" 
            for a in final_bo: 
                msg += format_stock_block(a, "🚀") 
                log_signal(a['ticker'], a['close']) 
                save_to_smart_memory( a['ticker'], a['close'], a['stop_loss'], a['risk_pct'], a['vol_ratio'], a['pivot'], a['close_strength'], a['rs_65'], a['tightness'], a['type'], "Sent" ) 
                
        if final_wl: 
            msg += f"👀 <b>מתבשלות למעקב ({len(final_wl)}):</b>\n\n" 
            for a in final_wl: 
                base_line = format_stock_block(a, "⏳") 
                base_line = base_line.replace( f"🎯 <b>פיבוט:</b> ${a['pivot']:.2f}", f"🎯 <b>פיבוט:</b> ${a['pivot']:.2f} (מרחק: {a['dist_to_pivot']:.1f}%)" ) 
                msg += base_line 
                log_signal(a['ticker'], a['close']) 
                save_to_smart_memory( a['ticker'], a['close'], a['stop_loss'], a['risk_pct'], a['vol_ratio'], a['pivot'], a['close_strength'], a['rs_65'], a['tightness'], a['type'], "Sent" ) 
                
        send_telegram(msg) 
    else: 
        print("💤 הסריקה הסתיימה. לא נמצאו מניות חדשות לשליחה בסיבוב זה.") 
        send_telegram( f"✅ הסריקה הסתיימה.\n\n{market_warning}" f"אין פריצות או מניות חדשות במעקב שלא נשלחו ב-5 הימים האחרונים." ) 
    print("=" * 50) 

if __name__ == "__main__": 
    scan_market()
