# -*- coding: utf-8 -*-
"""
SSHL Multi-Symbol Diagnostic Script - Latest Signal Version
Produces one row per symbol, using the most recent signal (if any)
Output CSV sorted by signal date descending
"""
import os
import re
import base64
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

# ===========================
# CONFIG
# ===========================
REPO_OWNER = "iamsrijit0"
REPO_NAME  = "Nepse"
BRANCH     = "main"

EXCLUDED_SYMBOLS = [
    "EBLD852", "EB89", "NABILD2089", "MBLD2085", "SBID89", "SBID2090",
    "SBLD2091", "NIMBD90", "RBBD2088", "CCBD88", "ICFCD88", "EBLD91",
    "GBILD84/85", "GBILD86/87", "NICD88", "CMF2", "GBIMESY2", "GIBF1",
    "GSY", "H8020", "HLICF", "KDBY", "KEF", "KSY", "LUK", "LVF2",
    "MBLEF", "MMF1", "MNMF1", "NBF2", "NBF3", "NIBLGF", "NIBLSTF",
    "NIBSF2", "NICBF", "NICFC", "NICGF2", "NICSF", "NMB50", "NMBHF2",
    "NSIF2", "PRSF", "PSF", "RMF1", "RMF2", "RSY", "SAGF", "SBCF",
    "SEF", "SFEF", "SIGS2", "SIGS3", "SLCF", "HEIP", "HIDCLP",
    "NIMBPO", "NLICLP", "RBCLPO", "C30MF", "ENL", "GWFD83", "HATHY",
    "HIDCL", "JBBD87", "PCBLP", "CZBILP", "HBLD83", "NLICP", "KBLPO",
    "JBLBP", "KMCDB", "ICFCD83", "ADBLD83", "GILB", "NIBD2082",
    "RBBD83", "SRBLD83", "SBID83", "RBBF40", "PBLD84", "GBBD85",
    "HBLD86", "SAND2085", "PBLD86", "NICAD2091", "CIZBD86", "EBLD85",
    "NMBD87/88", "PBD84", "NICAD85/86", "SBD87", "NBBD2085", "NBLD82",
    "NIBD84", "BOKD86KA", "NCCD86", "EBLEB89", "SBIBD86", "KSBBLD87",
    "MLBLD89", "NIFRAGED", "BOKD86", "PBLD87", "NMBD2085", "NBLD87",
    "MBLD87", "SDBD87", "NMBMF", "SBD89", "PBD88", "CBLD88", "KBLD89",
    "NMBD89/90", "LBBLD89", "NABILD87", "CIZBD90", "LBLD88", "SBLD89",
    "KBLD86", "MLBLPO", "KBLD90", "PROFLP"
]

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set in environment")

HEADERS = {"Authorization": f"token {GH_TOKEN}"}

# ===========================
# GITHUB HELPERS
# ===========================
def github_raw(path):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

def upload_to_github(filename, content):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"
    r = requests.get(url, headers=HEADERS)
    payload = {
        "message": f"Upload {filename}",
        "content": base64.b64encode(content.encode()).decode(),
        "branch": BRANCH
    }
    if r.status_code == 200:
        payload["sha"] = r.json()["sha"]
    res = requests.put(url, headers=HEADERS, json=payload)
    if res.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed: {res.text}")
    print(f"✅ Uploaded: {filename}")

def delete_old_files(prefix, keep_filename):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()
    for f in r.json():
        name = f["name"]
        if name.startswith(prefix) and name.endswith(".csv") and name != keep_filename:
            del_payload = {
                "message": f"Delete old file {name}",
                "sha": f["sha"],
                "branch": BRANCH
            }
            del_url = f"{url}/{name}"
            res = requests.delete(del_url, headers=HEADERS, json=del_payload)
            if res.status_code == 200:
                print(f"🗑️ Deleted: {name}")

# ===========================
# GET LATEST ESPEN CSV
# ===========================
def get_latest_espen_csv():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()
    
    espen_files = {}
    for f in r.json():
        name = f["name"]
        if name.startswith("espen_") and name.endswith(".csv"):
            m = re.search(r"espen_(\d{4}-\d{2}-\d{2})\.csv", name)
            if m:
                espen_files[m.group(1)] = name
    
    if not espen_files:
        raise FileNotFoundError("No espen_*.csv file found")
    
    latest_date = max(espen_files.keys())
    latest_file = espen_files[latest_date]
    
    print(f"📂 Using market data file: {latest_file}")
    return github_raw(latest_file), latest_date

# ===========================
# LOAD AND PREPARE DATA
# ===========================
def load_market_data():
    csv_url, latest_date = get_latest_espen_csv()
    response = requests.get(csv_url)
    csv_content = response.text
    
    lines = csv_content.strip().split('\n')
    header_line = None
    data_start_index = 0
    
    for i, line in enumerate(lines):
        if 'Symbol' in line and 'Date' in line and 'Close' in line:
            header_line = line
            data_start_index = i
            break
    
    if header_line is None:
        raise ValueError("Could not find header line")
    
    if data_start_index > 0:
        data_lines = lines[data_start_index:]
        reconstructed_csv = header_line + '\n' + '\n'.join(data_lines)
    else:
        reconstructed_csv = csv_content
    
    try:
        df = pd.read_csv(StringIO(reconstructed_csv), sep='\t')
        if len(df.columns) == 1:
            df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
    except:
        df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
    
    df.columns = df.columns.str.strip()
    
    original_dates = df["Date"].copy()
    df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y', errors='coerce')
    
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(original_dates, format='%Y-%m-%d', errors='coerce')
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(original_dates, errors='coerce')
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=["Symbol", "Date", "Close"])
    df = df.sort_values(["Symbol", "Date"])
    
    print(f"✅ Loaded {len(df)} rows, {df['Symbol'].nunique()} unique symbols")
    print(f"📅 Latest market date: {latest_date}")
    
    return df, latest_date

# ===========================
# NET VOLUME CALCULATIONS
# ===========================
def calculate_net_volume(df_symbol):
    df = df_symbol.copy()
    
    df['Net_Volume']       = (df['Close'] - df['Open']) * df['Volume']
    df['Cumulative_NV']    = df['Net_Volume'].cumsum()
    df['NV_3M_Low']        = df['Net_Volume'].rolling(window=63, min_periods=20).min()
    df['NV_Improvement_Pct'] = 0.0
    
    for i in range(1, len(df)):
        prev = df['Net_Volume'].iloc[i-1]
        if prev != 0:
            df.loc[df.index[i], 'NV_Improvement_Pct'] = (
                (df['Net_Volume'].iloc[i] - prev) / abs(prev) * 100
            )
    
    df['High_252D']     = df['High'].rolling(window=252, min_periods=50).max()
    df['Drawdown_Pct']  = ((df['Close'] - df['High_252D']) / df['High_252D']) * 100
    df['EMA_50']        = df['Close'].ewm(span=50, adjust=False).mean()
    df['Pct_From_EMA50']= ((df['Close'] - df['EMA_50']) / df['EMA_50']) * 100
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio']  = df['Volume'] / df['Volume_SMA_20']
    
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# ===========================
# SIGNAL DETECTION
# ===========================
def detect_standard_signals(df):
    signals = pd.Series(False, index=df.index)
    for i in range(50, len(df)):
        if (df['Net_Volume'].iloc[i] > df['Net_Volume'].iloc[i-1] and
            df['RSI'].iloc[i] < 40 and
            df['Drawdown_Pct'].iloc[i] < -10):
            signals.iloc[i] = True
    return signals

def detect_strict_signals(df, params=None):
    if params is None:
        params = {
            'nv_3m_low_pct': 10,
            'nv_improvement_pct': 30,
            'drawdown_min': -15,
            'ema50_min': -25,
            'ema50_max': -15,
            'volume_ratio_max': 0.5
        }
    
    signals = pd.Series(False, index=df.index)
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        if row['NV_3M_Low'] == 0:
            continue
        nv_pct = ((row['Net_Volume'] - row['NV_3M_Low']) / abs(row['NV_3M_Low'])) * 100
        
        if (nv_pct <= params['nv_3m_low_pct'] and
            row['NV_Improvement_Pct'] >= params['nv_improvement_pct'] and
            row['Drawdown_Pct'] <= params['drawdown_min'] and
            params['ema50_min'] <= row['Pct_From_EMA50'] <= params['ema50_max'] and
            row['Volume_Ratio'] < params['volume_ratio_max']):
            signals.iloc[i] = True
    
    return signals

# ===========================
# ANALYZE SINGLE SYMBOL → ONE ROW
# ===========================
def analyze_symbol(symbol, df_symbol):
    if len(df_symbol) < 100:
        return None
    
    df_symbol = df_symbol.set_index('Date').sort_index()
    df_ind = calculate_net_volume(df_symbol)
    
    std_signals  = detect_standard_signals(df_ind)
    str_signals  = detect_strict_signals(df_ind)
    
    num_std  = std_signals.sum()
    num_str  = str_signals.sum()
    
    latest_close = df_ind['Close'].iloc[-1]
    latest_rsi   = df_ind['RSI'].iloc[-1]
    
    # No signals → return placeholder row
    if num_std == 0:
        return {
            'Symbol': symbol,
            'Signal_Date': 'N/A',
            'Close_at_Signal': 0.0,
            'Latest_Close': round(latest_close, 2),
            'P/L_%': 0.0,
            'Status': 'NO_SIGNALS',
            'Standard_Signals': 0,
            'Strict_Signals': 0,
            'Latest_RSI': round(latest_rsi, 1) if pd.notna(latest_rsi) else 0.0,
            'NV_Range': 'N/A',
            'Recommended_NV_Threshold': 10,
            'Recommended_NV_Improvement': 30,
            'Recommended_Drawdown': -15,
            'Recommended_EMA50_Min': -25,
            'Recommended_EMA50_Max': -15,
            'Recommended_Volume_Ratio': 0.5
        }
    
    # ─── Has signals → take the LATEST one ───
    signal_dates = df_ind[std_signals].index
    if len(signal_dates) == 0:
        return None
    
    last_date = signal_dates.max()
    row = df_ind.loc[last_date]
    
    close_signal = row['Close']
    pl_pct = ((latest_close - close_signal) / close_signal * 100) if close_signal != 0 else 0.0
    
    status = 'STRICT_PASS' if str_signals.loc[last_date] else 'STANDARD_ONLY'
    
    # NV position at signal
    nv_pct_at_signal = 'N/A'
    if row['NV_3M_Low'] != 0:
        nv_pct_at_signal = ((row['Net_Volume'] - row['NV_3M_Low']) / abs(row['NV_3M_Low'])) * 100
    
    # Collect metrics from ALL signals for recommendation logic
    metrics = {
        'nv_3m_low_pct': [],
        'nv_improvement': [],
        'drawdown': [],
        'ema50_pct': [],
        'volume_ratio': []
    }
    
    for dt in signal_dates:
        r = df_ind.loc[dt]
        if r['NV_3M_Low'] != 0:
            metrics['nv_3m_low_pct'].append(
                ((r['Net_Volume'] - r['NV_3M_Low']) / abs(r['NV_3M_Low'])) * 100
            )
        metrics['nv_improvement'].append(r['NV_Improvement_Pct'])
        metrics['drawdown'].append(r['Drawdown_Pct'])
        metrics['ema50_pct'].append(r['Pct_From_EMA50'])
        metrics['volume_ratio'].append(r['Volume_Ratio'])
    
    rec = {}
    if metrics['nv_3m_low_pct']:
        rec['nv_3m_low_pct']     = max(metrics['nv_3m_low_pct']) + 5
    if metrics['nv_improvement']:
        rec['nv_improvement_pct'] = min(metrics['nv_improvement']) - 5
    if metrics['drawdown']:
        rec['drawdown_min']      = max(metrics['drawdown']) + 2
    if metrics['ema50_pct']:
        rec['ema50_min']         = min(metrics['ema50_pct']) - 5
        rec['ema50_max']         = max(metrics['ema50_pct']) + 5
    if metrics['volume_ratio']:
        rec['volume_ratio_max']   = max(metrics['volume_ratio']) + 0.1
    
    nv_range_str = f"{min(metrics['nv_3m_low_pct']):.1f} – {max(metrics['nv_3m_low_pct']):.1f}" \
                   if metrics['nv_3m_low_pct'] else 'N/A'
    
    return {
        'Symbol': symbol,
        'Signal_Date': last_date.strftime('%Y-%m-%d'),
        'Close_at_Signal': round(close_signal, 2),
        'Latest_Close': round(latest_close, 2),
        'P/L_%': round(pl_pct, 2),
        'Status': status,
        'Standard_Signals': num_std,
        'Strict_Signals': num_str,
        'Latest_RSI': round(latest_rsi, 1) if pd.notna(latest_rsi) else 0.0,
        'NV_Range': nv_range_str,
        'Recommended_NV_Threshold': round(rec.get('nv_3m_low_pct', 10), 0),
        'Recommended_NV_Improvement': round(rec.get('nv_improvement_pct', 30), 0),
        'Recommended_Drawdown': round(rec.get('drawdown_min', -15), 0),
        'Recommended_EMA50_Min': round(rec.get('ema50_min', -25), 0),
        'Recommended_EMA50_Max': round(rec.get('ema50_max', -15), 0),
        'Recommended_Volume_Ratio': round(rec.get('volume_ratio_max', 0.5), 2)
    }

# ===========================
# MAIN
# ===========================
def main():
    print("\n" + "="*80)
    print("SSHL DIAGNOSTIC - LATEST SIGNAL PER SYMBOL")
    print("One row per symbol • sorted by most recent signal date")
    print("="*80)
    
    df, market_date = load_market_data()
    symbols = [s for s in df['Symbol'].unique() if s not in EXCLUDED_SYMBOLS]
    
    print(f"\nAnalyzing {len(symbols)} symbols...")
    
    results = []
    for i, sym in enumerate(symbols, 1):
        if i % 50 == 0:
            print(f"  {i:3d}/{len(symbols)} ...")
        data = df[df['Symbol'] == sym].copy()
        row = analyze_symbol(sym, data)
        if row:
            results.append(row)
    
    if not results:
        print("No valid results generated.")
        return
    
    df_out = pd.DataFrame(results)
    
    # Split & sort
    has_signal = df_out['Signal_Date'] != 'N/A'
    with_signal = df_out[has_signal].copy()
    without     = df_out[~has_signal].copy()
    
    if not with_signal.empty:
        with_signal['dt'] = pd.to_datetime(with_signal['Signal_Date'])
        with_signal = with_signal.sort_values('dt', ascending=False).drop(columns=['dt'])
    
    final = pd.concat([with_signal, without], ignore_index=True)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total symbols:       {len(final):3d}")
    print(f"  With signals:      {len(with_signal):3d}")
    print(f"    • Strict pass    {len(final[(final['Status']=='STRICT_PASS') & has_signal]):3d}")
    print(f"    • Standard only  {len(final[(final['Status']=='STANDARD_ONLY') & has_signal]):3d}")
    print(f"  No signals:        {len(without):3d}")
    
    if not with_signal.empty:
        print(f"Avg P/L (latest signals): {with_signal['P/L_%'].mean():.2f}%")
    
    # Save
    filename = f"SSHL_LATEST_SIGNALS_{market_date}.csv"
    upload_to_github(filename, final.to_csv(index=False))
    delete_old_files("SSHL_LATEST_SIGNALS_", filename)
    delete_old_files("SSHL_DIAGNOSTIC_RESULTS_", "")
    
    # Preview top 10
    print("\nTop 10 most recent signals:")
    print(final.head(12)[[
        'Symbol','Signal_Date','P/L_%','Status','Latest_RSI','Close_at_Signal','Latest_Close'
    ]].to_string(index=False))
    
    print(f"\nResults saved → {filename}")
    print("="*80)

if __name__ == "__main__":
    main()
