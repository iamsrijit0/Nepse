# -*- coding: utf-8 -*-
"""
SSHL Multi-Symbol Diagnostic Script
Analyzes Net Volume signals across multiple symbols from espen_*.csv files
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
REPO_NAME = "Nepse"
BRANCH = "main"

# Symbols to exclude from analysis
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
    
    # Find header line
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
    
    # Reconstruct CSV
    if data_start_index > 0:
        data_lines = lines[:data_start_index]
        reconstructed_csv = header_line + '\n' + '\n'.join(data_lines)
    else:
        reconstructed_csv = csv_content
    
    # Parse CSV
    try:
        df = pd.read_csv(StringIO(reconstructed_csv), sep='\t')
        if len(df.columns) == 1:
            df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
    except:
        df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Parse dates
    original_dates = df["Date"].copy()
    df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y', errors='coerce')
    
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(original_dates, format='%Y-%m-%d', errors='coerce')
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(original_dates, errors='coerce')
    
    # Convert numeric columns
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
    """Calculate Net Volume and related metrics"""
    df = df_symbol.copy()
    
    # Net Volume = (Close - Open) * Volume
    df['Net_Volume'] = (df['Close'] - df['Open']) * df['Volume']
    
    # Cumulative Net Volume
    df['Cumulative_NV'] = df['Net_Volume'].cumsum()
    
    # Net Volume 3-month low
    df['NV_3M_Low'] = df['Net_Volume'].rolling(window=63, min_periods=20).min()
    
    # Net Volume improvement
    df['NV_Improvement_Pct'] = 0.0
    for i in range(1, len(df)):
        if df['Net_Volume'].iloc[i-1] != 0:
            df.loc[df.index[i], 'NV_Improvement_Pct'] = (
                (df['Net_Volume'].iloc[i] - df['Net_Volume'].iloc[i-1]) / 
                abs(df['Net_Volume'].iloc[i-1]) * 100
            )
    
    # Price metrics
    df['High_252D'] = df['High'].rolling(window=252, min_periods=50).max()
    df['Drawdown_Pct'] = ((df['Close'] - df['High_252D']) / df['High_252D']) * 100
    
    # EMAs
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Pct_From_EMA50'] = ((df['Close'] - df['EMA_50']) / df['EMA_50']) * 100
    
    # Volume ratio
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# ===========================
# SIGNAL DETECTION
# ===========================
def detect_standard_signals(df):
    """Detect signals using standard mode criteria"""
    signals = pd.Series(False, index=df.index)
    
    # Standard mode criteria (simplified example)
    for i in range(50, len(df)):
        if (df['Net_Volume'].iloc[i] > df['Net_Volume'].iloc[i-1] and
            df['RSI'].iloc[i] < 40 and
            df['Drawdown_Pct'].iloc[i] < -10):
            signals.iloc[i] = True
    
    return signals

def detect_strict_signals(df, params=None):
    """Detect signals using strict mode criteria"""
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
        
        # Calculate NV position relative to 3M low
        if row['NV_3M_Low'] != 0:
            nv_pct = ((row['Net_Volume'] - row['NV_3M_Low']) / abs(row['NV_3M_Low'])) * 100
        else:
            continue
        
        # Strict criteria
        if (nv_pct <= params['nv_3m_low_pct'] and
            row['NV_Improvement_Pct'] >= params['nv_improvement_pct'] and
            row['Drawdown_Pct'] <= params['drawdown_min'] and
            params['ema50_min'] <= row['Pct_From_EMA50'] <= params['ema50_max'] and
            row['Volume_Ratio'] < params['volume_ratio_max']):
            signals.iloc[i] = True
    
    return signals

# ===========================
# DIAGNOSTIC ANALYSIS
# ===========================
def analyze_symbol(symbol, df_symbol):
    """Run diagnostic analysis on a single symbol"""
    
    # Skip if insufficient data
    if len(df_symbol) < 100:
        return None
    
    # Calculate indicators
    df_with_indicators = calculate_net_volume(df_symbol)
    
    # Detect signals
    standard_signals = detect_standard_signals(df_with_indicators)
    strict_signals = detect_strict_signals(df_with_indicators)
    
    num_standard = standard_signals.sum()
    num_strict = strict_signals.sum()
    
    if num_standard == 0:
        return {
            'Symbol': symbol,
            'Status': 'NO_SIGNALS',
            'Standard_Signals': 0,
            'Strict_Signals': 0,
            'Latest_Close': df_with_indicators['Close'].iloc[-1],
            'Latest_RSI': df_with_indicators['RSI'].iloc[-1],
            'Issue': 'No signals found even in standard mode'
        }
    
    # Analyze why strict mode rejected signals
    standard_dates = df_with_indicators[standard_signals].index
    
    # Get metrics for all standard signals
    metrics = {
        'nv_3m_low_pct': [],
        'nv_improvement': [],
        'drawdown': [],
        'ema50_pct': [],
        'volume_ratio': []
    }
    
    for date in standard_dates:
        row = df_with_indicators.loc[date]
        
        if row['NV_3M_Low'] != 0:
            nv_pct = ((row['Net_Volume'] - row['NV_3M_Low']) / abs(row['NV_3M_Low'])) * 100
            metrics['nv_3m_low_pct'].append(nv_pct)
        
        metrics['nv_improvement'].append(row['NV_Improvement_Pct'])
        metrics['drawdown'].append(row['Drawdown_Pct'])
        metrics['ema50_pct'].append(row['Pct_From_EMA50'])
        metrics['volume_ratio'].append(row['Volume_Ratio'])
    
    # Calculate recommended parameters
    recommended_params = {}
    if metrics['nv_3m_low_pct']:
        recommended_params['nv_3m_low_pct'] = max(metrics['nv_3m_low_pct']) + 5
    if metrics['nv_improvement']:
        recommended_params['nv_improvement_pct'] = min(metrics['nv_improvement']) - 5
    if metrics['drawdown']:
        recommended_params['drawdown_min'] = max(metrics['drawdown']) + 2
    if metrics['ema50_pct']:
        recommended_params['ema50_min'] = min(metrics['ema50_pct']) - 5
        recommended_params['ema50_max'] = max(metrics['ema50_pct']) + 5
    if metrics['volume_ratio']:
        recommended_params['volume_ratio_max'] = max(metrics['volume_ratio']) + 0.1
    
    return {
        'Symbol': symbol,
        'Status': 'SIGNALS_FOUND',
        'Standard_Signals': num_standard,
        'Strict_Signals': num_strict,
        'Latest_Close': round(df_with_indicators['Close'].iloc[-1], 2),
        'Latest_RSI': round(df_with_indicators['RSI'].iloc[-1], 1),
        'NV_Range': f"{min(metrics['nv_3m_low_pct']):.1f} to {max(metrics['nv_3m_low_pct']):.1f}" if metrics['nv_3m_low_pct'] else 'N/A',
        'Recommended_NV_Threshold': round(recommended_params.get('nv_3m_low_pct', 10), 0),
        'Recommended_NV_Improvement': round(recommended_params.get('nv_improvement_pct', 30), 0),
        'Recommended_Drawdown': round(recommended_params.get('drawdown_min', -15), 0),
        'Recommended_EMA50_Min': round(recommended_params.get('ema50_min', -25), 0),
        'Recommended_EMA50_Max': round(recommended_params.get('ema50_max', -15), 0),
        'Recommended_Volume_Ratio': round(recommended_params.get('volume_ratio_max', 0.5), 2)
    }

# ===========================
# MAIN EXECUTION
# ===========================
def main():
    print("\n" + "="*80)
    print("SSHL MULTI-SYMBOL DIAGNOSTIC - FINDING SIGNAL ISSUES")
    print("="*80)
    
    # Load data
    df, latest_date = load_market_data()
    
    # Get unique symbols
    symbols = df['Symbol'].unique()
    symbols = [s for s in symbols if s not in EXCLUDED_SYMBOLS]
    
    print(f"\n📊 Analyzing {len(symbols)} symbols (excluding {len(EXCLUDED_SYMBOLS)} filtered symbols)")
    
    # Analyze each symbol
    results = []
    for i, symbol in enumerate(symbols, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(symbols)} symbols...")
        
        symbol_data = df[df['Symbol'] == symbol].copy().sort_values('Date')
        result = analyze_symbol(symbol, symbol_data)
        
        if result:
            results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    total_analyzed = len(results_df)
    with_signals = len(results_df[results_df['Status'] == 'SIGNALS_FOUND'])
    without_signals = len(results_df[results_df['Status'] == 'NO_SIGNALS'])
    
    print(f"Total symbols analyzed: {total_analyzed}")
    print(f"Symbols with signals: {with_signals}")
    print(f"Symbols without signals: {without_signals}")
    
    if with_signals > 0:
        avg_standard = results_df[results_df['Status'] == 'SIGNALS_FOUND']['Standard_Signals'].mean()
        avg_strict = results_df[results_df['Status'] == 'SIGNALS_FOUND']['Strict_Signals'].mean()
        print(f"Average standard signals: {avg_standard:.1f}")
        print(f"Average strict signals: {avg_strict:.1f}")
    
    # Sort by number of standard signals
    results_df = results_df.sort_values('Standard_Signals', ascending=False).reset_index(drop=True)
    
    # Upload results
    output_file = f"SSHL_DIAGNOSTIC_RESULTS_{latest_date}.csv"
    upload_to_github(output_file, results_df.to_csv(index=False))
    delete_old_files("SSHL_DIAGNOSTIC_RESULTS_", output_file)
    
    # Show top symbols with most signals
    print(f"\n🏆 TOP 10 SYMBOLS WITH MOST SIGNALS:")
    top_10 = results_df[results_df['Status'] == 'SIGNALS_FOUND'].head(10)
    print(top_10[['Symbol', 'Standard_Signals', 'Strict_Signals', 'Latest_RSI']].to_string(index=False))
    
    print(f"\n✅ Results uploaded to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
