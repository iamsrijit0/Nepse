# -*- coding: utf-8 -*-
"""
MULTI-SYMBOL DIAGNOSTIC SCRIPT
================================
Analyzes Net Volume signals for ALL symbols in espen CSV files
and uploads results to GitHub

Features:
- Processes all symbols from latest espen_*.csv file
- Excludes problematic symbols (bonds, debentures)
- Runs both STANDARD and STRICT mode diagnostics
- Calculates P&L for each signal (entry to latest close)
- Uploads comprehensive results CSV to GitHub

Author: Multi-Symbol Net Volume Analyzer
Date: 2026-02-13
"""

import os
import re
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
import warnings
import base64
import json
import traceback

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
REPO_OWNER = "iamsrijit0"
REPO_NAME = "Nepse"
BRANCH = "main"

# Excluded symbols (bonds, debentures, problematic tickers)
EXCLUDED_SYMBOLS = [
    "EBLD852", "EBL", "EB89", "NABILD2089", "MBLD2085", "SBID89",
    "SBID2090", "SBLD2091", "NIMBD90", "RBBD2088", "CCBD88", "ULBSL",
    "ICFCD88", "EBLD91", "ANLB", "GBILD84/85", "GBILD86/87", "NICD88"
]

# Output file (upload to root directory to avoid path issues)
DIAGNOSTIC_RESULTS_CSV = "diagnostic_results.csv"

# GitHub credentials
GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("❌ GH_TOKEN not set in environment")

HEADERS = {
    "Authorization": f"token {GH_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ===========================
# GITHUB UTILITIES
# ===========================
def github_raw(path):
    """Get raw GitHub URL for a file"""
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

def get_latest_espen_csv():
    """Find and return the latest espen_*.csv file from GitHub"""
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
        raise FileNotFoundError("No espen_*.csv file found in repository")
    
    latest_date = max(espen_files.keys())
    latest_file = espen_files[latest_date]
    print(f"📂 Latest data file: {latest_file} ({latest_date})")
    return github_raw(latest_file), latest_date

def upload_csv_to_github(df, repo_path, commit_message):
    """Upload a CSV file to GitHub"""
    try:
        print(f"\n📤 Preparing to upload to: {repo_path}")
        
        csv_content = df.to_csv(index=False)
        content_base64 = base64.b64encode(csv_content.encode()).decode()
        
        print(f"   CSV size: {len(csv_content)} bytes, {len(df)} rows")
        
        # Check if file exists to get SHA
        check_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
        print(f"   Checking if file exists: {check_url}")
        
        check_response = requests.get(check_url, headers=HEADERS, params={"ref": BRANCH})
        
        sha = None
        if check_response.status_code == 200:
            sha = check_response.json().get('sha')
            print(f"   File exists, SHA: {sha[:8]}...")
        elif check_response.status_code == 404:
            print(f"   File doesn't exist, will create new file")
        else:
            print(f"   Unexpected response: {check_response.status_code}")
            print(f"   Response: {check_response.text[:500]}")
        
        # Prepare upload data
        upload_data = {
            "message": commit_message,
            "content": content_base64,
            "branch": BRANCH
        }
        if sha:
            upload_data["sha"] = sha
        
        # Upload
        upload_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
        print(f"   Uploading to: {upload_url}")
        
        response = requests.put(upload_url, headers=HEADERS, data=json.dumps(upload_data))
        
        if response.status_code in [200, 201]:
            print(f"✅ Successfully uploaded {repo_path}")
            result = response.json()
            if 'content' in result and 'html_url' in result['content']:
                print(f"   URL: {result['content']['html_url']}")
            return True
        else:
            print(f"❌ Upload failed for {repo_path}")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text[:1000]}")
            
            # Try to parse error message
            try:
                error_data = response.json()
                if 'message' in error_data:
                    print(f"   Error message: {error_data['message']}")
                if 'errors' in error_data:
                    print(f"   Errors: {error_data['errors']}")
            except:
                pass
            
            return False
            
    except Exception as e:
        print(f"❌ Exception during upload of {repo_path}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

# ===========================
# TECHNICAL INDICATORS
# ===========================
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

# ===========================
# NET VOLUME CALCULATIONS
# ===========================
def calculate_net_volume_features(df):
    """Calculate all Net Volume related features"""
    data = df.copy()
    
    # Basic Net Volume
    data['Net_Volume'] = np.where(
        data['Close'] >= data['Open'],
        data['Volume'],
        -data['Volume']
    )
    
    # Cumulative Net Volume
    data['Cumulative_NV'] = data['Net_Volume'].cumsum()
    
    # Net Volume Moving Averages
    data['NV_SMA_20'] = data['Net_Volume'].rolling(window=20).mean()
    data['NV_SMA_50'] = data['Net_Volume'].rolling(window=50).mean()
    data['NV_EMA_20'] = data['Net_Volume'].ewm(span=20, adjust=False).mean()
    
    # 3-month (60 days) Net Volume metrics
    data['NV_3M_High'] = data['Net_Volume'].rolling(window=60).max()
    data['NV_3M_Low'] = data['Net_Volume'].rolling(window=60).min()
    data['NV_3M_Avg'] = data['Net_Volume'].rolling(window=60).mean()
    
    # Net Volume position relative to 3M range
    data['NV_Position_3M'] = np.where(
        (data['NV_3M_High'] - data['NV_3M_Low']) != 0,
        (data['Net_Volume'] - data['NV_3M_Low']) / (data['NV_3M_High'] - data['NV_3M_Low']),
        0.5
    )
    
    # Net Volume Improvement (7-day vs 30-day avg)
    data['NV_7D_Avg'] = data['Net_Volume'].rolling(window=7).mean()
    data['NV_30D_Avg'] = data['Net_Volume'].rolling(window=30).mean()
    data['NV_Improvement_Pct'] = np.where(
        data['NV_30D_Avg'] != 0,
        ((data['NV_7D_Avg'] - data['NV_30D_Avg']) / abs(data['NV_30D_Avg'])) * 100,
        0
    )
    
    # Price metrics
    data['EMA_50'] = calculate_ema(data['Close'], 50)
    data['EMA_200'] = calculate_ema(data['Close'], 200)
    data['Pct_From_EMA50'] = ((data['Close'] - data['EMA_50']) / data['EMA_50']) * 100
    
    # Drawdown from 52-week high
    data['52W_High'] = data['Close'].rolling(window=252, min_periods=50).max()
    data['Drawdown_Pct'] = ((data['Close'] - data['52W_High']) / data['52W_High']) * 100
    
    # Volume metrics
    data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
    
    # RSI
    data['RSI'] = calculate_rsi(data['Close'], 14)
    
    return data

# ===========================
# SIGNAL GENERATION
# ===========================
def generate_standard_signals(data):
    """Generate signals using STANDARD mode"""
    signals = pd.DataFrame(index=data.index)
    signals['Buy_Signal'] = False
    
    # Standard conditions
    condition_nv_low = data['Net_Volume'] <= data['NV_3M_Low'] * 1.2
    condition_nv_improvement = data['NV_Improvement_Pct'] >= 20
    condition_drawdown = data['Drawdown_Pct'] <= -10
    condition_below_ema = data['Pct_From_EMA50'] <= -10
    condition_volume = data['Volume_Ratio'] < 0.8
    
    signals['Buy_Signal'] = (
        condition_nv_low &
        condition_nv_improvement &
        condition_drawdown &
        condition_below_ema &
        condition_volume
    )
    
    return signals

def generate_strict_signals(data):
    """Generate signals using STRICT mode"""
    signals = pd.DataFrame(index=data.index)
    signals['Buy_Signal'] = False
    
    # Strict conditions (more conservative)
    condition_nv_low = data['Net_Volume'] <= data['NV_3M_Low'] * 1.1
    condition_nv_improvement = data['NV_Improvement_Pct'] >= 30
    condition_drawdown = data['Drawdown_Pct'] <= -15
    condition_below_ema = (data['Pct_From_EMA50'] >= -25) & (data['Pct_From_EMA50'] <= -15)
    condition_volume = data['Volume_Ratio'] < 0.5
    
    signals['Buy_Signal'] = (
        condition_nv_low &
        condition_nv_improvement &
        condition_drawdown &
        condition_below_ema &
        condition_volume
    )
    
    return signals

# ===========================
# MAIN DIAGNOSTIC FUNCTION
# ===========================
def analyze_symbol(df_symbol, symbol, latest_date):
    """Analyze a single symbol and return diagnostic results"""
    
    # Skip if insufficient data
    if len(df_symbol) < 60:
        return None
    
    # Calculate features
    try:
        data = calculate_net_volume_features(df_symbol)
    except Exception as e:
        print(f"  ⚠️ Error calculating features for {symbol}: {str(e)}")
        return None
    
    # Generate signals
    signals_std = generate_standard_signals(data)
    signals_strict = generate_strict_signals(data)
    
    num_std = signals_std['Buy_Signal'].sum()
    num_strict = signals_strict['Buy_Signal'].sum()
    
    results = []
    
    # Process STANDARD mode signals
    std_dates = data[signals_std['Buy_Signal']].index
    for signal_date in std_dates:
        entry_price = data.loc[signal_date, 'Close']
        latest_price = df_symbol[df_symbol['Date'] == latest_date]['Close'].iloc[0] if len(df_symbol[df_symbol['Date'] == latest_date]) > 0 else entry_price
        
        profit_pct = ((latest_price - entry_price) / entry_price) * 100
        
        results.append({
            'Symbol': symbol,
            'Mode': 'STANDARD',
            'Signal_Date': signal_date.strftime('%Y-%m-%d') if hasattr(signal_date, 'strftime') else str(signal_date),
            'Entry_Price': round(entry_price, 2),
            'Latest_Date': latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date),
            'Latest_Price': round(latest_price, 2),
            'Profit_Loss_Pct': round(profit_pct, 2),
            'Net_Volume': round(data.loc[signal_date, 'Net_Volume'], 0),
            'NV_Improvement_Pct': round(data.loc[signal_date, 'NV_Improvement_Pct'], 1),
            'Drawdown_Pct': round(data.loc[signal_date, 'Drawdown_Pct'], 1),
            'Pct_From_EMA50': round(data.loc[signal_date, 'Pct_From_EMA50'], 1),
            'Volume_Ratio': round(data.loc[signal_date, 'Volume_Ratio'], 2),
            'RSI': round(data.loc[signal_date, 'RSI'], 1)
        })
    
    # Process STRICT mode signals
    strict_dates = data[signals_strict['Buy_Signal']].index
    for signal_date in strict_dates:
        entry_price = data.loc[signal_date, 'Close']
        latest_price = df_symbol[df_symbol['Date'] == latest_date]['Close'].iloc[0] if len(df_symbol[df_symbol['Date'] == latest_date]) > 0 else entry_price
        
        profit_pct = ((latest_price - entry_price) / entry_price) * 100
        
        results.append({
            'Symbol': symbol,
            'Mode': 'STRICT',
            'Signal_Date': signal_date.strftime('%Y-%m-%d') if hasattr(signal_date, 'strftime') else str(signal_date),
            'Entry_Price': round(entry_price, 2),
            'Latest_Date': latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date),
            'Latest_Price': round(latest_price, 2),
            'Profit_Loss_Pct': round(profit_pct, 2),
            'Net_Volume': round(data.loc[signal_date, 'Net_Volume'], 0),
            'NV_Improvement_Pct': round(data.loc[signal_date, 'NV_Improvement_Pct'], 1),
            'Drawdown_Pct': round(data.loc[signal_date, 'Drawdown_Pct'], 1),
            'Pct_From_EMA50': round(data.loc[signal_date, 'Pct_From_EMA50'], 1),
            'Volume_Ratio': round(data.loc[signal_date, 'Volume_Ratio'], 2),
            'RSI': round(data.loc[signal_date, 'RSI'], 1)
        })
    
    if len(results) > 0:
        print(f"  ✅ {symbol}: {num_std} STANDARD signals, {num_strict} STRICT signals")
    
    return results

# ===========================
# MAIN EXECUTION
# ===========================
def main():
    print("\n" + "="*80)
    print("🔍 MULTI-SYMBOL NET VOLUME DIAGNOSTIC")
    print("="*80)
    print(f"🕒 Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate GitHub token
    print("\n🔑 Validating GitHub credentials...")
    if not GH_TOKEN:
        print("❌ ERROR: GH_TOKEN environment variable not set!")
        print("   Set it with: export GH_TOKEN='your_token_here'")
        return None
    
    # Test GitHub API access
    test_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
    try:
        test_response = requests.get(test_url, headers=HEADERS)
        if test_response.status_code == 200:
            print(f"✅ GitHub API access verified")
            repo_info = test_response.json()
            print(f"   Repository: {repo_info['full_name']}")
            print(f"   Default branch: {repo_info['default_branch']}")
        else:
            print(f"⚠️ GitHub API returned status {test_response.status_code}")
            print(f"   Response: {test_response.text[:200]}")
            if test_response.status_code == 401:
                print("   ERROR: Invalid or expired GitHub token!")
                return None
    except Exception as e:
        print(f"❌ GitHub API test failed: {str(e)}")
        return None
    
    # Get latest espen CSV from GitHub
    print("\n📥 Fetching latest NEPSE data from GitHub...")
    csv_url, data_date = get_latest_espen_csv()
    
    response = requests.get(csv_url)
    csv_content = response.text
    
    # Parse CSV (handle both tab and comma delimiters)
    lines = csv_content.strip().split('\n')
    header_line = None
    data_start_index = 0
    
    for i, line in enumerate(lines):
        if 'Symbol' in line and 'Date' in line and 'Close' in line:
            header_line = line
            data_start_index = i
            break
    
    if data_start_index > 0:
        reconstructed_csv = header_line + '\n' + '\n'.join(lines[:data_start_index])
    else:
        reconstructed_csv = csv_content
    
    try:
        df = pd.read_csv(StringIO(reconstructed_csv), sep='\t')
        if len(df.columns) == 1:
            df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
    except:
        df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
    
    df.columns = df.columns.str.strip()
    
    # Parse dates
    original_dates = df["Date"].copy()
    df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y', errors='coerce')
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(original_dates, format='%Y-%m-%d', errors='coerce')
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(original_dates, errors='coerce')
    
    # Convert numeric columns
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=["Symbol", "Date", "Close"])
    df = df.sort_values(["Symbol", "Date"])
    
    latest_date = df['Date'].max()
    
    print(f"✅ Data loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")
    print(f"📅 Date range: {df['Date'].min().date()} to {latest_date.date()}")
    
    # Get all symbols (excluding problematic ones)
    all_symbols = [s for s in df['Symbol'].unique() if s not in EXCLUDED_SYMBOLS]
    print(f"📊 Processing {len(all_symbols)} symbols (excluded {len(EXCLUDED_SYMBOLS)} problematic symbols)")
    
    # Analyze each symbol
    print("\n🔄 Analyzing symbols...")
    print("-"*80)
    
    all_results = []
    processed_count = 0
    error_count = 0
    
    for i, symbol in enumerate(all_symbols, 1):
        if i % 20 == 0:
            print(f"   Progress: {i}/{len(all_symbols)} symbols processed...")
        
        try:
            df_symbol = df[df['Symbol'] == symbol].copy()
            df_symbol = df_symbol.sort_values('Date').reset_index(drop=True)
            
            results = analyze_symbol(df_symbol, symbol, latest_date)
            
            if results:
                all_results.extend(results)
                processed_count += 1
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Only show first 5 errors
                print(f"  ⚠️ Error processing {symbol}: {str(e)}")
    
    print(f"\n✅ Processing complete: {processed_count} symbols analyzed, {error_count} errors")
    
    # Create results DataFrame
    if len(all_results) == 0:
        print("\n❌ No signals found across all symbols!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Sort by profit/loss percentage (descending)
    results_df = results_df.sort_values('Profit_Loss_Pct', ascending=False)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("📊 SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    total_signals = len(results_df)
    std_signals = len(results_df[results_df['Mode'] == 'STANDARD'])
    strict_signals = len(results_df[results_df['Mode'] == 'STRICT'])
    
    print(f"\nTotal Signals Found: {total_signals}")
    print(f"  STANDARD mode: {std_signals}")
    print(f"  STRICT mode: {strict_signals}")
    
    profitable = len(results_df[results_df['Profit_Loss_Pct'] > 0])
    unprofitable = len(results_df[results_df['Profit_Loss_Pct'] <= 0])
    
    print(f"\nP&L Distribution:")
    print(f"  Profitable: {profitable} ({profitable/total_signals*100:.1f}%)")
    print(f"  Unprofitable: {unprofitable} ({unprofitable/total_signals*100:.1f}%)")
    
    print(f"\nAverage P&L: {results_df['Profit_Loss_Pct'].mean():+.2f}%")
    print(f"Median P&L: {results_df['Profit_Loss_Pct'].median():+.2f}%")
    print(f"Best Signal: {results_df['Profit_Loss_Pct'].max():+.2f}%")
    print(f"Worst Signal: {results_df['Profit_Loss_Pct'].min():+.2f}%")
    
    # Top performers
    print(f"\n{'='*80}")
    print("🏆 TOP 10 SIGNALS BY P&L")
    print(f"{'='*80}")
    
    top_10 = results_df.head(10)
    print(f"\n{'#':<3} {'Symbol':<10} {'Mode':<10} {'Signal Date':<12} {'Entry':<8} {'Latest':<8} {'P&L %':<8}")
    print("-"*80)
    
    for i, row in enumerate(top_10.itertuples(), 1):
        print(f"{i:<3} {row.Symbol:<10} {row.Mode:<10} {row.Signal_Date:<12} "
              f"{row.Entry_Price:<8.2f} {row.Latest_Price:<8.2f} {row.Profit_Loss_Pct:+8.2f}%")
    
    # Bottom performers
    print(f"\n{'='*80}")
    print("⚠️ BOTTOM 10 SIGNALS BY P&L")
    print(f"{'='*80}")
    
    bottom_10 = results_df.tail(10)
    print(f"\n{'#':<3} {'Symbol':<10} {'Mode':<10} {'Signal Date':<12} {'Entry':<8} {'Latest':<8} {'P&L %':<8}")
    print("-"*80)
    
    for i, row in enumerate(bottom_10.itertuples(), 1):
        print(f"{i:<3} {row.Symbol:<10} {row.Mode:<10} {row.Signal_Date:<12} "
              f"{row.Entry_Price:<8.2f} {row.Latest_Price:<8.2f} {row.Profit_Loss_Pct:+8.2f}%")
    
    # Upload to GitHub
    print(f"\n{'='*80}")
    print("📤 UPLOADING RESULTS TO GITHUB")
    print(f"{'='*80}")
    
    # Always save a local backup first
    local_filename = f'diagnostic_results_{latest_date.strftime("%Y%m%d")}.csv'
    results_df.to_csv(local_filename, index=False)
    print(f"💾 Local backup saved: {local_filename}")
    
    commit_message = f"🔍 Multi-Symbol Diagnostic Results - {latest_date.strftime('%Y-%m-%d')} - {total_signals} signals"
    
    print(f"\n📤 Attempting GitHub upload...")
    success = upload_csv_to_github(results_df, DIAGNOSTIC_RESULTS_CSV, commit_message)
    
    if success:
        print(f"\n✅ Results uploaded successfully to GitHub!")
        print(f"🌐 View at: https://github.com/{REPO_OWNER}/{REPO_NAME}/blob/{BRANCH}/{DIAGNOSTIC_RESULTS_CSV}")
        print(f"\n💡 You can also download the local backup: {local_filename}")
    else:
        print(f"\n❌ GitHub upload failed!")
        print(f"💾 Results saved locally to: {local_filename}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check your GH_TOKEN is valid and has write permissions")
        print(f"   2. Verify repository name: {REPO_OWNER}/{REPO_NAME}")
        print(f"   3. Check branch exists: {BRANCH}")
        print(f"   4. You can manually upload the file: {local_filename}")
    
    # Mode comparison
    print(f"\n{'='*80}")
    print("📊 MODE COMPARISON")
    print(f"{'='*80}")
    
    for mode in ['STANDARD', 'STRICT']:
        mode_df = results_df[results_df['Mode'] == mode]
        if len(mode_df) > 0:
            print(f"\n{mode} MODE:")
            print(f"  Signals: {len(mode_df)}")
            print(f"  Win Rate: {len(mode_df[mode_df['Profit_Loss_Pct'] > 0])/len(mode_df)*100:.1f}%")
            print(f"  Avg P&L: {mode_df['Profit_Loss_Pct'].mean():+.2f}%")
            print(f"  Median P&L: {mode_df['Profit_Loss_Pct'].median():+.2f}%")
    
    print(f"\n{'='*80}")
    print("✅ DIAGNOSTIC COMPLETE!")
    print(f"{'='*80}")
    
    return results_df

if __name__ == "__main__":
    results = main()
