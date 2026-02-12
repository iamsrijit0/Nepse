# -*- coding: utf-8 -*-
"""
DAILY ML SIGNAL GENERATOR FOR GITHUB ACTIONS
- Loads trained models from CSV
- Generates buy signals for today
- Records outcomes and updates model performance
- Uses 2 CSV files: model_data.csv and signals_history.csv
- No local downloads - everything runs in GitHub
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
warnings.filterwarnings('ignore')

# ===========================
# CONFIG
# ===========================
REPO_OWNER = "iamsrijit0"
REPO_NAME = "Nepse"
BRANCH = "main"

EXCLUDED_SYMBOLS = [
    "EBLD852", "EBL", "EB89", "NABILD2089", "MBLD2085", "SBID89",
    "SBID2090", "SBLD2091", "NIMBD90", "RBBD2088", "CCBD88", "ULBSL",
    "ICFCD88", "EBLD91", "ANLB", "GBILD84/85", "GBILD86/87", "NICD88"
]

# ML SETTINGS
PROFIT_THRESHOLD = 0.02  # 2% gain = success
MIN_CONFIDENCE = 0.65  # Only signals with 65%+ confidence
MIN_MODEL_ACCURACY = 0.55  # Minimum historical accuracy
MIN_TRADES = 3  # Minimum trades to trust a model
LOOKBACK_DAYS = 365  # Use 1 year of training data
MIN_TRAINING_SAMPLES = 50

# CSV FILES IN GITHUB
MODEL_DATA_CSV = "ml_data/model_data.csv"  # Stores model performance & parameters
SIGNALS_HISTORY_CSV = "ml_data/signals_history.csv"  # Stores all signal history
DAILY_SIGNALS_CSV = "ml_data/daily_signals.csv"  # Today's signals

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set in environment")

HEADERS = {
    "Authorization": f"token {GH_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ===========================
# GITHUB FUNCTIONS
# ===========================
def github_raw(path):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

def get_latest_espen_csv():
    """Find and return the latest espen_YYYYMMDD.csv file"""
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
    
    print(f"📂 Latest data file: {latest_file} ({latest_date})")
    return github_raw(latest_file), latest_date

def load_csv_from_github(repo_path, create_if_missing=False, columns=None):
    """Load CSV from GitHub, optionally create if missing"""
    try:
        csv_url = github_raw(repo_path)
        response = requests.get(csv_url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            print(f"✅ Loaded {repo_path}: {len(df)} rows")
            return df
        elif create_if_missing and columns:
            print(f"🆕 Creating new {repo_path}")
            return pd.DataFrame(columns=columns)
        else:
            print(f"⚠️  Could not load {repo_path}")
            return None
    except Exception as e:
        if create_if_missing and columns:
            print(f"🆕 Creating new {repo_path}")
            return pd.DataFrame(columns=columns)
        print(f"❌ Error loading {repo_path}: {str(e)}")
        return None

def upload_csv_to_github(df, repo_path, commit_message):
    """Upload CSV DataFrame to GitHub"""
    try:
        # Convert DataFrame to CSV
        csv_content = df.to_csv(index=False)
        content_base64 = base64.b64encode(csv_content.encode()).decode()
        
        # Check if file exists
        check_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
        check_response = requests.get(check_url, headers=HEADERS, params={"ref": BRANCH})
        
        sha = None
        if check_response.status_code == 200:
            sha = check_response.json().get('sha')
        
        upload_data = {
            "message": commit_message,
            "content": content_base64,
            "branch": BRANCH
        }
        
        if sha:
            upload_data["sha"] = sha
        
        upload_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
        response = requests.put(upload_url, headers=HEADERS, data=json.dumps(upload_data))
        
        if response.status_code in [200, 201]:
            print(f"✅ Uploaded {repo_path}")
            return True
        else:
            print(f"⚠️  Upload failed for {repo_path}: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading {repo_path}: {str(e)}")
        return False

# ===========================
# TECHNICAL INDICATORS
# ===========================
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_bollinger_bands(prices, period=20):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    return sma, std

# ===========================
# FEATURE ENGINEERING
# ===========================
def create_features(df_symbol):
    """Create comprehensive feature set"""
    df = df_symbol.copy()
    features = pd.DataFrame(index=df.index)
    
    # RSI
    for period in [7, 14, 21]:
        features[f'RSI_{period}'] = calculate_rsi(df['Close'], period)
    
    # EMA
    for period in [5, 9, 21, 50]:
        features[f'EMA_{period}'] = calculate_ema(df['Close'], period)
        features[f'Close_to_EMA_{period}'] = (df['Close'] - features[f'EMA_{period}']) / features[f'EMA_{period}']
    
    # EMA crossovers
    features['EMA_5_9_cross'] = (features['EMA_5'] - features['EMA_9']) / features['EMA_9']
    features['EMA_9_21_cross'] = (features['EMA_9'] - features['EMA_21']) / features['EMA_21']
    features['EMA_21_50_cross'] = (features['EMA_21'] - features['EMA_50']) / features['EMA_50']
    
    # Bollinger Bands
    bb_sma, bb_std = calculate_bollinger_bands(df['Close'], 20)
    features['BB_position'] = (df['Close'] - bb_sma) / (2 * bb_std)
    features['BB_width'] = (4 * bb_std) / bb_sma
    
    # Volume
    if 'Volume' in df.columns:
        for period in [5, 10, 20]:
            vol_sma = df['Volume'].rolling(window=period).mean()
            features[f'Volume_ratio_{period}'] = df['Volume'] / vol_sma
    
    # Momentum
    for period in [5, 10, 20]:
        features[f'Momentum_{period}'] = df['Close'].pct_change(period) * 100
    
    # Volatility
    features['Volatility_10'] = df['Close'].pct_change().rolling(window=10).std() * 100
    features['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * 100
    
    # 52-week position
    features['52W_High'] = df['Close'].rolling(window=252, min_periods=50).max()
    features['52W_Low'] = df['Close'].rolling(window=252, min_periods=50).min()
    features['52W_position'] = (df['Close'] - features['52W_Low']) / (features['52W_High'] - features['52W_Low'])
    
    # Daily patterns
    features['Daily_return'] = df['Close'].pct_change() * 100
    if 'High' in df.columns and 'Low' in df.columns:
        features['High_Low_range'] = ((df['High'] - df['Low']) / df['Close']) * 100
    
    return features

# ===========================
# SIMPLE RULE-BASED MODEL (CSV-friendly)
# ===========================
def generate_signal_simple(df_symbol, model_data=None):
    """
    Generate buy signal based on technical indicators
    This is a simplified model that doesn't require sklearn
    Returns: (signal, confidence, features_dict)
    """
    if len(df_symbol) < 50:
        return 0, 0.0, {}
    
    # Get latest data
    latest = df_symbol.iloc[-1]
    
    # Create features
    features = create_features(df_symbol)
    latest_features = features.iloc[-1]
    
    # Check for NaN
    if latest_features.isna().any():
        return 0, 0.0, {}
    
    # SIGNAL RULES
    score = 0.0
    max_score = 0.0
    
    # Rule 1: RSI oversold (weight: 0.15)
    if latest_features['RSI_14'] < 30:
        score += 0.15
    elif latest_features['RSI_14'] < 40:
        score += 0.10
    max_score += 0.15
    
    # Rule 2: Price below EMA (weight: 0.20)
    if latest_features['Close_to_EMA_21'] < -0.02:  # 2% below
        score += 0.20
    elif latest_features['Close_to_EMA_21'] < 0:
        score += 0.10
    max_score += 0.20
    
    # Rule 3: EMA crossover potential (weight: 0.15)
    if latest_features['EMA_5_9_cross'] > 0 and latest_features['EMA_9_21_cross'] > -0.01:
        score += 0.15
    elif latest_features['EMA_5_9_cross'] > -0.01:
        score += 0.08
    max_score += 0.15
    
    # Rule 4: Bollinger Band position (weight: 0.15)
    if latest_features['BB_position'] < -0.8:  # Near lower band
        score += 0.15
    elif latest_features['BB_position'] < -0.5:
        score += 0.10
    max_score += 0.15
    
    # Rule 5: Momentum turning positive (weight: 0.10)
    if latest_features['Momentum_5'] > 0 and latest_features['Momentum_10'] < 0:
        score += 0.10
    elif latest_features['Momentum_5'] > -2:
        score += 0.05
    max_score += 0.10
    
    # Rule 6: Volume surge (weight: 0.10)
    if 'Volume_ratio_5' in latest_features:
        if latest_features['Volume_ratio_5'] > 1.5:
            score += 0.10
        elif latest_features['Volume_ratio_5'] > 1.2:
            score += 0.05
    max_score += 0.10
    
    # Rule 7: 52-week position (weight: 0.15)
    if latest_features['52W_position'] < 0.3:  # Near 52-week low
        score += 0.15
    elif latest_features['52W_position'] < 0.5:
        score += 0.08
    max_score += 0.15
    
    # Calculate confidence
    confidence = score / max_score if max_score > 0 else 0.0
    
    # Adjust confidence based on historical performance
    if model_data is not None:
        hist_accuracy = model_data.get('accuracy', 0.5)
        confidence = confidence * 0.7 + hist_accuracy * 0.3
    
    # Generate signal
    signal = 1 if confidence >= MIN_CONFIDENCE else 0
    
    # Feature dict for logging
    features_dict = {
        'RSI_14': latest_features['RSI_14'],
        'Close_to_EMA_21': latest_features['Close_to_EMA_21'],
        'EMA_5_9_cross': latest_features['EMA_5_9_cross'],
        'BB_position': latest_features['BB_position'],
        'Momentum_5': latest_features['Momentum_5'],
        '52W_position': latest_features['52W_position']
    }
    
    return signal, confidence, features_dict

# ===========================
# MAIN DAILY RUNNER
# ===========================
def main():
    print("="*80)
    print("🤖 DAILY ML SIGNAL GENERATOR (GitHub Actions)")
    print(f"🕒 Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load CSV data from GitHub
    print("\n📥 Loading data from GitHub...")
    
    # Load model performance data
    model_data_df = load_csv_from_github(
        MODEL_DATA_CSV,
        create_if_missing=True,
        columns=['Symbol', 'Last_Updated', 'Total_Signals', 'Successful_Signals', 
                 'Accuracy', 'Avg_Profit', 'Last_Signal_Date', 'Last_Signal_Price',
                 'Last_Signal_Confidence']
    )
    
    # Load signals history
    signals_history_df = load_csv_from_github(
        SIGNALS_HISTORY_CSV,
        create_if_missing=True,
        columns=['Date', 'Symbol', 'Price', 'Confidence', 'Prediction',
                 'Exit_Date', 'Exit_Price', 'Return_Pct', 'Success']
    )
    
    # Load market data
    csv_url, data_date = get_latest_espen_csv()
    response = requests.get(csv_url)
    csv_content = response.text
    
    # Parse CSV
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
    
    # Clean numeric columns
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=["Symbol", "Date", "Close"])
    df = df.sort_values(["Symbol", "Date"])
    
    latest_date = df['Date'].max()
    
    print(f"✅ Data loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")
    print(f"📅 Data date range: {df['Date'].min().date()} to {latest_date.date()}")
    
    # ===========================
    # STEP 1: Record yesterday's outcomes
    # ===========================
    print("\n📊 Checking yesterday's signals...")
    
    if signals_history_df is not None and len(signals_history_df) > 0:
        # Get pending signals (no exit yet)
        pending = signals_history_df[signals_history_df['Exit_Date'].isna()].copy()
        
        if len(pending) > 0:
            print(f"   Found {len(pending)} pending signals to check")
            
            updated_count = 0
            for idx, signal in pending.iterrows():
                symbol = signal['Symbol']
                entry_date = pd.to_datetime(signal['Date'])
                entry_price = signal['Price']
                
                # Get today's data
                df_symbol = df[df['Symbol'] == symbol]
                today_data = df_symbol[df_symbol['Date'] == latest_date]
                
                if len(today_data) > 0:
                    exit_price = today_data.iloc[0]['Close']
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    success = 1 if (return_pct / 100) >= PROFIT_THRESHOLD else 0
                    
                    # Update signals history
                    signals_history_df.at[idx, 'Exit_Date'] = latest_date.strftime('%Y-%m-%d')
                    signals_history_df.at[idx, 'Exit_Price'] = exit_price
                    signals_history_df.at[idx, 'Return_Pct'] = return_pct
                    signals_history_df.at[idx, 'Success'] = success
                    
                    # Update model data
                    if model_data_df is not None:
                        model_row = model_data_df[model_data_df['Symbol'] == symbol]
                        if len(model_row) > 0:
                            idx_model = model_row.index[0]
                            total = model_data_df.at[idx_model, 'Total_Signals']
                            successful = model_data_df.at[idx_model, 'Successful_Signals']
                            
                            model_data_df.at[idx_model, 'Total_Signals'] = total + 1
                            if success:
                                model_data_df.at[idx_model, 'Successful_Signals'] = successful + 1
                            
                            # Recalculate accuracy and avg profit
                            new_total = total + 1
                            new_successful = successful + (1 if success else 0)
                            model_data_df.at[idx_model, 'Accuracy'] = new_successful / new_total
                            
                            # Update average profit (running average)
                            old_avg = model_data_df.at[idx_model, 'Avg_Profit']
                            model_data_df.at[idx_model, 'Avg_Profit'] = (old_avg * total + return_pct) / new_total
                            model_data_df.at[idx_model, 'Last_Updated'] = latest_date.strftime('%Y-%m-%d')
                    
                    status = "✅" if success else "❌"
                    print(f"   {status} {symbol}: {return_pct:+.2f}%")
                    updated_count += 1
            
            if updated_count > 0:
                print(f"   📝 Updated {updated_count} signal outcomes")
    
    # ===========================
    # STEP 2: Generate today's signals
    # ===========================
    print(f"\n🎯 Generating signals for {latest_date.date()}...")
    
    symbols = df['Symbol'].unique()
    symbols = [s for s in symbols if s not in EXCLUDED_SYMBOLS]
    
    today_signals = []
    
    for symbol in symbols:
        df_symbol = df[df['Symbol'] == symbol].copy()
        df_symbol = df_symbol[df_symbol['Date'] <= latest_date]
        
        if len(df_symbol) < MIN_TRAINING_SAMPLES:
            continue
        
        # Get model data for this symbol
        model_info = None
        if model_data_df is not None and len(model_data_df) > 0:
            model_row = model_data_df[model_data_df['Symbol'] == symbol]
            if len(model_row) > 0:
                model_info = model_row.iloc[0].to_dict()
                
                # Skip if model is not performing well
                if model_info['Total_Signals'] >= MIN_TRADES:
                    if model_info['Accuracy'] < MIN_MODEL_ACCURACY:
                        continue
        
        # Generate signal
        signal, confidence, features = generate_signal_simple(df_symbol, model_info)
        
        if signal == 1:
            latest_price = df_symbol.iloc[-1]['Close']
            
            # Calculate model accuracy (if available)
            model_accuracy = model_info['Accuracy'] if model_info and 'Accuracy' in model_info else 0.5
            trades_count = int(model_info['Total_Signals']) if model_info and 'Total_Signals' in model_info else 0
            
            today_signals.append({
                'Date': latest_date.strftime('%Y-%m-%d'),
                'Symbol': symbol,
                'Price': latest_price,
                'Confidence': confidence,
                'Model_Accuracy': model_accuracy,
                'Historical_Trades': trades_count
            })
    
    # ===========================
    # STEP 3: Save results
    # ===========================
    if len(today_signals) > 0:
        print(f"\n{'='*80}")
        print(f"🚨 {len(today_signals)} BUY SIGNALS GENERATED!")
        print(f"{'='*80}")
        
        # Sort by confidence
        today_signals_sorted = sorted(today_signals, key=lambda x: x['Confidence'], reverse=True)
        
        for i, sig in enumerate(today_signals_sorted, 1):
            print(f"\n{i}. 📈 {sig['Symbol']}")
            print(f"   Price: NPR {sig['Price']:.2f}")
            print(f"   Confidence: {sig['Confidence']*100:.1f}%")
            print(f"   Model Accuracy: {sig['Model_Accuracy']*100:.1f}%")
            print(f"   Historical Trades: {sig['Historical_Trades']}")
        
        # Save to daily signals CSV
        daily_df = pd.DataFrame(today_signals_sorted)
        upload_csv_to_github(
            daily_df,
            DAILY_SIGNALS_CSV,
            f"🤖 ML Signals for {latest_date.strftime('%Y-%m-%d')} - {len(today_signals)} signals"
        )
        
        # Add to signals history (pending outcomes)
        new_history = pd.DataFrame([{
            'Date': sig['Date'],
            'Symbol': sig['Symbol'],
            'Price': sig['Price'],
            'Confidence': sig['Confidence'],
            'Prediction': 1,
            'Exit_Date': None,
            'Exit_Price': None,
            'Return_Pct': None,
            'Success': None
        } for sig in today_signals_sorted])
        
        if signals_history_df is not None:
            signals_history_df = pd.concat([signals_history_df, new_history], ignore_index=True)
        else:
            signals_history_df = new_history
        
        # Initialize model data for new symbols
        if model_data_df is not None:
            for sig in today_signals_sorted:
                if len(model_data_df[model_data_df['Symbol'] == sig['Symbol']]) == 0:
                    new_model = pd.DataFrame([{
                        'Symbol': sig['Symbol'],
                        'Last_Updated': latest_date.strftime('%Y-%m-%d'),
                        'Total_Signals': 0,
                        'Successful_Signals': 0,
                        'Accuracy': 0.5,
                        'Avg_Profit': 0.0,
                        'Last_Signal_Date': sig['Date'],
                        'Last_Signal_Price': sig['Price'],
                        'Last_Signal_Confidence': sig['Confidence']
                    }])
                    model_data_df = pd.concat([model_data_df, new_model], ignore_index=True)
                else:
                    # Update last signal info
                    idx = model_data_df[model_data_df['Symbol'] == sig['Symbol']].index[0]
                    model_data_df.at[idx, 'Last_Signal_Date'] = sig['Date']
                    model_data_df.at[idx, 'Last_Signal_Price'] = sig['Price']
                    model_data_df.at[idx, 'Last_Signal_Confidence'] = sig['Confidence']
    else:
        print("\n⏸️  No signals generated today")
        print("   Models need more confidence or better conditions")
    
    # Upload updated CSVs
    print("\n📤 Uploading updated data to GitHub...")
    
    if signals_history_df is not None:
        upload_csv_to_github(
            signals_history_df,
            SIGNALS_HISTORY_CSV,
            f"📊 Updated signals history - {latest_date.strftime('%Y-%m-%d')}"
        )
    
    if model_data_df is not None:
        upload_csv_to_github(
            model_data_df,
            MODEL_DATA_CSV,
            f"📈 Updated model performance data - {latest_date.strftime('%Y-%m-%d')}"
        )
    
    # Show top performers
    if model_data_df is not None and len(model_data_df) > 0:
        print(f"\n{'='*80}")
        print("🏆 TOP PERFORMING MODELS")
        print(f"{'='*80}")
        
        # Filter models with enough history
        performers = model_data_df[model_data_df['Total_Signals'] >= MIN_TRADES].copy()
        performers = performers.sort_values('Accuracy', ascending=False).head(10)
        
        if len(performers) > 0:
            for i, row in performers.iterrows():
                print(f"{row['Symbol']:12} | Accuracy: {row['Accuracy']*100:5.1f}% | "
                      f"Avg Profit: {row['Avg_Profit']:+6.2f}% | Trades: {int(row['Total_Signals'])}")
        else:
            print("Not enough trade history yet")
    
    print(f"\n{'='*80}")
    print("✅ Daily run complete!")
    print(f"{'='*80}")
    print(f"\n🌐 View results: https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/{BRANCH}/ml_data")
    print("\n💡 Next steps:")
    print("   1. Check today's signals in ml_data/daily_signals.csv")
    print("   2. Tomorrow's run will check outcomes & generate new signals")
    print("   3. Model accuracy improves with each trade")

if __name__ == "__main__":
    main()
