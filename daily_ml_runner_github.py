# -*- coding: utf-8 -*-
"""
DAILY ML SIGNAL GENERATOR FOR GITHUB ACTIONS (WITH BACKTESTING AND LEARNING)
- Uses statsmodels for logistic regression models per symbol
- Performs walk-forward backtesting to populate historical performance
- Trains on historical data for predictions
- Adjusts confidence with historical accuracy
- Uses CSV files: model_data.csv and signals_history.csv
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
import statsmodels.api as sm
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
PROFIT_THRESHOLD = 0.02 # 2% gain = success
MIN_CONFIDENCE = 0.65 # Only signals with 65%+ confidence
MIN_MODEL_ACCURACY = 0.55 # Minimum historical accuracy
MIN_TRADES = 3 # Minimum trades to trust a model
LOOKBACK_DAYS = 365 # Use 1 year of training data
MIN_TRAINING_SAMPLES = 50
# CSV FILES IN GITHUB
MODEL_DATA_CSV = "ml_data/model_data.csv" # Stores model performance & parameters
SIGNALS_HISTORY_CSV = "ml_data/signals_history.csv" # Stores all signal history
DAILY_SIGNALS_CSV = "ml_data/daily_signals.csv" # Today's signals
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
            print(f"⚠️ Could not load {repo_path}")
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
            print(f"⚠️ Upload failed for {repo_path}: {response.status_code}")
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
   
    return features.dropna(how='all', axis=1)  # Remove all-NaN columns if any
# ===========================
# DATA PREPARATION FOR ML
# ===========================
def prepare_data(df):
    if len(df) < 2:
        return pd.DataFrame()
    features = create_features(df)
    target = ((df['Close'].shift(-1) - df['Close']) / df['Close'] >= PROFIT_THRESHOLD).astype(int)
    data = pd.concat([features, target.rename('target')], axis=1)
    return data.dropna()
# ===========================
# TRAIN MODEL
# ===========================
def train_model(df_symbol):
    # Use recent data if LOOKBACK_DAYS specified
    if LOOKBACK_DAYS > 0 and len(df_symbol) > LOOKBACK_DAYS + 1:
        df_train = df_symbol.tail(LOOKBACK_DAYS + 1)
    else:
        df_train = df_symbol
    data = prepare_data(df_train)
    if len(data) < MIN_TRAINING_SAMPLES:
        return None, 0.5
    X = sm.add_constant(data.drop('target', axis=1))
    y = data['target']
    try:
        model = sm.Logit(y, X).fit(disp=0)
        pred = model.predict(X) > 0.5
        train_acc = (pred.astype(int) == y).mean()
        return model, train_acc
    except Exception as e:
        print(f"❌ Model fit error: {e}")
        return None, 0.5
# ===========================
# GENERATE SIGNAL WITH ML
# ===========================
def generate_signal_ml(df_symbol, model_info=None):
    if len(df_symbol) < 50:
        return 0, 0.0, {}
    model, _ = train_model(df_symbol)
    if model is None:
        return 0, 0.0, {}
    features = create_features(df_symbol)
    latest_features = features.iloc[-1]
    if pd.isna(latest_features).any():
        return 0, 0.0, {}
    # Ensure columns match
    pred_df = pd.DataFrame([latest_features])
    X_pred = sm.add_constant(pred_df)[model.exog_names]
    prob = model.predict(X_pred)[0]
    confidence = prob
    # Adjust with historical accuracy
    if model_info:
        hist_acc = model_info.get('Accuracy', 0.5)
        confidence = prob * 0.7 + hist_acc * 0.3
    signal = 1 if confidence >= MIN_CONFIDENCE else 0
    features_dict = latest_features.to_dict()
    return signal, confidence, features_dict
# ===========================
# BACKTEST SYMBOL
# ===========================
def backtest_symbol(symbol, df_symbol, signals_history_df, model_data_df):
    df = df_symbol.sort_values('Date').reset_index(drop=True)
    min_start = MIN_TRAINING_SAMPLES + 10
    if len(df) < min_start:
        return signals_history_df, model_data_df
    history = []
    for test_idx in range(min_start, len(df) - 1):
        df_available = df.iloc[:test_idx + 1]
        model, _ = train_model(df_available)
        if model is None:
            continue
        features = create_features(df_available)
        latest_features = features.iloc[-1]
        if pd.isna(latest_features).any():
            continue
        pred_df = pd.DataFrame([latest_features])
        X_pred = sm.add_constant(pred_df)[model.exog_names]
        try:
            prob = model.predict(X_pred)[0]
        except:
            continue
        if prob >= MIN_CONFIDENCE:
            entry_date = df.iloc[test_idx]['Date']
            entry_price = df.iloc[test_idx]['Close']
            exit_date = df.iloc[test_idx + 1]['Date']
            exit_price = df.iloc[test_idx + 1]['Close']
            return_pct = (exit_price - entry_price) / entry_price * 100
            success = 1 if return_pct / 100 >= PROFIT_THRESHOLD else 0
            history.append({
                'Date': entry_date.strftime('%Y-%m-%d'),
                'Symbol': symbol,
                'Price': entry_price,
                'Confidence': prob,
                'Prediction': 1,
                'Exit_Date': exit_date.strftime('%Y-%m-%d'),
                'Exit_Price': exit_price,
                'Return_Pct': return_pct,
                'Success': success
            })
    if len(history) > 0:
        new_history = pd.DataFrame(history)
        existing = signals_history_df[signals_history_df['Symbol'] == symbol]
        all_history = pd.concat([existing, new_history], ignore_index=True)
        total = len(all_history)
        successful = all_history['Success'].sum()
        accuracy = successful / total if total > 0 else 0.5
        avg_profit = all_history['Return_Pct'].mean() if total > 0 else 0.0
        model_row = model_data_df[model_data_df['Symbol'] == symbol]
        if len(model_row) == 0:
            last_sig = all_history.sort_values('Date').iloc[-1]
            new_model = pd.DataFrame([{
                'Symbol': symbol,
                'Last_Updated': last_sig['Date'],
                'Total_Signals': total,
                'Successful_Signals': successful,
                'Accuracy': accuracy,
                'Avg_Profit': avg_profit,
                'Last_Signal_Date': last_sig['Date'],
                'Last_Signal_Price': last_sig['Price'],
                'Last_Signal_Confidence': last_sig['Confidence'],
                'Backtested': True
            }])
            model_data_df = pd.concat([model_data_df, new_model], ignore_index=True)
        else:
            idx = model_row.index[0]
            model_data_df.at[idx, 'Total_Signals'] = total
            model_data_df.at[idx, 'Successful_Signals'] = successful
            model_data_df.at[idx, 'Accuracy'] = accuracy
            model_data_df.at[idx, 'Avg_Profit'] = avg_profit
            all_history = all_history.sort_values('Date')
            last_sig = all_history.iloc[-1]
            model_data_df.at[idx, 'Last_Signal_Date'] = last_sig['Date']
            model_data_df.at[idx, 'Last_Signal_Price'] = last_sig['Price']
            model_data_df.at[idx, 'Last_Signal_Confidence'] = last_sig['Confidence']
            model_data_df.at[idx, 'Last_Updated'] = datetime.now().strftime('%Y-%m-%d')
            model_data_df.at[idx, 'Backtested'] = True
        signals_history_df = pd.concat([signals_history_df[signals_history_df['Symbol'] != symbol], all_history], ignore_index=True)
    return signals_history_df, model_data_df
# ===========================
# MAIN DAILY RUNNER
# ===========================
def main():
    print("="*80)
    print("🤖 DAILY ML SIGNAL GENERATOR WITH BACKTESTING (GitHub Actions)")
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
                 'Last_Signal_Confidence', 'Backtested']
    )
    if 'Backtested' not in model_data_df.columns:
        model_data_df['Backtested'] = False
   
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
    # PERFORM BACKTESTS IF NEEDED
    # ===========================
    print("\n🧪 Checking for symbols needing backtest...")
    symbols = df['Symbol'].unique()
    symbols = [s for s in symbols if s not in EXCLUDED_SYMBOLS]
    for symbol in symbols:
        df_sym = df[df['Symbol'] == symbol].copy()
        if len(df_sym) < MIN_TRAINING_SAMPLES + 10:
            continue
        model_row = model_data_df[model_data_df['Symbol'] == symbol]
        if len(model_row) == 0 or not model_row.iloc[0]['Backtested']:
            print(f"🔄 Performing backtest for {symbol}")
            signals_history_df, model_data_df = backtest_symbol(symbol, df_sym, signals_history_df, model_data_df)
   
    # ===========================
    # STEP 1: Record yesterday's outcomes
    # ===========================
    print("\n📊 Checking yesterday's signals...")
   
    if len(signals_history_df) > 0:
        pending = signals_history_df[signals_history_df['Exit_Date'].isna()].copy()
        if len(pending) > 0:
            print(f" Found {len(pending)} pending signals to check")
            updated_count = 0
            for idx, signal in pending.iterrows():
                symbol = signal['Symbol']
                entry_date = pd.to_datetime(signal['Date'])
                entry_price = signal['Price']
                df_sym = df[df['Symbol'] == symbol]
                today_data = df_sym[df_sym['Date'] == latest_date]
                if len(today_data) > 0:
                    exit_price = today_data.iloc[0]['Close']
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    success = 1 if return_pct / 100 >= PROFIT_THRESHOLD else 0
                    signals_history_df.at[idx, 'Exit_Date'] = latest_date.strftime('%Y-%m-%d')
                    signals_history_df.at[idx, 'Exit_Price'] = exit_price
                    signals_history_df.at[idx, 'Return_Pct'] = return_pct
                    signals_history_df.at[idx, 'Success'] = success
                    model_row = model_data_df[model_data_df['Symbol'] == symbol]
                    if len(model_row) > 0:
                        idx_model = model_row.index[0]
                        total = model_data_df.at[idx_model, 'Total_Signals']
                        successful = model_data_df.at[idx_model, 'Successful_Signals']
                        model_data_df.at[idx_model, 'Total_Signals'] = total + 1
                        model_data_df.at[idx_model, 'Successful_Signals'] = successful + success
                        new_total = total + 1
                        model_data_df.at[idx_model, 'Accuracy'] = (successful + success) / new_total
                        old_avg = model_data_df.at[idx_model, 'Avg_Profit']
                        model_data_df.at[idx_model, 'Avg_Profit'] = (old_avg * total + return_pct) / new_total
                        model_data_df.at[idx_model, 'Last_Updated'] = latest_date.strftime('%Y-%m-%d')
                    status = "✅" if success else "❌"
                    print(f" {status} {symbol}: {return_pct:+.2f}%")
                    updated_count += 1
            if updated_count > 0:
                print(f" 📝 Updated {updated_count} signal outcomes")
   
    # ===========================
    # STEP 2: Generate today's signals
    # ===========================
    print(f"\n🎯 Generating signals for {latest_date.date()}...")
   
    today_signals = []
   
    for symbol in symbols:
        df_sym = df[df['Symbol'] == symbol].copy()
        df_sym = df_sym[df_sym['Date'] <= latest_date]
        if len(df_sym) < MIN_TRAINING_SAMPLES:
            continue
        model_row = model_data_df[model_data_df['Symbol'] == symbol]
        model_info = model_row.iloc[0].to_dict() if len(model_row) > 0 else None
        if model_info and model_info['Total_Signals'] >= MIN_TRADES and model_info['Accuracy'] < MIN_MODEL_ACCURACY:
            continue
        signal, confidence, features = generate_signal_ml(df_sym, model_info)
        if signal == 1:
            latest_price = df_sym.iloc[-1]['Close']
            model_accuracy = model_info['Accuracy'] if model_info else 0.5
            trades_count = model_info['Total_Signals'] if model_info else 0
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
        today_signals_sorted = sorted(today_signals, key=lambda x: x['Confidence'], reverse=True)
        for i, sig in enumerate(today_signals_sorted, 1):
            print(f"\n{i}. 📈 {sig['Symbol']}")
            print(f" Price: NPR {sig['Price']:.2f}")
            print(f" Confidence: {sig['Confidence']*100:.1f}%")
            print(f" Model Accuracy: {sig['Model_Accuracy']*100:.1f}%")
            print(f" Historical Trades: {sig['Historical_Trades']}")
        daily_df = pd.DataFrame(today_signals_sorted)
        upload_csv_to_github(
            daily_df,
            DAILY_SIGNALS_CSV,
            f"🤖 ML Signals for {latest_date.strftime('%Y-%m-%d')} - {len(today_signals)} signals"
        )
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
        signals_history_df = pd.concat([signals_history_df, new_history], ignore_index=True)
        for sig in today_signals_sorted:
            model_row = model_data_df[model_data_df['Symbol'] == sig['Symbol']]
            if len(model_row) == 0:
                new_model = pd.DataFrame([{
                    'Symbol': sig['Symbol'],
                    'Last_Updated': latest_date.strftime('%Y-%m-%d'),
                    'Total_Signals': 0,
                    'Successful_Signals': 0,
                    'Accuracy': 0.5,
                    'Avg_Profit': 0.0,
                    'Last_Signal_Date': sig['Date'],
                    'Last_Signal_Price': sig['Price'],
                    'Last_Signal_Confidence': sig['Confidence'],
                    'Backtested': False
                }])
                model_data_df = pd.concat([model_data_df, new_model], ignore_index=True)
            else:
                idx = model_row.index[0]
                model_data_df.at[idx, 'Last_Signal_Date'] = sig['Date']
                model_data_df.at[idx, 'Last_Signal_Price'] = sig['Price']
                model_data_df.at[idx, 'Last_Signal_Confidence'] = sig['Confidence']
    else:
        print("\n⏸️ No signals generated today")
   
    # Upload updated CSVs
    print("\n📤 Uploading updated data to GitHub...")
   
    upload_csv_to_github(
        signals_history_df,
        SIGNALS_HISTORY_CSV,
        f"📊 Updated signals history - {latest_date.strftime('%Y-%m-%d')}"
    )
   
    upload_csv_to_github(
        model_data_df,
        MODEL_DATA_CSV,
        f"📈 Updated model performance data - {latest_date.strftime('%Y-%m-%d')}"
    )
   
    # Show top performers
    if len(model_data_df) > 0:
        print(f"\n{'='*80}")
        print("🏆 TOP PERFORMING MODELS")
        print(f"{'='*80}")
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
if __name__ == "__main__":
    main()
