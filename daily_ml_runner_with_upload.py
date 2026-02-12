# -*- coding: utf-8 -*-
"""
DAILY ML RUNNER WITH GITHUB AUTO-UPLOAD
- Loads ALL historical data for each symbol (oldest to newest)
- Trains on complete history up to yesterday
- Generates signals for today
- Uploads results to GitHub
- Deletes old signal files
- Tracks performance over time
"""
import os
import re
import pickle
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import base64
import json
warnings.filterwarnings('ignore')

# ===========================
# CONFIG
# ===========================
REPO_OWNER = "iamsrijit"
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
LOOKBACK_DAYS = 365  # Use 1 year of training data
MIN_TRAINING_SAMPLES = 50
RETRAIN_FREQUENCY = 5  # Retrain every N days

# GITHUB UPLOAD SETTINGS
UPLOAD_TO_GITHUB = True  # Set to False to disable auto-upload
DELETE_OLD_SIGNALS = True  # Delete yesterday's signal files from GitHub
SIGNALS_FOLDER = "ml_signals/daily"  # Where to store daily signals
TRACKER_FILE = "ml_signals/ML_PERFORMANCE_TRACKER.csv"
STATS_FILE = "ml_signals/MODEL_STATISTICS.csv"

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

def upload_file_to_github(file_path, repo_path, commit_message):
    """Upload a file to GitHub repository"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        content_base64 = base64.b64encode(content.encode()).decode()
        
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
            return True
        else:
            print(f"   ⚠️  Upload failed for {repo_path}: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error uploading {repo_path}: {str(e)}")
        return False

def delete_file_from_github(repo_path):
    """Delete a file from GitHub repository"""
    try:
        # Get file SHA
        check_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
        check_response = requests.get(check_url, headers=HEADERS, params={"ref": BRANCH})
        
        if check_response.status_code != 200:
            return False  # File doesn't exist
        
        sha = check_response.json().get('sha')
        
        # Delete file
        delete_data = {
            "message": f"🗑️ Cleanup: Remove old signal file",
            "sha": sha,
            "branch": BRANCH
        }
        
        delete_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
        response = requests.delete(delete_url, headers=HEADERS, data=json.dumps(delete_data))
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"   ⚠️  Error deleting {repo_path}: {str(e)}")
        return False

def cleanup_old_signals(current_date_str):
    """Delete signal files older than today"""
    try:
        # List all files in signals folder
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{SIGNALS_FOLDER}"
        response = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
        
        if response.status_code != 200:
            return
        
        files = response.json()
        deleted_count = 0
        
        for file in files:
            if file['type'] == 'file' and file['name'].startswith('signals_'):
                # Extract date from filename
                m = re.search(r'signals_(\d{8})\.csv', file['name'])
                if m:
                    file_date = m.group(1)
                    # Delete if not today
                    if file_date != current_date_str:
                        repo_path = f"{SIGNALS_FOLDER}/{file['name']}"
                        if delete_file_from_github(repo_path):
                            print(f"   🗑️  Deleted old file: {file['name']}")
                            deleted_count += 1
        
        if deleted_count > 0:
            print(f"   ✅ Cleaned up {deleted_count} old signal files")
        
    except Exception as e:
        print(f"   ⚠️  Cleanup error: {str(e)}")

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
    
    # EMAs
    for period in [5, 9, 21, 50]:
        features[f'EMA_{period}'] = calculate_ema(df['Close'], period)
        features[f'Close_to_EMA_{period}'] = (df['Close'] - features[f'EMA_{period}']) / features[f'EMA_{period}']
    
    # EMA crossovers
    features['EMA_5_9_cross'] = (features['EMA_5'] - features['EMA_9']) / features['EMA_9']
    features['EMA_9_21_cross'] = (features['EMA_9'] - features['EMA_21']) / features['EMA_21']
    
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

def create_target(df_symbol, profit_threshold=0.02):
    """Create binary target: 1 if next day gain >= threshold"""
    next_day_return = df_symbol['Close'].shift(-1) / df_symbol['Close'] - 1
    return (next_day_return >= profit_threshold).astype(int)

# ===========================
# SYMBOL MODEL CLASS
# ===========================
class SymbolModel:
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        self.total_profit = 0.0
        self.trade_history = []
        
    def train(self, X, y):
        if len(X) < MIN_TRAINING_SAMPLES:
            return False
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < MIN_TRAINING_SAMPLES:
            return False
        
        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y_clean)
        self.is_trained = True
        self.feature_names = X.columns.tolist()
        
        return True
    
    def predict(self, X):
        if not self.is_trained or list(X.columns) != self.feature_names:
            return None, 0.0
        
        if X.isna().any().any():
            return None, 0.0
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        confidence = self.model.predict_proba(X_scaled)[0][1]
        
        return prediction, confidence
    
    def record_outcome(self, predicted, actual_return):
        self.predictions_made += 1
        actual_success = 1 if actual_return >= PROFIT_THRESHOLD else 0
        
        if predicted == actual_success:
            self.correct_predictions += 1
        
        self.total_profit += actual_return
        self.trade_history.append({
            'predicted': predicted,
            'actual_return': actual_return,
            'success': actual_success
        })
    
    def get_accuracy(self):
        if self.predictions_made == 0:
            return 0.0
        return self.correct_predictions / self.predictions_made
    
    def get_stats(self):
        return {
            'Symbol': self.symbol,
            'Predictions': self.predictions_made,
            'Accuracy_%': self.get_accuracy() * 100,
            'Total_Profit_%': self.total_profit * 100,
            'Avg_Profit_%': (self.total_profit / self.predictions_made * 100) if self.predictions_made > 0 else 0
        }

# ===========================
# ADAPTIVE STRATEGY ENGINE
# ===========================
class AdaptiveMLStrategy:
    def __init__(self):
        self.models = {}
        self.last_retrain_day = {}
        
    def get_or_create_model(self, symbol):
        if symbol not in self.models:
            self.models[symbol] = SymbolModel(symbol)
        return self.models[symbol]
    
    def prepare_training_data(self, df_symbol, current_date):
        """Use ALL historical data up to current date"""
        historical = df_symbol[df_symbol['Date'] < current_date].copy()
        
        if len(historical) < MIN_TRAINING_SAMPLES + 50:
            return None, None
        
        features = create_features(historical)
        target = create_target(historical, PROFIT_THRESHOLD)
        
        # Align indices
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]
        
        # Use last LOOKBACK_DAYS for training
        if len(features) > LOOKBACK_DAYS:
            features = features.iloc[-LOOKBACK_DAYS:]
            target = target.iloc[-LOOKBACK_DAYS:]
        
        # Remove last row (no target)
        features = features.iloc[:-1]
        target = target.iloc[:-1]
        
        return features, target
    
    def should_retrain(self, symbol, current_date):
        if symbol not in self.last_retrain_day:
            return True
        
        last_retrain = self.last_retrain_day[symbol]
        days_since = (current_date - last_retrain).days
        
        return days_since >= RETRAIN_FREQUENCY
    
    def generate_signals(self, df, current_date):
        signals = []
        symbols_today = df[df['Date'] == current_date]['Symbol'].unique()
        
        for symbol in symbols_today:
            if symbol in EXCLUDED_SYMBOLS:
                continue
            
            df_symbol = df[df['Symbol'] == symbol].copy()
            model = self.get_or_create_model(symbol)
            
            # Retrain if needed
            if self.should_retrain(symbol, current_date):
                X_train, y_train = self.prepare_training_data(df_symbol, current_date)
                
                if X_train is not None and len(X_train) >= MIN_TRAINING_SAMPLES:
                    success = model.train(X_train, y_train)
                    if success:
                        self.last_retrain_day[symbol] = current_date
            
            # Generate prediction
            if model.is_trained:
                historical = df_symbol[df_symbol['Date'] <= current_date].copy()
                features = create_features(historical)
                
                if len(features) > 0:
                    today_features = features.iloc[[-1]]
                    prediction, confidence = model.predict(today_features)
                    
                    if prediction == 1 and confidence >= MIN_CONFIDENCE:
                        current_price = df_symbol[df_symbol['Date'] == current_date].iloc[0]['Close']
                        
                        signals.append({
                            'Symbol': symbol,
                            'Date': current_date,
                            'Price': current_price,
                            'Confidence_%': confidence * 100,
                            'Model_Accuracy_%': model.get_accuracy() * 100,
                            'Predictions_Made': model.predictions_made
                        })
        
        return signals
    
    def record_outcomes(self, df, signals, signal_date):
        next_date = signal_date + pd.Timedelta(days=1)
        
        for signal in signals:
            symbol = signal['Symbol']
            entry_price = signal['Price']
            
            df_symbol = df[df['Symbol'] == symbol]
            next_day_data = df_symbol[df_symbol['Date'] == next_date]
            
            if len(next_day_data) > 0:
                exit_price = next_day_data.iloc[0]['Close']
                actual_return = (exit_price - entry_price) / entry_price
                
                model = self.models[symbol]
                model.record_outcome(predicted=1, actual_return=actual_return)
                
                signal['Next_Day_Return_%'] = actual_return * 100
                signal['Success'] = 1 if actual_return >= PROFIT_THRESHOLD else 0
    
    def get_model_stats(self):
        stats = []
        for symbol, model in self.models.items():
            if model.predictions_made > 0:
                stats.append(model.get_stats())
        return pd.DataFrame(stats).sort_values('Accuracy_%', ascending=False)
    
    def save_models(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'last_retrain_day': self.last_retrain_day
            }, f)
    
    def load_models(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.last_retrain_day = data['last_retrain_day']
            return True
        except:
            return False

# ===========================
# MAIN DAILY RUNNER
# ===========================
def main():
    print("="*80)
    print("🤖 DAILY ML SIGNAL GENERATOR WITH GITHUB AUTO-UPLOAD")
    print(f"🕒 Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Get latest data
    print("\n📥 Fetching latest NEPSE data...")
    csv_url, data_date = get_latest_espen_csv()
    
    response = requests.get(csv_url)
    csv_content = response.text
    
    # Parse CSV - handle the unusual format
    lines = csv_content.strip().split('\n')
    
    # Find the header line
    header_line = None
    data_start_index = 0
    
    for i, line in enumerate(lines):
        if 'Symbol' in line and 'Date' in line and 'Close' in line:
            header_line = line
            data_start_index = i
            break
    
    # Reconstruct CSV properly
    if data_start_index > 0:
        # Data before header = actual data rows
        data_lines = lines[:data_start_index]
        reconstructed_csv = header_line + '\n' + '\n'.join(data_lines)
    else:
        reconstructed_csv = csv_content
    
    # Try tab-separated first, then comma
    try:
        df = pd.read_csv(StringIO(reconstructed_csv), sep='\t')
        if len(df.columns) == 1:
            df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
    except:
        df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
    
    df.columns = df.columns.str.strip()
    
    # Parse dates - handle both formats
    original_dates = df["Date"].copy()
    df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y', errors='coerce')
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(original_dates, format='%Y-%m-%d', errors='coerce')
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(original_dates, errors='coerce')
    
    # Clean numeric columns
    df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(',', ''), errors='coerce')
    df["Open"] = pd.to_numeric(df["Open"].astype(str).str.replace(',', ''), errors='coerce')
    if 'High' in df.columns:
        df["High"] = pd.to_numeric(df["High"].astype(str).str.replace(',', ''), errors='coerce')
    if 'Low' in df.columns:
        df["Low"] = pd.to_numeric(df["Low"].astype(str).str.replace(',', ''), errors='coerce')
    if 'Volume' in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=["Symbol", "Date", "Close"])
    df = df.sort_values(["Symbol", "Date"])
    
    latest_date = df['Date'].max()
    
    print(f"✅ Data loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")
    print(f"📅 Data date range: {df['Date'].min().date()} to {latest_date.date()}")
    print(f"📊 Using ALL historical data per symbol for training")
    
    # Initialize strategy
    strategy = AdaptiveMLStrategy()
    model_file = "/home/claude/adaptive_models.pkl"
    
    if os.path.exists(model_file):
        print(f"\n🔄 Loading existing models...")
        loaded = strategy.load_models(model_file)
        if loaded:
            print(f"✅ Loaded {len(strategy.models)} trained models")
    else:
        print("\n🆕 No existing models, will train from scratch")
    
    # Record yesterday's outcomes
    yesterday_signals_file = "/home/claude/yesterday_signals.csv"
    if os.path.exists(yesterday_signals_file):
        print("\n📊 Checking yesterday's signals...")
        yesterday_signals = pd.read_csv(yesterday_signals_file)
        yesterday_signals['Date'] = pd.to_datetime(yesterday_signals['Date'])
        
        outcomes_recorded = 0
        for _, signal in yesterday_signals.iterrows():
            symbol = signal['Symbol']
            entry_date = signal['Date']
            entry_price = signal['Price']
            
            df_symbol = df[df['Symbol'] == symbol]
            today_data = df_symbol[df_symbol['Date'] == latest_date]
            
            if len(today_data) > 0 and symbol in strategy.models:
                exit_price = today_data.iloc[0]['Close']
                actual_return = (exit_price - entry_price) / entry_price
                
                model = strategy.models[symbol]
                model.record_outcome(predicted=1, actual_return=actual_return)
                
                success = "✅" if actual_return >= PROFIT_THRESHOLD else "❌"
                print(f"   {success} {symbol}: {actual_return*100:+.2f}%")
                outcomes_recorded += 1
        
        if outcomes_recorded > 0:
            print(f"   📝 Recorded {outcomes_recorded} outcomes")
    
    # Generate today's signals
    print(f"\n🎯 Generating signals for {latest_date.date()}...")
    signals = strategy.generate_signals(df, latest_date)
    
    # Save models
    strategy.save_models(model_file)
    print(f"💾 Models saved to: {model_file}")
    
    if signals:
        print(f"\n{'='*80}")
        print(f"🚨 {len(signals)} BUY SIGNALS GENERATED!")
        print(f"{'='*80}")
        
        signals_sorted = sorted(signals, key=lambda x: x['Confidence_%'], reverse=True)
        
        for i, sig in enumerate(signals_sorted, 1):
            print(f"\n{i}. 📈 {sig['Symbol']}")
            print(f"   Price: NPR {sig['Price']:.2f}")
            print(f"   Confidence: {sig['Confidence_%']:.1f}%")
            print(f"   Model Accuracy: {sig['Model_Accuracy_%']:.1f}%")
            print(f"   Historical Trades: {sig['Predictions_Made']}")
        
        # Save signals locally
        signals_df = pd.DataFrame(signals)
        signals_df['Date'] = signals_df['Date'].dt.strftime('%Y-%m-%d')
        
        date_str = latest_date.strftime('%Y%m%d')
        local_signals_file = f"/home/claude/signals_{date_str}.csv"
        signals_df.to_csv(local_signals_file, index=False)
        signals_df.to_csv(yesterday_signals_file, index=False)  # For tomorrow's check
        
        # Upload to GitHub
        if UPLOAD_TO_GITHUB:
            print(f"\n📤 Uploading to GitHub...")
            
            # Delete old signal files
            if DELETE_OLD_SIGNALS:
                print("   🗑️  Cleaning up old signal files...")
                cleanup_old_signals(date_str)
            
            # Upload today's signals
            repo_signals_path = f"{SIGNALS_FOLDER}/signals_{date_str}.csv"
            commit_msg = f"🤖 ML Signals for {latest_date.strftime('%Y-%m-%d')} - {len(signals)} signals generated"
            
            if upload_file_to_github(local_signals_file, repo_signals_path, commit_msg):
                print(f"   ✅ Uploaded: {repo_signals_path}")
            
            # Upload model statistics
            model_stats_df = strategy.get_model_stats()
            if len(model_stats_df) > 0:
                stats_file = "/home/claude/model_statistics.csv"
                model_stats_df.to_csv(stats_file, index=False)
                
                commit_msg = f"📊 Model Statistics Update - {latest_date.strftime('%Y-%m-%d')}"
                if upload_file_to_github(stats_file, STATS_FILE, commit_msg):
                    print(f"   ✅ Uploaded: {STATS_FILE}")
            
            print(f"\n   🌐 View signals: https://github.com/{REPO_OWNER}/{REPO_NAME}/blob/{BRANCH}/{repo_signals_path}")
        
    else:
        print("\n⏸️  No signals generated today")
        print("   Models need more confidence or training data")
    
    # Show top performers
    print(f"\n{'='*80}")
    print("🏆 TOP PERFORMING MODELS")
    print(f"{'='*80}")
    
    model_stats_df = strategy.get_model_stats()
    if len(model_stats_df) > 0:
        top_10 = model_stats_df.head(10)
        print(top_10.to_string(index=False))
    else:
        print("Not enough trade history yet")
    
    print(f"\n{'='*80}")
    print("✅ Daily run complete!")
    print(f"{'='*80}")
    print("\n💡 Next steps:")
    print("   1. Check GitHub for uploaded signals")
    print("   2. Execute top 2-3 signals")
    print("   3. Run again tomorrow to check outcomes & get new signals")

if __name__ == "__main__":
    main()
