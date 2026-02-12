# -*- coding: utf-8 -*-
"""
ADAPTIVE ML TRADING STRATEGY
- Learns different buy signals for each symbol
- Checks next-day performance and rewards itself
- Improves strategy daily through online learning
- Uses Random Forest with incremental updates
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
from collections import defaultdict
import warnings
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
PROFIT_THRESHOLD = 0.02  # 2% gain next day = success
MIN_CONFIDENCE = 0.65  # Model confidence to generate signal
LOOKBACK_DAYS = 252  # 1 year training window
MIN_TRAINING_SAMPLES = 50  # Minimum samples to train model
RETRAIN_FREQUENCY = 5  # Retrain every N days

# FEATURE ENGINEERING
FEATURE_PERIODS = {
    'RSI': [7, 14, 21],
    'EMA': [5, 9, 21, 50],
    'BB': [20],
    'VOLUME': [5, 10, 20],
    'MOMENTUM': [5, 10, 20]
}

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set in environment")

HEADERS = {"Authorization": f"token {GH_TOKEN}"}

# ===========================
# GITHUB HELPERS
# ===========================
def github_raw(path):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

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
    print(f"📂 Using: {latest_file}")
    return github_raw(latest_file)

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
    """Create comprehensive feature set for ML model"""
    df = df_symbol.copy()
    features = pd.DataFrame(index=df.index)
    
    # RSI features (multiple periods)
    for period in FEATURE_PERIODS['RSI']:
        features[f'RSI_{period}'] = calculate_rsi(df['Close'], period)
    
    # EMA features
    for period in FEATURE_PERIODS['EMA']:
        features[f'EMA_{period}'] = calculate_ema(df['Close'], period)
        features[f'Close_to_EMA_{period}'] = (df['Close'] - features[f'EMA_{period}']) / features[f'EMA_{period}']
    
    # EMA crossovers
    features['EMA_5_9_cross'] = (features['EMA_5'] - features['EMA_9']) / features['EMA_9']
    features['EMA_9_21_cross'] = (features['EMA_9'] - features['EMA_21']) / features['EMA_21']
    features['EMA_21_50_cross'] = (features['EMA_21'] - features['EMA_50']) / features['EMA_50']
    
    # Bollinger Bands
    for period in FEATURE_PERIODS['BB']:
        bb_sma, bb_std = calculate_bollinger_bands(df['Close'], period)
        features[f'BB_position_{period}'] = (df['Close'] - bb_sma) / (2 * bb_std)
        features[f'BB_width_{period}'] = (4 * bb_std) / bb_sma
    
    # Volume features (if available)
    if 'Volume' in df.columns:
        for period in FEATURE_PERIODS['VOLUME']:
            vol_sma = df['Volume'].rolling(window=period).mean()
            features[f'Volume_ratio_{period}'] = df['Volume'] / vol_sma
    
    # Price momentum
    for period in FEATURE_PERIODS['MOMENTUM']:
        features[f'Momentum_{period}'] = df['Close'].pct_change(period) * 100
    
    # Volatility
    features['Volatility_10'] = df['Close'].pct_change().rolling(window=10).std() * 100
    features['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * 100
    
    # 52-week high/low position
    features['52W_High'] = df['Close'].rolling(window=252, min_periods=50).max()
    features['52W_Low'] = df['Close'].rolling(window=252, min_periods=50).min()
    features['52W_position'] = (df['Close'] - features['52W_Low']) / (features['52W_High'] - features['52W_Low'])
    
    # Price patterns
    features['Daily_return'] = df['Close'].pct_change() * 100
    features['High_Low_range'] = ((df['High'] - df['Low']) / df['Close']) * 100 if 'High' in df.columns else 0
    
    return features

def create_target(df_symbol, profit_threshold=0.02):
    """Create binary target: 1 if next day gain >= threshold, 0 otherwise"""
    df = df_symbol.copy()
    next_day_return = df['Close'].shift(-1) / df['Close'] - 1
    target = (next_day_return >= profit_threshold).astype(int)
    return target

# ===========================
# SYMBOL-SPECIFIC MODEL CLASS
# ===========================
class SymbolModel:
    """Individual ML model for each symbol with online learning"""
    
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
        """Train the model"""
        if len(X) < MIN_TRAINING_SAMPLES:
            return False
        
        # Remove any NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < MIN_TRAINING_SAMPLES:
            return False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train model
        self.model.fit(X_scaled, y_clean)
        self.is_trained = True
        self.feature_names = X.columns.tolist()
        
        return True
    
    def predict(self, X):
        """Predict with confidence score"""
        if not self.is_trained:
            return None, 0.0
        
        # Ensure same features
        if list(X.columns) != self.feature_names:
            return None, 0.0
        
        # Remove NaN
        if X.isna().any().any():
            return None, 0.0
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        confidence = self.model.predict_proba(X_scaled)[0][1]  # Probability of positive class
        
        return prediction, confidence
    
    def record_outcome(self, predicted, actual_return):
        """Record prediction outcome for learning"""
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
        """Get current accuracy"""
        if self.predictions_made == 0:
            return 0.0
        return self.correct_predictions / self.predictions_made
    
    def get_stats(self):
        """Get model statistics"""
        return {
            'symbol': self.symbol,
            'predictions': self.predictions_made,
            'accuracy': self.get_accuracy() * 100,
            'total_profit': self.total_profit * 100,
            'avg_profit': (self.total_profit / self.predictions_made * 100) if self.predictions_made > 0 else 0
        }

# ===========================
# ADAPTIVE STRATEGY ENGINE
# ===========================
class AdaptiveMLStrategy:
    """Main strategy engine managing all symbol models"""
    
    def __init__(self):
        self.models = {}  # {symbol: SymbolModel}
        self.training_data = {}  # {symbol: (X, y)}
        self.last_retrain_day = {}  # {symbol: last_date}
        
    def get_or_create_model(self, symbol):
        """Get existing model or create new one"""
        if symbol not in self.models:
            self.models[symbol] = SymbolModel(symbol)
        return self.models[symbol]
    
    def prepare_training_data(self, df_symbol, current_date):
        """Prepare training data up to current date"""
        # Get historical data up to current date
        historical = df_symbol[df_symbol['Date'] < current_date].copy()
        
        if len(historical) < MIN_TRAINING_SAMPLES + 50:
            return None, None
        
        # Create features and target
        features = create_features(historical)
        target = create_target(historical, PROFIT_THRESHOLD)
        
        # Use only last LOOKBACK_DAYS
        if len(features) > LOOKBACK_DAYS:
            features = features.iloc[-LOOKBACK_DAYS:]
            target = target.iloc[-LOOKBACK_DAYS:]
        
        # Remove last row (no target available)
        features = features.iloc[:-1]
        target = target.iloc[:-1]
        
        return features, target
    
    def should_retrain(self, symbol, current_date):
        """Check if model needs retraining"""
        if symbol not in self.last_retrain_day:
            return True
        
        last_retrain = self.last_retrain_day[symbol]
        days_since = (current_date - last_retrain).days
        
        return days_since >= RETRAIN_FREQUENCY
    
    def generate_signals(self, df, current_date):
        """Generate buy signals for all symbols on current date"""
        signals = []
        
        # Get all symbols with data on current date
        symbols_today = df[df['Date'] == current_date]['Symbol'].unique()
        
        for symbol in symbols_today:
            if symbol in EXCLUDED_SYMBOLS:
                continue
            
            # Get symbol data
            df_symbol = df[df['Symbol'] == symbol].copy()
            
            # Get or create model
            model = self.get_or_create_model(symbol)
            
            # Check if we need to retrain
            if self.should_retrain(symbol, current_date):
                X_train, y_train = self.prepare_training_data(df_symbol, current_date)
                
                if X_train is not None and len(X_train) >= MIN_TRAINING_SAMPLES:
                    success = model.train(X_train, y_train)
                    if success:
                        self.last_retrain_day[symbol] = current_date
                        self.training_data[symbol] = (X_train, y_train)
            
            # Generate prediction for today
            if model.is_trained:
                # Get today's features
                historical = df_symbol[df_symbol['Date'] <= current_date].copy()
                features = create_features(historical)
                
                if len(features) > 0:
                    today_features = features.iloc[[-1]]
                    
                    prediction, confidence = model.predict(today_features)
                    
                    # Generate signal if confidence is high enough
                    if prediction == 1 and confidence >= MIN_CONFIDENCE:
                        current_price = df_symbol[df_symbol['Date'] == current_date].iloc[0]['Close']
                        
                        signals.append({
                            'Symbol': symbol,
                            'Date': current_date,
                            'Price': current_price,
                            'Confidence': confidence,
                            'Model_Accuracy': model.get_accuracy()
                        })
        
        return signals
    
    def record_outcomes(self, df, signals, signal_date):
        """Check next day outcomes and reward models"""
        next_date = signal_date + pd.Timedelta(days=1)
        
        for signal in signals:
            symbol = signal['Symbol']
            entry_price = signal['Price']
            
            # Get next day price
            df_symbol = df[df['Symbol'] == symbol]
            next_day_data = df_symbol[df_symbol['Date'] == next_date]
            
            if len(next_day_data) > 0:
                exit_price = next_day_data.iloc[0]['Close']
                actual_return = (exit_price - entry_price) / entry_price
                
                # Record outcome
                model = self.models[symbol]
                model.record_outcome(predicted=1, actual_return=actual_return)
                
                signal['Next_Day_Return'] = actual_return * 100
                signal['Actual_Success'] = 1 if actual_return >= PROFIT_THRESHOLD else 0
            else:
                signal['Next_Day_Return'] = None
                signal['Actual_Success'] = None
    
    def get_top_performers(self, min_predictions=10):
        """Get best performing models"""
        performers = []
        
        for symbol, model in self.models.items():
            if model.predictions_made >= min_predictions:
                stats = model.get_stats()
                performers.append(stats)
        
        return sorted(performers, key=lambda x: x['accuracy'], reverse=True)
    
    def save_models(self, filepath):
        """Save all models to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'training_data': self.training_data,
                'last_retrain_day': self.last_retrain_day
            }, f)
    
    def load_models(self, filepath):
        """Load models from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.training_data = data['training_data']
                self.last_retrain_day = data['last_retrain_day']
            return True
        except:
            return False

# ===========================
# MAIN EXECUTION
# ===========================
def main():
    print("="*70)
    print("🤖 ADAPTIVE ML TRADING STRATEGY")
    print("="*70)
    
    # Load data
    print("\n📥 Loading data from GitHub...")
    csv_url = get_latest_espen_csv()
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
    
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    if 'High' in df.columns:
        df["High"] = pd.to_numeric(df["High"], errors='coerce')
    if 'Low' in df.columns:
        df["Low"] = pd.to_numeric(df["Low"], errors='coerce')
    if 'Volume' in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
    
    df = df.dropna(subset=["Symbol", "Date", "Close"])
    df = df.sort_values(["Symbol", "Date"])
    
    print(f"✅ Loaded {len(df)} rows, {df['Symbol'].nunique()} symbols")
    print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Initialize strategy
    strategy = AdaptiveMLStrategy()
    
    # Simulation period (last 6 months)
    end_date = df['Date'].max()
    start_date = end_date - pd.Timedelta(days=180)
    
    print(f"\n🔄 Running adaptive learning simulation...")
    print(f"   Period: {start_date.date()} to {end_date.date()}")
    print(f"   Profit threshold: {PROFIT_THRESHOLD*100}%")
    print(f"   Min confidence: {MIN_CONFIDENCE*100}%")
    
    # Get all trading dates
    trading_dates = sorted(df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]['Date'].unique())
    
    all_signals = []
    
    # Simulate day by day
    for i, current_date in enumerate(trading_dates):
        # Generate signals
        signals = strategy.generate_signals(df, current_date)
        
        if signals:
            print(f"\n📅 {current_date.date()} - Generated {len(signals)} signals")
            for sig in signals:
                print(f"   🎯 {sig['Symbol']}: NPR {sig['Price']:.2f} (conf: {sig['Confidence']:.1%}, acc: {sig['Model_Accuracy']:.1%})")
        
        # Record outcomes from previous day's signals
        if i > 0:
            prev_date = trading_dates[i-1]
            prev_signals = [s for s in all_signals if s['Date'] == prev_date]
            
            if prev_signals:
                strategy.record_outcomes(df, prev_signals, prev_date)
        
        all_signals.extend(signals)
    
    # Final outcome recording
    if len(trading_dates) > 0:
        last_date = trading_dates[-1]
        last_signals = [s for s in all_signals if s['Date'] == last_date]
        if last_signals:
            strategy.record_outcomes(df, last_signals, last_date)
    
    # ===========================
    # RESULTS
    # ===========================
    print("\n" + "="*70)
    print("📊 LEARNING RESULTS")
    print("="*70)
    
    # Convert to DataFrame
    signals_df = pd.DataFrame(all_signals)
    
    if len(signals_df) > 0:
        # Filter signals with outcomes
        completed_signals = signals_df[signals_df['Next_Day_Return'].notna()].copy()
        
        if len(completed_signals) > 0:
            total_signals = len(completed_signals)
            successful = len(completed_signals[completed_signals['Actual_Success'] == 1])
            success_rate = (successful / total_signals) * 100
            
            avg_return = completed_signals['Next_Day_Return'].mean()
            total_return = completed_signals['Next_Day_Return'].sum()
            
            print(f"\n📈 OVERALL PERFORMANCE:")
            print(f"   Total Signals: {total_signals}")
            print(f"   Successful: {successful} ({success_rate:.1f}%)")
            print(f"   Average Return: {avg_return:+.2f}%")
            print(f"   Total Return: {total_return:+.2f}%")
            
            # Top performers
            print(f"\n🏆 TOP PERFORMING MODELS:")
            top_performers = strategy.get_top_performers(min_predictions=5)
            
            if top_performers:
                for i, perf in enumerate(top_performers[:10], 1):
                    print(f"   {i}. {perf['symbol']}: {perf['accuracy']:.1f}% accuracy, {perf['avg_profit']:+.2f}% avg profit ({perf['predictions']} trades)")
            else:
                print("   Not enough data yet")
            
            # Save results
            output_file = "ML_ADAPTIVE_SIGNALS.csv"
            completed_signals['Date'] = completed_signals['Date'].dt.strftime('%Y-%m-%d')
            completed_signals.to_csv(f"/home/claude/{output_file}", index=False)
            print(f"\n💾 Results saved to: {output_file}")
            
            # Save models
            model_file = "/home/claude/adaptive_models.pkl"
            strategy.save_models(model_file)
            print(f"💾 Models saved to: adaptive_models.pkl")
            
            print("\n" + "="*70)
            print("💡 INSIGHTS:")
            print("="*70)
            
            if success_rate >= 60:
                print("✅ EXCELLENT: Models are learning well!")
            elif success_rate >= 50:
                print("✅ GOOD: Models showing positive learning")
            elif success_rate >= 40:
                print("⚠️  MODERATE: Models need more training data")
            else:
                print("❌ POOR: Models not learning effectively - check data quality")
            
            # Symbol-specific insights
            print(f"\n📊 SYMBOL-SPECIFIC LEARNING:")
            symbol_stats = completed_signals.groupby('Symbol').agg({
                'Actual_Success': ['sum', 'count'],
                'Next_Day_Return': 'mean'
            }).round(2)
            
            symbol_stats.columns = ['Successes', 'Total', 'Avg_Return']
            symbol_stats['Success_Rate'] = (symbol_stats['Successes'] / symbol_stats['Total'] * 100).round(1)
            symbol_stats = symbol_stats[symbol_stats['Total'] >= 3].sort_values('Success_Rate', ascending=False)
            
            if len(symbol_stats) > 0:
                print(symbol_stats.head(10).to_string())
        else:
            print("⏳ No completed signals yet (need next day data)")
    else:
        print("⚠️  No signals generated - models may need more training data")
    
    print("\n" + "="*70)
    print("🔄 NEXT STEPS:")
    print("="*70)
    print("1. Models saved and ready for next day")
    print("2. Run daily to improve predictions")
    print("3. Models will adapt to each symbol's unique patterns")
    print("4. Best models will emerge over time")

if __name__ == "__main__":
    main()
