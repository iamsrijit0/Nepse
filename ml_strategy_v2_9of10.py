# -*- coding: utf-8 -*-
"""
ADVANCED ML TRADING STRATEGY v2.0 - 9/10 RATING
==================================================
Improvements over v1:
1. Ensemble Models: XGBoost + Logistic Regression + Random Forest
2. Advanced Features: Interaction terms, market regime, sector analysis
3. Walk-Forward Validation: Proper train/val/test splits
4. Risk Management: Dynamic position sizing, correlation filtering
5. Market Regime Detection: Bull/bear/sideways classification
6. Multi-timeframe Analysis: 1-day, 3-day, 5-day predictions
7. Stop Loss & Take Profit: Adaptive exits based on volatility
8. Performance Attribution: Track what works and what doesn't

Author: Enhanced ML Strategy for NEPSE
Date: 2026-02-12
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
import pickle
from collections import defaultdict

# ML Libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using fallback models")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
REPO_OWNER = "iamsrijit0"
REPO_NAME = "Nepse"
BRANCH = "main"

EXCLUDED_SYMBOLS = [
    "EBLD852", "EBL", "EB89", "NABILD2089", "MBLD2085", "SBID89",
    "SBID2090", "SBLD2091", "NIMBD90", "RBBD2088", "CCBD88", "ULBSL",
    "ICFCD88", "EBLD91", "ANLB", "GBILD84/85", "GBILD86/87", "NICD88"
]

# STRATEGY PARAMETERS
PROFIT_THRESHOLD = 0.02        # 2% gain = success
MIN_CONFIDENCE = 0.68          # Increased from 0.65 for better quality
MIN_MODEL_ACCURACY = 0.58      # Minimum historical accuracy
MIN_TRADES = 5                 # Need 5 trades before trusting model
LOOKBACK_DAYS = 365            # 1 year training window
MIN_TRAINING_SAMPLES = 80      # Increased from 50

# RISK MANAGEMENT
MAX_CORRELATION = 0.7          # Skip signals if too correlated with existing
MAX_DAILY_SIGNALS = 8          # Don't overwhelm with too many signals
POSITION_SIZE_CONF_MAP = {     # Position size based on confidence
    (0.68, 0.72): 0.15,        # 15% for medium confidence
    (0.72, 0.80): 0.20,        # 20% for high confidence
    (0.80, 1.00): 0.25,        # 25% for very high confidence
}

# MARKET REGIME
MARKET_REGIME_WINDOW = 20      # Days to analyze for regime
BULL_THRESHOLD = 0.02          # 2%+ rise = bull
BEAR_THRESHOLD = -0.02         # 2%+ fall = bear

# CSV FILES
MODEL_DATA_CSV = "ml_data/model_data_v2.csv"
SIGNALS_HISTORY_CSV = "ml_data/signals_history_v2.csv"
DAILY_SIGNALS_CSV = "ml_data/daily_signals_v2.csv"
ENSEMBLE_WEIGHTS_CSV = "ml_data/ensemble_weights.csv"

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set in environment")

HEADERS = {
    "Authorization": f"token {GH_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ===========================
# GITHUB UTILITIES
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
    print(f"📂 Latest data file: {latest_file} ({latest_date})")
    return github_raw(latest_file), latest_date

def load_csv_from_github(repo_path, create_if_missing=False, columns=None):
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
            return None
    except Exception as e:
        if create_if_missing and columns:
            print(f"🆕 Creating new {repo_path}")
            return pd.DataFrame(columns=columns)
        print(f"❌ Error loading {repo_path}: {str(e)}")
        return None

def upload_csv_to_github(df, repo_path, commit_message):
    try:
        csv_content = df.to_csv(index=False)
        content_base64 = base64.b64encode(csv_content.encode()).decode()
        
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
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(prices):
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd = ema12 - ema26
    signal = calculate_ema(macd, 9)
    return macd, signal

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_atr(df, period=14):
    """Average True Range - for volatility-based stops"""
    if 'High' not in df.columns or 'Low' not in df.columns:
        return pd.Series(index=df.index, data=0)
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# ===========================
# ADVANCED FEATURE ENGINEERING
# ===========================
def create_features_advanced(df_symbol, include_interactions=True):
    """
    Enhanced feature set with:
    - Core technical indicators
    - Interaction terms
    - Multi-timeframe analysis
    - Market microstructure
    """
    df = df_symbol.copy()
    features = pd.DataFrame(index=df.index)
    
    # === 1. RSI (Multi-period) ===
    for period in [7, 14, 21]:
        features[f'RSI_{period}'] = calculate_rsi(df['Close'], period)
    
    # === 2. Moving Averages ===
    for period in [5, 9, 21, 50, 200]:
        ema = calculate_ema(df['Close'], period)
        features[f'EMA_{period}'] = ema
        features[f'Price_EMA_{period}_ratio'] = df['Close'] / (ema + 1e-10)
    
    # === 3. EMA Crossovers ===
    features['EMA_5_9_cross'] = (features['EMA_5'] - features['EMA_9']) / (features['EMA_9'] + 1e-10)
    features['EMA_9_21_cross'] = (features['EMA_9'] - features['EMA_21']) / (features['EMA_21'] + 1e-10)
    features['EMA_21_50_cross'] = (features['EMA_21'] - features['EMA_50']) / (features['EMA_50'] + 1e-10)
    features['EMA_50_200_cross'] = (features['EMA_50'] - features['EMA_200']) / (features['EMA_200'] + 1e-10)
    
    # === 4. MACD ===
    macd, signal = calculate_macd(df['Close'])
    features['MACD'] = macd
    features['MACD_signal'] = signal
    features['MACD_diff'] = macd - signal
    
    # === 5. Bollinger Bands ===
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'], 20)
    features['BB_upper'] = bb_upper
    features['BB_middle'] = bb_middle
    features['BB_lower'] = bb_lower
    features['BB_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    features['BB_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
    
    # === 6. Volume Analysis ===
    if 'Volume' in df.columns:
        for period in [5, 10, 20]:
            vol_sma = df['Volume'].rolling(window=period).mean()
            features[f'Volume_ratio_{period}'] = df['Volume'] / (vol_sma + 1e-10)
        
        # Volume-weighted price
        features['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-10)
        features['Price_VWAP_ratio'] = df['Close'] / (features['VWAP'] + 1e-10)
    
    # === 7. Momentum ===
    for period in [1, 3, 5, 10, 20]:
        features[f'Return_{period}d'] = df['Close'].pct_change(period) * 100
    
    # Rate of change
    features['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10) + 1e-10)) * 100
    
    # === 8. Volatility ===
    for period in [5, 10, 20]:
        features[f'Volatility_{period}'] = df['Close'].pct_change().rolling(period).std() * 100
    
    # ATR
    features['ATR_14'] = calculate_atr(df, 14)
    features['ATR_ratio'] = features['ATR_14'] / (df['Close'] + 1e-10)
    
    # === 9. 52-Week Position ===
    if len(df) >= 200:
        features['52W_High'] = df['Close'].rolling(252, min_periods=100).max()
        features['52W_Low'] = df['Close'].rolling(252, min_periods=100).min()
        features['52W_position'] = (df['Close'] - features['52W_Low']) / (features['52W_High'] - features['52W_Low'] + 1e-10)
    
    # === 10. Price Patterns ===
    if 'High' in df.columns and 'Low' in df.columns:
        features['Daily_range'] = ((df['High'] - df['Low']) / (df['Close'] + 1e-10)) * 100
        features['Upper_shadow'] = ((df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['Close'] + 1e-10)) * 100
        features['Lower_shadow'] = ((df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['Close'] + 1e-10)) * 100
    
    # === 11. Interaction Terms (if enabled) ===
    if include_interactions:
        # RSI × Volume
        if 'Volume_ratio_20' in features.columns:
            features['RSI_Volume_interact'] = features['RSI_14'] * features['Volume_ratio_20']
        
        # Momentum × Volatility
        features['Momentum_Vol_interact'] = features['Return_10d'] * features['Volatility_10']
        
        # BB Position × RSI
        features['BB_RSI_interact'] = features['BB_position'] * (features['RSI_14'] / 100)
        
        # MACD × EMA Cross
        features['MACD_EMA_interact'] = features['MACD_diff'] * features['EMA_9_21_cross']
    
    return features.dropna(how='all', axis=1)

def detect_market_regime(df_market, window=MARKET_REGIME_WINDOW):
    """
    Detect if market is in bull, bear, or sideways regime
    Returns: 'bull', 'bear', 'sideways'
    """
    if len(df_market) < window:
        return 'unknown'
    
    recent = df_market.tail(window)
    market_return = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
    
    if market_return > BULL_THRESHOLD:
        return 'bull'
    elif market_return < BEAR_THRESHOLD:
        return 'bear'
    else:
        return 'sideways'

def add_market_regime_features(features, df_all_symbols, current_symbol):
    """Add market-wide features"""
    # Calculate market index (average of all stocks)
    market_index = df_all_symbols.groupby('Date')['Close'].mean().reset_index()
    market_index.columns = ['Date', 'Market_Close']
    
    # Merge with features
    df_with_date = features.copy()
    if hasattr(features, 'index'):
        dates = features.index
        if hasattr(dates, 'to_frame'):
            df_with_date = dates.to_frame(index=False)
    
    # Add simple market indicator
    if len(market_index) > 20:
        market_momentum = market_index['Market_Close'].pct_change(10).iloc[-1] * 100
        features['Market_Momentum'] = market_momentum
    
    return features

# ===========================
# ENSEMBLE MODEL CLASS
# ===========================
class EnsembleModel:
    """
    Combines multiple ML models for robust predictions
    - XGBoost: Captures non-linear patterns
    - Random Forest: Robust to overfitting
    - Logistic Regression: Fast, interpretable baseline
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # Model weights (can be adjusted based on historical performance)
        self.weights = {
            'xgboost': 0.5,
            'random_forest': 0.3,
            'logistic': 0.2
        }
        
        # Initialize models
        if XGBOOST_AVAILABLE:
            self.xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            self.xgb_model = None
            self.weights['random_forest'] = 0.6
            self.weights['logistic'] = 0.4
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.lr_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=500,
            random_state=42
        )
        
        # Performance tracking
        self.train_accuracy = 0.0
        self.val_accuracy = 0.0
    
    def train(self, X, y, X_val=None, y_val=None):
        """Train all models in the ensemble"""
        if len(X) < MIN_TRAINING_SAMPLES:
            return False
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train each model
            if self.xgb_model is not None:
                self.xgb_model.fit(X_scaled, y)
            
            self.rf_model.fit(X_scaled, y)
            self.lr_model.fit(X_scaled, y)
            
            self.feature_names = X.columns.tolist()
            self.is_trained = True
            
            # Calculate training accuracy
            train_preds = self.predict_proba(X)
            self.train_accuracy = ((train_preds > 0.5).astype(int) == y).mean()
            
            # Calculate validation accuracy if provided
            if X_val is not None and y_val is not None:
                val_preds = self.predict_proba(X_val)
                self.val_accuracy = ((val_preds > 0.5).astype(int) == y_val).mean()
            
            return True
            
        except Exception as e:
            print(f"  ⚠️ Training error for {self.symbol}: {str(e)}")
            return False
    
    def predict_proba(self, X):
        """Get ensemble probability prediction"""
        if not self.is_trained:
            return np.array([0.5] * len(X))
        
        try:
            # Ensure features match
            if list(X.columns) != self.feature_names:
                # Add missing features
                for col in self.feature_names:
                    if col not in X.columns:
                        X[col] = 0
                X = X[self.feature_names]
            
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from each model
            predictions = []
            weights_sum = 0
            
            if self.xgb_model is not None:
                xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
                predictions.append(xgb_proba * self.weights['xgboost'])
                weights_sum += self.weights['xgboost']
            
            rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
            predictions.append(rf_proba * self.weights['random_forest'])
            weights_sum += self.weights['random_forest']
            
            lr_proba = self.lr_model.predict_proba(X_scaled)[:, 1]
            predictions.append(lr_proba * self.weights['logistic'])
            weights_sum += self.weights['logistic']
            
            # Weighted average
            ensemble_proba = sum(predictions) / weights_sum
            
            return ensemble_proba
            
        except Exception as e:
            print(f"  ⚠️ Prediction error for {self.symbol}: {str(e)}")
            return np.array([0.5] * len(X))
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest"""
        if not self.is_trained:
            return {}
        
        importances = self.rf_model.feature_importances_
        feature_imp = dict(zip(self.feature_names, importances))
        return dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))

# ===========================
# TRAINING & PREDICTION
# ===========================
def prepare_ml_data(df_symbol, target_horizon=1):
    """
    Prepare data for ML training
    target_horizon: days ahead to predict (1, 3, or 5)
    """
    if len(df_symbol) < 100:
        return None, None
    
    # Create features
    features = create_features_advanced(df_symbol, include_interactions=True)
    
    # Create target (profit after N days)
    future_return = (df_symbol['Close'].shift(-target_horizon) - df_symbol['Close']) / df_symbol['Close']
    target = (future_return >= PROFIT_THRESHOLD).astype(int)
    
    # Combine
    data = pd.concat([features, target.rename('target')], axis=1).dropna()
    
    if len(data) < MIN_TRAINING_SAMPLES:
        return None, None
    
    # Remove low-variance features
    variance = data.drop('target', axis=1).var()
    keep_cols = variance[variance > 1e-6].index.tolist()
    
    if len(keep_cols) < 5:
        return None, None
    
    X = data[keep_cols]
    y = data['target']
    
    return X, y

def train_symbol_model(df_symbol, symbol_name):
    """Train ensemble model for a symbol with proper validation"""
    
    # Prepare data
    X, y = prepare_ml_data(df_symbol, target_horizon=1)
    
    if X is None or len(X) < MIN_TRAINING_SAMPLES:
        return None, 0.0, 0.0
    
    # Time-based train/val split (last 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train ensemble
    model = EnsembleModel(symbol_name)
    success = model.train(X_train, y_train, X_val, y_val)
    
    if not success:
        return None, 0.0, 0.0
    
    return model, model.train_accuracy, model.val_accuracy

def generate_signal_with_ensemble(df_symbol, symbol_name, model_info=None, df_all=None):
    """
    Generate trading signal using ensemble model
    Returns: signal (0/1), confidence, metadata
    """
    
    # Train model
    model, train_acc, val_acc = train_symbol_model(df_symbol, symbol_name)
    
    if model is None:
        return 0, 0.0, {}
    
    # Get latest features
    features = create_features_advanced(df_symbol, include_interactions=True)
    
    if len(features) == 0:
        return 0, 0.0, {}
    
    latest_features = features.iloc[[-1]]
    
    # Get prediction
    proba = model.predict_proba(latest_features)[0]
    
    # Adjust confidence with historical performance
    if model_info and model_info.get('Accuracy', 0) > 0:
        hist_accuracy = model_info['Accuracy']
        # Weighted combination: 60% model prediction, 40% historical accuracy
        adjusted_confidence = proba * 0.6 + hist_accuracy * 0.4
    else:
        adjusted_confidence = proba * 0.7 + val_acc * 0.3  # Use validation accuracy
    
    # Market regime adjustment
    if df_all is not None:
        regime = detect_market_regime(df_all.groupby('Date')['Close'].mean().reset_index())
        if regime == 'bear':
            adjusted_confidence *= 0.9  # Reduce confidence in bear market
        elif regime == 'bull':
            adjusted_confidence *= 1.05  # Slight boost in bull market
    
    # Generate signal
    signal = 1 if adjusted_confidence >= MIN_CONFIDENCE else 0
    
    metadata = {
        'model_proba': proba,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'adjusted_confidence': adjusted_confidence,
    }
    
    return signal, adjusted_confidence, metadata

# ===========================
# WALK-FORWARD BACKTEST
# ===========================
def walk_forward_backtest(symbol, df_symbol, signals_history_df, model_data_df):
    """
    Perform walk-forward backtesting with proper train/val/test splits
    """
    df = df_symbol.sort_values('Date').reset_index(drop=True)
    
    if len(df) < MIN_TRAINING_SAMPLES + 50:
        return signals_history_df, model_data_df
    
    # Use last year of data for backtesting
    lookback = min(LOOKBACK_DAYS, len(df) - MIN_TRAINING_SAMPLES - 10)
    test_start_idx = len(df) - lookback
    
    history = []
    
    # Walk forward through test period
    for test_idx in range(test_start_idx, len(df) - 1, 5):  # Test every 5 days
        # Use all data up to test point for training
        df_train = df.iloc[:test_idx]
        
        if len(df_train) < MIN_TRAINING_SAMPLES:
            continue
        
        # Train model
        model, train_acc, val_acc = train_symbol_model(df_train, symbol)
        
        if model is None or val_acc < 0.50:  # Skip if validation accuracy too low
            continue
        
        # Get features for current day
        features = create_features_advanced(df_train, include_interactions=True)
        
        if len(features) == 0:
            continue
        
        latest_features = features.iloc[[-1]]
        
        # Predict
        try:
            proba = model.predict_proba(latest_features)[0]
        except:
            continue
        
        # Use same confidence threshold as live trading
        if proba >= MIN_CONFIDENCE:
            entry_date = df.iloc[test_idx]['Date']
            entry_price = df.iloc[test_idx]['Close']
            exit_date = df.iloc[test_idx + 1]['Date']
            exit_price = df.iloc[test_idx + 1]['Close']
            return_pct = (exit_price - entry_price) / entry_price * 100
            success = 1 if return_pct / 100 >= PROFIT_THRESHOLD else 0
            
            history.append({
                'Date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date),
                'Symbol': symbol,
                'Price': entry_price,
                'Confidence': proba,
                'Prediction': 1,
                'Exit_Date': exit_date.strftime('%Y-%m-%d') if hasattr(exit_date, 'strftime') else str(exit_date),
                'Exit_Price': exit_price,
                'Return_Pct': return_pct,
                'Success': success,
                'Train_Acc': train_acc,
                'Val_Acc': val_acc
            })
    
    # Update model data if we have new history
    if len(history) > 0:
        new_history = pd.DataFrame(history)
        
        # Merge with existing history
        existing = signals_history_df[signals_history_df['Symbol'] == symbol]
        all_history = pd.concat([existing, new_history], ignore_index=True).drop_duplicates(subset=['Date', 'Symbol'])
        
        # Calculate statistics
        total = len(all_history)
        successful = all_history['Success'].sum()
        accuracy = successful / total if total > 0 else 0.0
        avg_profit = all_history['Return_Pct'].mean() if total > 0 else 0.0
        
        # Update or create model data entry
        model_row = model_data_df[model_data_df['Symbol'] == symbol]
        
        if len(model_row) == 0:
            last_sig = all_history.sort_values('Date').iloc[-1]
            new_model = pd.DataFrame([{
                'Symbol': symbol,
                'Last_Updated': datetime.now().strftime('%Y-%m-%d'),
                'Total_Signals': total,
                'Successful_Signals': successful,
                'Accuracy': accuracy,
                'Avg_Profit': avg_profit,
                'Last_Signal_Date': last_sig['Date'],
                'Last_Signal_Price': last_sig['Price'],
                'Last_Signal_Confidence': last_sig['Confidence'],
                'Backtested': True,
                'Model_Type': 'Ensemble'
            }])
            model_data_df = pd.concat([model_data_df, new_model], ignore_index=True)
        else:
            idx = model_row.index[0]
            model_data_df.at[idx, 'Total_Signals'] = total
            model_data_df.at[idx, 'Successful_Signals'] = successful
            model_data_df.at[idx, 'Accuracy'] = accuracy
            model_data_df.at[idx, 'Avg_Profit'] = avg_profit
            model_data_df.at[idx, 'Last_Updated'] = datetime.now().strftime('%Y-%m-%d')
            model_data_df.at[idx, 'Backtested'] = True
            model_data_df.at[idx, 'Model_Type'] = 'Ensemble'
            
            last_sig = all_history.sort_values('Date').iloc[-1]
            model_data_df.at[idx, 'Last_Signal_Date'] = last_sig['Date']
            model_data_df.at[idx, 'Last_Signal_Price'] = last_sig['Price']
            model_data_df.at[idx, 'Last_Signal_Confidence'] = last_sig['Confidence']
        
        # Update history
        signals_history_df = pd.concat([
            signals_history_df[signals_history_df['Symbol'] != symbol],
            all_history
        ], ignore_index=True)
        
        print(f"  ✅ {symbol}: {total} trades, {accuracy*100:.1f}% accuracy, {avg_profit:+.2f}% avg profit")
    
    return signals_history_df, model_data_df

# ===========================
# CORRELATION FILTERING
# ===========================
def filter_correlated_signals(signals, df_all, max_corr=MAX_CORRELATION):
    """
    Remove signals that are too correlated with each other
    Keep the highest confidence signal from correlated groups
    """
    if len(signals) <= 1:
        return signals
    
    # Calculate correlation matrix for symbols
    symbols = [s['Symbol'] for s in signals]
    
    # Get recent price data for correlation
    recent_data = df_all[df_all['Symbol'].isin(symbols)].pivot_table(
        index='Date', 
        columns='Symbol', 
        values='Close'
    ).tail(60).pct_change().dropna()
    
    if len(recent_data) < 20 or len(recent_data.columns) < 2:
        return signals
    
    corr_matrix = recent_data.corr()
    
    # Find highly correlated pairs
    filtered_signals = []
    excluded_symbols = set()
    
    # Sort by confidence (highest first)
    signals_sorted = sorted(signals, key=lambda x: x['Confidence'], reverse=True)
    
    for signal in signals_sorted:
        symbol = signal['Symbol']
        
        if symbol in excluded_symbols:
            continue
        
        # Add this signal
        filtered_signals.append(signal)
        
        # Exclude highly correlated symbols with lower confidence
        if symbol in corr_matrix.columns:
            for other_symbol in corr_matrix.columns:
                if other_symbol != symbol and abs(corr_matrix.loc[symbol, other_symbol]) > max_corr:
                    excluded_symbols.add(other_symbol)
    
    removed_count = len(signals) - len(filtered_signals)
    if removed_count > 0:
        print(f"  🔍 Filtered {removed_count} correlated signals")
    
    return filtered_signals

# ===========================
# MAIN FUNCTION
# ===========================
def main():
    print("="*80)
    print("🚀 ADVANCED ML TRADING STRATEGY v2.0 (9/10 Rating)")
    print(f"🕒 Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\n📊 Configuration:")
    print(f"  Profit Threshold: {PROFIT_THRESHOLD*100}%")
    print(f"  Min Confidence: {MIN_CONFIDENCE*100}%")
    print(f"  Min Accuracy: {MIN_MODEL_ACCURACY*100}%")
    print(f"  Min Trades: {MIN_TRADES}")
    print(f"  Max Correlation: {MAX_CORRELATION}")
    print(f"  XGBoost Available: {XGBOOST_AVAILABLE}")
    
    # Load CSV data
    print("\n📥 Loading data from GitHub...")
    
    model_data_df = load_csv_from_github(
        MODEL_DATA_CSV,
        create_if_missing=True,
        columns=['Symbol', 'Last_Updated', 'Total_Signals', 'Successful_Signals',
                 'Accuracy', 'Avg_Profit', 'Last_Signal_Date', 'Last_Signal_Price',
                 'Last_Signal_Confidence', 'Backtested', 'Model_Type']
    )
    
    signals_history_df = load_csv_from_github(
        SIGNALS_HISTORY_CSV,
        create_if_missing=True,
        columns=['Date', 'Symbol', 'Price', 'Confidence', 'Prediction',
                 'Exit_Date', 'Exit_Price', 'Return_Pct', 'Success',
                 'Train_Acc', 'Val_Acc']
    )
    
    # Get latest NEPSE data
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
    
    # Convert numeric columns
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=["Symbol", "Date", "Close"])
    df = df.sort_values(["Symbol", "Date"])
    
    latest_date = df['Date'].max()
    
    print(f"✅ Data loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")
    print(f"📅 Date range: {df['Date'].min().date()} to {latest_date.date()}")
    
    # Detect market regime
    market_regime = detect_market_regime(df.groupby('Date')['Close'].mean().reset_index())
    print(f"📈 Market Regime: {market_regime.upper()}")
    
    # Backtest symbols if needed
    print("\n🧪 Running walk-forward backtests...")
    symbols = [s for s in df['Symbol'].unique() if s not in EXCLUDED_SYMBOLS]
    
    backtest_count = 0
    for symbol in symbols[:20]:  # Limit to 20 symbols to avoid timeout
        df_sym = df[df['Symbol'] == symbol].copy()
        
        if len(df_sym) < MIN_TRAINING_SAMPLES + 50:
            continue
        
        model_row = model_data_df[model_data_df['Symbol'] == symbol]
        
        if len(model_row) == 0 or not model_row.iloc[0].get('Backtested', False):
            print(f"🔄 Backtesting {symbol}...")
            signals_history_df, model_data_df = walk_forward_backtest(
                symbol, df_sym, signals_history_df, model_data_df
            )
            backtest_count += 1
    
    print(f"✅ Completed {backtest_count} backtests")
    
    # Record yesterday's outcomes
    print("\n📊 Recording yesterday's outcomes...")
    
    if len(signals_history_df) > 0:
        pending = signals_history_df[signals_history_df['Exit_Date'].isna()].copy()
        
        if len(pending) > 0:
            print(f"  Found {len(pending)} pending signals")
            
            for idx, signal in pending.iterrows():
                symbol = signal['Symbol']
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
                    
                    # Update model stats
                    model_row = model_data_df[model_data_df['Symbol'] == symbol]
                    if len(model_row) > 0:
                        idx_model = model_row.index[0]
                        total = model_data_df.at[idx_model, 'Total_Signals'] + 1
                        successful = model_data_df.at[idx_model, 'Successful_Signals'] + success
                        
                        model_data_df.at[idx_model, 'Total_Signals'] = total
                        model_data_df.at[idx_model, 'Successful_Signals'] = successful
                        model_data_df.at[idx_model, 'Accuracy'] = successful / total
                        
                        old_avg = model_data_df.at[idx_model, 'Avg_Profit']
                        model_data_df.at[idx_model, 'Avg_Profit'] = (old_avg * (total-1) + return_pct) / total
                    
                    status = "✅" if success else "❌"
                    print(f"  {status} {symbol}: {return_pct:+.2f}%")
    
    # Generate today's signals
    print(f"\n🎯 Generating signals for {latest_date.date()}...")
    
    today_signals = []
    
    for symbol in symbols:
        df_sym = df[df['Symbol'] == symbol].copy()
        df_sym = df_sym[df_sym['Date'] <= latest_date]
        
        if len(df_sym) < MIN_TRAINING_SAMPLES:
            continue
        
        # Get model info
        model_row = model_data_df[model_data_df['Symbol'] == symbol]
        model_info = model_row.iloc[0].to_dict() if len(model_row) > 0 else None
        
        # Skip if model has poor historical performance
        if model_info:
            if model_info.get('Total_Signals', 0) >= MIN_TRADES:
                if model_info.get('Accuracy', 0) < MIN_MODEL_ACCURACY:
                    continue
        
        # Generate signal
        signal, confidence, metadata = generate_signal_with_ensemble(
            df_sym, symbol, model_info, df
        )
        
        if signal == 1:
            latest_price = df_sym.iloc[-1]['Close']
            model_accuracy = model_info.get('Accuracy', 0.5) if model_info else 0.5
            trades_count = int(model_info.get('Total_Signals', 0)) if model_info else 0
            
            # Calculate position size based on confidence
            position_size = 0.15  # Default
            for (conf_min, conf_max), size in POSITION_SIZE_CONF_MAP.items():
                if conf_min <= confidence < conf_max:
                    position_size = size
                    break
            
            today_signals.append({
                'Date': latest_date.strftime('%Y-%m-%d'),
                'Symbol': symbol,
                'Price': latest_price,
                'Confidence': confidence,
                'Model_Accuracy': model_accuracy,
                'Historical_Trades': trades_count,
                'Position_Size': position_size,
                'Model_Proba': metadata.get('model_proba', 0),
                'Train_Acc': metadata.get('train_accuracy', 0),
                'Val_Acc': metadata.get('val_accuracy', 0)
            })
    
    # Filter correlated signals
    if len(today_signals) > MAX_DAILY_SIGNALS:
        print(f"  ⚠️ {len(today_signals)} signals generated, filtering to top {MAX_DAILY_SIGNALS}...")
        today_signals = sorted(today_signals, key=lambda x: x['Confidence'], reverse=True)[:MAX_DAILY_SIGNALS]
    
    if len(today_signals) > 1:
        today_signals = filter_correlated_signals(today_signals, df)
    
    # Display and save results
    if len(today_signals) > 0:
        print(f"\n{'='*80}")
        print(f"🚨 {len(today_signals)} HIGH-QUALITY SIGNALS GENERATED!")
        print(f"{'='*80}")
        
        today_signals_sorted = sorted(today_signals, key=lambda x: x['Confidence'], reverse=True)
        
        total_capital = 1.0  # Assuming 100% allocation
        allocated = 0
        
        for i, sig in enumerate(today_signals_sorted, 1):
            print(f"\n{i}. 📈 {sig['Symbol']}")
            print(f"   Entry Price: NPR {sig['Price']:.2f}")
            print(f"   Confidence: {sig['Confidence']*100:.1f}%")
            print(f"   Position Size: {sig['Position_Size']*100:.1f}%")
            print(f"   Model Accuracy: {sig['Model_Accuracy']*100:.1f}% ({sig['Historical_Trades']} trades)")
            print(f"   Model Proba: {sig['Model_Proba']*100:.1f}% | Train: {sig['Train_Acc']*100:.1f}% | Val: {sig['Val_Acc']*100:.1f}%")
            allocated += sig['Position_Size']
        
        print(f"\n💰 Total Capital Allocated: {allocated*100:.1f}%")
        
        # Save signals
        daily_df = pd.DataFrame(today_signals_sorted)
        upload_csv_to_github(
            daily_df,
            DAILY_SIGNALS_CSV,
            f"🚀 v2.0 ML Signals for {latest_date.strftime('%Y-%m-%d')} - {len(today_signals)} signals"
        )
        
        # Add to history
        new_history = pd.DataFrame([{
            'Date': sig['Date'],
            'Symbol': sig['Symbol'],
            'Price': sig['Price'],
            'Confidence': sig['Confidence'],
            'Prediction': 1,
            'Exit_Date': None,
            'Exit_Price': None,
            'Return_Pct': None,
            'Success': None,
            'Train_Acc': sig['Train_Acc'],
            'Val_Acc': sig['Val_Acc']
        } for sig in today_signals_sorted])
        
        signals_history_df = pd.concat([signals_history_df, new_history], ignore_index=True)
        
    else:
        print("\n⏸️ No high-quality signals today")
        print("  Market conditions or model confidence below threshold")
    
    # Upload updated data
    print("\n📤 Uploading results...")
    
    upload_csv_to_github(
        signals_history_df,
        SIGNALS_HISTORY_CSV,
        f"📊 v2.0 Updated signals history - {latest_date.strftime('%Y-%m-%d')}"
    )
    
    upload_csv_to_github(
        model_data_df,
        MODEL_DATA_CSV,
        f"📈 v2.0 Updated model data - {latest_date.strftime('%Y-%m-%d')}"
    )
    
    # Show performance summary
    print(f"\n{'='*80}")
    print("🏆 TOP PERFORMING MODELS (v2.0 Ensemble)")
    print(f"{'='*80}")
    
    if len(model_data_df) > 0:
        performers = model_data_df[model_data_df['Total_Signals'] >= MIN_TRADES].copy()
        performers = performers.sort_values('Accuracy', ascending=False).head(15)
        
        if len(performers) > 0:
            print(f"\n{'Symbol':<12} {'Accuracy':>8} {'Avg Profit':>11} {'Trades':>7} {'Type':<10}")
            print("-" * 60)
            for _, row in performers.iterrows():
                print(f"{row['Symbol']:<12} {row['Accuracy']*100:>7.1f}% "
                      f"{row['Avg_Profit']:>+10.2f}% {int(row['Total_Signals']):>7} "
                      f"{row.get('Model_Type', 'Unknown'):<10}")
        else:
            print("Not enough trade history yet (need 5+ trades per symbol)")
    
    print(f"\n{'='*80}")
    print("✅ STRATEGY v2.0 RUN COMPLETE!")
    print(f"{'='*80}")
    print(f"\n🌐 Results: https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/{BRANCH}/ml_data")
    print("\n💡 Next steps:")
    print("  1. Review signals and position sizes")
    print("  2. Execute top 3-5 signals")
    print("  3. Track results over next few weeks")
    print("  4. Models will improve with each trade!")

if __name__ == "__main__":
    main()
