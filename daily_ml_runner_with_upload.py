# -*- coding: utf-8 -*-
"""
GITHUB-NATIVE DAILY ML RUNNER
- Loads latest espen CSV
- Trains per-symbol RandomForest
- Stores models directly in GitHub (no local file dependency)
- Generates daily signals
- Uploads signals + model statistics
"""

import os
import re
import pickle
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import base64
import json

warnings.filterwarnings("ignore")

# ===========================
# CONFIG
# ===========================

REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"

SIGNALS_FOLDER = "ml_signals/daily"
MODEL_PATH = "ml_signals/adaptive_models.pkl"
STATS_FILE = "ml_signals/MODEL_STATISTICS.csv"

PROFIT_THRESHOLD = 0.02
MIN_CONFIDENCE = 0.65
MIN_TRAINING_SAMPLES = 50
LOOKBACK_DAYS = 365

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set")

HEADERS = {
    "Authorization": f"token {GH_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ===========================
# GITHUB HELPERS
# ===========================

def github_api_url(path):
    return f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"

def github_raw(path):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

def upload_binary_to_github(repo_path, binary_data, commit_message):
    content_base64 = base64.b64encode(binary_data).decode()

    check = requests.get(github_api_url(repo_path), headers=HEADERS, params={"ref": BRANCH})
    sha = check.json().get("sha") if check.status_code == 200 else None

    data = {
        "message": commit_message,
        "content": content_base64,
        "branch": BRANCH
    }
    if sha:
        data["sha"] = sha

    response = requests.put(github_api_url(repo_path),
                            headers=HEADERS,
                            data=json.dumps(data))

    return response.status_code in [200, 201]

def upload_csv_to_github(df, repo_path, commit_message):
    csv_bytes = df.to_csv(index=False).encode()
    return upload_binary_to_github(repo_path, csv_bytes, commit_message)

def load_model_from_github(repo_path):
    response = requests.get(github_api_url(repo_path), headers=HEADERS, params={"ref": BRANCH})
    if response.status_code != 200:
        print("🆕 No existing model in GitHub")
        return None

    content = response.json()["content"]
    binary = base64.b64decode(content)
    return pickle.loads(binary)

# ===========================
# FEATURE ENGINEERING
# ===========================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_features(df):
    features = pd.DataFrame(index=df.index)

    features["RSI"] = calculate_rsi(df["Close"], 14)
    features["Momentum_5"] = df["Close"].pct_change(5)
    features["Volatility_10"] = df["Close"].pct_change().rolling(10).std()
    features["EMA_10"] = df["Close"].ewm(span=10).mean()
    features["Close_to_EMA"] = (df["Close"] - features["EMA_10"]) / features["EMA_10"]

    return features

def create_target(df):
    return (df["Close"].shift(-1) / df["Close"] - 1 >= PROFIT_THRESHOLD).astype(int)

# ===========================
# MODEL CLASS
# ===========================

class SymbolModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.trained = False

    def train(self, X, y):
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < MIN_TRAINING_SAMPLES:
            return False

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True
        return True

    def predict(self, X):
        if not self.trained or X.isna().any().any():
            return None, 0

        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0][1]
        pred = 1 if prob >= MIN_CONFIDENCE else 0
        return pred, prob

# ===========================
# MAIN
# ===========================

def main():
    print("🚀 GitHub Native ML Runner")
    print("Run time:", datetime.now())

    # Load latest espen file
    root = requests.get(
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents",
        headers=HEADERS,
        params={"ref": BRANCH}
    ).json()

    espen_files = [f["name"] for f in root if f["name"].startswith("espen_")]
    latest_file = sorted(espen_files)[-1]

    print("📂 Using:", latest_file)

    csv_data = requests.get(github_raw(latest_file)).text
    df = pd.read_csv(StringIO(csv_data))

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])
    latest_date = df["Date"].max()

    # Load models from GitHub
    model_store = load_model_from_github(MODEL_PATH)
    models = model_store if model_store else {}

    signals = []

    for symbol in df["Symbol"].unique():
        df_symbol = df[df["Symbol"] == symbol].copy()

        features = create_features(df_symbol)
        target = create_target(df_symbol)

        X_train = features.iloc[:-1].tail(LOOKBACK_DAYS)
        y_train = target.iloc[:-1].tail(LOOKBACK_DAYS)

        model = models.get(symbol, SymbolModel())

        trained = model.train(X_train, y_train)

        if trained:
            today_features = features.iloc[[-1]]
            pred, prob = model.predict(today_features)

            if pred == 1:
                price = df_symbol.iloc[-1]["Close"]
                signals.append({
                    "Symbol": symbol,
                    "Date": latest_date,
                    "Price": price,
                    "Confidence_%": prob * 100
                })

            models[symbol] = model

    # Save models to GitHub
    binary = pickle.dumps(models)
    upload_binary_to_github(
        MODEL_PATH,
        binary,
        f"🧠 Model Update {latest_date.date()}"
    )

    print("🧠 Models saved to GitHub")

    # Save signals
    if signals:
        signals_df = pd.DataFrame(signals)
        date_str = latest_date.strftime("%Y%m%d")
        repo_path = f"{SIGNALS_FOLDER}/signals_{date_str}.csv"

        upload_csv_to_github(
            signals_df,
            repo_path,
            f"📈 Signals {latest_date.date()}"
        )

        print(f"🚨 {len(signals)} signals uploaded")

    else:
        print("⏸️ No signals today")

    print("✅ Done")

if __name__ == "__main__":
    main()
