# SSHL Multi-Symbol Diagnostic System

This system automatically analyzes Net Volume signals across all symbols in the Nepse market to diagnose why SSHL (Strict Signal High Loss) mode isn't finding signals.

## 🎯 What This Does

The diagnostic system:

1. **Loads Market Data**: Fetches the latest `espen_*.csv` file from your GitHub repository
2. **Analyzes All Symbols**: Processes each symbol (excluding bonds/debentures) to detect Net Volume signals
3. **Compares Modes**: Runs both Standard and Strict signal detection modes
4. **Identifies Issues**: Finds why Strict mode rejects signals that Standard mode finds
5. **Recommends Parameters**: Suggests optimal parameter values for each symbol
6. **Generates Report**: Uploads a comprehensive CSV with diagnostic results

## 📊 Output File

The system generates: `SSHL_DIAGNOSTIC_RESULTS_YYYY-MM-DD.csv`

### Columns:
- **Symbol**: Stock symbol
- **Status**: Either `SIGNALS_FOUND` or `NO_SIGNALS`
- **Standard_Signals**: Number of signals detected in standard mode
- **Strict_Signals**: Number of signals detected in strict mode
- **Latest_Close**: Current closing price
- **Latest_RSI**: Current RSI value
- **NV_Range**: Range of Net Volume percentages
- **Recommended_NV_Threshold**: Suggested parameter value
- **Recommended_NV_Improvement**: Suggested parameter value
- **Recommended_Drawdown**: Suggested parameter value
- **Recommended_EMA50_Min**: Suggested parameter value
- **Recommended_EMA50_Max**: Suggested parameter value
- **Recommended_Volume_Ratio**: Suggested parameter value

## 🚀 Setup Instructions

### Step 1: Add the Files to Your Repository

1. Copy `sshl_diagnostics.py` to the root of your repository
2. Create the `.github/workflows/` directory if it doesn't exist
3. Copy `sshl_diagnostics.yml` to `.github/workflows/`

### Step 2: Verify GitHub Token

Make sure you have a GitHub Personal Access Token set up:

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` scope (full control of private repositories)
3. Go to your repository → Settings → Secrets and variables → Actions
4. Add a new repository secret named `GH_TOKEN` with your token value

### Step 3: Enable GitHub Actions

1. Go to your repository → Actions tab
2. Enable workflows if prompted
3. The workflow will now run automatically every day at 6:30 PM Nepal Time

## 📅 Schedule

The workflow runs automatically:
- **Time**: 6:30 PM Nepal Time (12:45 UTC) daily
- **Trigger**: Can also be run manually from the Actions tab

## 🔧 Manual Execution

To run the diagnostic manually:

1. Go to your repository on GitHub
2. Click the "Actions" tab
3. Select "SSHL Diagnostics - Multi-Symbol Analysis" from the left sidebar
4. Click "Run workflow" button
5. Click the green "Run workflow" button in the dropdown

## 📈 Understanding the Results

### High Standard Signals, Zero Strict Signals
This indicates the strict parameters are too restrictive. Use the recommended parameter values to adjust.

### No Signals in Both Modes
This suggests:
- Insufficient data (less than 100 trading days)
- The symbol doesn't match the signal criteria
- Data quality issues

### Example Usage

Once you have the results CSV, you can:

1. **Find Best Candidates**: Sort by `Standard_Signals` to find stocks with most potential
2. **Adjust Parameters**: Use the `Recommended_*` columns to create custom parameters
3. **Focus on Specific Symbols**: Filter by specific symbols you're interested in

```python
# Example: Use recommended parameters for a specific symbol
import pandas as pd

results = pd.read_csv('SSHL_DIAGNOSTIC_RESULTS_2025-02-13.csv')
symbol_data = results[results['Symbol'] == 'NABBC'].iloc[0]

custom_params = {
    'nv_3m_low_pct': symbol_data['Recommended_NV_Threshold'],
    'nv_improvement_pct': symbol_data['Recommended_NV_Improvement'],
    'drawdown_min': symbol_data['Recommended_Drawdown'],
    'ema50_min': symbol_data['Recommended_EMA50_Min'],
    'ema50_max': symbol_data['Recommended_EMA50_Max'],
    'volume_ratio_max': symbol_data['Recommended_Volume_Ratio']
}
```

## 🛠️ Customization

### Exclude Additional Symbols

Edit the `EXCLUDED_SYMBOLS` list in `sshl_diagnostics.py`:

```python
EXCLUDED_SYMBOLS = [
    "EBLD852", "EB89", # ... add more symbols here
]
```

### Change Schedule

Edit the cron expression in `.github/workflows/sshl_diagnostics.yml`:

```yaml
schedule:
  - cron: '45 12 * * *'  # Change this line
```

Use [crontab.guru](https://crontab.guru/) to generate cron expressions.

### Adjust Signal Detection Criteria

Modify the `detect_standard_signals()` and `detect_strict_signals()` functions in `sshl_diagnostics.py` to change the signal detection logic.

## 📝 Notes

- The system automatically deletes old diagnostic files to keep the repository clean
- Results are sorted by number of standard signals (descending)
- Only symbols with at least 100 trading days are analyzed
- The analysis uses the most recent `espen_*.csv` file available

## 🐛 Troubleshooting

### Workflow Fails with Authentication Error
- Check that `GH_TOKEN` secret is properly set in repository settings
- Verify the token has `repo` scope permissions

### No Results Generated
- Check the Actions log for error messages
- Verify that `espen_*.csv` file exists in the repository
- Ensure the CSV has the required columns: Symbol, Date, Open, High, Low, Close, Volume

### Incorrect Date Format
- The script supports both `M/D/YYYY` and `YYYY-MM-DD` formats
- Check the Actions log to see which format was detected

## 📧 Support

For issues or questions, please open an issue in the GitHub repository.

## 📜 License

This project follows the same license as the parent repository.
