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
- **Date**: Date when the signal was generated
- **Close_at_Signal**: Closing price on the signal date (buy price)
- **Latest_Close**: Current closing price (sell price)
- **P/L_%**: Profit/Loss percentage (assuming buy at signal, sell at current price)
- **Status**: Either `STRICT_PASS`, `STANDARD_ONLY`, or `NO_SIGNALS`
- **Standard_Signals**: Total number of signals detected in standard mode for this symbol
- **Strict_Signals**: Total number of signals detected in strict mode for this symbol
- **Latest_RSI**: Current RSI value
- **NV_Range**: Range of Net Volume percentages across all signals
- **Recommended_NV_Threshold**: Suggested parameter value
- **Recommended_NV_Improvement**: Suggested parameter value
- **Recommended_Drawdown**: Suggested parameter value
- **Recommended_EMA50_Min**: Suggested parameter value
- **Recommended_EMA50_Max**: Suggested parameter value
- **Recommended_Volume_Ratio**: Suggested parameter value

**Note**: The CSV contains one row per signal, not per symbol. If a symbol has multiple signals, it will have multiple rows.

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

### Signal Types
- **STRICT_PASS**: Signal meets both standard AND strict mode criteria
- **STANDARD_ONLY**: Signal meets standard mode criteria but fails strict mode
- **NO_SIGNALS**: No signals detected for this symbol

### Profit/Loss Calculation
P/L% is calculated assuming:
- **Buy** at the signal date (Close_at_Signal)
- **Sell** at the current market price (Latest_Close)
- Formula: `((Latest_Close - Close_at_Signal) / Close_at_Signal) * 100`

### Interpreting Results

**Positive P/L%**: The signal would have been profitable if you bought at the signal and held until today

**Negative P/L%**: The signal would have resulted in a loss

**Multiple signals per symbol**: If a symbol generated signals on different dates, each signal is shown as a separate row with its own P/L calculation

### Example Usage

Once you have the results CSV, you can:

1. **Find Most Profitable Signals**: Sort by `P/L_%` to see which signals have performed best
2. **Find Symbols with Most Signals**: Group by `Symbol` and count signals
3. **Analyze Signal Quality**: Compare `STRICT_PASS` vs `STANDARD_ONLY` performance
4. **Filter by Date Range**: Focus on recent signals or historical performance

```python
# Example: Analyze results
import pandas as pd

results = pd.read_csv('SSHL_DIAGNOSTIC_RESULTS_2025-02-13.csv')

# Find most profitable signals
profitable = results[results['P/L_%'] > 0].sort_values('P/L_%', ascending=False)
print("Top 10 most profitable signals:")
print(profitable[['Symbol', 'Date', 'Close_at_Signal', 'Latest_Close', 'P/L_%']].head(10))

# Find symbols with most signals
symbol_counts = results[results['Status'] != 'NO_SIGNALS'].groupby('Symbol').size()
print("\nSymbols with most signals:")
print(symbol_counts.sort_values(ascending=False).head(10))

# Compare strict vs standard performance
strict_pl = results[results['Status'] == 'STRICT_PASS']['P/L_%'].mean()
standard_pl = results[results['Status'] == 'STANDARD_ONLY']['P/L_%'].mean()
print(f"\nAverage P/L for STRICT_PASS: {strict_pl:.2f}%")
print(f"Average P/L for STANDARD_ONLY: {standard_pl:.2f}%")

# Find recent signals (last 30 days)
results['Date'] = pd.to_datetime(results['Date'])
recent = results[results['Date'] >= results['Date'].max() - pd.Timedelta(days=30)]
print(f"\nSignals in last 30 days: {len(recent)}")
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
