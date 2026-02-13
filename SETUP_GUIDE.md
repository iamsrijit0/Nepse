# 🚀 Quick Setup Guide for SSHL Diagnostics

Follow these steps to get the SSHL diagnostic system running on your GitHub repository.

## ✅ Step-by-Step Setup

### 1. Create GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name like "SSHL Diagnostics Token"
4. Select the following scopes:
   - ✅ **repo** (all sub-options)
   - ✅ **workflow** (to update workflows)
5. Click "Generate token"
6. **IMPORTANT**: Copy the token immediately - you won't see it again!

### 2. Add Token to Your Repository

1. Go to your repository: https://github.com/iamsrijit0/Nepse
2. Click "Settings" tab
3. In the left sidebar, click "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Name: `GH_TOKEN`
6. Value: Paste your token from Step 1
7. Click "Add secret"

### 3. Upload Files to Your Repository

You need to add these files to your repository:

```
your-repo/
├── sshl_diagnostics.py          # Main diagnostic script
├── config_sshl.py                # Configuration file
├── requirements.txt              # Python dependencies
├── README_SSHL_DIAGNOSTICS.md   # Documentation
└── .github/
    └── workflows/
        └── sshl_diagnostics.yml  # GitHub Actions workflow
```

#### Option A: Upload via GitHub Web Interface

1. Go to your repository on GitHub
2. Click "Add file" → "Upload files"
3. Drag and drop the files (create `.github/workflows/` folder first)
4. Commit the changes

#### Option B: Upload via Git Command Line

```bash
# Clone your repository
git clone https://github.com/iamsrijit0/Nepse.git
cd Nepse

# Create workflow directory
mkdir -p .github/workflows

# Copy files (assuming files are in ~/Downloads/)
cp ~/Downloads/sshl_diagnostics.py .
cp ~/Downloads/config_sshl.py .
cp ~/Downloads/requirements.txt .
cp ~/Downloads/README_SSHL_DIAGNOSTICS.md .
cp ~/Downloads/sshl_diagnostics.yml .github/workflows/

# Commit and push
git add .
git commit -m "Add SSHL diagnostic system"
git push
```

### 4. Verify the Setup

1. Go to your repository
2. Click the "Actions" tab
3. You should see "SSHL Diagnostics - Multi-Symbol Analysis" in the left sidebar
4. If you see a message about workflows being disabled, click "I understand my workflows, go ahead and enable them"

### 5. Run Your First Diagnostic

#### Option A: Wait for Automatic Run
The workflow will automatically run every day at 6:30 PM Nepal Time.

#### Option B: Run Manually (Recommended for First Time)

1. Go to "Actions" tab
2. Click "SSHL Diagnostics - Multi-Symbol Analysis"
3. Click "Run workflow" button (top right)
4. Select "Branch: main"
5. Click the green "Run workflow" button

### 6. View Results

1. Wait for the workflow to complete (usually 2-5 minutes)
2. Go back to your repository's main page
3. Look for a new file: `SSHL_DIAGNOSTIC_RESULTS_YYYY-MM-DD.csv`
4. Click on the file to view or download it

## 🎯 What Happens Next?

Once setup is complete:

- ✅ Diagnostics run automatically every day at 6:30 PM Nepal Time
- ✅ Results are saved as CSV files in your repository
- ✅ Old files are automatically deleted to keep things clean
- ✅ You can run diagnostics manually anytime from the Actions tab

## 📊 Understanding Your Results

The CSV file contains these columns:

| Column | Description |
|--------|-------------|
| **Symbol** | Stock ticker symbol |
| **Status** | SIGNALS_FOUND or NO_SIGNALS |
| **Standard_Signals** | Number of signals in relaxed mode |
| **Strict_Signals** | Number of signals in strict mode |
| **Latest_Close** | Current stock price |
| **Latest_RSI** | Current RSI value |
| **Recommended_*** | Suggested parameter values |

### Example: Finding Good Opportunities

```python
import pandas as pd

# Load results
df = pd.read_csv('SSHL_DIAGNOSTIC_RESULTS_2025-02-13.csv')

# Find stocks with signals
with_signals = df[df['Status'] == 'SIGNALS_FOUND']

# Sort by number of signals
best_opportunities = with_signals.sort_values('Standard_Signals', ascending=False)

# View top 10
print(best_opportunities.head(10))
```

## 🔧 Customization

### Change Analysis Schedule

Edit `.github/workflows/sshl_diagnostics.yml`:

```yaml
schedule:
  # Run at 6:30 PM Nepal Time (12:45 UTC)
  - cron: '45 12 * * *'
  
  # Examples:
  # Every 6 hours: '0 */6 * * *'
  # Twice daily (9 AM and 6 PM Nepal): '15 3,12 * * *'
  # Only on weekdays: '45 12 * * 1-5'
```

### Adjust Signal Parameters

Edit `config_sshl.py`:

```python
STRICT_PARAMS = {
    'nv_3m_low_pct': 10,        # Change this
    'nv_improvement_pct': 30,   # Change this
    # ... etc
}
```

### Exclude Additional Symbols

Add to `EXCLUDED_SYMBOLS` list in `config_sshl.py`:

```python
EXCLUDED_SYMBOLS = [
    "EBLD852", 
    "EB89",
    "YOUR_SYMBOL_HERE",  # Add your symbols
]
```

## ⚠️ Troubleshooting

### Problem: Workflow doesn't appear in Actions tab

**Solution**: 
- Make sure the workflow file is in `.github/workflows/` directory
- Check file name is `sshl_diagnostics.yml`
- Enable workflows if prompted

### Problem: Workflow fails with "Authentication failed"

**Solution**:
- Verify `GH_TOKEN` secret is set correctly
- Make sure token has `repo` scope
- Token might have expired - generate a new one

### Problem: No CSV file generated

**Solution**:
1. Click on the failed workflow run
2. Click on "run-sshl-diagnostics" job
3. Read the error message
4. Common causes:
   - Missing `espen_*.csv` file in repository
   - Incorrect CSV format
   - Python dependency issues

### Problem: CSV is empty or has no signals

**Solution**:
- This is normal if no symbols meet the criteria
- Try adjusting parameters in `config_sshl.py`
- Check that `espen_*.csv` has recent data

## 📞 Getting Help

If you encounter issues:

1. Check the Actions log for detailed error messages
2. Verify all files are in the correct locations
3. Make sure your `espen_*.csv` file has these columns:
   - Symbol, Date, Open, High, Low, Close, Volume
4. Open an issue on GitHub with:
   - The error message from Actions log
   - Your workflow file contents
   - Sample data from your CSV

## 🎉 You're All Set!

Your SSHL diagnostic system is now running. Check back daily for new results, or run it manually whenever you want to analyze the market.

**Happy Trading! 📈**
