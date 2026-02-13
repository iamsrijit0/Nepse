# SSHL Diagnostics Configuration File
# Customize these parameters without modifying the main script

# =======================
# STRICT MODE PARAMETERS
# =======================
# Default parameters for strict signal detection
STRICT_PARAMS = {
    # Net Volume must be within this % of 3-month low
    'nv_3m_low_pct': 10,
    
    # Net Volume must improve by at least this %
    'nv_improvement_pct': 30,
    
    # Drawdown must be at least this severe (negative %)
    'drawdown_min': -15,
    
    # Price must be between these % below EMA 50
    'ema50_min': -25,
    'ema50_max': -15,
    
    # Volume ratio must be below this threshold
    'volume_ratio_max': 0.5
}

# =======================
# STANDARD MODE PARAMETERS
# =======================
# Looser criteria for standard signal detection
STANDARD_PARAMS = {
    'min_rsi': 40,           # RSI must be below this
    'min_drawdown': -10,     # Minimum drawdown (negative %)
    'min_nv_improvement': True  # Net Volume must be improving
}

# =======================
# DATA REQUIREMENTS
# =======================
# Minimum number of trading days required for analysis
MIN_TRADING_DAYS = 100

# Minimum number of days for calculating rolling metrics
MIN_ROLLING_WINDOW = 50

# =======================
# ANALYSIS SETTINGS
# =======================
# Number of top symbols to display in summary
TOP_N_SYMBOLS = 10

# Date range for analysis (set to None to use all available data)
ANALYSIS_START_DATE = None  # Example: '2024-01-01'
ANALYSIS_END_DATE = None     # Example: '2025-12-31'

# =======================
# OUTPUT SETTINGS
# =======================
# Include detailed signal dates in output
INCLUDE_SIGNAL_DATES = False

# Include individual metric values for each signal
INCLUDE_SIGNAL_METRICS = True

# Maximum number of signals to detail per symbol
MAX_SIGNALS_PER_SYMBOL = 5

# =======================
# TECHNICAL INDICATORS
# =======================
# EMA periods
EMA_SHORT_PERIOD = 9
EMA_MEDIUM_PERIOD = 21
EMA_LONG_PERIOD = 50

# SMA period for 200-day moving average
SMA_LONG_PERIOD = 200

# RSI period
RSI_PERIOD = 14

# Volume SMA period
VOLUME_SMA_PERIOD = 20

# Bollinger Bands settings
BB_PERIOD = 20
BB_STD_DEV = 2

# 52-week high/low period (in trading days)
PERIOD_52_WEEK = 252

# =======================
# GITHUB SETTINGS
# =======================
# Repository configuration
REPO_OWNER = "iamsrijit0"
REPO_NAME = "Nepse"
BRANCH = "main"

# File naming patterns
ESPEN_FILE_PREFIX = "espen_"
OUTPUT_FILE_PREFIX = "SSHL_DIAGNOSTIC_RESULTS_"

# =======================
# SYMBOLS TO EXCLUDE
# =======================
# Add symbols here that should be excluded from analysis
# (bonds, debentures, mutual funds, etc.)
EXCLUDED_SYMBOLS = [
    # Bonds and Debentures
    "EBLD852", "EB89", "NABILD2089", "MBLD2085", "SBID89", "SBID2090",
    "SBLD2091", "NIMBD90", "RBBD2088", "CCBD88", "ICFCD88", "EBLD91",
    "GBILD84/85", "GBILD86/87", "NICD88", "CCBD88", "EBLD85", "EBLD91",
    "NMBD87/88", "PBD84", "NICAD85/86", "SBD87", "NBBD2085", "NBLD82",
    "NIBD84", "BOKD86KA", "NCCD86", "EBLEB89", "SBIBD86", "KSBBLD87",
    "MLBLD89", "NIFRAGED", "BOKD86", "PBLD87", "NMBD2085", "NBLD87",
    "MBLD87", "SDBD87", "SBD89", "PBD88", "CBLD88", "KBLD89",
    "NMBD89/90", "LBBLD89", "NABILD87", "CIZBD90", "LBLD88", "SBLD89",
    "KBLD86", "KBLD90", "GWFD83", "JBBD87", "HBLD83", "ICFCD83",
    "ADBLD83", "GILB", "NIBD2082", "RBBD83", "SRBLD83", "SBID83",
    "RBBF40", "PBLD84", "GBBD85", "HBLD86", "SAND2085", "PBLD86",
    "NICAD2091", "CIZBD86",
    
    # Mutual Funds
    "CMF2", "GBIMESY2", "GIBF1", "GSY", "H8020", "HLICF", "KDBY",
    "KEF", "KSY", "LUK", "LVF2", "MBLEF", "MMF1", "MNMF1", "NBF2",
    "NBF3", "NIBLGF", "NIBLSTF", "NIBSF2", "NICBF", "NICFC", "NICGF2",
    "NICSF", "NMB50", "NMBHF2", "NSIF2", "PRSF", "PSF", "RMF1", "RMF2",
    "RSY", "SAGF", "SBCF", "SEF", "SFEF", "SIGS2", "SIGS3", "SLCF",
    "C30MF", "NMBMF",
    
    # Preference Shares
    "HEIP", "HIDCLP", "NIMBPO", "NLICLP", "RBCLPO", "PCBLP", "CZBILP",
    "NLICP", "KBLPO", "JBLBP", "KMCDB", "MLBLPO", "PROFLP",
    
    # Others
    "ENL", "HATHY", "HIDCL"
]

# =======================
# ADVANCED SETTINGS
# =======================
# Enable verbose logging
VERBOSE = True

# Generate individual symbol reports (separate CSV per symbol)
GENERATE_INDIVIDUAL_REPORTS = False

# Compare against historical performance
ENABLE_HISTORICAL_COMPARISON = False

# Calculate confidence scores for recommendations
CALCULATE_CONFIDENCE_SCORES = True

# Use percentile-based parameter recommendations
USE_PERCENTILE_RECOMMENDATIONS = True
PERCENTILE_VALUE = 75  # Use 75th percentile for recommended parameters
