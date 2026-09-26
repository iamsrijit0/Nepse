import subprocess
import sys
import os
import requests
import base64
import re
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup

# ── Install required packages ────────────────────────────────────────────────
packages = [
    "nepse-scraper",
    "xlsxwriter",
    "gitpython",
    "pandas",
    "matplotlib",
    "joblib"
]

subprocess.check_call(
    [sys.executable, "-m", "pip", "install"] + packages,
    stdout=subprocess.DEVNULL
)

from nepse_scraper import Nepse_scraper
from joblib import Parallel, delayed


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

STANDARD_COLS = [
    'Symbol',
    'Date',
    'Open',
    'High',
    'Low',
    'Close',
    'Percent Change',
    'Volume',
    '52High',
    '52Low'
]

GITHUB_REPO = 'iamsrijit0/Nepse'
GH_TOKEN = os.getenv("GH_TOKEN")


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def format_date(dt):
    """Return M/D/YYYY with no leading zeros."""
    return f"{dt.month}/{dt.day}/{dt.year}"


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 – FETCH TODAY'S PRICE FROM NEPSE API
# ════════════════════════════════════════════════════════════════════════════

print("Fetching today's price from NEPSE ...")

request_obj = Nepse_scraper(verify_ssl=False)
today_price = request_obj.get_today_price()

content_data = (
    today_price.get('content', [])
    if isinstance(today_price, dict)
    else today_price
)

filtered_data = []

for item in content_data:

    symbol = item.get('symbol', '')
    date = item.get('businessDate', '')

    open_price = to_float(item.get('openPrice'))
    high_price = to_float(item.get('highPrice'))
    low_price = to_float(item.get('lowPrice'))
    close_price = to_float(item.get('closePrice'))

    volume = to_float(
        item.get('totalTradedQuantity')
    )

    high52 = to_float(
        item.get('fiftyTwoWeekHigh')
    )

    low52 = to_float(
        item.get('fiftyTwoWeekLow')
    )

    pct_chg = round(
        ((close_price - open_price) / open_price * 100)
        if open_price > 0 else 0,
        2
    )

    filtered_data.append({

        'Symbol': symbol,
        'Date': date,
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Percent Change': pct_chg,
        'Volume': volume,
        '52High': high52,
        '52Low': low52

    })


first = pd.DataFrame(filtered_data)

first['Date'] = pd.to_datetime(
    first['Date'],
    errors='coerce'
)


# ════════════════════════════════════════════════════════════════════════════
# LIVE 52 WEEK HIGH / LOW
# ════════════════════════════════════════════════════════════════════════════

live_52 = (

    first
    .dropna(subset=['Date'])
    .sort_values('Date', ascending=False)
    .drop_duplicates('Symbol')
    .set_index('Symbol')[['52High', '52Low']]

)


if first.empty:

    print(
        "No data available from NEPSE API today."
    )

    raise SystemExit(1)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 – GET LATEST HISTORICAL ESPEN FILE
# ════════════════════════════════════════════════════════════════════════════

def get_latest_espen_url(repo_url: str) -> str:

    response = requests.get(
        repo_url,
        timeout=15
    )

    soup = BeautifulSoup(
        response.content,
        'html.parser'
    )

    file_urls = {}

    for link in soup.find_all('a', href=True):

        href = link['href']

        if (
            'espen_' in href
            and href.endswith('.csv')
        ):

            date_match = re.search(
                r'(\d{4}-\d{2}-\d{2})',
                href
            )

            if date_match:

                file_date = date_match.group(1)

                raw_url = (
                    f"https://raw.githubusercontent.com/"
                    f"{GITHUB_REPO}/main/"
                    f"{href.split('/')[-1]}"
                )

                file_urls[file_date] = raw_url


    if not file_urls:

        raise ValueError(
            "No espen_ CSV files found in repository."
        )


    latest = max(file_urls.keys())

    print(
        f"Latest espen_ file date: {latest}"
    )

    print(
        f"URL: {file_urls[latest]}"
    )

    return file_urls[latest]


secondss = pd.DataFrame()

try:

    latest_url = get_latest_espen_url(
        f'https://github.com/{GITHUB_REPO}/tree/main'
    )

    raw = pd.read_csv(latest_url)

    # Keep only standard columns
    available = [
        c for c in STANDARD_COLS
        if c in raw.columns
    ]

    secondss = raw[available].copy()

    # Add missing columns
    for col in STANDARD_COLS:

        if col not in secondss.columns:
            secondss[col] = np.nan


    secondss = secondss[STANDARD_COLS]

    secondss['Date'] = pd.to_datetime(
        secondss['Date'],
        errors='coerce'
    )

    print(
        f"Historical data loaded: "
        f"{len(secondss)} rows"
    )

except Exception as e:

    print(
        f"Could not load historical data: {e}"
    )


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 – MERGE TODAY + HISTORY
# ════════════════════════════════════════════════════════════════════════════

dfs = [
    df
    for df in [first, secondss]
    if not df.empty
]

if not dfs:

    print(
        "No data to process."
    )

    raise SystemExit(1)


combined_df = pd.concat(
    dfs,
    ignore_index=True,
    join='outer'
)


combined_df['Date'] = pd.to_datetime(
    combined_df['Date'],
    errors='coerce'
)

combined_df.dropna(
    subset=['Date'],
    inplace=True
)


finall_df = pd.DataFrame()


for symbol, grp in combined_df.groupby('Symbol'):

    grp = (
        grp
        .sort_values('Date', ascending=False)
        .drop_duplicates(
            'Date',
            keep='first'
        )
    )


    # Use latest live 52-week values
    if symbol in live_52.index:

        grp['52High'] = live_52.loc[
            symbol,
            '52High'
        ]

        grp['52Low'] = live_52.loc[
            symbol,
            '52Low'
        ]


    finall_df = pd.concat(
        [finall_df, grp],
        ignore_index=True
    )


# Format date
finall_df['Date'] = finall_df['Date'].apply(
    format_date
)


# Ensure column order
finall_df = finall_df[
    STANDARD_COLS
]


print(
    f"\nCombined data ready "
    f"({len(finall_df)} rows)"
)

print(
    finall_df.head(10).to_string(
        index=False
    )
)


# ════════════════════════════════════════════════════════════════════════════
# GITHUB UPLOAD FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def github_put(
    file_name: str,
    df: pd.DataFrame
):

    csv_b64 = base64.b64encode(
        df.to_csv(index=False).encode()
    ).decode()

    url = (
        f'https://api.github.com/repos/'
        f'{GITHUB_REPO}/contents/{file_name}'
    )

    headers = {
        'Authorization': f'token {GH_TOKEN}'
    }

    # Check if file already exists
    r = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    sha = (
        r.json().get('sha')
        if r.status_code == 200
        else None
    )

    payload = {

        'message': f'Add {file_name}',
        'content': csv_b64,
        'branch': 'main'

    }

    if sha:
        payload['sha'] = sha


    resp = requests.put(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )


    if resp.status_code in (200, 201):

        print(
            f"Uploaded '{file_name}' "
            f"successfully."
        )

    else:

        print(
            f"Upload failed "
            f"({resp.status_code}): "
            f"{resp.text}"
        )


# ════════════════════════════════════════════════════════════════════════════
# DELETE OLD GITHUB FILES
# ════════════════════════════════════════════════════════════════════════════

def delete_old_github_files(
    prefix: str,
    keep: int = 1
):

    headers = {
        'Authorization': f'token {GH_TOKEN}'
    }

    list_url = (
        f'https://api.github.com/repos/'
        f'{GITHUB_REPO}/contents/'
    )

    r = requests.get(
        list_url,
        headers=headers,
        timeout=15
    )


    if r.status_code != 200:

        print(
            f"Could not list repo contents: "
            f"{r.status_code}"
        )

        return


    all_files = r.json()


    matched = sorted(

        [
            f
            for f in all_files

            if (
                isinstance(f, dict)
                and f.get('name', '').startswith(prefix)
                and f.get('name', '').endswith('.csv')
            )
        ],

        key=lambda f: f['name'],
        reverse=True

    )


    to_delete = matched[keep:]


    for f in to_delete:

        del_url = f['url']
        sha = f['sha']

        payload = {

            'message':
                f'Auto-cleanup: remove {f["name"]}',

            'sha': sha,

            'branch': 'main'

        }


        dr = requests.delete(
            del_url,
            headers=headers,
            json=payload,
            timeout=15
        )


        if dr.status_code == 200:

            print(
                f"Deleted old file: "
                f"{f['name']}"
            )

        else:

            print(
                f"Failed to delete "
                f"{f['name']}: "
                f"{dr.status_code} "
                f"{dr.text}"
            )


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 – UPLOAD DAILY HISTORICAL FILE
# ════════════════════════════════════════════════════════════════════════════

today_str = datetime.today().strftime(
    '%Y-%m-%d'
)


github_put(
    f'espen_{today_str}.csv',
    finall_df
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 – DAILY EMA 20 / EMA 50 CROSSOVER
# ════════════════════════════════════════════════════════════════════════════

print(
    "\nRunning DAILY EMA 20 / EMA 50 analysis ..."
)


daily_data = finall_df.copy()


daily_data['Date'] = pd.to_datetime(
    daily_data['Date'],
    errors='coerce'
)


daily_data['Close'] = pd.to_numeric(

    daily_data['Close']
    .astype(str)
    .str.replace(',', ''),

    errors='coerce'

)


daily_data['Volume'] = pd.to_numeric(

    daily_data['Volume']
    .astype(str)
    .str.replace(',', '')
    .replace('-', np.nan),

    errors='coerce'

)


daily_data = daily_data.dropna(
    subset=['Symbol', 'Date', 'Close']
)


daily_data = daily_data.sort_values(
    ['Symbol', 'Date']
).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# DAILY PROCESSING
# ────────────────────────────────────────────────────────────────────────────

def process_daily_symbol(
    symbol,
    group
):

    group = group.copy()

    group = group.sort_values(
        'Date'
    ).reset_index(drop=True)


    # EMA 20
    group['EMA_20'] = (
        group['Close']
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )


    # EMA 50
    group['EMA_50'] = (
        group['Close']
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )


    # RSI
    delta = group['Close'].diff()

    gain = (
        delta
        .where(delta > 0, 0)
        .rolling(
            14,
            min_periods=1
        )
        .mean()
    )

    loss = (
        -delta
        .where(delta < 0, 0)
        .rolling(
            14,
            min_periods=1
        )
        .mean()
    )

    rs = gain / loss

    group['RSI'] = (
        100 -
        (100 / (1 + rs))
    )


    # 30-day average volume
    group['30D_Avg_Volume'] = (
        group['Volume']
        .rolling(
            30,
            min_periods=1
        )
        .mean()
    )


    # EMA slopes
    group['Slope_20'] = (
        group['EMA_20'].diff(5) / 5
    )

    group['Slope_50'] = (
        group['EMA_50'].diff(5) / 5
    )


    # Bullish crossover
    group['Crossover'] = (

        (group['EMA_20'] > group['EMA_50'])

        &

        (
            group['EMA_20'].shift(1)
            <=
            group['EMA_50'].shift(1)
        )

    )


    # Existing filters
    valid = group[

        group['Crossover']

        &

        (
            group['Slope_20']
            >
            group['Slope_50']
        )

        &

        group['RSI'].between(
            30,
            70
        )

        &

        (
            group['Volume']
            >=
            0.3 *
            group['30D_Avg_Volume']
        )

        &

        (
            group['Close']
            >
            group['Close']
            .rolling(
                60,
                min_periods=1
            )
            .max()
            .shift(1)
            * 0.95
        )

    ]


    return valid


daily_results = Parallel(
    n_jobs=-1
)(
    delayed(process_daily_symbol)(
        sym,
        grp
    )

    for sym, grp
    in daily_data.groupby('Symbol')
)


daily_results = [
    x
    for x in daily_results
    if not x.empty
]


if daily_results:

    all_valid = pd.concat(
        daily_results,
        ignore_index=True
    )

    daily_crossovers = (

        all_valid
        .sort_values(
            'Date',
            ascending=False
        )
        .drop_duplicates(
            'Symbol'
        )
        .reset_index(drop=True)

    )

else:

    daily_crossovers = pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# DISPLAY DAILY SIGNALS
# ════════════════════════════════════════════════════════════════════════════

print(
    "\n" +
    "=" * 80
)

print(
    "LATEST DAILY EMA 20 / EMA 50 CROSSOVERS"
)

print(
    "=" * 80
)


if daily_crossovers.empty:

    print(
        "No daily EMA 20 / EMA 50 crossover found."
    )

else:

    print(

        daily_crossovers[
            [
                'Symbol',
                'Date',
                'Close',
                'EMA_20',
                'EMA_50',
                'RSI'
            ]
        ]
        .to_string(index=False)

    )


# ════════════════════════════════════════════════════════════════════════════
# UPLOAD DAILY EMA FILE
# ════════════════════════════════════════════════════════════════════════════

daily_file_name = (
    f'EMA_Cross_for_{today_str}.csv'
)


github_put(
    daily_file_name,
    daily_crossovers
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 – CREATE MONTHLY DATA
# ════════════════════════════════════════════════════════════════════════════

print(
    "\nCreating MONTHLY OHLC data ..."
)


monthly_source = finall_df.copy()


monthly_source['Date'] = pd.to_datetime(
    monthly_source['Date'],
    errors='coerce'
)


for col in [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume'
]:

    monthly_source[col] = pd.to_numeric(

        monthly_source[col]
        .astype(str)
        .str.replace(',', '')
        .replace('-', np.nan),

        errors='coerce'

    )


monthly_source = monthly_source.dropna(
    subset=[
        'Symbol',
        'Date',
        'Close'
    ]
)


monthly_source = monthly_source.sort_values(
    ['Symbol', 'Date']
)


# ════════════════════════════════════════════════════════════════════════════
# BUILD MONTHLY OHLCV
# ════════════════════════════════════════════════════════════════════════════

def make_monthly_data(group):

    group = group.sort_values(
        'Date'
    ).copy()


    group = group.set_index(
        'Date'
    )


    monthly = group.resample(
        'ME'
    ).agg({

        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'

    })


    monthly['Symbol'] = (
        group['Symbol'].iloc[0]
    )


    return monthly.reset_index()


monthly_results = Parallel(
    n_jobs=-1
)(
    delayed(make_monthly_data)(group)

    for symbol, group
    in monthly_source.groupby('Symbol')
)


monthly_df = pd.concat(
    monthly_results,
    ignore_index=True
)


monthly_df = monthly_df.sort_values(
    ['Symbol', 'Date']
).reset_index(drop=True)


print(
    f"Monthly records created: "
    f"{len(monthly_df)}"
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 – MONTHLY EMA 10 / EMA 25
# ════════════════════════════════════════════════════════════════════════════

print(
    "\nRunning MONTHLY EMA 10 / EMA 25 analysis ..."
)


def process_monthly_symbol(
    symbol,
    group
):

    group = group.copy()

    group = group.sort_values(
        'Date'
    ).reset_index(drop=True)


    # EMA 10
    group['EMA_10'] = (

        group['Close']
        .ewm(
            span=10,
            adjust=False
        )
        .mean()

    )


    # EMA 25
    group['EMA_25'] = (

        group['Close']
        .ewm(
            span=25,
            adjust=False
        )
        .mean()

    )


    # EMA difference
    group['EMA_Difference'] = (

        group['EMA_10']
        -
        group['EMA_25']

    )


    # Bullish monthly crossover
    group['Monthly_Crossover'] = (

        (group['EMA_10'] > group['EMA_25'])

        &

        (
            group['EMA_10'].shift(1)
            <=
            group['EMA_25'].shift(1)
        )

    )


    # Percentage gap
    group['EMA_Gap_%'] = (

        (
            group['EMA_10']
            -
            group['EMA_25']
        )

        /

        group['EMA_25']

        *

        100

    )


    valid = group[
        group['Monthly_Crossover']
    ].copy()


    return valid


monthly_results = Parallel(
    n_jobs=-1
)(
    delayed(process_monthly_symbol)(
        sym,
        grp
    )

    for sym, grp
    in monthly_df.groupby('Symbol')
)


monthly_results = [
    x
    for x in monthly_results
    if not x.empty
]


if monthly_results:

    monthly_crossovers = pd.concat(
        monthly_results,
        ignore_index=True
    )


    monthly_crossovers = (

        monthly_crossovers
        .sort_values(
            'Date',
            ascending=False
        )
        .drop_duplicates(
            'Symbol'
        )
        .reset_index(drop=True)

    )

else:

    monthly_crossovers = pd.DataFrame(
        columns=[

            'Symbol',
            'Date',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'EMA_10',
            'EMA_25',
            'EMA_Difference',
            'Monthly_Crossover',
            'EMA_Gap_%'

        ]
    )


# ════════════════════════════════════════════════════════════════════════════
# FORMAT MONTHLY OUTPUT
# ════════════════════════════════════════════════════════════════════════════

if not monthly_crossovers.empty:

    monthly_crossovers['Date'] = (

        monthly_crossovers['Date']
        .dt.strftime('%Y-%m-%d')

    )


    for col in [
        'EMA_10',
        'EMA_25',
        'EMA_Difference',
        'EMA_Gap_%',
        'Close'
    ]:

        monthly_crossovers[col] = (
            monthly_crossovers[col]
            .round(2)
        )


    monthly_crossovers['Volume'] = (
        monthly_crossovers['Volume']
        .round(0)
    )


# ════════════════════════════════════════════════════════════════════════════
# DISPLAY MONTHLY SIGNALS
# ════════════════════════════════════════════════════════════════════════════

print(
    "\n" +
    "=" * 80
)

print(
    "LATEST MONTHLY EMA 10 / EMA 25 CROSSOVERS"
)

print(
    "=" * 80
)


if monthly_crossovers.empty:

    print(
        "No monthly EMA 10 / EMA 25 crossover found."
    )

else:

    print(

        monthly_crossovers[
            [
                'Symbol',
                'Date',
                'Close',
                'EMA_10',
                'EMA_25',
                'EMA_Difference',
                'EMA_Gap_%'
            ]
        ]
        .to_string(index=False)

    )


# ════════════════════════════════════════════════════════════════════════════
# UPLOAD MONTHLY EMA FILE
# ════════════════════════════════════════════════════════════════════════════

monthly_file_name = (
    f'Monthly_EMA_Cross_for_{today_str}.csv'
)


github_put(
    monthly_file_name,
    monthly_crossovers
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 – CLEAN OLD EMA FILES
# ════════════════════════════════════════════════════════════════════════════

print(
    "\nCleaning up old EMA files ..."
)


# Keep latest DAILY EMA file
delete_old_github_files(
    'EMA_Cross_for_',
    keep=1
)


# Keep latest MONTHLY EMA file
delete_old_github_files(
    'Monthly_EMA_Cross_for_',
    keep=1
)


# Keep latest ESPEN file
delete_old_github_files(
    'espen_',
    keep=1
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 9 – REMOVE OLD JUNK CSV FILES
# ════════════════════════════════════════════════════════════════════════════

JUNK_PATTERNS = [

    r'^nepse_\w+\.csv$',
    r'^combined_data\.csv$',
    r'^valid_ema_crossovers\.csv$',
    r'^latest_valid_ema_crossovers\.csv$'

]


headers = {
    'Authorization': f'token {GH_TOKEN}'
}


r = requests.get(

    f'https://api.github.com/repos/'
    f'{GITHUB_REPO}/contents/',

    headers=headers,

    timeout=15

)


if r.status_code == 200:

    for f in r.json():

        if (

            isinstance(f, dict)

            and

            any(

                re.match(
                    p,
                    f.get('name', '')
                )

                for p in JUNK_PATTERNS

            )

        ):

            dr = requests.delete(

                f['url'],

                headers=headers,

                timeout=15,

                json={

                    'message':
                        f'Cleanup: remove {f["name"]}',

                    'sha': f['sha'],

                    'branch': 'main'

                }

            )


            if dr.status_code == 200:

                print(
                    f"Removed from GitHub: "
                    f"{f['name']}"
                )

            else:

                print(
                    f"Could not remove "
                    f"{f['name']}: "
                    f"{dr.status_code}"
                )


# ════════════════════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════════════════════

print(
    "\n" +
    "=" * 80
)

print(
    "DONE"
)

print(
    "=" * 80
)

print(
    f"Daily EMA file:   {daily_file_name}"
)

print(
    f"Monthly EMA file: {monthly_file_name}"
)

print(
    f"Historical file:  espen_{today_str}.csv"
)

print(
    "=" * 80
)
