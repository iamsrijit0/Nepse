
import subprocess
import sys
import os
import requests
import base64
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo


# ════════════════════════════════════════════════════════════════════════════
# INSTALL REQUIRED PACKAGES
# ════════════════════════════════════════════════════════════════════════════

packages = [
    "nepse-scraper",
    "xlsxwriter",
    "gitpython",
    "pandas",
    "matplotlib",
    "joblib",
    "beautifulsoup4"
]

print("Checking required packages...")

subprocess.check_call(
    [sys.executable, "-m", "pip", "install"] + packages,
    stdout=subprocess.DEVNULL
)

from nepse_scraper import Nepse_scraper
from joblib import Parallel, delayed


# ════════════════════════════════════════════════════════════════════════════
# SETTINGS
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

# Minimum number of COMPLETED monthly candles required
MIN_MONTHS_REQUIRED = 36


# ════════════════════════════════════════════════════════════════════════════
# CHECK GITHUB TOKEN
# ════════════════════════════════════════════════════════════════════════════

if not GH_TOKEN:

    print("\nERROR: GH_TOKEN environment variable is not set.")

    raise SystemExit(1)


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def to_float(value):

    try:

        if value is None:
            return 0.0

        if isinstance(value, str):
            value = value.replace(',', '').strip()

        return float(value)

    except (TypeError, ValueError):

        return 0.0


def clean_numeric_series(series):

    return pd.to_numeric(

        series
        .astype(str)
        .str.replace(',', '', regex=False)
        .replace('-', np.nan),

        errors='coerce'

    )


def format_date(dt):

    return f"{dt.month}/{dt.day}/{dt.year}"


# ════════════════════════════════════════════════════════════════════════════
# STEP 1
# FETCH TODAY'S NEPSE DATA
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 1 - FETCHING TODAY'S NEPSE DATA")
print("=" * 90)


request_obj = Nepse_scraper(
    verify_ssl=False
)


today_price = request_obj.get_today_price()


if isinstance(today_price, dict):

    content_data = today_price.get(
        'content',
        []
    )

else:

    content_data = today_price


filtered_data = []


for item in content_data:

    symbol = item.get(
        'symbol',
        ''
    )

    date = item.get(
        'businessDate',
        ''
    )


    open_price = to_float(
        item.get('openPrice')
    )

    high_price = to_float(
        item.get('highPrice')
    )

    low_price = to_float(
        item.get('lowPrice')
    )

    close_price = to_float(
        item.get('closePrice')
    )

    volume = to_float(
        item.get('totalTradedQuantity')
    )

    high52 = to_float(
        item.get('fiftyTwoWeekHigh')
    )

    low52 = to_float(
        item.get('fiftyTwoWeekLow')
    )


    if open_price > 0:

        pct_change = (

            (
                close_price
                -
                open_price
            )
            /
            open_price
            *
            100

        )

    else:

        pct_change = 0


    filtered_data.append({

        'Symbol': symbol,

        'Date': date,

        'Open': open_price,

        'High': high_price,

        'Low': low_price,

        'Close': close_price,

        'Percent Change': round(
            pct_change,
            2
        ),

        'Volume': volume,

        '52High': high52,

        '52Low': low52

    })


first = pd.DataFrame(
    filtered_data
)


if first.empty:

    print(
        "ERROR: No NEPSE data returned."
    )

    raise SystemExit(1)


first['Date'] = pd.to_datetime(
    first['Date'],
    errors='coerce'
)


first = first.dropna(
    subset=['Date']
)


print(
    f"Today's records: {len(first):,}"
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2
# FIND LATEST ESPEN FILE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 2 - LOADING HISTORICAL DATA")
print("=" * 90)


def get_latest_espen_url():

    repo_url = (

        f'https://github.com/'
        f'{GITHUB_REPO}/tree/main'

    )


    response = requests.get(

        repo_url,

        timeout=30

    )


    response.raise_for_status()


    soup = BeautifulSoup(

        response.content,

        'html.parser'

    )


    files = {}


    for link in soup.find_all(
        'a',
        href=True
    ):

        href = link['href']


        if (

            'espen_' in href

            and

            href.endswith('.csv')

        ):

            match = re.search(

                r'espen_(\d{4}-\d{2}-\d{2})\.csv',

                href

            )


            if match:

                file_date = match.group(1)

                file_name = (
                    f'espen_{file_date}.csv'
                )


                files[file_date] = (

                    f'https://raw.githubusercontent.com/'
                    f'{GITHUB_REPO}/main/'
                    f'{file_name}'

                )


    if not files:

        raise ValueError(
            "No espen_ CSV file found in GitHub repository."
        )


    latest_date = max(
        files.keys()
    )


    print(
        f"Latest historical file: "
        f"espen_{latest_date}.csv"
    )


    return files[latest_date]


secondss = pd.DataFrame()


try:

    latest_url = get_latest_espen_url()


    raw = pd.read_csv(
        latest_url
    )


    for col in STANDARD_COLS:

        if col not in raw.columns:

            raw[col] = np.nan


    secondss = raw[
        STANDARD_COLS
    ].copy()


    secondss['Date'] = pd.to_datetime(

        secondss['Date'],

        errors='coerce'

    )


    secondss = secondss.dropna(

        subset=['Date']

    )


    print(
        f"Historical rows loaded: "
        f"{len(secondss):,}"
    )


except Exception as e:

    print(
        f"WARNING: Could not load historical data."
    )

    print(
        str(e)
    )


# ════════════════════════════════════════════════════════════════════════════
# STEP 3
# MERGE TODAY WITH HISTORICAL DATA
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 3 - MERGING HISTORICAL + TODAY")
print("=" * 90)


frames = [

    df

    for df in [
        secondss,
        first
    ]

    if not df.empty

]


if not frames:

    print(
        "ERROR: No data available."
    )

    raise SystemExit(1)


combined_df = pd.concat(

    frames,

    ignore_index=True

)


combined_df['Date'] = pd.to_datetime(

    combined_df['Date'],

    errors='coerce'

)


combined_df = combined_df.dropna(

    subset=[
        'Symbol',
        'Date'
    ]

)


# ════════════════════════════════════════════════════════════════════════════
# REMOVE DUPLICATE SYMBOL + DATE
# ════════════════════════════════════════════════════════════════════════════

combined_df = (

    combined_df

    .sort_values(
        'Date'
    )

    .drop_duplicates(

        subset=[
            'Symbol',
            'Date'
        ],

        keep='last'

    )

)


# ════════════════════════════════════════════════════════════════════════════
# UPDATE LIVE 52 WEEK VALUES
# ════════════════════════════════════════════════════════════════════════════

live_52 = (

    first

    .drop_duplicates(
        'Symbol',
        keep='last'
    )

    .set_index(
        'Symbol'
    )

)


for symbol in combined_df['Symbol'].unique():

    if symbol in live_52.index:

        mask = (
            combined_df['Symbol']
            == symbol
        )


        combined_df.loc[
            mask,
            '52High'
        ] = live_52.loc[
            symbol,
            '52High'
        ]


        combined_df.loc[
            mask,
            '52Low'
        ] = live_52.loc[
            symbol,
            '52Low'
        ]


# ════════════════════════════════════════════════════════════════════════════
# SORT HISTORICAL DATA
# ════════════════════════════════════════════════════════════════════════════

combined_df = combined_df.sort_values(

    [
        'Symbol',
        'Date'
    ]

).reset_index(
    drop=True
)


# ════════════════════════════════════════════════════════════════════════════
# SAVE DATE AS ORIGINAL DISPLAY FORMAT
# ════════════════════════════════════════════════════════════════════════════

combined_for_upload = combined_df.copy()


combined_for_upload['Date'] = (

    combined_for_upload['Date']
    .apply(format_date)

)


combined_for_upload = combined_for_upload[
    STANDARD_COLS
]


print(
    f"Total accumulated rows: "
    f"{len(combined_for_upload):,}"
)


# ════════════════════════════════════════════════════════════════════════════
# GITHUB UPLOAD
# ════════════════════════════════════════════════════════════════════════════

def github_put(
    file_name,
    df
):

    print(
        f"\nUploading {file_name} ..."
    )


    csv_content = df.to_csv(
        index=False
    )


    encoded = base64.b64encode(

        csv_content.encode()

    ).decode()


    url = (

        f'https://api.github.com/repos/'
        f'{GITHUB_REPO}/contents/'
        f'{file_name}'

    )


    headers = {

        'Authorization':
            f'token {GH_TOKEN}',

        'Accept':
            'application/vnd.github+json'

    }


    existing = requests.get(

        url,

        headers=headers,

        timeout=30

    )


    sha = None


    if existing.status_code == 200:

        sha = existing.json().get(
            'sha'
        )


    payload = {

        'message':
            f'Update {file_name}',

        'content':
            encoded,

        'branch':
            'main'

    }


    if sha:

        payload['sha'] = sha


    response = requests.put(

        url,

        headers=headers,

        json=payload,

        timeout=30

    )


    if response.status_code in (
        200,
        201
    ):

        print(
            f"SUCCESS: {file_name}"
        )

    else:

        print(
            f"ERROR uploading {file_name}"
        )

        print(
            response.status_code
        )

        print(
            response.text
        )


# ════════════════════════════════════════════════════════════════════════════
# DELETE OLD FILES
# ════════════════════════════════════════════════════════════════════════════

def delete_old_github_files(
    prefix,
    keep=1
):

    headers = {

        'Authorization':
            f'token {GH_TOKEN}',

        'Accept':
            'application/vnd.github+json'

    }


    url = (

        f'https://api.github.com/repos/'
        f'{GITHUB_REPO}/contents/'

    )


    response = requests.get(

        url,

        headers=headers,

        timeout=30

    )


    if response.status_code != 200:

        print(
            f"Could not list GitHub files: "
            f"{response.status_code}"
        )

        return


    files = response.json()


    matching = [

        f

        for f in files

        if (

            isinstance(f, dict)

            and

            f.get(
                'name',
                ''
            ).startswith(prefix)

            and

            f.get(
                'name',
                ''
            ).endswith('.csv')

        )

    ]


    matching.sort(

        key=lambda x: x['name'],

        reverse=True

    )


    old_files = matching[
        keep:
    ]


    for file_info in old_files:

        delete_response = requests.delete(

            file_info['url'],

            headers=headers,

            json={

                'message':
                    f'Cleanup {file_info["name"]}',

                'sha':
                    file_info['sha'],

                'branch':
                    'main'

            },

            timeout=30

        )


        if delete_response.status_code == 200:

            print(
                f"Deleted old file: "
                f"{file_info['name']}"
            )

        else:

            print(
                f"Could not delete: "
                f"{file_info['name']}"
            )


# ════════════════════════════════════════════════════════════════════════════
# NEPAL DATE
# ════════════════════════════════════════════════════════════════════════════

nepal_now = pd.Timestamp.now(
    tz=ZoneInfo("Asia/Kathmandu")
)


nepal_today = (

    nepal_now

    .normalize()

    .tz_localize(None)

)


today_str = nepal_today.strftime(
    '%Y-%m-%d'
)


print(
    f"\nNepal date: {today_str}"
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4
# UPLOAD ACCUMULATED HISTORY
# ════════════════════════════════════════════════════════════════════════════

historical_file = (

    f'espen_{today_str}.csv'

)


github_put(

    historical_file,

    combined_for_upload

)


# ════════════════════════════════════════════════════════════════════════════
# STEP 5
# DAILY EMA 20 / 50
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 5 - DAILY EMA 20 / EMA 50")
print("=" * 90)


daily_data = combined_df.copy()


for col in [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume'
]:

    daily_data[col] = clean_numeric_series(

        daily_data[col]

    )


daily_data = daily_data.dropna(

    subset=[
        'Symbol',
        'Date',
        'Close'
    ]

)


def process_daily_symbol(
    symbol,
    group
):

    group = (

        group
        .sort_values('Date')
        .copy()
        .reset_index(drop=True)

    )


    if len(group) < 50:

        return pd.DataFrame()


    # EMA 20

    group['EMA_20'] = (

        group['Close']
        .ewm(
            span=20,
            adjust=False,
            min_periods=20
        )
        .mean()

    )


    # EMA 50

    group['EMA_50'] = (

        group['Close']
        .ewm(
            span=50,
            adjust=False,
            min_periods=50
        )
        .mean()

    )


    # RSI 14

    delta = group['Close'].diff()


    gain = (

        delta
        .where(
            delta > 0,
            0
        )
        .rolling(
            14,
            min_periods=14
        )
        .mean()

    )


    loss = (

        -delta
        .where(
            delta < 0,
            0
        )
        .rolling(
            14,
            min_periods=14
        )
        .mean()

    )


    rs = gain / loss


    group['RSI'] = (

        100
        -
        (
            100
            /
            (1 + rs)
        )

    )


    # 30 day average volume

    group['30D_Avg_Volume'] = (

        group['Volume']
        .rolling(
            30,
            min_periods=30
        )
        .mean()

    )


    # EMA slopes

    group['Slope_20'] = (

        group['EMA_20']
        .diff(5)
        / 5

    )


    group['Slope_50'] = (

        group['EMA_50']
        .diff(5)
        / 5

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
                min_periods=60
            )
            .max()
            .shift(1)
            * 0.95
        )

    ].copy()


    return valid


daily_results = Parallel(
    n_jobs=-1
)(

    delayed(
        process_daily_symbol
    )(
        symbol,
        group
    )

    for symbol, group
    in daily_data.groupby(
        'Symbol'
    )

)


daily_results = [

    x

    for x in daily_results

    if not x.empty

]


if daily_results:

    daily_crossovers = pd.concat(

        daily_results,

        ignore_index=True

    )


    daily_crossovers = (

        daily_crossovers

        .sort_values(
            'Date',
            ascending=False
        )

        .drop_duplicates(
            'Symbol'
        )

        .reset_index(
            drop=True
        )

    )

else:

    daily_crossovers = pd.DataFrame()


print(
    "\nDAILY EMA CROSSOVERS:"
)


if daily_crossovers.empty:

    print(
        "No daily EMA crossover found."
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
        ].to_string(
            index=False
        )

    )


daily_file = (

    f'EMA_Cross_for_{today_str}.csv'

)


github_put(

    daily_file,

    daily_crossovers

)


# ════════════════════════════════════════════════════════════════════════════
# STEP 6
# CREATE MONTHLY DATA USING ONLY ACTUAL CSV DATES
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 6 - MONTHLY DATA FROM ACTUAL CSV TRADING DATES")
print("=" * 90)


monthly_source = combined_df.copy()


for col in [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume'
]:

    monthly_source[col] = clean_numeric_series(

        monthly_source[col]

    )


monthly_source = monthly_source.dropna(

    subset=[
        'Symbol',
        'Date',
        'Close'
    ]

)


monthly_source = monthly_source.sort_values(

    [
        'Symbol',
        'Date'
    ]

)


# ════════════════════════════════════════════════════════════════════════════
# IMPORTANT:
#
# We DO NOT use:
#
#     resample('ME')
#
# because that creates a calendar month-end date.
#
# Instead:
#
# 1. Group actual CSV rows by year/month.
# 2. Aggregate those actual rows.
# 3. Use the LAST ACTUAL DATE in the CSV as the monthly Date.
#
# Therefore a monthly Date can ONLY be a date that exists in the CSV.
# ════════════════════════════════════════════════════════════════════════════


def make_monthly_data(
    symbol,
    group
):

    group = (

        group
        .sort_values('Date')
        .copy()

    )


    # ------------------------------------------------------------
    # Remove current incomplete month
    # ------------------------------------------------------------

    group = group[

        ~(

            (group['Date'].dt.year
             == nepal_today.year)

            &

            (group['Date'].dt.month
             == nepal_today.month)

        )

    ].copy()


    if group.empty:

        return pd.DataFrame()


    # ------------------------------------------------------------
    # Create year/month grouping WITHOUT resample
    # ------------------------------------------------------------

    group['_Year'] = (
        group['Date'].dt.year
    )

    group['_Month'] = (
        group['Date'].dt.month
    )


    monthly_rows = []


    for (
        year,
        month
    ), month_group in group.groupby(

        [
            '_Year',
            '_Month'
        ],

        sort=True

    ):

        month_group = month_group.sort_values(
            'Date'
        )


        # ========================================================
        # THIS IS THE CRITICAL PART
        #
        # monthly_date = LAST ACTUAL DATE IN CSV
        #
        # No artificial month-end date.
        # ========================================================

        actual_last_date = (

            month_group['Date']
            .iloc[-1]

        )


        monthly_rows.append({

            'Symbol':
                symbol,

            'Date':
                actual_last_date,

            'Open':
                month_group['Open'].iloc[0],

            'High':
                month_group['High'].max(),

            'Low':
                month_group['Low'].min(),

            'Close':
                month_group['Close'].iloc[-1],

            'Volume':
                month_group['Volume'].sum()

        })


    return pd.DataFrame(
        monthly_rows
    )


monthly_results = Parallel(
    n_jobs=-1
)(

    delayed(
        make_monthly_data
    )(
        symbol,
        group
    )

    for symbol, group
    in monthly_source.groupby(
        'Symbol'
    )

)


monthly_results = [

    x

    for x in monthly_results

    if not x.empty

]


if monthly_results:

    monthly_df = pd.concat(

        monthly_results,

        ignore_index=True

    )

else:

    monthly_df = pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# SORT MONTHLY DATA
# ════════════════════════════════════════════════════════════════════════════

if monthly_df.empty:

    print(
        "ERROR: No completed monthly data available."
    )

    monthly_crossovers = pd.DataFrame()

else:

    monthly_df = monthly_df.sort_values(

        [
            'Symbol',
            'Date'
        ]

    ).reset_index(
        drop=True
    )


    # ════════════════════════════════════════════════════════════════════════
    # VERIFY EVERY MONTHLY DATE EXISTS IN ORIGINAL CSV
    # ════════════════════════════════════════════════════════════════════════

    actual_csv_dates = set(

        monthly_source['Date']
        .dt.normalize()
        .unique()

    )


    monthly_dates = set(

        monthly_df['Date']
        .dt.normalize()
        .unique()

    )


    invalid_dates = (

        monthly_dates
        -
        actual_csv_dates

    )


    if invalid_dates:

        print(
            "\nERROR: Artificial monthly dates detected!"
        )

        print(
            invalid_dates
        )

        raise SystemExit(1)


    print(
        "DATE VALIDATION PASSED:"
    )

    print(
        "Every monthly Date exists "
        "in the original CSV history."
    )


    # ════════════════════════════════════════════════════════════════════════
    # SHOW SAMPLE MONTHLY DATES
    # ════════════════════════════════════════════════════════════════════════

    print(
        "\nSample monthly dates:"
    )


    sample = (

        monthly_df

        .groupby('Symbol')

        .head(12)

        .tail(20)

        [
            [
                'Symbol',
                'Date',
                'Close'
            ]
        ]

    )


    if not sample.empty:

        print(
            sample.to_string(
                index=False
            )
        )


    # ════════════════════════════════════════════════════════════════════════
    # STEP 7
    # MONTHLY EMA 10 / EMA 25
    # ════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("STEP 7 - MONTHLY EMA 10 / EMA 25")
    print("=" * 90)


    def process_monthly_symbol(
        symbol,
        group
    ):

        group = (

            group
            .sort_values('Date')
            .copy()
            .reset_index(drop=True)

        )


        # Need at least 36 completed months

        if len(group) < MIN_MONTHS_REQUIRED:

            return pd.DataFrame()


        # Number of completed monthly observations

        group['Months_Used'] = (

            np.arange(
                1,
                len(group) + 1
            )

        )


        # ═══════════════════════════════════════════════════════════
        # EMA 10
        # ═══════════════════════════════════════════════════════════

        group['EMA_10'] = (

            group['Close']

            .ewm(

                span=10,

                adjust=False,

                min_periods=10

            )

            .mean()

        )


        # ═══════════════════════════════════════════════════════════
        # EMA 25
        # ═══════════════════════════════════════════════════════════

        group['EMA_25'] = (

            group['Close']

            .ewm(

                span=25,

                adjust=False,

                min_periods=25

            )

            .mean()

        )


        # ═══════════════════════════════════════════════════════════
        # PREVIOUS MONTH VALUES
        # ═══════════════════════════════════════════════════════════

        group['Previous_EMA_10'] = (

            group['EMA_10']
            .shift(1)

        )


        group['Previous_EMA_25'] = (

            group['EMA_25']
            .shift(1)

        )


        # ═══════════════════════════════════════════════════════════
        # EMA DIFFERENCE
        # ═══════════════════════════════════════════════════════════

        group['EMA_Difference'] = (

            group['EMA_10']
            -
            group['EMA_25']

        )


        # ═══════════════════════════════════════════════════════════
        # EMA GAP %
        # ═══════════════════════════════════════════════════════════

        group['EMA_Gap_%'] = np.where(

            group['EMA_25'] != 0,

            (

                group['EMA_Difference']
                /
                group['EMA_25']
                *
                100

            ),

            np.nan

        )


        # ═══════════════════════════════════════════════════════════
        # BULLISH MONTHLY CROSSOVER
        #
        # Previous ACTUAL completed month:
        #
        # EMA10 <= EMA25
        #
        # Current ACTUAL completed month:
        #
        # EMA10 > EMA25
        # ═══════════════════════════════════════════════════════════

        group['Monthly_Crossover'] = (

            (

                group['EMA_10']
                >
                group['EMA_25']

            )

            &

            (

                group['Previous_EMA_10']
                <=
                group['Previous_EMA_25']

            )

        )


        # Keep only crossover rows

        valid = group[

            group['Monthly_Crossover']

        ].copy()


        return valid


    monthly_results = Parallel(
        n_jobs=-1
    )(

        delayed(
            process_monthly_symbol
        )(
            symbol,
            group
        )

        for symbol, group
        in monthly_df.groupby(
            'Symbol'
        )

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

            .reset_index(

                drop=True

            )

        )

    else:

        monthly_crossovers = pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# STEP 8
# FORMAT MONTHLY RESULTS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 8 - MONTHLY CROSSOVER RESULTS")
print("=" * 90)


if not monthly_crossovers.empty:

    # ------------------------------------------------------------
    # FINAL SAFETY CHECK:
    # Every crossover date MUST exist in original CSV.
    # ------------------------------------------------------------

    original_dates = set(

        monthly_source['Date']
        .dt.normalize()
        .unique()

    )


    crossover_dates = set(

        pd.to_datetime(
            monthly_crossovers['Date']
        )
        .dt.normalize()
        .unique()

    )


    bad_dates = (

        crossover_dates
        -
        original_dates

    )


    if bad_dates:

        print(
            "ERROR: Crossover contains a date "
            "that does not exist in CSV."
        )

        print(
            bad_dates
        )

        raise SystemExit(1)


    # ------------------------------------------------------------
    # Format date
    # ------------------------------------------------------------

    monthly_crossovers['Date'] = (

        pd.to_datetime(

            monthly_crossovers['Date']

        )

        .dt.strftime(
            '%Y-%m-%d'
        )

    )


    # ------------------------------------------------------------
    # Round values
    # ------------------------------------------------------------

    for col in [

        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'EMA_10',
        'EMA_25',
        'Previous_EMA_10',
        'Previous_EMA_25',
        'EMA_Difference',
        'EMA_Gap_%'

    ]:

        if col in monthly_crossovers.columns:

            monthly_crossovers[col] = (

                pd.to_numeric(

                    monthly_crossovers[col],

                    errors='coerce'

                )

                .round(2)

            )


    print(
        "\nCONFIRMED MONTHLY EMA 10 / EMA 25 CROSSOVERS:"
    )


    print(

        monthly_crossovers[

            [

                'Symbol',

                'Date',

                'Close',

                'EMA_10',

                'EMA_25',

                'Previous_EMA_10',

                'Previous_EMA_25',

                'EMA_Difference',

                'EMA_Gap_%',

                'Months_Used'

            ]

        ].to_string(
            index=False
        )

    )

else:

    print(
        "No confirmed monthly EMA crossover found."
    )


# ════════════════════════════════════════════════════════════════════════════
# FINAL MONTHLY COLUMN ORDER
# ════════════════════════════════════════════════════════════════════════════

monthly_output_columns = [

    'Symbol',
    'Date',
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'EMA_10',
    'EMA_25',
    'Previous_EMA_10',
    'Previous_EMA_25',
    'EMA_Difference',
    'Monthly_Crossover',
    'EMA_Gap_%',
    'Months_Used'

]


if not monthly_crossovers.empty:

    monthly_crossovers = (

        monthly_crossovers[

            [

                col

                for col
                in monthly_output_columns

                if col
                in monthly_crossovers.columns

            ]

        ]

    )

else:

    monthly_crossovers = pd.DataFrame(

        columns=monthly_output_columns

    )


# ════════════════════════════════════════════════════════════════════════════
# STEP 9
# UPLOAD MONTHLY EMA FILE
# ════════════════════════════════════════════════════════════════════════════

monthly_file = (

    f'Monthly_EMA_Cross_for_{today_str}.csv'

)


github_put(

    monthly_file,

    monthly_crossovers

)


# ════════════════════════════════════════════════════════════════════════════
# STEP 10
# DELETE OLD FILES
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 10 - CLEANING OLD FILES")
print("=" * 90)


delete_old_github_files(

    'espen_',

    keep=1

)


delete_old_github_files(

    'EMA_Cross_for_',

    keep=1

)


delete_old_github_files(

    'Monthly_EMA_Cross_for_',

    keep=1

)


# ════════════════════════════════════════════════════════════════════════════
# REMOVE OLD JUNK CSV FILES
# ════════════════════════════════════════════════════════════════════════════

JUNK_PATTERNS = [

    r'^nepse_\w+\.csv$',

    r'^combined_data\.csv$',

    r'^valid_ema_crossovers\.csv$',

    r'^latest_valid_ema_crossovers\.csv$'

]


headers = {

    'Authorization':
        f'token {GH_TOKEN}',

    'Accept':
        'application/vnd.github+json'

}


repo_url = (

    f'https://api.github.com/repos/'
    f'{GITHUB_REPO}/contents/'

)


response = requests.get(

    repo_url,

    headers=headers,

    timeout=30

)


if response.status_code == 200:

    repo_files = response.json()


    for file_info in repo_files:

        file_name = file_info.get(
            'name',
            ''
        )


        should_delete = any(

            re.match(
                pattern,
                file_name
            )

            for pattern
            in JUNK_PATTERNS

        )


        if should_delete:

            delete_response = requests.delete(

                file_info['url'],

                headers=headers,

                json={

                    'message':
                        f'Cleanup {file_name}',

                    'sha':
                        file_info['sha'],

                    'branch':
                        'main'

                },

                timeout=30

            )


            if delete_response.status_code == 200:

                print(
                    f"Removed junk file: "
                    f"{file_name}"
                )


# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 100)

print(
    f"Historical file:  {historical_file}"
)

print(
    f"Daily EMA file:   {daily_file}"
)

print(
    f"Monthly EMA file: {monthly_file}"
)

print(
    f"Minimum monthly history: "
    f"{MIN_MONTHS_REQUIRED} completed months"
)

print(
    "Monthly dates: ACTUAL CSV TRADING DATES ONLY"
)

print(
    "Current incomplete month: EXCLUDED"
)

print(
    "Calendar month-end dates: NOT GENERATED"
)

print("=" * 100)
print("DONE.")

