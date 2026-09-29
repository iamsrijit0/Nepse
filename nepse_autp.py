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
    "Symbol",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Percent Change",
    "Volume",
    "52High",
    "52Low"
]

GITHUB_REPO = "iamsrijit0/Nepse"

GH_TOKEN = os.getenv("GH_TOKEN")


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
            value = value.replace(",", "").strip()

        return float(value)

    except (TypeError, ValueError):
        return 0.0


def clean_numeric_series(series):

    return pd.to_numeric(
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("-", np.nan),
        errors="coerce"
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
        "content",
        []
    )

else:

    content_data = today_price


filtered_data = []


for item in content_data:

    symbol = item.get(
        "symbol",
        ""
    )

    date = item.get(
        "businessDate",
        ""
    )

    open_price = to_float(
        item.get("openPrice")
    )

    high_price = to_float(
        item.get("highPrice")
    )

    low_price = to_float(
        item.get("lowPrice")
    )

    close_price = to_float(
        item.get("closePrice")
    )

    volume = to_float(
        item.get("totalTradedQuantity")
    )

    high52 = to_float(
        item.get("fiftyTwoWeekHigh")
    )

    low52 = to_float(
        item.get("fiftyTwoWeekLow")
    )


    if open_price > 0:

        pct_change = (
            (
                close_price - open_price
            )
            /
            open_price
            *
            100
        )

    else:

        pct_change = 0


    filtered_data.append({

        "Symbol": symbol,

        "Date": date,

        "Open": open_price,

        "High": high_price,

        "Low": low_price,

        "Close": close_price,

        "Percent Change": round(
            pct_change,
            2
        ),

        "Volume": volume,

        "52High": high52,

        "52Low": low52

    })


first = pd.DataFrame(
    filtered_data
)


if first.empty:

    print(
        "ERROR: No NEPSE data returned."
    )

    raise SystemExit(1)


first["Date"] = pd.to_datetime(
    first["Date"],
    errors="coerce"
)


first = first.dropna(
    subset=["Date"]
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
        f"https://github.com/"
        f"{GITHUB_REPO}/tree/main"
    )

    response = requests.get(
        repo_url,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    files = {}

    for link in soup.find_all(
        "a",
        href=True
    ):

        href = link["href"]

        if (
            "espen_" in href
            and
            href.endswith(".csv")
        ):

            match = re.search(
                r"espen_(\d{4}-\d{2}-\d{2})\.csv",
                href
            )

            if match:

                file_date = match.group(1)

                file_name = (
                    f"espen_{file_date}.csv"
                )

                files[file_date] = (
                    f"https://raw.githubusercontent.com/"
                    f"{GITHUB_REPO}/main/"
                    f"{file_name}"
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

    secondss["Date"] = pd.to_datetime(
        secondss["Date"],
        errors="coerce"
    )

    secondss = secondss.dropna(
        subset=["Date"]
    )

    print(
        f"Historical rows loaded: "
        f"{len(secondss):,}"
    )

except Exception as e:

    print(
        "WARNING: Could not load historical data."
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


combined_df["Date"] = pd.to_datetime(
    combined_df["Date"],
    errors="coerce"
)


combined_df = combined_df.dropna(
    subset=[
        "Symbol",
        "Date"
    ]
)


# ════════════════════════════════════════════════════════════════════════════
# REMOVE DUPLICATE SYMBOL + DATE
# ════════════════════════════════════════════════════════════════════════════

combined_df = (

    combined_df

    .sort_values(
        "Date"
    )

    .drop_duplicates(
        subset=[
            "Symbol",
            "Date"
        ],
        keep="last"
    )

)


# ════════════════════════════════════════════════════════════════════════════
# UPDATE LIVE 52 WEEK VALUES
# ════════════════════════════════════════════════════════════════════════════

live_52 = (

    first

    .drop_duplicates(
        "Symbol",
        keep="last"
    )

    .set_index(
        "Symbol"
    )

)


for symbol in combined_df["Symbol"].unique():

    if symbol in live_52.index:

        mask = (
            combined_df["Symbol"]
            == symbol
        )

        combined_df.loc[
            mask,
            "52High"
        ] = live_52.loc[
            symbol,
            "52High"
        ]

        combined_df.loc[
            mask,
            "52Low"
        ] = live_52.loc[
            symbol,
            "52Low"
        ]


# ════════════════════════════════════════════════════════════════════════════
# SORT HISTORICAL DATA
# ════════════════════════════════════════════════════════════════════════════

combined_df = combined_df.sort_values(

    [
        "Symbol",
        "Date"
    ]

).reset_index(
    drop=True
)


# ════════════════════════════════════════════════════════════════════════════
# SAVE DATE AS ORIGINAL DISPLAY FORMAT
# ════════════════════════════════════════════════════════════════════════════

combined_for_upload = combined_df.copy()


combined_for_upload["Date"] = (
    combined_for_upload["Date"]
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
# GITHUB UPLOAD FUNCTION
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
        f"https://api.github.com/repos/"
        f"{GITHUB_REPO}/contents/"
        f"{file_name}"
    )

    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    existing = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    sha = None

    if existing.status_code == 200:

        sha = existing.json().get(
            "sha"
        )

    payload = {

        "message":
            f"Update {file_name}",

        "content":
            encoded,

        "branch":
            "main"

    }

    if sha:
        payload["sha"] = sha

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

        "Authorization":
            f"token {GH_TOKEN}",

        "Accept":
            "application/vnd.github+json"

    }

    url = (
        f"https://api.github.com/repos/"
        f"{GITHUB_REPO}/contents/"
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
                "name",
                ""
            ).startswith(prefix)

            and

            f.get(
                "name",
                ""
            ).endswith(".csv")

        )

    ]

    matching.sort(
        key=lambda x: x["name"],
        reverse=True
    )

    old_files = matching[
        keep:
    ]

    for file_info in old_files:

        delete_response = requests.delete(

            file_info["url"],

            headers=headers,

            json={

                "message":
                    f"Cleanup {file_info['name']}",

                "sha":
                    file_info["sha"],

                "branch":
                    "main"

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
    "%Y-%m-%d"
)


print(
    f"\nNepal date: {today_str}"
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4
# UPLOAD ACCUMULATED HISTORY
# ════════════════════════════════════════════════════════════════════════════

historical_file = (
    f"espen_{today_str}.csv"
)


github_put(
    historical_file,
    combined_for_upload
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 5
# CREATE 2-TRADING-DAY CANDLES
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 5 - CREATING 2-TRADING-DAY CANDLES")
print("=" * 90)


two_day_source = combined_df.copy()


for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:

    two_day_source[col] = clean_numeric_series(
        two_day_source[col]
    )


two_day_source = two_day_source.dropna(
    subset=[
        "Symbol",
        "Date",
        "Close"
    ]
)


two_day_source = two_day_source.sort_values(
    [
        "Symbol",
        "Date"
    ]
)


# ════════════════════════════════════════════════════════════════════════════
# IMPORTANT
#
# We DO NOT use calendar-day resampling.
#
# We use ACTUAL rows from the CSV.
#
# Example:
#
# Trading Day 1 + Trading Day 2 = Candle 1
# Trading Day 3 + Trading Day 4 = Candle 2
# Trading Day 5 + Trading Day 6 = Candle 3
#
# Weekend / holiday = NOT counted.
#
# If a symbol has an odd number of trading rows, the final single day
# is NOT used because it is not a complete 2-trading-day candle.
# ════════════════════════════════════════════════════════════════════════════


def make_two_day_data(
    symbol,
    group
):

    group = (

        group

        .sort_values(
            "Date"
        )

        .copy()

        .reset_index(
            drop=True
        )

    )


    # Need at least 2 actual trading days
    if len(group) < 2:
        return pd.DataFrame()


    # ------------------------------------------------------------
    # Keep only COMPLETE pairs
    # ------------------------------------------------------------

    complete_count = (
        len(group) // 2
    ) * 2

    group = group.iloc[
        :complete_count
    ].copy()


    if len(group) < 2:
        return pd.DataFrame()


    # ------------------------------------------------------------
    # Create sequential 2-day groups
    #
    # 0,1   -> pair 1
    # 2,3   -> pair 2
    # 4,5   -> pair 3
    # ------------------------------------------------------------

    group["_2Day_Group"] = (
        np.arange(
            len(group)
        ) // 2
    )


    two_day_rows = []


    for pair_number, pair in group.groupby(
        "_2Day_Group",
        sort=True
    ):

        pair = pair.sort_values(
            "Date"
        )


        # Safety: every candle MUST contain exactly 2 rows
        if len(pair) != 2:
            continue


        first_day = pair.iloc[0]
        second_day = pair.iloc[1]


        two_day_rows.append({

            "Symbol":
                symbol,

            # The candle date is the SECOND
            # actual trading date
            "Date":
                second_day["Date"],

            # First trading day's Open
            "Open":
                first_day["Open"],

            # Highest price over both days
            "High":
                pair["High"].max(),

            # Lowest price over both days
            "Low":
                pair["Low"].min(),

            # Second trading day's Close
            "Close":
                second_day["Close"],

            # Total volume over both days
            "Volume":
                pair["Volume"].sum(),

            # Useful verification fields
            "First_Trading_Date":
                first_day["Date"],

            "Second_Trading_Date":
                second_day["Date"],

            "Trading_Days":
                2

        })


    return pd.DataFrame(
        two_day_rows
    )


two_day_results = Parallel(
    n_jobs=-1
)(
    delayed(
        make_two_day_data
    )(
        symbol,
        group
    )

    for symbol, group
    in two_day_source.groupby(
        "Symbol"
    )
)


two_day_results = [
    x
    for x in two_day_results
    if not x.empty
]


if two_day_results:

    two_day_df = pd.concat(
        two_day_results,
        ignore_index=True
    )

else:

    two_day_df = pd.DataFrame()


if two_day_df.empty:

    print(
        "ERROR: Could not create 2-day candles."
    )

    raise SystemExit(1)


two_day_df = two_day_df.sort_values(
    [
        "Symbol",
        "Date"
    ]
).reset_index(
    drop=True
)


print(
    f"2-day candles created: "
    f"{len(two_day_df):,}"
)


# ════════════════════════════════════════════════════════════════════════════
# VERIFY 2-DAY CANDLES
# ════════════════════════════════════════════════════════════════════════════

print(
    "\n2-DAY CANDLE VALIDATION:"
)


invalid_two_day = two_day_df[
    two_day_df["Trading_Days"] != 2
]


if not invalid_two_day.empty:

    print(
        "ERROR: Invalid 2-day candle detected."
    )

    raise SystemExit(1)


print(
    "Every generated candle contains exactly "
    "2 actual trading days."
)


# Verify the candle date is an actual CSV date

actual_csv_dates = set(
    two_day_source["Date"]
    .dt.normalize()
    .unique()
)


two_day_dates = set(
    two_day_df["Date"]
    .dt.normalize()
    .unique()
)


invalid_dates = (
    two_day_dates
    -
    actual_csv_dates
)


if invalid_dates:

    print(
        "ERROR: Artificial dates detected!"
    )

    print(
        invalid_dates
    )

    raise SystemExit(1)


print(
    "DATE VALIDATION PASSED."
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 6
# 2-DAY EMA 20 / EMA 50
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 6 - 2-DAY EMA 20 / EMA 50")
print("=" * 90)


def process_two_day_symbol(
    symbol,
    group
):

    group = (

        group

        .sort_values(
            "Date"
        )

        .copy()

        .reset_index(
            drop=True
        )

    )


    # EMA 50 requires at least 50 two-day candles
    if len(group) < 50:

        return pd.DataFrame()


    # ════════════════════════════════════════════════════════════════════════
    # EMA 20
    # ════════════════════════════════════════════════════════════════════════

    group["EMA_20"] = (

        group["Close"]

        .ewm(
            span=20,
            adjust=False,
            min_periods=20
        )

        .mean()

    )


    # ════════════════════════════════════════════════════════════════════════
    # EMA 50
    # ════════════════════════════════════════════════════════════════════════

    group["EMA_50"] = (

        group["Close"]

        .ewm(
            span=50,
            adjust=False,
            min_periods=50
        )

        .mean()

    )


    # ════════════════════════════════════════════════════════════════════════
    # RSI 14
    #
    # Calculated using 2-day candles
    # ════════════════════════════════════════════════════════════════════════

    delta = group["Close"].diff()


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


    group["RSI"] = (

        100
        -
        (
            100
            /
            (1 + rs)
        )

    )


    # ════════════════════════════════════════════════════════════════════════
    # 30-CANDLE AVERAGE VOLUME
    #
    # Each candle = 2 trading days
    # Therefore approximately 60 trading days.
    # ════════════════════════════════════════════════════════════════════════

    group["30_2Day_Avg_Volume"] = (

        group["Volume"]

        .rolling(
            30,
            min_periods=30
        )

        .mean()

    )


    # ════════════════════════════════════════════════════════════════════════
    # EMA SLOPES
    #
    # 5 two-day candles = approximately 10 trading days
    # ════════════════════════════════════════════════════════════════════════

    group["Slope_20"] = (

        group["EMA_20"]
        .diff(5)
        /
        5

    )


    group["Slope_50"] = (

        group["EMA_50"]
        .diff(5)
        /
        5

    )


    # ════════════════════════════════════════════════════════════════════════
    # PREVIOUS EMA VALUES
    # ════════════════════════════════════════════════════════════════════════

    group["Previous_EMA_20"] = (
        group["EMA_20"].shift(1)
    )


    group["Previous_EMA_50"] = (
        group["EMA_50"].shift(1)
    )


    # ════════════════════════════════════════════════════════════════════════
    # EMA DIFFERENCE
    # ════════════════════════════════════════════════════════════════════════

    group["EMA_Difference"] = (

        group["EMA_20"]
        -
        group["EMA_50"]

    )


    # ════════════════════════════════════════════════════════════════════════
    # EMA GAP %
    # ════════════════════════════════════════════════════════════════════════

    group["EMA_Gap_%"] = np.where(

        group["EMA_50"] != 0,

        (
            group["EMA_Difference"]
            /
            group["EMA_50"]
            *
            100
        ),

        np.nan

    )


    # ════════════════════════════════════════════════════════════════════════
    # BULLISH 2-DAY EMA CROSSOVER
    #
    # Previous 2-day candle:
    #
    # EMA20 <= EMA50
    #
    # Current 2-day candle:
    #
    # EMA20 > EMA50
    # ════════════════════════════════════════════════════════════════════════

    group["Two_Day_Crossover"] = (

        (
            group["EMA_20"]
            >
            group["EMA_50"]
        )

        &

        (
            group["Previous_EMA_20"]
            <=
            group["Previous_EMA_50"]
        )

    )


    # ════════════════════════════════════════════════════════════════════════
    # EXISTING FILTERS
    #
    # These are now calculated on 2-day candles.
    # ════════════════════════════════════════════════════════════════════════

    valid = group[

        group["Two_Day_Crossover"]

        &

        (
            group["Slope_20"]
            >
            group["Slope_50"]
        )

        &

        group["RSI"].between(
            30,
            70
        )

        &

        (
            group["Volume"]
            >=
            0.3
            *
            group["30_2Day_Avg_Volume"]
        )

        &

        (
            group["Close"]
            >
            group["Close"]
            .rolling(
                60,
                min_periods=60
            )
            .max()
            .shift(1)
            *
            0.95
        )

    ].copy()


    return valid


two_day_ema_results = Parallel(
    n_jobs=-1
)(
    delayed(
        process_two_day_symbol
    )(
        symbol,
        group
    )

    for symbol, group
    in two_day_df.groupby(
        "Symbol"
    )
)


two_day_ema_results = [

    x

    for x in two_day_ema_results

    if not x.empty

]


if two_day_ema_results:

    two_day_crossovers = pd.concat(

        two_day_ema_results,

        ignore_index=True

    )


    # Keep the latest confirmed crossover for each symbol

    two_day_crossovers = (

        two_day_crossovers

        .sort_values(
            "Date",
            ascending=False
        )

        .drop_duplicates(
            "Symbol"
        )

        .reset_index(
            drop=True
        )

    )

else:

    two_day_crossovers = pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# STEP 7
# DISPLAY 2-DAY EMA CROSSOVERS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 7 - 2-DAY EMA CROSSOVER RESULTS")
print("=" * 90)


if two_day_crossovers.empty:

    print(
        "No confirmed 2-day EMA crossover found."
    )

else:

    print(
        "\nCONFIRMED 2-DAY EMA 20 / EMA 50 CROSSOVERS:"
    )


    display_columns = [

        "Symbol",

        "Date",

        "First_Trading_Date",

        "Second_Trading_Date",

        "Close",

        "EMA_20",

        "EMA_50",

        "Previous_EMA_20",

        "Previous_EMA_50",

        "EMA_Difference",

        "EMA_Gap_%",

        "RSI"

    ]


    available_display_columns = [

        col

        for col
        in display_columns

        if col
        in two_day_crossovers.columns

    ]


    print(

        two_day_crossovers[
            available_display_columns
        ].to_string(
            index=False
        )

    )


# ════════════════════════════════════════════════════════════════════════════
# STEP 8
# FORMAT 2-DAY RESULTS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 8 - FORMATTING 2-DAY EMA RESULTS")
print("=" * 90)


# Final safety check:
# every crossover date must exist in original CSV

if not two_day_crossovers.empty:

    original_dates = set(

        two_day_source["Date"]
        .dt.normalize()
        .unique()

    )


    crossover_dates = set(

        pd.to_datetime(
            two_day_crossovers["Date"]
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


    # Format date

    two_day_crossovers["Date"] = (

        pd.to_datetime(
            two_day_crossovers["Date"]
        )

        .dt.strftime(
            "%Y-%m-%d"
        )

    )


    two_day_crossovers["First_Trading_Date"] = (

        pd.to_datetime(
            two_day_crossovers["First_Trading_Date"]
        )

        .dt.strftime(
            "%Y-%m-%d"
        )

    )


    two_day_crossovers["Second_Trading_Date"] = (

        pd.to_datetime(
            two_day_crossovers["Second_Trading_Date"]
        )

        .dt.strftime(
            "%Y-%m-%d"
        )

    )


    # Round numerical values

    numeric_columns = [

        "Open",
        "High",
        "Low",
        "Close",
        "Volume",

        "EMA_20",
        "EMA_50",

        "Previous_EMA_20",
        "Previous_EMA_50",

        "EMA_Difference",
        "EMA_Gap_%",

        "RSI",

        "Slope_20",
        "Slope_50",

        "30_2Day_Avg_Volume"

    ]


    for col in numeric_columns:

        if col in two_day_crossovers.columns:

            two_day_crossovers[col] = (

                pd.to_numeric(

                    two_day_crossovers[col],

                    errors="coerce"

                )

                .round(2)

            )


# ════════════════════════════════════════════════════════════════════════════
# FINAL 2-DAY OUTPUT COLUMN ORDER
# ════════════════════════════════════════════════════════════════════════════

two_day_output_columns = [

    "Symbol",

    "Date",

    "First_Trading_Date",

    "Second_Trading_Date",

    "Open",

    "High",

    "Low",

    "Close",

    "Volume",

    "EMA_20",

    "EMA_50",

    "Previous_EMA_20",

    "Previous_EMA_50",

    "EMA_Difference",

    "Two_Day_Crossover",

    "EMA_Gap_%",

    "RSI",

    "30_2Day_Avg_Volume",

    "Slope_20",

    "Slope_50",

    "Trading_Days"

]


if not two_day_crossovers.empty:

    two_day_crossovers = (

        two_day_crossovers[

            [

                col

                for col
                in two_day_output_columns

                if col
                in two_day_crossovers.columns

            ]

        ]

    )

else:

    two_day_crossovers = pd.DataFrame(

        columns=two_day_output_columns

    )


# ════════════════════════════════════════════════════════════════════════════
# STEP 9
# UPLOAD 2-DAY EMA FILE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 9 - UPLOADING 2-DAY EMA CROSSOVER FILE")
print("=" * 90)


two_day_file = (
    f"2Day_EMA_Cross_for_{today_str}.csv"
)


github_put(
    two_day_file,
    two_day_crossovers
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 10
# DELETE OLD FILES
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("STEP 10 - CLEANING OLD FILES")
print("=" * 90)


# Keep latest historical file

delete_old_github_files(
    "espen_",
    keep=1
)


# Keep latest 2-day EMA file

delete_old_github_files(
    "2Day_EMA_Cross_for_",
    keep=1
)


# Also clean old daily EMA files

delete_old_github_files(
    "EMA_Cross_for_",
    keep=0
)


# Remove old monthly EMA files

delete_old_github_files(
    "Monthly_EMA_Cross_for_",
    keep=0
)


# ════════════════════════════════════════════════════════════════════════════
# REMOVE OLD JUNK CSV FILES
# ════════════════════════════════════════════════════════════════════════════

JUNK_PATTERNS = [

    r"^nepse_\w+\.csv$",

    r"^combined_data\.csv$",

    r"^valid_ema_crossovers\.csv$",

    r"^latest_valid_ema_crossovers\.csv$"

]


headers = {

    "Authorization":
        f"token {GH_TOKEN}",

    "Accept":
        "application/vnd.github+json"

}


repo_url = (
    f"https://api.github.com/repos/"
    f"{GITHUB_REPO}/contents/"
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
            "name",
            ""
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

                file_info["url"],

                headers=headers,

                json={

                    "message":
                        f"Cleanup {file_name}",

                    "sha":
                        file_info["sha"],

                    "branch":
                        "main"

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
    f"Historical file: "
    f"{historical_file}"
)


print(
    f"2-Day EMA file:   "
    f"{two_day_file}"
)


print(
    "EMA calculation: EMA 20 / EMA 50"
)


print(
    "Candle period: 2 ACTUAL NEPSE TRADING DAYS"
)


print(
    "Weekends/holidays: NOT COUNTED"
)


print(
    "Incomplete final 1-day candle: EXCLUDED"
)


print(
    "2-day candle date: SECOND ACTUAL TRADING DATE"
)


print(
    "Monthly EMA calculation: REMOVED"
)


print(
    "Old daily EMA files: REMOVED"
)


print(
    "Old monthly EMA files: REMOVED"
)


print("=" * 100)
print("DONE.")
