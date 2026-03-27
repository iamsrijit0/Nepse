"""
NEPSE Fundamental Stock Screener — GitHub Actions version
- Reads latest Fundamental_*.csv from: iamsrijit/Nepse (public repo, no token needed)
- Scores stocks on 24-point system (P/E, P/B, D/E, ROE, ROA, growth metrics)
- Uploads scored CSV to: iamsrijit0/Nepse / Claude Fundamentals/ (uses GH_TOKEN)
"""

import os
import re
import io
import json
import base64
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ── Constants ────────────────────────────────────────────────────────────────
SOURCE_REPO   = "iamsrijit/Nepse"
SOURCE_FOLDER = "Fundamental"
DEST_REPO     = "iamsrijit0/Nepse"
DEST_FOLDER   = "Claude Fundamentals"
GH_TOKEN      = os.environ["GH_TOKEN"]          # set in repo secrets
GH_HEADERS    = {"Authorization": f"token {GH_TOKEN}",
                 "Accept": "application/vnd.github.v3+json"}

FINANCIAL = {
    "Commercial Banks", "Development Bank", "Finance",
    "Microcredit", "Life Insurance", "Non Life Insurance", "ReInsurance"
}
EXCLUDE = {"Bond", "Mutual Fund", "Organized Fund"}


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 – Fetch latest Fundamental_*.csv from source repo (public, no token)
# ════════════════════════════════════════════════════════════════════════════
def get_latest_fundamental_url() -> tuple[str, str]:
    """Return (raw_url, filename) for the most recent Fundamental_*.csv."""
    api_url  = f"https://api.github.com/repos/{SOURCE_REPO}/contents/{SOURCE_FOLDER}"
    r = requests.get(api_url, timeout=20)
    r.raise_for_status()

    files = [
        f for f in r.json()
        if isinstance(f, dict)
        and f.get("name", "").startswith("Fundamental_")
        and f.get("name", "").endswith(".csv")
    ]
    if not files:
        raise ValueError(f"No Fundamental_*.csv files found in {SOURCE_REPO}/{SOURCE_FOLDER}")

    # Sort by name descending (date is embedded: Fundamental_YYYY-MM-DD.csv)
    files.sort(key=lambda f: f["name"], reverse=True)
    latest = files[0]
    print(f"[source] Latest file: {latest['name']}")
    print(f"[source] URL        : {latest['download_url']}")
    return latest["download_url"], latest["name"]


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 – Parse & normalise CSV
# ════════════════════════════════════════════════════════════════════════════
def clean_num(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").replace("%", "").replace("+", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


COLUMN_ALIASES = {
    "Stock name":              "Stock Name",
    "EPS":                     "EPS (Trailing)",
    "PE ratio":                "P/E Ratio",
    "PB ratio":                "P/B Ratio",
    "T Rev L":                 "Total Revenue (Latest Quarter)",
    "T Rev P":                 "Total Revenue (Previous Quarter)",
    "Gross Profit L":          "Gross Profit (Latest)",
    "Gross Profit P":          "Gross Profit (Previous)",
    "Net Profit L":            "Net Profit (Latest)",
    "Net Profit P":            "Net Profit (Previous)",
    "% change in Net Profit":  "Net Profit % Change",
    "Eps Annualized L":        "Annualized EPS (Latest)",
    "Eps Annualized P":        "Annualized EPS (Previous)",
    "Book Value Per Share L":  "Book Value per Share (Latest)",
    "Book Value Per Share P":  "Book Value per Share (Previous)",
    "Total Asset L":           "Total Assets (Latest)",
    "Total Asset P":           "Total Assets (Previous)",
    "Total Liabilities L":     "Total Liabilities (Latest)",
    "Total Liabilities P":     "Total Liabilities (Previous)",
    "Paid Up Capital L":       "Paid-up Capital (Latest)",
    "Paid Up Capital P":       "Paid-up Capital (Previous)",
    "Reserves L":              "Reserves (Latest)",
    "Reserves P":              "Reserves (Previous)",
}

GROWTH_PAIRS = [
    ("Annualized EPS (Latest)",        "Annualized EPS (Previous)",        "Annualized EPS % Change"),
    ("Total Revenue (Latest Quarter)", "Total Revenue (Previous Quarter)", "Total Revenue % Change"),
    ("Net Profit (Latest)",            "Net Profit (Previous)",            "Net Profit % Change"),
    ("Book Value per Share (Latest)",  "Book Value per Share (Previous)",  "Book Value per Share % Change"),
]

NUM_COLS = {
    "EPS (Trailing)":               "eps",
    "P/E Ratio":                    "pe",
    "P/B Ratio":                    "pb",
    "Net Profit % Change":          "np_chg",
    "Total Revenue % Change":       "rev_chg",
    "Annualized EPS % Change":      "eps_chg",
    "Book Value per Share % Change":"bv_chg",
    "Monthly Change (%)":           "monthly",
    "3-Month Change (%)":           "three_m",
    "Yearly Change (%)":            "yearly",
    "Today's Price":                "price",
    "Annualized EPS (Latest)":      "eps_ann",
    "Book Value per Share (Latest)":"bv",
    "Total Assets (Latest)":        "total_assets",
    "Total Liabilities (Latest)":   "total_liab",
}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    # Derive % change columns if missing
    for l_col, p_col, pct_col in GROWTH_PAIRS:
        if pct_col not in df.columns and l_col in df.columns and p_col in df.columns:
            l = df[l_col].apply(clean_num)
            p = df[p_col].apply(clean_num)
            df[pct_col] = ((l - p) / p.abs() * 100).replace([np.inf, -np.inf], np.nan)

    # Normalise price column name
    if "Today's Price" not in df.columns:
        for alt in ["Price", "LTP", "Close"]:
            if alt in df.columns:
                df["Today's Price"] = df[alt]
                break

    # Ensure momentum columns exist
    for col in ["Daily Change (%)", "Weekly Change (%)", "Monthly Change (%)",
                "3-Month Change (%)", "Yearly Change (%)"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def build_numeric_aliases(df: pd.DataFrame) -> pd.DataFrame:
    for col, alias in NUM_COLS.items():
        df[alias] = df[col].apply(clean_num) if col in df.columns else np.nan
    return df


def derive_ratios(df: pd.DataFrame) -> pd.DataFrame:
    has_bs  = df["total_assets"].notna().any() and df["total_liab"].notna().any()
    has_ebv = df["eps_ann"].notna().any()       and df["bv"].notna().any()

    if has_bs:
        df["equity"] = df["total_assets"] - df["total_liab"]
        df["DE"] = df.apply(
            lambda r: r["total_liab"] / r["equity"]
            if pd.notna(r["equity"]) and r["equity"] > 0 else np.nan, axis=1)
    else:
        df["DE"] = np.nan

    if has_ebv:
        df["ROE"] = df.apply(
            lambda r: (r["eps_ann"] / r["bv"] * 100)
            if pd.notna(r["eps_ann"]) and pd.notna(r["bv"]) and r["bv"] > 0 else np.nan, axis=1)
    else:
        df["ROE"] = np.nan

    if has_bs and has_ebv:
        df["ROA"] = df.apply(
            lambda r: r["ROE"] / (1 + r["DE"])
            if pd.notna(r.get("ROE")) and pd.notna(r.get("DE")) else np.nan, axis=1)
    else:
        df["ROA"] = np.nan

    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 – Score (max 24 pts)
# ════════════════════════════════════════════════════════════════════════════
def score_row(r) -> int:
    s   = 0
    fin = r.get("Sector", "") in FINANCIAL

    def v(k):
        val = r.get(k, np.nan)
        return np.nan if (val is None or (isinstance(val, float) and np.isnan(val))) else val

    pe  = v("pe");      pb  = v("pb");      npg = v("np_chg")
    rev = v("rev_chg"); eg  = v("eps_chg"); bvc = v("bv_chg")
    mo  = v("monthly"); tm  = v("three_m")
    de  = v("DE");      roe = v("ROE");     roa = v("ROA")

    # Valuation
    if pd.notna(pe):
        s += 3 if pe < 15 else 2 if pe < 25 else 1 if pe < 40 else 0
    if pd.notna(pb):
        s += 3 if pb < 2  else 2 if pb < 4  else 1 if pb < 6  else 0
    # Growth
    if pd.notna(npg):
        s += 3 if npg > 50 else 2 if npg > 10 else 1 if npg >= 0 else 0
    if pd.notna(rev):
        s += 2 if rev > 50 else 1 if rev > 10 else 0
    if pd.notna(eg):
        s += 2 if eg > 20  else 1 if eg >= 0  else 0
    # Momentum
    if pd.notna(mo)  and mo  > 0: s += 1
    if pd.notna(tm)  and tm  > 0: s += 1
    if pd.notna(bvc) and bvc > 0: s += 1
    # Balance-sheet ratios (sector-aware)
    if pd.notna(de):
        if fin:
            s += 3 if de < 8  else 2 if de < 12 else 1 if de < 18 else 0
        else:
            s += 3 if de < 0.5 else 2 if de < 1.5 else 1 if de < 3  else 0
    if pd.notna(roe):
        if fin:
            s += 3 if roe > 15 else 2 if roe > 10 else 1 if roe > 8 else 0
        else:
            s += 3 if roe > 15 else 2 if roe > 10 else 1 if roe > 5 else 0
    if pd.notna(roa):
        if fin:
            s += 2 if roa > 1.5 else 1 if roa > 0.8 else 0
        else:
            s += 2 if roa > 5   else 1 if roa > 2   else 0
    return s


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 – Upload result to destination repo (iamsrijit0/Nepse)
# ════════════════════════════════════════════════════════════════════════════
def github_put(file_name: str, df: pd.DataFrame):
    """Upload df as CSV to DEST_REPO / DEST_FOLDER / file_name."""
    path    = f"{DEST_FOLDER}/{file_name}"
    csv_b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    url     = f"https://api.github.com/repos/{DEST_REPO}/contents/{requests.utils.quote(path)}"

    r   = requests.get(url, headers=GH_HEADERS, timeout=15)
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {
        "message": f"screener: add {file_name}",
        "content": csv_b64,
        "branch":  "main",
    }
    if sha:
        payload["sha"] = sha

    resp = requests.put(url, headers=GH_HEADERS, json=payload, timeout=30)
    if resp.status_code in (200, 201):
        print(f"[upload] '{path}' → {DEST_REPO}  ✓")
    else:
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")


def delete_old_dest_files(prefix: str, keep: int = 1):
    """Keep only the `keep` most recent files matching prefix in DEST_FOLDER."""
    url = (f"https://api.github.com/repos/{DEST_REPO}/contents/"
           f"{requests.utils.quote(DEST_FOLDER)}")
    r   = requests.get(url, headers=GH_HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"[cleanup] Could not list {DEST_FOLDER}: {r.status_code}")
        return

    matched = sorted(
        [f for f in r.json()
         if isinstance(f, dict)
         and f.get("name", "").startswith(prefix)
         and f.get("name", "").endswith(".csv")],
        key=lambda f: f["name"],
        reverse=True,
    )

    for f in matched[keep:]:
        dr = requests.delete(
            f["url"], headers=GH_HEADERS, timeout=15,
            json={"message": f"cleanup: remove {f['name']}",
                  "sha":     f["sha"],
                  "branch":  "main"},
        )
        status = "✓" if dr.status_code == 200 else f"✗ {dr.status_code}"
        print(f"[cleanup] Deleted {f['name']}  {status}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    # 1. Download source CSV
    raw_url, source_filename = get_latest_fundamental_url()
    resp = requests.get(raw_url, timeout=30)
    resp.raise_for_status()
    df_raw = pd.read_csv(io.StringIO(resp.text))
    print(f"[parse]  Rows loaded from source: {len(df_raw)}")

    # 2. Normalise
    df = normalize_df(df_raw.copy())

    # 3. Filter: exclude non-fundamental sectors, require positive EPS & P/E
    df = df[df["Sector"].notna() & ~df["Sector"].isin(EXCLUDE)]
    df = df[df.get("EPS (Trailing)", pd.Series(dtype=float)).apply(clean_num) > 0] \
        if "EPS (Trailing)" in df.columns else df
    df = df[df.get("P/E Ratio", pd.Series(dtype=float)).apply(clean_num) > 0] \
        if "P/E Ratio" in df.columns else df
    df = df.copy()

    # 4. Build numeric aliases & derived ratios
    df = build_numeric_aliases(df)
    df = derive_ratios(df)

    # 5. Score
    df["Score"]  = df.apply(score_row, axis=1)
    df["Tier"]   = df["Score"].apply(
        lambda s: "Strong" if s >= 20 else "Good" if s >= 14 else "Watch"
    )

    total  = len(df)
    strong = (df["Tier"] == "Strong").sum()
    good   = (df["Tier"] == "Good").sum()

    print(f"\n{'='*55}")
    print(f"  NEPSE Screener  —  {today_str}")
    print(f"{'='*55}")
    print(f"  Stocks screened : {total}")
    print(f"  ★ Strong (≥20)  : {strong}")
    print(f"  ✓ Good   (14-19): {good}")
    print(f"  ~ Watch  (<14)  : {total - strong - good}")
    print(f"\n  D/E available   : {df['DE'].notna().sum()}")
    print(f"  ROE available   : {df['ROE'].notna().sum()}")
    print(f"  ROA available   : {df['ROA'].notna().sum()}")

    # 6. Build output: top 50 scored, clean column set
    output_cols = [
        "Ticker", "Stock Name", "Sector",
        "Today's Price", "EPS (Trailing)", "P/E Ratio", "P/B Ratio",
        "DE", "ROE", "ROA",
        "Net Profit % Change", "Total Revenue % Change",
        "Annualized EPS % Change", "Book Value per Share % Change",
        "Monthly Change (%)", "3-Month Change (%)", "Yearly Change (%)",
        "Score", "Tier",
    ]
    # Only keep columns that actually exist
    output_cols = [c for c in output_cols if c in df.columns]

    # Round float columns for readability
    for col in ["DE", "ROE", "ROA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    top50 = (
        df.nlargest(50, "Score")[output_cols]
          .reset_index(drop=True)
    )
    top50.index += 1   # rank starts at 1

    print(f"\n  Top 10 preview:")
    print(top50[["Ticker", "Sector", "Score", "Tier",
                  "P/E Ratio", "P/B Ratio", "DE", "ROE", "ROA"]
               ].head(10).to_string())
    print(f"{'='*55}")

    # 7. Upload to destination repo
    out_filename = f"Fundamental_Screener_{today_str}.csv"
    github_put(out_filename, top50)

    # 8. Keep only latest file in destination folder
    delete_old_dest_files("Fundamental_Screener_", keep=1)

    print("\nDone.")


if __name__ == "__main__":
    main()
