"""
NEPSE 5-Layer Fundamental Filter — GitHub Actions version
- Reads latest Fundamental_*.csv from: iamsrijit/Nepse/Fundamental/ (public)
- Applies same normalise → score → 5-layer filter pipeline as the Colab widget
- Uploads top-15 results as Filter_YYYY-MM-DD.csv to:
    iamsrijit0/Nepse / Claude Fundamentals/  (uses GH_TOKEN)
"""

import os
import io
import re
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
GH_TOKEN      = os.environ["GH_TOKEN"]
GH_HEADERS    = {"Authorization": f"token {GH_TOKEN}",
                 "Accept": "application/vnd.github.v3+json"}

# ── Default filter thresholds (mirror widget defaults) ────────────────────────
LAYER1_MIN_SCORE     = 18      # ① min composite score
LAYER2_MAX_PE_FIN    = 25.0    # ② max P/E — financial
LAYER2_MAX_PE_NONF   = 30.0    # ② max P/E — non-financial
LAYER3_MIN_ROA_FIN   = 0.8     # ③ min ROA % — financial
LAYER3_MIN_ROA_NONF  = 5.0     # ③ min ROA % — non-financial
LAYER4_MAX_PER_SEC   = 2       # ④ max stocks per sector
LAYER5_MAX_NPG       = 500.0   # ⑤ max net-profit growth % (base-effect flag)
TOP_N                = 15      # final rows written to CSV

FINANCIAL = {
    "Commercial Banks", "Development Bank", "Finance",
    "Microcredit", "Life Insurance", "Non Life Insurance", "ReInsurance"
}
EXCLUDE = {"Bond", "Mutual Fund", "Organized Fund"}


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 – Fetch latest Fundamental_*.csv from source repo
# ════════════════════════════════════════════════════════════════════════════
def get_latest_fundamental_url() -> tuple[str, str]:
    api_url = f"https://api.github.com/repos/{SOURCE_REPO}/contents/{SOURCE_FOLDER}"
    r = requests.get(api_url, timeout=20)
    r.raise_for_status()

    files = [
        f for f in r.json()
        if isinstance(f, dict)
        and f.get("name", "").startswith("Fundamental_")
        and f.get("name", "").endswith(".csv")
    ]
    if not files:
        raise ValueError(f"No Fundamental_*.csv found in {SOURCE_REPO}/{SOURCE_FOLDER}")

    files.sort(key=lambda f: f["name"], reverse=True)
    latest = files[0]
    print(f"[source] Latest file : {latest['name']}")
    print(f"[source] Download URL: {latest['download_url']}")
    return latest["download_url"], latest["name"]


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 – Parse & normalise
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
    "EPS (Trailing)":                "eps",
    "P/E Ratio":                     "pe",
    "P/B Ratio":                     "pb",
    "Net Profit % Change":           "np_chg",
    "Total Revenue % Change":        "rev_chg",
    "Annualized EPS % Change":       "eps_chg",
    "Book Value per Share % Change": "bv_chg",
    "Monthly Change (%)":            "monthly",
    "3-Month Change (%)":            "three_m",
    "Yearly Change (%)":             "yearly",
    "Today's Price":                 "price",
    "Annualized EPS (Latest)":       "eps_ann",
    "Book Value per Share (Latest)": "bv",
    "Total Assets (Latest)":         "total_assets",
    "Total Liabilities (Latest)":    "total_liab",
}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    for l_col, p_col, pct_col in GROWTH_PAIRS:
        if pct_col not in df.columns and l_col in df.columns and p_col in df.columns:
            l = df[l_col].apply(clean_num)
            p = df[p_col].apply(clean_num)
            df[pct_col] = ((l - p) / p.abs() * 100).replace([np.inf, -np.inf], np.nan)

    if "Today's Price" not in df.columns:
        for alt in ["Price", "LTP", "Close"]:
            if alt in df.columns:
                df["Today's Price"] = df[alt]
                break

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

    if pd.notna(pe):
        s += 3 if pe < 15 else 2 if pe < 25 else 1 if pe < 40 else 0
    if pd.notna(pb):
        s += 3 if pb < 2  else 2 if pb < 4  else 1 if pb < 6  else 0
    if pd.notna(npg):
        s += 3 if npg > 50 else 2 if npg > 10 else 1 if npg >= 0 else 0
    if pd.notna(rev):
        s += 2 if rev > 50 else 1 if rev > 10 else 0
    if pd.notna(eg):
        s += 2 if eg > 20  else 1 if eg >= 0  else 0
    if pd.notna(mo)  and mo  > 0: s += 1
    if pd.notna(tm)  and tm  > 0: s += 1
    if pd.notna(bvc) and bvc > 0: s += 1
    if pd.notna(de):
        if fin:
            s += 3 if de < 8   else 2 if de < 12 else 1 if de < 18 else 0
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
# STEP 4 – 5-Layer filter (mirrors the Colab widget logic exactly)
# ════════════════════════════════════════════════════════════════════════════
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only the rows that pass all 5 layers.
    Layer 5 (base-effect flag) removes stocks as 'warn' — excluded from final picks.
    """
    d = df.copy()
    d["_status"] = "pass"
    d["_reason"] = ""

    # Layer 1 — Score gate
    mask = d["score"] < LAYER1_MIN_SCORE
    d.loc[mask, "_status"] = "fail"
    d.loc[mask, "_reason"] = d.loc[mask, "score"].apply(
        lambda s: f"Score {int(s)} < {LAYER1_MIN_SCORE}"
    )
    alive = d[d["_status"] == "pass"]
    print(f"[L1] Score gate      : {mask.sum():3d} removed  | {len(alive):3d} remain")

    # Layer 2 — Valuation sanity (P/E)
    def pe_bad(r):
        val = r.get("pe", np.nan)
        if pd.isna(val): return False
        lim = LAYER2_MAX_PE_FIN if r.get("Sector", "") in FINANCIAL else LAYER2_MAX_PE_NONF
        return val > lim

    mask2 = alive.apply(pe_bad, axis=1)
    bad2  = alive.index[mask2]
    d.loc[bad2, "_status"] = "fail"
    d.loc[bad2, "_reason"] = alive.loc[mask2, "pe"].apply(
        lambda v: f"P/E {v:.1f}x above limit"
    )
    alive = d[d["_status"] == "pass"]
    print(f"[L2] Valuation (P/E) : {mask2.sum():3d} removed  | {len(alive):3d} remain")

    # Layer 3 — Capital efficiency (ROA)
    def roa_bad(r):
        val = r.get("ROA", np.nan)
        if pd.isna(val): return False
        lim = LAYER3_MIN_ROA_FIN if r.get("Sector", "") in FINANCIAL else LAYER3_MIN_ROA_NONF
        return val < lim

    mask3 = alive.apply(roa_bad, axis=1)
    bad3  = alive.index[mask3]
    d.loc[bad3, "_status"] = "fail"
    d.loc[bad3, "_reason"] = alive.loc[mask3, "ROA"].apply(
        lambda v: f"ROA {v:.2f}% below limit"
    )
    alive = d[d["_status"] == "pass"]
    print(f"[L3] ROA gate        : {mask3.sum():3d} removed  | {len(alive):3d} remain")

    # Layer 4 — Sector overlap (keep top N per sector by score)
    remove_l4 = []
    for sec, grp in alive.groupby("Sector"):
        if len(grp) > LAYER4_MAX_PER_SEC:
            keep = set(grp.nlargest(LAYER4_MAX_PER_SEC, "score").index)
            remove_l4 += [i for i in grp.index if i not in keep]

    for idx in remove_l4:
        sec_name = d.at[idx, "Sector"]
        d.at[idx, "_status"] = "fail"
        d.at[idx, "_reason"] = f"Overlap: >{LAYER4_MAX_PER_SEC} stocks in {sec_name}"
    alive = d[d["_status"] == "pass"]
    print(f"[L4] Sector overlap  : {len(remove_l4):3d} removed  | {len(alive):3d} remain")

    # Layer 5 — Growth quality check (base-effect flag → excluded)
    mask5 = alive["np_chg"].apply(lambda v: pd.notna(v) and v > LAYER5_MAX_NPG)
    bad5  = alive.index[mask5]
    d.loc[bad5, "_status"] = "warn"
    d.loc[bad5, "_reason"] = alive.loc[mask5, "np_chg"].apply(
        lambda v: f"Profit growth +{v:.0f}% — likely base effect"
    )
    alive = d[d["_status"] == "pass"]
    print(f"[L5] Growth quality  : {mask5.sum():3d} flagged   | {len(alive):3d} remain (final)")

    return alive.sort_values("score", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 – Upload to destination repo
# ════════════════════════════════════════════════════════════════════════════
def github_put(file_name: str, df: pd.DataFrame):
    path    = f"{DEST_FOLDER}/{file_name}"
    csv_b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    url     = (f"https://api.github.com/repos/{DEST_REPO}/contents/"
               f"{requests.utils.quote(path)}")

    r   = requests.get(url, headers=GH_HEADERS, timeout=15)
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {"message": f"filter: add {file_name}", "content": csv_b64, "branch": "main"}
    if sha:
        payload["sha"] = sha

    resp = requests.put(url, headers=GH_HEADERS, json=payload, timeout=30)
    if resp.status_code in (200, 201):
        print(f"[upload] '{path}' → {DEST_REPO}  ✓")
    else:
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")


def delete_old_dest_files(prefix: str, keep: int = 1):
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
                  "sha": f["sha"], "branch": "main"},
        )
        status = "✓" if dr.status_code == 200 else f"✗ {dr.status_code}"
        print(f"[cleanup] Deleted {f['name']}  {status}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    # 1. Download latest Fundamental CSV
    raw_url, source_filename = get_latest_fundamental_url()
    resp = requests.get(raw_url, timeout=30)
    resp.raise_for_status()
    df_raw = pd.read_csv(io.StringIO(resp.text))
    print(f"[parse]  Rows loaded: {len(df_raw)}")

    # 2. Normalise
    df = normalize_df(df_raw.copy())

    # 3. Pre-filter: drop excluded sectors, require positive EPS & P/E
    df = df[df["Sector"].notna() & ~df["Sector"].isin(EXCLUDE)].copy()
    if "EPS (Trailing)" in df.columns:
        df = df[df["EPS (Trailing)"].apply(clean_num) > 0].copy()
    if "P/E Ratio" in df.columns:
        df = df[df["P/E Ratio"].apply(clean_num) > 0].copy()

    # 4. Build aliases and derived ratios
    df = build_numeric_aliases(df)
    df = derive_ratios(df)

    # 5. Score
    df["score"] = df.apply(score_row, axis=1)
    df["Tier"]  = df["score"].apply(
        lambda s: "Strong" if s >= 20 else "Good" if s >= 14 else "Watch"
    )
    print(f"[score]  {len(df)} stocks scored | "
          f"Strong: {(df['Tier']=='Strong').sum()} | "
          f"Good: {(df['Tier']=='Good').sum()}")

    # 6. Apply 5-layer filter
    print(f"\n{'='*50}")
    print(f"  5-Layer Filter  —  {today_str}")
    print(f"{'='*50}")
    final = apply_filters(df)
    print(f"{'='*50}")

    # 7. Build output CSV — top 15 picks only
    output_cols = [
        "Ticker", "Stock Name", "Sector",
        "Today's Price", "EPS (Trailing)", "P/E Ratio", "P/B Ratio",
        "DE", "ROE", "ROA",
        "Net Profit % Change", "Total Revenue % Change",
        "Annualized EPS % Change", "Book Value per Share % Change",
        "Monthly Change (%)", "3-Month Change (%)", "Yearly Change (%)",
        "score", "Tier",
    ]
    output_cols = [c for c in output_cols if c in final.columns]

    for col in ["DE", "ROE", "ROA"]:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors="coerce").round(2)

    top15 = final[output_cols].head(TOP_N).reset_index(drop=True)
    top15.index += 1  # rank from 1

    print(f"\n  Top {TOP_N} picks after all 5 layers:")
    print(top15[["Ticker", "Sector", "score", "Tier",
                  "P/E Ratio", "ROE", "ROA"]].to_string())
    print(f"{'='*50}")

    # 8. Upload
    out_filename = f"Filter_{today_str}.csv"
    github_put(out_filename, top15)

    # 9. Clean up old Filter_ files — keep only the latest
    delete_old_dest_files("Filter_", keep=1)

    print("\nDone.")


if __name__ == "__main__":
    main()
