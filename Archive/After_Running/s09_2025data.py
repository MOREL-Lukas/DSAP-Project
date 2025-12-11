# s07_2025data.py
"""
Downloads 2025 YTD monthly CAPM data (S&P 500, ^GSPC, ^IRX).
Each dataset is saved separately in Data/2025/.
Includes caching and a --force flag to force re-downloads.
"""

import os
import sys
import pandas as pd
import yfinance as yf
from sp500_symbols import sp500_symbols

# ========================= Config =========================
TICKERS   = sp500_symbols
BENCHMARK = "^GSPC"
RF_SYMBOL = "^IRX"
START     = "2024-12-01"
END       = "2025-09-30"
INTERVAL  = "1mo"
BATCH_SZ  = 80

DATA_DIR   = "Data"
OUTPUT_DIR = os.path.join(DATA_DIR, "2025")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORCE = "--force" in sys.argv  # run with: python s07_2025data.py --force

# ====================== Helper Functions ======================
def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex and numeric columns; drop non-date rows; sort."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()


def save_clean_csv(df: pd.DataFrame, path: str, index_label: str = "Date"):
    """Ensure datetime index and numeric columns; save with consistent header."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.to_csv(path, index=True, index_label=index_label, float_format="%.8f")


def to_month_period_last(df: pd.DataFrame) -> pd.DataFrame:
    """Group by monthly PeriodIndex and take last row in each month."""
    df = df.copy()
    df["__P"] = df.index.to_period("M")
    df = df.groupby("__P").last()
    df.index.name = "Month"
    return df


def normalize_month_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index uses consistent month-end timestamps."""
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp("M")
    else:
        df.index = pd.to_datetime(df.index)
    return df


def load_or_download_csv(filename, download_func):
    """Use cache if valid, otherwise redownload."""
    path = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(path) and not FORCE:
        try:
            # Read file
            df = pd.read_csv(path, index_col=0)
            
            # Parse index safely and explicitly
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
            
            # Drop any invalid rows (e.g., header artifacts)
            df = df[~df.index.isna()]
            
            # Ensure not empty
            if df.empty:
                raise ValueError("File is empty after cleaning.")
            
            print(f"✅ Using cached file: {filename}")
            return df
        
        except Exception as e:
            print(f"⚠️ Cached file '{filename}' invalid ({e}), re-downloading...")
    
    # Redownload if missing or invalid
    print(f"⬇️ Downloading new data: {filename}")
    df = download_func()
    save_clean_csv(df, path)
    print(f"💾 Saved to {path}")
    return df
# ==================== Market Index (^GSPC) ====================
def download_market():
    print(f"⬇️ Downloading {BENCHMARK} market index...")
    df = yf.download(
        BENCHMARK, start=START, end=END, interval=INTERVAL,
        auto_adjust=True, progress=False
    )

    # Retry with daily interval if monthly is empty
    if df.empty:
        print("⚠️ No monthly data returned, retrying with daily interval...")
        df = yf.download(
            BENCHMARK, start=START, end=END, interval="1d",
            auto_adjust=True, progress=False
        )

    if df.empty:
        raise ValueError(f"No data returned for {BENCHMARK} even after retry.")

    # Handle MultiIndex or single-level columns
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", BENCHMARK) in df.columns:
            df = df[("Close", BENCHMARK)]
        elif ("Adj Close", BENCHMARK) in df.columns:
            df = df[("Adj Close", BENCHMARK)]
        else:
            df = df[df.columns[0]]
    else:
        if "Close" in df.columns:
            df = df["Close"]
        elif "Adj Close" in df.columns:
            df = df["Adj Close"]
        else:
            df = df.iloc[:, 0]

    df = df.to_frame("MKT")
    df = coerce_numeric_df(df)
    df = to_month_period_last(df)
    df = normalize_month_index(df)
    print(f"✅ Downloaded {len(df)} monthly rows for {BENCHMARK}")
    return df


mkt_mp = load_or_download_csv("market_adj_close_2025.csv", download_market)

# ==================== Risk-Free Rate (^IRX) ===================
rf_daily = load_or_download_csv(
    "rf_irx_daily_2025.csv",
    lambda: yf.download(
        RF_SYMBOL, start=START, end=END, interval="1d",
        auto_adjust=False, progress=False
    )[["Adj Close"]].rename(columns={"Adj Close": "RF"})
)
rf_daily = coerce_numeric_df(rf_daily)

# Convert annualized % → effective monthly rate
rf_yield_pct = rf_daily["RF"].dropna().resample("ME").last()
rf_monthly = ((1 + (rf_yield_pct / 100.0)) ** (1 / 12)) - 1.0
if not isinstance(rf_monthly, pd.DataFrame):
    rf_monthly = rf_monthly.to_frame(name="RFm")
else:
    rf_monthly.columns = ["RFm"]
save_clean_csv(rf_monthly, os.path.join(OUTPUT_DIR, "rf_irx_monthly_2025.csv"))

# ==================== S&P 500 Stock Prices ====================
def download_sp500():
    frames = []
    for block in chunks(TICKERS, BATCH_SZ):
        df = yf.download(
            block, start=START, end=END, interval=INTERVAL,
            auto_adjust=False, progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Adj Close"]
        else:
            df = df[["Adj Close"]]
        frames.append(df)
    df_all = pd.concat(frames, axis=1)
    df_all = coerce_numeric_df(df_all)
    df_all = to_month_period_last(df_all)
    df_all = normalize_month_index(df_all)
    return df_all


prices_mp = load_or_download_csv("sp500_adj_close_2025.csv", download_sp500)

# ==================== Summary ===============================
print("\n✅ Download complete — saved in 'Data/2025':")
for fname in [
    "market_adj_close_2025.csv",
    "rf_irx_daily_2025.csv",
    "rf_irx_monthly_2025.csv",
    "sp500_adj_close_2025.csv",
]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(f" - {fname}: {'✅' if os.path.exists(fpath) else '⚠️ Missing'}")

if not prices_mp.empty:
    print(f"\n📅 Coverage: {prices_mp.index.min().date()} → {prices_mp.index.max().date()}")
else:
    print("\n⚠️ No price data found — check ticker list or internet connection.")
