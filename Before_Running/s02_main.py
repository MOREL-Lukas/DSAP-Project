# ============================================================
# s02_main.py — Vectorized CAPM computation + Robust Data Fetch
# ============================================================

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sp500_symbols import sp500_symbols

# ========================= Config =========================
TICKERS   = sp500_symbols
BENCHMARK = "^GSPC"         # S&P 500 Index
RF_SYMBOL = "^IRX"          # 13-week T-bill yield (% annualized)
START     = "2020-01-01"
END       = "2024-12-31"
INTERVAL  = "1mo"
BATCH_SZ  = 80
MIN_MONTHS_COVERAGE = 48

DATA_DIR   = "Data"
OUTPUT_DIR = "Output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== Helpers ===========================

def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Force DataFrame to numeric with datetime index."""
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_index()

def read_csv_smart(path: str) -> pd.DataFrame:
    """
    Read a CSV file, automatically cleaning messy headers like
    'Price,MKT' / 'Ticker,^GSPC' / 'Date,' style.
    """
    import pandas as pd
    
    # Read normally first
    df = pd.read_csv(path)
    
    # Detect nonstandard headers (like Yahoo or Excel exports)
    if "Date" not in df.columns and df.iloc[0, 0].lower() in ["ticker", "price"]:
        # Skip first two metadata rows
        df = pd.read_csv(path, skiprows=2)
    
    # Normalize headers
    df.columns = [c.strip() for c in df.columns]
    
    # Parse dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    
    # Coerce numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    return df.dropna(how="all")

def save_clean_csv(df: pd.DataFrame, path: str, index_label: str = "Date"):
    df = coerce_numeric_df(df)
    df.to_csv(path, index=True, index_label=index_label)

def load_or_download_csv(filename: str, download_func) -> pd.DataFrame:
    """Load cached CSV if valid, otherwise download anew."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"✅ Using cached file: {filename}")
        try:
            df = read_csv_smart(path)
            if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Cached file unusable.")
            return df
        except Exception as e:
            print(f"⚠️ Cached '{filename}' invalid ({e}). Re-downloading...")
            try: os.remove(path)
            except: pass
    print(f"⬇️ Downloading new data: {filename}")
    df = download_func()
    save_clean_csv(df, path)
    print(f"💾 Saved to {path}")
    return read_csv_smart(path)

def to_month_period_last(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to month-end frequency (last value each month)."""
    tmp = df.copy()
    tmp["__P"] = tmp.index.to_period("M")
    tmp = tmp.groupby("__P").last()
    tmp.index.name = "Month"
    return tmp

# =================== Download / Prepare Data =====================

# 1) Market Benchmark (^GSPC)
def fetch_market():
    df = yf.download(
        BENCHMARK, start=START, end=END, interval=INTERVAL,
        progress=False, auto_adjust=False
    )
    # Handle both single and MultiIndex column structures
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", BENCHMARK) in df.columns:
            s = df[("Adj Close", BENCHMARK)]
        elif ("Close", BENCHMARK) in df.columns:
            s = df[("Close", BENCHMARK)]
        else:
            raise KeyError(f"Could not find 'Adj Close' or 'Close' for {BENCHMARK} in {df.columns}")
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            raise KeyError(f"Could not find 'Adj Close' or 'Close' columns in {df.columns}")

    return s.to_frame(name="MKT")
    
mkt_df = load_or_download_csv("market_adj_close.csv", fetch_market)
mkt = mkt_df["MKT"].dropna()
# 2) Risk-Free Rate (^IRX)
rf_daily_path = os.path.join(DATA_DIR, "rf_irx_daily.csv")

def fetch_rf():
    df = yf.download(RF_SYMBOL, start=START, end=END, interval="1d", progress=False)
    if df.empty or all(c not in df.columns for c in ["Adj Close", "Close"]):
        print("⚠️ ^IRX unavailable, falling back to ^FVX (5-year yield).")
        df = yf.download("^FVX", start=START, end=END, interval="1d", progress=False)
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[[col]].rename(columns={col: "RF"})

if os.path.exists(rf_daily_path):
    print("✅ Using cached file: rf_irx_daily.csv")
    rf_daily = read_csv_smart(rf_daily_path)
else:
    print("⬇️ Downloading daily ^IRX...")
    rf_daily = fetch_rf()
    save_clean_csv(rf_daily, rf_daily_path)

if rf_daily.empty:
    raise ValueError("❌ Risk-free rate data is empty.")

# --- Compute monthly risk-free yields ---
rf_obj = rf_daily["RF"]
if isinstance(rf_obj, pd.DataFrame):
    rf_obj = rf_obj.squeeze()  # convert to Series if single column

rf_yield_pct = rf_obj.dropna().resample("ME").last()
if isinstance(rf_yield_pct, pd.DataFrame):
    rf_yield_pct = rf_yield_pct.squeeze()  # ensure Series

# Effective monthly risk-free rate
rf_monthly = ((1 + rf_yield_pct / 100.0) ** (1/12)) - 1.0
rf_monthly.name = "RFm"
rf_monthly.index = rf_monthly.index.to_period("M").to_timestamp("M")

# Save both versions
save_clean_csv(rf_yield_pct.to_frame(name="RF_yield_pct"), os.path.join(DATA_DIR, "rf_irx_yield_monthly.csv"))
save_clean_csv(rf_monthly.to_frame(name="RFm"), os.path.join(DATA_DIR, "rf_irx_monthly.csv"))
print("💾 Saved effective monthly RF rate → Data/rf_irx_monthly.csv")

# 3) S&P 500 Stock Prices
def fetch_prices(block):
    df = yf.download(block, start=START, end=END, interval=INTERVAL, progress=False)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Adj Close"] if "Adj Close" in df.columns.levels[0] else df["Close"]
    elif "Adj Close" in df.columns:
        df = df["Adj Close"]
    elif "Close" in df.columns:
        df = df["Close"]
    return df

prices_path = os.path.join(DATA_DIR, "sp500_adj_close.csv")
if os.path.exists(prices_path):
    print(f"✅ Using cached S&P 500 prices: {prices_path}")
    prices = read_csv_smart(prices_path)
else:
    print("⬇️ Downloading S&P 500 stock prices (batched)...")
    frames = []
    for block in chunks(TICKERS, BATCH_SZ):
        frames.append(fetch_prices(block))
    prices = pd.concat(frames, axis=1)
    save_clean_csv(prices, prices_path)
    print(f"💾 Saved all prices to {prices_path}")

# =================== Align Monthly Period ==================
prices_mp = to_month_period_last(prices)
mkt_mp    = to_month_period_last(mkt.to_frame())["MKT"]
rf_mp = rf_monthly.copy()
rf_mp.index = rf_mp.index.to_period("M")

# =================== Returns ================================
rets     = np.log(prices_mp / prices_mp.shift(1)).drop(columns=["__P"], errors="ignore")
mkt_rets = np.log(mkt_mp / mkt_mp.shift(1))

valid_cols = rets.columns[rets.count() >= MIN_MONTHS_COVERAGE]
rets = rets[valid_cols]

if isinstance(mkt_rets, pd.DataFrame):
    mkt_rets = mkt_rets.squeeze("columns")
if isinstance(rf_mp, pd.DataFrame):
    rf_mp = rf_mp.squeeze("columns")

common_idx = rets.index.intersection(mkt_rets.index).intersection(rf_mp.index)
rets = rets.loc[common_idx]
mkt_rets = mkt_rets.loc[common_idx]
rf_mp = rf_mp.loc[common_idx]

# =================== Excess Returns ========================
excess_rets = rets.sub(rf_mp.astype(float), axis=0)
mkt_excess  = (mkt_rets.astype(float) - rf_mp.astype(float))

# =================== Save intermediates ====================
month_end_index = rets.index.to_timestamp("M")
prices_mp.index = month_end_index
rets.index = month_end_index
excess_rets.index = month_end_index

prices_mp.to_csv(os.path.join(OUTPUT_DIR, "sp500_prices_monthly_2020_2024.csv"), index_label="Date")
rets.to_csv(os.path.join(OUTPUT_DIR, "sp500_logreturns_monthly_2020_2024.csv"), index_label="Date")
excess_rets.to_csv(os.path.join(OUTPUT_DIR, "sp500_excess_logreturns_monthly_2020_2024.csv"), index_label="Date")

# =================== Vectorized CAPM Computation ===================
print("\n📈 Computing betas and alphas (vectorized)...")

if isinstance(mkt_excess.index, pd.PeriodIndex):
    mkt_excess.index = mkt_excess.index.to_timestamp("M")
if isinstance(excess_rets.index, pd.PeriodIndex):
    excess_rets.index = excess_rets.index.to_timestamp("M")

var_m = float(mkt_excess.var(ddof=1))
valid_mask = excess_rets.count() >= 6
excess_rets = excess_rets.loc[:, valid_mask]

ri_centered = excess_rets - excess_rets.mean()
mx_centered = mkt_excess - mkt_excess.mean()
mx_centered = mx_centered.reindex(excess_rets.index)

covs = (ri_centered.mul(mx_centered.values, axis=0)).sum(axis=0) / (len(mx_centered) - 1)
betas = covs / var_m

alpha_m = excess_rets.mean() - betas * mkt_excess.mean()
alpha_annual = (1 + alpha_m) ** 12 - 1

corrs = excess_rets.corrwith(mkt_excess)
r2 = corrs ** 2

results = pd.DataFrame({
    "Symbol": excess_rets.columns,
    "Beta": betas,
    "Alpha_monthly": alpha_m,
    "Alpha_annual": alpha_annual,
    "Corr": corrs,
    "R2": r2,
    "N_obs": excess_rets.notna().sum().values
}).reset_index(drop=True).sort_values("Symbol")

out_file = os.path.join(OUTPUT_DIR, "sp500_betas_alphas_2020_2024_monthly_excess.csv")
results.to_csv(out_file, index=False)

# =================== Diagnostics ===========================
print("\n--- Diagnostics ---")
print(f"Months (prices Period): {len(prices_mp.index)}")
print(f"Months (market): {len(mkt_mp.index)}")
print(f"Months (RF monthly): {len(rf_mp.index)}")
print(f"Coverage filter (>= {MIN_MONTHS_COVERAGE}) → tickers kept: {len(valid_cols)}")
print(f"Final months after align/dropna: {len(rets)} ({rets.index.min()} → {rets.index.max()})")
print(f"\n✅ Done — computed {len(results)} betas/alphas.")
print(f"📁 Saved outputs in '{OUTPUT_DIR}':")
print("   - sp500_prices_monthly_2020_2024.csv")
print("   - sp500_logreturns_monthly_2020_2024.csv")
print("   - sp500_excess_logreturns_monthly_2020_2024.csv")
print("   - sp500_betas_alphas_2020_2024_monthly_excess.csv")
print(results.head(10))
