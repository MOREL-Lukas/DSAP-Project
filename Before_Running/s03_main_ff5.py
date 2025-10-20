# ============================================================
# s02_main_ff5.py — Vectorized Fama–French 5-Factor Model
# ============================================================

import os
import numpy as np
import pandas as pd
from sp500_symbols import sp500_symbols

DATA_DIR   = "Data"
OUTPUT_DIR = "Output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

START = "2020-01-01"
END   = "2024-12-31"

# ============================================================
# Utilities (reuse from s02_main.py)
# ============================================================

def read_csv_smart(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def save_clean_csv(df: pd.DataFrame, path: str, index_label="Date"):
    df.to_csv(path, index=True, index_label=index_label)

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(how="all")

# ============================================================
# Step 1. Fetch Fama–French 5-Factor Data
# ============================================================

def fetch_ff5_factors():
    """
    Download Fama-French 5-Factor (monthly) data from Kenneth French's data library.
    Returns columns: [MKT_RF, SMB, HML, RMW, CMA, RF].
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    print("⬇️ Downloading Fama-French 5-Factor monthly data...")

    df = pd.read_csv(url, skiprows=3)

    # Try to find the first blank line (end of data section)
    blank_rows = df[df.iloc[:, 0].astype(str).str.strip().eq("")].index
    if len(blank_rows) > 0:
        stop_idx = blank_rows.min()
        df = df.iloc[:stop_idx]
    # else: no blank line — keep full df

    # Rename and clean
    df.columns = ["Date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m", errors="coerce")
    df = df.set_index("Date").sort_index()

    # Convert percentages to decimals
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0

    # Filter to desired period
    df = df.loc[(df.index >= START) & (df.index <= END)]
    df.index = df.index.to_period("M").to_timestamp("M")
    return df.dropna(how="all")
    """
    Download Fama-French 5-Factor (monthly) data from Kenneth French's data library.
    Returns columns: [MKT_RF, SMB, HML, RMW, CMA, RF]
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    print("⬇️ Downloading Fama-French 5-Factor monthly data...")
    
    df = pd.read_csv(url, skiprows=3)
    stop_idx = df[df.iloc[:, 0].astype(str).str.strip().eq("")].index.min()
    df = df.iloc[:stop_idx]
    
    df.columns = ["Date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m", errors="coerce")
    df = df.set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df = df.loc[(df.index >= START) & (df.index <= END)]
    df.index = df.index.to_period("M").to_timestamp("M")
    return df.dropna()

ff5_path = os.path.join(DATA_DIR, "fama_french_5factors_monthly.csv")
if os.path.exists(ff5_path):
    print(f"✅ Using cached Fama-French 5-Factor data: {ff5_path}")
    ff5_factors = read_csv_smart(ff5_path)
else:
    ff5_factors = fetch_ff5_factors()
    save_clean_csv(ff5_factors, ff5_path)
    print(f"💾 Saved to {ff5_path}")

# ============================================================
# Step 2. Load Excess Returns from CAPM Script
# ============================================================

excess_path = os.path.join(OUTPUT_DIR, "sp500_excess_logreturns_monthly_2020_2024.csv")
if not os.path.exists(excess_path):
    raise FileNotFoundError("❌ Missing CAPM excess returns file. Run s02_main.py first.")

excess_rets = read_csv_smart(excess_path)
excess_rets = coerce_numeric_df(excess_rets)

# Align with FF5 factors
common_idx = excess_rets.index.intersection(ff5_factors.index)
Y = excess_rets.loc[common_idx]
X = ff5_factors.loc[common_idx, ["MKT_RF", "SMB", "HML", "RMW", "CMA"]]
rf = ff5_factors.loc[common_idx, "RF"]

print(f"\n🧩 Running Fama–French 5-Factor model for {Y.shape[1]} stocks...")
print(f"Common months: {len(common_idx)} ({common_idx.min().date()} → {common_idx.max().date()})")

# ============================================================
# Step 3. Vectorized Regression (OLS)
# ============================================================

# Add constant for alpha
X = X.assign(const=1.0)
cols = ["const", "MKT_RF", "SMB", "HML", "RMW", "CMA"]
X_mat = X[cols].to_numpy()

XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
XtY = X_mat.T @ Y.to_numpy()
B = XtX_inv @ XtY  # shape (6, N_stocks)

alphas = pd.Series(B[0, :], index=Y.columns, name="Alpha")
betas = pd.DataFrame(B[1:, :].T, index=Y.columns, columns=["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA"])

# Predicted values & R²
Y_hat = X_mat @ B
resid = Y.to_numpy() - Y_hat
rss = np.sum(resid**2, axis=0)
tss = np.sum((Y.to_numpy() - Y.to_numpy().mean(axis=0))**2, axis=0)
r2 = 1 - rss / tss

results_ff5 = pd.concat([alphas, betas], axis=1)
results_ff5["Alpha_annual"] = (1 + results_ff5["Alpha"]) ** 12 - 1
results_ff5["R2"] = r2
results_ff5["N_obs"] = len(X)

out_path = os.path.join(OUTPUT_DIR, "sp500_ff5_betas_2020_2024.csv")
results_ff5.to_csv(out_path, index_label="Symbol")

# ============================================================
# Step 4. Diagnostics
# ============================================================

print("\n--- Diagnostics ---")
print(f"Stocks estimated: {len(results_ff5)}")
print(f"Period: {common_idx.min().date()} → {common_idx.max().date()}")
print(f"Saved results → {out_path}")
print(results_ff5.head(10))
print("\n✅ Fama–French 5-Factor model estimation complete.")