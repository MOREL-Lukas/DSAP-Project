# ============================================================
# s03_make_sp500_sectors.py — Fetch Sector Info for S&P 500 Symbols (only if missing)
# ============================================================

import os
import sys
import time
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# === Paths ===
DATA_DIR = "Data"
OUTPUT_DIR = "Output"
os.makedirs(DATA_DIR, exist_ok=True)

capm_path = os.path.join(OUTPUT_DIR, "sp500_betas_alphas_2020_2024_monthly_excess.csv")
sector_out_path = os.path.join(DATA_DIR, "sp500_sectors.csv")

# === tqdm settings ===
tqdm_kwargs = dict(
    desc="Fetching Sectors",
    ncols=80,
    leave=True,
    file=sys.stdout,
    dynamic_ncols=True,
    mininterval=0.3,
    ascii=True
)

# === Skip if file already exists ===
if os.path.exists(sector_out_path):
    print(f"✅ Sector file already exists: {sector_out_path}")
    print("➡️  Skipping download.")
    sys.exit(0)

# === Load symbols from CAPM results ===
print("📂 Loading CAPM results...")
capm_df = pd.read_csv(capm_path)

if "Symbol" not in capm_df.columns:
    raise KeyError("❌ 'Symbol' column not found in CAPM results.")

symbols = sorted(capm_df["Symbol"].unique())
print(f"✅ Found {len(symbols)} unique S&P 500 symbols.")

# === Fetch sectors from Yahoo Finance ===
sectors = {}
start_time = time.time()

for sym in tqdm(symbols, **tqdm_kwargs):
    try:
        ticker = yf.Ticker(sym)
        info = ticker.info if hasattr(ticker, "info") else ticker.get_info()
        sector = info.get("sector", None)
        sectors[sym] = sector
    except Exception as e:
        sectors[sym] = None
    time.sleep(0.4)  # avoid rate limiting

elapsed = time.time() - start_time
print(f"⏱️  Completed in {elapsed:.1f} seconds.")

# === Build DataFrame and save ===
sector_df = pd.DataFrame(list(sectors.items()), columns=["Symbol", "Sector"])
sector_df.to_csv(sector_out_path, index=False)

print("\n✅ Done — sector info saved to:")
print(f"📁 {sector_out_path}")
print("\nSample:")
print(sector_df.head(10))
