import os
import pandas as pd
import numpy as np
# --- Configuration ---
DATA_DIR = "Stock Data Cleaned"

# --- Step 1: Load all cleaned CSVs into a dictionary ---
price_data = {}

for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".csv"):
        continue

    symbol = filename.split("_")[0]  # Extract symbol prefix (e.g., AAPL from AAPL_historical_stock_data.csv)
    file_path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    # Ensure expected column exists
    if "adjusted close" not in df.columns:
        print(f"⚠️ {symbol}: Missing 'adjusted close' column — skipped.")
        continue

    df = df[["timestamp", "adjusted close"]].rename(columns={"adjusted close": symbol})
    price_data[symbol] = df

print(f"📈 Loaded {len(price_data)} stocks")

# --- Step 2: Merge all stocks on timestamp (inner join keeps only matching weeks) ---
merged_df = None
for symbol, df in price_data.items():
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on="timestamp", how="outer")

merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)
print(f"✅ Merged data shape: {merged_df.shape}")

# --- Step 3: Compute weekly log returns ---
returns_df = merged_df.set_index("timestamp").apply(lambda x: np.log(x / x.shift(1)))
returns_df = returns_df.dropna(how="any")

# --- Step 4: Calculate covariance matrix ---
cov_matrix = returns_df.cov()

# --- Step 5: Save results ---
cov_matrix.to_csv("weekly_covariance_matrix.csv")
print("💾 Covariance matrix saved to 'weekly_covariance_matrix.csv'")

# --- Optional: display summary ---
print("\n📊 Covariance matrix (first 5x5 block):")
print(cov_matrix.iloc[:10, :10])
