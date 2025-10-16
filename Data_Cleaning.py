# --- Import necessary libraries ---
import os
import pandas as pd

# --- Configuration ---
DATA_DIR = "Stock Data"
OUTPUT_DIR = "Stock Data Cleaned"  # optional: store cleaned copies separately
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 2 & 3: Process each CSV in the folder ---
for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".csv"):
        continue  # skip non-CSV files

    file_path = os.path.join(DATA_DIR, filename)
    print(f"🧹 Processing {filename}...")

    try:
        # --- Load CSV ---
        df = pd.read_csv(file_path)

        # --- Step 2: Clean and prepare the data ---
        if "timestamp" not in df.columns:
            print(f"⚠️ Skipping {filename} (missing 'timestamp' column)")
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")

        # --- Step 3: Filter only data from 2000 to 2024 ---
        df = df[(df["timestamp"] < "2025-01-01") & (df["timestamp"] >= "2000-01-01")]

        # --- Save cleaned file ---
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False)
        print(f"✅ Cleaned data saved to '{output_path}'")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

print("🎉 All files cleaned and filtered successfully!")
