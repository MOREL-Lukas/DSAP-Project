# --- Import necessary libraries ---
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
# import time  # (optional) use if you want to sleep between API calls

# --- Configuration ---
api_key = 'RW1VBO2DLFVA9OUH'  # 'GZWEY4238K97DK19'
DATA_DIR = "Stock Data"       # <— folder for all CSVs
os.makedirs(DATA_DIR, exist_ok=True)

# List of stock symbols to process (30 symbols) highest market cap from S&P 500
symbols = [
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
    "BRK.B", "JPM", "LLY", "UNH", "V", "MA", "PG", "HD", "XOM", "JNJ",
    "COST", "PEP", "KO", "CVX", "MRK", "ABBV", "MCD", "WMT", "NFLX",
    "ADBE", "CSCO", "CRM"
]

# --- Step 1: Loop through each symbol ---
for symbol in symbols:
    safe_symbol = symbol.replace('.', '-')  # e.g., BRK.B -> BRK-B
    csv_path = os.path.join(DATA_DIR, f"{safe_symbol}_historical_stock_data.csv")

    # --- Check if local CSV exists ---
    if os.path.exists(csv_path):
        print(f"✅ Using cached data for {symbol} ({csv_path})")
        df = pd.read_csv(csv_path)
    else:
        print(f"⬇️ Fetching new data for {symbol} from Alpha Vantage...")
        url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={symbol}"
            f"&apikey={api_key}&datatype=csv"
        )
        response = requests.get(url)

        # Error handling
        if response.status_code != 200 or "Error Message" in response.text:
            print(f"⚠️ Failed to fetch data for {symbol}")
            continue

        # Save the response as CSV
        with open(csv_path, 'w', newline='') as file:
            file.write(response.text)

        print(f"💾 Data saved to '{csv_path}'")
        df = pd.read_csv(csv_path)

print("✅ All symbols processed!")

# --- Delete any CSV that doesn't start with a valid header ---
Validity = True
for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(DATA_DIR, filename)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().lower()

        # If the first line doesn't start with 'timestamp,', it's invalid
        if not first_line.startswith("timestamp,"):
            os.remove(file_path)
            print(f"🗑️ Removed invalid CSV: {filename}")
            Validity = False
        else:
            print(f"✅ Valid CSV: {filename}")

    except Exception as e:
        print(f"❌ Could not read {filename}: {e}")
if Validity:
    print("✅ Sanity check complete.")
else:
    print("❌ Some files were invalid and have been removed.")