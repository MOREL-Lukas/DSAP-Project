# fetch_sp500_symbols.py
import pandas as pd
import requests
from time import sleep

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

def fetch_html(url: str, headers: dict, retries: int = 3, pause: float = 1.5) -> str:
    """Fetch the Wikipedia HTML page with retry logic."""
    for i in range(retries):
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.text
        sleep(pause)
    raise RuntimeError(f"Failed to fetch page. Last status: {resp.status_code}")

# --- Main fetch ---
html = fetch_html(URL, HEADERS)
tables = pd.read_html(html)  # requires lxml installed ✅

sp500 = tables[0].copy()     # first table = current S&P 500 constituents

# Some versions of the table use "Symbol" or "Ticker symbol"
symbol_col = "Symbol" if "Symbol" in sp500.columns else "Ticker symbol"
symbols = sp500[symbol_col].astype(str).str.strip().tolist()

# Normalize tickers: replace '.' with '-' for Yahoo compatibility (e.g. BRK.B → BRK-B)
symbols_clean = [s.replace(".", "-").replace(" ", "") for s in symbols]

print(f"✅ Found {len(symbols_clean)} tickers.")
print("🔹 First 10:", symbols_clean[:10])

# Save as an importable Python module
with open("sp500_symbols.py", "w", encoding="utf-8") as f:
    f.write("sp500_symbols = " + repr(symbols_clean) + "\n")

print("💾 Saved tickers to 'sp500_symbols.py'")
