import pandas as pd

# Yahoo Finance keeps an updated CSV
url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
tickers = pd.read_csv(url)["Symbol"]
tickers.to_csv("sp500_tickers.txt", index=False, header=False)
print(f"Saved {len(tickers)} tickers to sp500_tickers.txt")
