import os
import zipfile
from io import BytesIO
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm


def load_sp500_companies():
    """
    Load S&P 500 company data from DataHub and export:
    - data/raw/sp500_tickers.csv : one ticker per line (Yahoo-compatible symbols)
    - data/raw/sp500_companies.csv : full table

    Returns
    -------
    df : pandas.DataFrame
        Full S&P 500 companies table.
    """
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"

    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Load the DataHub CSV
    df = pd.read_csv(url)

    # Validate expected columns
    required_column = {"Symbol"}
    if not required_column.issubset(df.columns):
        raise ValueError(f"CSV is missing required columns: {required_column}")

    # Create Yahoo-compatible tickers (BRK.B -> BRK-B)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

    # Export tickers only (one per line, no header)
    df["Symbol"].to_csv(
        "data/raw/sp500_tickers.csv", index=False, header=False
    )

    # Export the full company table
    df.to_csv("data/raw/sp500_companies.csv", index=False)

    print(f"Exported {len(df)} companies to data/raw/sp500_companies.csv")
    return df


def load_rf():
    """
    Download Fama-French 5-factor monthly data and return only:
    - Date (datetime)
    - RF (decimal)
    - Mkt-RF (decimal)
    - Mkt (raw market return = (Mkt-RF) + RF)

    Saves:
    - data/raw/French_Library_data.csv       (raw FF5 table)
    - data/processed/Fama_French.csv            (cleaned subset)

    	Rm-Rf is the excess return on the market, or Market Risk Premium. It is the value-weight return 
        of all CRSP (Center for Research in Security Prices) firms incorporated in the US and 
        listed on the NYSE, AMEX, or NASDAQ that have a CRSP share code of 10 or 11 (U.S. common stocks)
        at the beginning of month t, good shares and price data at the beginning of t, 
        and good return data for t minus the one-month Treasury bill rate. 
        The one-month Treasury bill rate data through May 2024 are from Ibbotson Associates. 
        Starting from June 2024, the one-month Treasury bill rate is from ICE BofA US 1-Month Treasury Bill Index.
    """

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    )

    # Download ZIP file
    response = requests.get(url)
    response.raise_for_status()

    # Open ZIP in memory
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        # Find the CSV inside
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]

        # Read the CSV, skipping descriptive header text
        df = pd.read_csv(z.open(csv_name), skiprows=3)

    # Save raw file as-is
    df.to_csv("data/raw/French_Library_data.csv", index=False, float_format="%.6f")

    # The first column is the date (YYYYMM / YYYY)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Date"})

    # Drop footer / non-data rows (where Date is not numeric)
    df = df[pd.to_numeric(df["Date"], errors="coerce").notnull()]

    # Keep only monthly rows: Date has 6 digits (YYYYMM)
    df["Date"] = df["Date"].astype(int).astype(str)
    df = df[df["Date"].str.len() == 6]

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")

    # Convert factor columns from percent → decimal
    for col in ["RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    # Compute market return:
    df["Mkt"] = df["Mkt-RF"] + df["RF"]

    # Keep the columns we want
    rf = df[["Date", "RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mkt"]].copy()

    # Save cleaned file
    rf.to_csv("data/processed/Fama_French.csv", index=False, float_format="%.6f")

    print("Saved monthly RF + Mkt data to data/processed/Fama_French.csv")
    return rf


# ---------- Helpers for SP500 monthly returns ----------

def extract_monthly_close(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the monthly 'Close' price for each ticker from the yfinance output.
    Assumes columns are a MultiIndex: (ticker, field).
    """
    monthly_prices = {}

    for ticker in raw.columns.levels[0]:
        try:
            monthly_prices[ticker] = raw[ticker]["Close"]
        except KeyError:
            continue

    return pd.DataFrame(monthly_prices)


def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly percentage returns from monthly prices."""
    return monthly_prices.pct_change(fill_method=None).dropna(how="all")


def download_sp500_prices(tickers, start="2024-11-01", end="2025-11-01"):
    """
    Optional helper: download monthly SP500 prices for given tickers.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1mo",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        timeout=30
    )
    return data


def load_sp500_monthly_returns(
    start: str = "2024-11-01",
    end: str = "2025-11-01",
):
    """
    Load S&P 500 tickers, download monthly prices from Yahoo, compute
    monthly returns, and save to data/processed.

    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    monthly_returns : pandas.DataFrame
        Monthly % returns for each S&P 500 ticker.
    """

    # Load tickers written by load_sp500_companies()
    tickers = pd.read_csv("data/raw/sp500_tickers.csv", header=None)[0].tolist()

    # Download MONTHLY data directly
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    # Extract monthly closing prices
    monthly_prices = extract_monthly_close(raw)

    # Compute monthly returns
    monthly_returns = compute_monthly_returns(monthly_prices)

    # Drop columns with all NaN values
    monthly_returns = monthly_returns.dropna(axis=1, how="all")

    # Ensure processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Set index names
    monthly_prices.index.name = "Date"
    monthly_returns.index.name = "Date"
    # Save results
    monthly_prices.to_csv("data/processed/sp500_monthly_prices.csv")
    monthly_returns.to_csv("data/processed/sp500_monthly_returns.csv")

    print("Saved monthly S&P 500 prices and returns to data/processed/")
    return monthly_returns


def classify_sp500_factors(tickers, force_recompute: bool = False):
    """
    Fast approximate classification of S&P 500 companies into:
    - Size: Small / Big            (by marketCap)
    - Value: High / Low            (by 1 / priceToBook as B/M proxy)
    - Profitability: Robust / Weak (by returnOnEquity)
    - Investment: Conservative / Aggressive (by revenueGrowth)

    Uses only yfinance .info (1 HTTP call per ticker).

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    force_recompute : bool
        If False and cached file exists, load from disk instead of redownloading.

    Returns
    -------
    df : pandas.DataFrame
        Columns: Ticker, ME, PB, BM_proxy, ROE, RevGrowth,
                 Size, Value, Profitability, Investment
    """
    os.makedirs("data/processed", exist_ok=True)
    rows = []

    print("\nFetching fundamentals from Yahoo Finance...")
    for t in tqdm(tickers, desc="Classifying S&P 500", unit="ticker"):
        try:
            tk = yf.Ticker(t)
            info = tk.info  # single quoteSummary call

            me = info.get("marketCap", np.nan)
            pb = info.get("priceToBook", np.nan)              # P/B
            roe = info.get("returnOnEquity", np.nan)          # profitability proxy
            rev_growth = info.get("revenueGrowth", np.nan)    # investment / growth proxy

            rows.append([t, me, pb, roe, rev_growth])

        except Exception as e:
            print(f"Failed for {t}: {e}")
            rows.append([t, np.nan, np.nan, np.nan, np.nan])

    df = pd.DataFrame(
        rows,
        columns=["Ticker", "ME", "PB", "ROE", "RevGrowth"]
    )

    # Drop rows with missing ME (cannot classify size)
    df = df.dropna(subset=["ME"])

    # ---- Build proxies ----

    # Value: use inverse of P/B as a crude B/M proxy
    df["PB"] = df["PB"].replace(0, np.nan)
    df["BM_proxy"] = 1.0 / df["PB"]

    # ---- Cross-sectional thresholds ----
    size_threshold = df["ME"].median()
    bm_threshold = df["BM_proxy"].median(skipna=True)
    roe_threshold = df["ROE"].median(skipna=True)
    inv_threshold = df["RevGrowth"].median(skipna=True)

    # ---- Classifications ----
    df["Size"] = np.where(df["ME"] <= size_threshold, "Small", "Big")
    df["Value"] = np.where(df["BM_proxy"] >= bm_threshold, "High", "Low")
    df["Profitability"] = np.where(df["ROE"] >= roe_threshold, "Robust", "Weak")
    df["Investment"] = np.where(df["RevGrowth"] <= inv_threshold,
                                "Conservative", "Aggressive")

    return df

