"""
Data Processing Module
Combines data loading, preprocessing, and feature engineering
"""
import os
import zipfile
import time
from io import BytesIO
import numpy as np
import pandas as pd 
import requests
import yfinance as yf
from tqdm import tqdm
import random

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


def extract_monthly_close(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the monthly 'Close' price for each ticker from the yfinance output.
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


def _download_in_chunks(
    tickers,
    start,
    end,
    interval="1mo",
    chunk_size=50,
    timeout=60,
    threads=False,
    max_retries=4,
    backoff_base=2.0,
):
    """
    Download yfinance data in chunks with retries.
    """
    all_chunks = []

    for i in tqdm(range(0, len(tickers), chunk_size), desc="Downloading Yahoo data", unit="chunk"):
        chunk = tickers[i : i + chunk_size]

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=threads,
                    timeout=timeout,
                    progress=False,   # we already show tqdm
                )
                # If Yahoo returned something, accept
                if raw is not None and not raw.empty:
                    all_chunks.append(raw)
                    last_err = None
                    break
                else:
                    last_err = RuntimeError("Empty response from Yahoo Finance")

            except Exception as e:
                last_err = e

            # exponential backoff between retries
            sleep_s = backoff_base ** (attempt - 1)
            time.sleep(sleep_s)

        if last_err is not None:
            print(f"\nWarning: chunk {i//chunk_size + 1} failed after {max_retries} retries: {last_err}")

    if not all_chunks:
        return pd.DataFrame()

    # Concatenate along columns (MultiIndex columns expected)
    combined = pd.concat(all_chunks, axis=1)
    return combined


def load_sp500_monthly_returns(
    start: str = "1990-01-01",
    end: str = "2025-12-01",
    force_download: bool = False,
):
    """
    - Otherwise download with chunking + retries + conservative concurrency.
    """

    os.makedirs("data/processed", exist_ok=True)

    returns_path = "data/processed/sp500_monthly_returns.csv"
    prices_path = "data/processed/sp500_monthly_prices.csv"

    # 1) Cache-first: if already computed, never download again
    if (not force_download) and os.path.exists(returns_path):
        df = pd.read_csv(returns_path, parse_dates=["Date"]).set_index("Date").sort_index()
        print(f"Loaded cached monthly returns from {returns_path} (no Yahoo download).")
        return df

    # 2) Load tickers
    tickers = pd.read_csv("data/raw/sp500_tickers.csv", header=None)[0].tolist()

    # Permanently exclude known bad symbols (quote missing/delisted)
    bad_symbols = {"WBA"}
    tickers = [t for t in tickers if t not in bad_symbols]

    # 3) Download in chunks with retries
    raw = _download_in_chunks(
        tickers=tickers,
        start=start,
        end=end,
        interval="1mo",
        chunk_size=50,      
        timeout=90,         
        threads=False,      
        max_retries=4,
        backoff_base=2.0,
    )

    if raw.empty:
        raise RuntimeError("Yahoo download failed completely (no data returned).")

    # 4) Extract monthly close & compute returns
    monthly_prices = extract_monthly_close(raw)
    monthly_returns = compute_monthly_returns(monthly_prices)
    monthly_returns = monthly_returns.dropna(axis=1, how="all")

    monthly_prices.index.name = "Date"
    monthly_returns.index.name = "Date"

    monthly_prices.to_csv(prices_path)
    monthly_returns.to_csv(returns_path)

    print("Saved monthly S&P 500 prices and returns to data/processed/")
    return monthly_returns


def classify_sp500_factors(
    tickers,
    force_recompute: bool = False,
    output_path: str = "data/processed/sp500_ff5_classifications.csv",
    failed_path: str = "data/processed/fundamentals_failed.csv",
    blacklist: set = None,
    max_retries: int = 4,
    backoff_base: float = 2.0,
    polite_sleep=(0.05, 0.15),
):
    """
    Robust approximate classification of S&P 500 companies into:
    - Size: Small / Big            (by marketCap)
    - Value: High / Low            (by 1 / priceToBook as B/M proxy)
    - Profitability: Robust / Weak (by returnOnEquity)
    - Investment: Conservative / Aggressive (by revenueGrowth)
    """

    os.makedirs("data/processed", exist_ok=True)

    if blacklist is None:
        blacklist = {"WBA"}
    tickers = [t for t in tickers if t not in blacklist]

    # Cache-first behavior
    cached_df = None
    cached_tickers = set()
    if (not force_recompute) and os.path.exists(output_path):
        try:
            cached_df = pd.read_csv(output_path)
            if "Ticker" in cached_df.columns:
                cached_tickers = set(cached_df["Ticker"].astype(str))
        except Exception:
            cached_df = None
            cached_tickers = set()

    # Only fetch missing tickers
    to_fetch = [t for t in tickers if t not in cached_tickers]

    # If nothing to fetch, return cached directly
    if cached_df is not None and len(to_fetch) == 0:
        return cached_df

    rows = []
    failed = []

    def fetch_one(ticker: str):
        """Fetch required fundamental fields with retries/backoff."""
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                tk = yf.Ticker(ticker)
                info = tk.info

                me = info.get("marketCap", np.nan)
                pb = info.get("priceToBook", np.nan)
                roe = info.get("returnOnEquity", np.nan)
                rev_growth = info.get("revenueGrowth", np.nan)

                return {"Ticker": ticker, "ME": me, "PB": pb, "ROE": roe, "RevGrowth": rev_growth, "error": ""}

            except Exception as e:
                last_err = str(e)
                sleep_s = (backoff_base ** (attempt - 1)) + random.uniform(0, 0.25)
                time.sleep(sleep_s)

        return {"Ticker": ticker, "ME": np.nan, "PB": np.nan, "ROE": np.nan, "RevGrowth": np.nan, "error": last_err or "unknown"}

    # Simple processing with single progress bar
    for t in tqdm(to_fetch, desc="Fetching fundamentals", unit="ticker"):
        res = fetch_one(t)

        if res.get("error"):
            failed.append(res)
        else:
            rows.append(res)

        time.sleep(random.uniform(*polite_sleep))

    fetched_df = pd.DataFrame(rows, columns=["Ticker", "ME", "PB", "ROE", "RevGrowth"])
    
    if failed:
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"Warning: {len(failed)} tickers failed fundamentals fetch. See {failed_path}")

    # Combine cached + fetched
    if cached_df is not None and not cached_df.empty:
        merged = pd.concat([cached_df, fetched_df], ignore_index=True)
    else:
        merged = fetched_df

    merged = merged.drop_duplicates(subset=["Ticker"], keep="first")
    merged = merged.dropna(subset=["ME"])

    # Build proxies
    merged["PB"] = merged["PB"].replace(0, np.nan)
    merged["BM_proxy"] = 1.0 / merged["PB"]

    # Cross-sectional thresholds
    size_threshold = merged["ME"].median()
    bm_threshold = merged["BM_proxy"].median(skipna=True)
    roe_threshold = merged["ROE"].median(skipna=True)
    inv_threshold = merged["RevGrowth"].median(skipna=True)

    # Classifications
    merged["Size"] = np.where(merged["ME"] <= size_threshold, "Small", "Big")
    merged["Value"] = np.where(merged["BM_proxy"] >= bm_threshold, "High", "Low")
    merged["Profitability"] = np.where(merged["ROE"] >= roe_threshold, "Robust", "Weak")
    merged["Investment"] = np.where(merged["RevGrowth"] <= inv_threshold, "Conservative", "Aggressive")

    merged = merged.sort_values("Ticker")
    merged.to_csv(output_path, index=False)
    return merged

def build_factor_ml_dataset(
    returns_path="data/processed/sp500_monthly_returns.csv",
    class_path="data/processed/sp500_ff5_classifications.csv",
    factor_path="data/processed/Fama_French.csv",
    out_path="data/processed/factor_ml_dataset.csv",
):
    """
    Build a monthly dataset for predicting 3-month ahead (t+3) FF5 factor returns.
    """

    # Load data
    rets = pd.read_csv(returns_path, parse_dates=["Date"]).set_index("Date").sort_index()
    classes = pd.read_csv(class_path).set_index("Ticker")
    
    # Align tickers
    common = sorted(set(rets.columns) & set(classes.index))
    rets = rets[common]
    classes = classes.loc[common]

    # Cross-sectional features
    features = pd.DataFrame(index=rets.index)
    features["ret_all_mean"] = rets.mean(axis=1)
    features["ret_all_std"] = rets.std(axis=1)
    features["ret_all_dispersion"] = rets.quantile(0.9, axis=1) - rets.quantile(0.1, axis=1)
    features["ret_pos_ratio"] = (rets > 0).mean(axis=1)

    def cols_for(mask):
        return classes.index[mask]

    # Size: Small vs Big
    small_cols = cols_for(classes["Size"] == "Small")
    big_cols = cols_for(classes["Size"] == "Big")
    features["ret_small"] = rets[small_cols].mean(axis=1)
    features["ret_big"] = rets[big_cols].mean(axis=1)
    features["SMB_proxy"] = features["ret_small"] - features["ret_big"]

    # Value: High vs Low
    high_cols = cols_for(classes["Value"] == "High")
    low_cols = cols_for(classes["Value"] == "Low")
    if len(high_cols) and len(low_cols):
        features["ret_highBM"] = rets[high_cols].mean(axis=1)
        features["ret_lowBM"] = rets[low_cols].mean(axis=1)
        features["HML_proxy"] = features["ret_highBM"] - features["ret_lowBM"]

    # Profitability: Robust vs Weak
    robust_cols = cols_for(classes["Profitability"] == "Robust")
    weak_cols = cols_for(classes["Profitability"] == "Weak")
    if len(robust_cols) and len(weak_cols):
        features["ret_robust"] = rets[robust_cols].mean(axis=1)
        features["ret_weak"] = rets[weak_cols].mean(axis=1)
        features["RMW_proxy"] = features["ret_robust"] - features["ret_weak"]

    # Investment: Conservative vs Aggressive
    cons_cols = cols_for(classes["Investment"] == "Conservative")
    aggr_cols = cols_for(classes["Investment"] == "Aggressive")
    if len(cons_cols) and len(aggr_cols):
        features["ret_cons"] = rets[cons_cols].mean(axis=1)
        features["ret_aggr"] = rets[aggr_cols].mean(axis=1)
        features["CMA_proxy"] = features["ret_cons"] - features["ret_aggr"]

    # Load factor data
    ff = pd.read_csv(factor_path, parse_dates=["Date"]).set_index("Date").sort_index()
    dataset = features.join(ff, how="inner").sort_index()

    # 6-month ahead targets
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    dataset[factors] = dataset[factors].shift(-6)
    dataset = dataset.dropna(subset=factors)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dataset.to_csv(out_path, index=True)
    return dataset


def add_lagged_factors(dataset: pd.DataFrame, factor_cols, lags):
    """Add lagged factor features."""
    lagged = pd.DataFrame(index=dataset.index)

    for fac in factor_cols:
        if fac not in dataset.columns:
            continue

        for lag in lags:
            lagged[f"{fac}_lag{lag}"] = dataset[fac].shift(lag)

        lagged[f"{fac}_ma3"] = dataset[fac].shift(1).rolling(3).mean()
        lagged[f"{fac}_vol6"] = dataset[fac].shift(1).rolling(6).std()
        lagged[f"{fac}_mom12"] = dataset[fac].shift(1).rolling(12).mean()

    return dataset.join(lagged, how="left")


def add_market_conditions(dataset: pd.DataFrame):
    """Add market regime indicators."""
    dataset["market_trend_3m"] = dataset["ret_all_mean"].rolling(3).mean()
    dataset["market_trend_6m"] = dataset["ret_all_mean"].rolling(6).mean()
    dataset["volatility_3m"] = dataset["ret_all_std"].rolling(3).mean()
    dataset["volatility_6m"] = dataset["ret_all_std"].rolling(6).mean()

    dataset["breadth_ma3"] = dataset["ret_pos_ratio"].rolling(3).mean()
    dataset["breadth_change"] = dataset["ret_pos_ratio"].diff()

    dataset["dispersion_ma3"] = dataset["ret_all_dispersion"].rolling(3).mean()
    dataset["dispersion_change"] = dataset["ret_all_dispersion"].pct_change()

    for proxy in ["SMB_proxy", "HML_proxy", "RMW_proxy", "CMA_proxy"]:
        if proxy in dataset.columns:
            dataset[f"{proxy}_ma3"] = dataset[proxy].rolling(3).mean()

    return dataset


def add_macro_features(dataset: pd.DataFrame, fred_api_key=None):
    """Add minimal macro features: VIX + Oil."""
    start = dataset.index.min()
    end = dataset.index.max() + pd.offsets.Day(31)

    # VIX
    try:
        vix = yf.download("^VIX", start=start, end=end, interval="1mo", progress=False)["Close"]
        if not vix.empty:
            vix.index = vix.index.to_period("M").to_timestamp()
            dataset["vix"] = vix.reindex(dataset.index, method="ffill")
            dataset["vix_change"] = dataset["vix"].pct_change()
        else:
            dataset["vix"] = np.nan
            dataset["vix_change"] = np.nan
    except Exception:
        dataset["vix"] = np.nan
        dataset["vix_change"] = np.nan

    # Oil
    try:
        oil = yf.download("CL=F", start=start, end=end, interval="1mo", progress=False)["Close"]
        if not oil.empty:
            oil.index = oil.index.to_period("M").to_timestamp()
            dataset["oil_price"] = oil.reindex(dataset.index, method="ffill")
            dataset["oil_change"] = dataset["oil_price"].pct_change()
        else:
            dataset["oil_price"] = np.nan
            dataset["oil_change"] = np.nan
    except Exception:
        dataset["oil_price"] = np.nan
        dataset["oil_change"] = np.nan

    return dataset.ffill().bfill()


def build_enhanced_factor_ml_dataset(
    returns_path="data/processed/sp500_monthly_returns.csv",
    class_path="data/processed/sp500_ff5_classifications.csv",
    factor_path="data/processed/Fama_French.csv",
    out_path="data/processed/factor_ml_dataset_enhanced.csv",
    fred_api_key=None,
    factor_lags=[3, 6, 12],
):
    """
    Enhanced dataset with lagged factors, market conditions, and macro features.
    """

    temp_path = out_path.replace(".csv", "_temp.csv")

    # Build base dataset
    build_factor_ml_dataset(
        returns_path=returns_path,
        class_path=class_path,
        factor_path=factor_path,
        out_path=temp_path,
    )

    dataset = pd.read_csv(temp_path, parse_dates=["Date"]).set_index("Date").sort_index()
    targets = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    # Add realized factor history as features
    ff = pd.read_csv(factor_path, parse_dates=["Date"]).set_index("Date").sort_index()
    ff_hist = ff[targets].rename(columns={c: f"{c}_hist" for c in targets})

    dataset = dataset.join(ff_hist, how="left")
    hist_cols = list(ff_hist.columns)

    dataset = add_lagged_factors(dataset, factor_cols=hist_cols, lags=factor_lags)
    dataset = dataset.drop(columns=hist_cols, errors="ignore")

    dataset = add_market_conditions(dataset)
    dataset = add_macro_features(dataset, fred_api_key=fred_api_key)

    dataset = dataset.dropna(subset=targets)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dataset.to_csv(out_path, index=True)

    try:
        os.remove(temp_path)
    except FileNotFoundError:
        pass

    return dataset