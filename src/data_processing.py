"""
Data Processing Module with Automatic Raw Data Caching

SIMPLE CACHING LOGIC:
- If raw data exists in data/raw/ → use it (fast)
- If raw data missing → download it once, cache it (slower first time)
- Just run main.py and it handles everything automatically

RAW DATA FILES (auto-cached):
1. data/raw/sp500_companies.csv - S&P 500 constituent list
2. data/raw/sp500_tickers.csv - Just the tickers
3. data/raw/French_Library_data.csv - Raw Fama-French factor data
4. data/raw/sp500_raw_yfinance.pkl - Raw price data from yfinance (~50-100MB)
5. data/raw/sp500_fundamentals_raw.csv - Raw fundamental data (ME, PB, ROE, etc.)
6. data/raw/vix_raw.csv - Raw VIX data
7. data/raw/oil_raw.csv - Raw Oil price data

TO GET FRESH DATA:
Just delete the raw files and run again:
  python main.py

FOR REPRODUCIBILITY:
Keep the data/raw/ directory intact and commit or backup those files.
"""

import os, time, zipfile, random
from io import BytesIO
import numpy as np
import pandas as pd
import requests, yfinance as yf
from tqdm import tqdm

DATA_RAW, DATA_PROC = "data/raw", "data/processed"
TICKER_FIX = (".", "-")  # Replace dots with dashes in tickers
BAD_SYMBOLS_DEFAULT = {"WBA"}  # Known bad symbols for yfinance
FF5_ZIP = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)  # Fama-French 5-factor data
SP500_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"  # S&P 500 companies


def _mk(*paths):
    [os.makedirs(p, exist_ok=True) for p in paths]


def _read_csv(path, **kw):
    return pd.read_csv(path, **kw)


def _save(df, path, **kw):
    _mk(os.path.dirname(path))
    df.to_csv(path, **kw)
    return df


def load_sp500_companies():
    """
    Load S&P 500 company data.

    Cache-first behavior:
    - If data/raw/sp500_companies.csv exists -> use it (no internet).
    - If not, download from SP500_URL, normalize tickers, and cache both:
        * data/raw/sp500_companies.csv
        * data/raw/sp500_tickers.csv
    """
    _mk(DATA_RAW, DATA_PROC)
    raw_path = f"{DATA_RAW}/sp500_companies.csv"
    tickers_path = f"{DATA_RAW}/sp500_tickers.csv"

    # 1) Pure cache path (no external calls)
    if os.path.exists(raw_path):
        df = _read_csv(raw_path)

        # Normalize ticker symbol format
        if "Symbol" not in df.columns:
            raise ValueError(
                "Cached sp500_companies.csv is missing required column 'Symbol'"
            )

        df["Symbol"] = df["Symbol"].astype(str).str.replace(*TICKER_FIX, regex=False)

        # Ensure tickers file also exists and is consistent
        if not os.path.exists(tickers_path):
            df["Symbol"].to_csv(tickers_path, index=False, header=False)

        return df

    # 2) If no cache exists, fall back to a single download and cache it
    df = _read_csv(SP500_URL)
    if "Symbol" not in df.columns:
        raise ValueError("Downloaded CSV is missing required column 'Symbol'")

    df["Symbol"] = df["Symbol"].str.replace(*TICKER_FIX, regex=False)

    # Cache both full company table and plain tickers list
    df["Symbol"].to_csv(tickers_path, index=False, header=False)
    _save(df, raw_path, index=False)
    print(f"Downloaded and exported {len(df)} companies to {raw_path}")
    return df


def download_ff_raw():
    """
    Fetch the canonical Fama-French 5-factor library once and cache it
    locally so subsequent experiments can run fully offline and against
    a stable historical dataset.
    """
    _mk(DATA_RAW)
    raw_path = f"{DATA_RAW}/French_Library_data.csv"

    # Check cache first
    if os.path.exists(raw_path):
        print(f"Using cached FF5 raw data from {raw_path}")
        return _read_csv(raw_path)

    # Download if not cached
    print(f"Downloading FF5 data...")
    r = requests.get(FF5_ZIP)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(z.open(csv_name), skiprows=3)

    _save(df, raw_path, index=False, float_format="%.6f")
    print(f"Saved raw FF5 data to {raw_path}")
    return df


def process_ff_data(raw_df=None):
    """
    Transform the raw Fama-French library into a clean monthly panel in
    decimal returns, providing a single source of truth for all factor-
    based calculations in the project.
    """
    _mk(DATA_PROC)

    # Load raw data if not provided
    if raw_df is None:
        raw_df = _read_csv(f"{DATA_RAW}/French_Library_data.csv")

    df = raw_df.copy()
    df = df.rename(columns={df.columns[0]: "Date"})
    df = df[pd.to_numeric(df["Date"], errors="coerce").notnull()]
    df["Date"] = df["Date"].astype(int).astype(str)
    df = df[df["Date"].str.len() == 6]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")

    cols = ["RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df["Mkt"] = df["Mkt-RF"] + df["RF"]

    rf = df[["Date", *cols, "Mkt"]].copy()
    _save(rf, f"{DATA_PROC}/Fama_French.csv", index=False, float_format="%.6f")
    return rf


def load_rf():
    """Download (if needed) and process Fama-French 5-factor data."""
    raw_df = download_ff_raw()
    return process_ff_data(raw_df)


def extract_monthly_close(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract monthly 'Close' for each ticker from yfinance output (MultiIndex columns)."""
    return pd.DataFrame(
        {
            t: raw[t]["Close"]
            for t in raw.columns.levels[0]
            if (t in raw and "Close" in raw[t])
        }
    )


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
    Download large yfinance universes in manageable batches with retries
    and exponential backoff to reduce rate-limit issues and intermittent
    network failures.
    """
    chunks = []
    for i in tqdm(
        range(0, len(tickers), chunk_size), desc="Downloading Yahoo data", unit="chunk"
    ):
        batch, last_err = tickers[i : i + chunk_size], None
        for attempt in range(1, max_retries + 1):
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=threads,
                    timeout=timeout,
                    progress=False,
                )
                if raw is not None and not raw.empty:
                    chunks.append(raw)
                    last_err = None
                    break
                last_err = RuntimeError("Empty response from Yahoo Finance")
            except Exception as e:
                last_err = e
            time.sleep(backoff_base ** (attempt - 1))
        if last_err is not None:
            print(
                f"\nWarning: chunk {i//chunk_size + 1} failed after {max_retries} retries: {last_err}"
            )
    return pd.concat(chunks, axis=1) if chunks else pd.DataFrame()


def download_sp500_raw(start="1990-01-01", end="2025-12-01"):
    """
    Materialize the full S&P 500 price history into a local pickle once
    so that all subsequent runs read from disk instead of repeatedly
    hitting the yfinance API.
    """
    _mk(DATA_RAW)
    raw_path = f"{DATA_RAW}/sp500_raw_yfinance.pkl"

    # Check cache first
    if os.path.exists(raw_path):
        print(f"Using cached yfinance data from {raw_path}")
        return pd.read_pickle(raw_path)

    # Download if not cached
    print(f"Downloading yfinance data for {start} to {end}...")
    tickers = _read_csv(f"{DATA_RAW}/sp500_tickers.csv", header=None)[0].tolist()
    tickers = [t for t in tickers if t not in BAD_SYMBOLS_DEFAULT]

    raw = _download_in_chunks(
        tickers,
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

    # Save raw MultiIndex DataFrame as pickle (preserves structure)
    raw.to_pickle(raw_path)
    print(f"Saved raw yfinance data to {raw_path}")
    return raw


def process_sp500_returns(raw_df=None):
    """
    Standardize yfinance output into a tidy monthly price/return panel
    that can be reused consistently by beta estimation, ML, and backtests.
    """
    _mk(DATA_PROC)

    # Load raw data if not provided
    if raw_df is None:
        raw_df = pd.read_pickle(f"{DATA_RAW}/sp500_raw_yfinance.pkl")

    prices = extract_monthly_close(raw_df)
    rets = compute_monthly_returns(prices).dropna(axis=1, how="all")
    prices.index.name = rets.index.name = "Date"

    prices_path = f"{DATA_PROC}/sp500_monthly_prices.csv"
    returns_path = f"{DATA_PROC}/sp500_monthly_returns.csv"

    _save(prices, prices_path)
    _save(rets, returns_path)
    print(f"Saved monthly S&P 500 prices and returns to {DATA_PROC}/")
    return rets


def load_sp500_monthly_returns(start="1990-01-01", end="2025-12-01"):
    """Download (if needed) and process S&P 500 monthly returns."""
    raw_df = download_sp500_raw(start=start, end=end)
    return process_sp500_returns(raw_df)


def download_fundamentals_raw(
    tickers, blacklist=None, max_retries=4, backoff_base=2.0, polite_sleep=(0.05, 0.15)
):
    """
    Download raw fundamental data from yfinance (cache in data/raw).

    Revised behavior for full offline reproducibility:
    - If data/raw/sp500_fundamentals_raw.csv exists -> use it as-is (NO external calls).
    - If not, download fundamentals for the provided tickers once, then cache.
    - No incremental "top-up" for missing tickers in subsequent runs.
    """
    _mk(DATA_RAW)
    raw_path = f"{DATA_RAW}/sp500_fundamentals_raw.csv"
    failed_path = f"{DATA_RAW}/fundamentals_failed.csv"

    # 1) Pure cache path: if fundamentals file exists, trust it completely.
    if os.path.exists(raw_path):
        print(f"Using cached fundamentals from {raw_path}")
        return _read_csv(raw_path)

    # 2) First-time download only (one-off)
    blacklist = BAD_SYMBOLS_DEFAULT if blacklist is None else set(blacklist)
    tickers = [t for t in tickers if t not in blacklist]

    print(f"Downloading fundamentals for {len(tickers)} tickers...")

    def fetch_one(t):
        last = None
        for attempt in range(1, max_retries + 1):
            try:
                info = yf.Ticker(t).info
                return dict(
                    Ticker=t,
                    ME=info.get("marketCap", np.nan),
                    PB=info.get("priceToBook", np.nan),
                    ROE=info.get("returnOnEquity", np.nan),
                    RevGrowth=info.get("revenueGrowth", np.nan),
                    error="",
                )
            except Exception as e:
                last = str(e)
                time.sleep((backoff_base ** (attempt - 1)) + random.uniform(0, 0.25))
        return dict(
            Ticker=t,
            ME=np.nan,
            PB=np.nan,
            ROE=np.nan,
            RevGrowth=np.nan,
            error=last or "unknown",
        )

    rows, failed = [], []
    for t in tqdm(tickers, desc="Fetching fundamentals", unit="ticker"):
        r = fetch_one(t)
        (failed if r.get("error") else rows).append(r)
        time.sleep(random.uniform(*polite_sleep))

    fetched = pd.DataFrame(rows, columns=["Ticker", "ME", "PB", "ROE", "RevGrowth"])

    if failed:
        _save(pd.DataFrame(failed), failed_path, index=False)
        print(
            f"Warning: {len(failed)} tickers failed fundamentals fetch. See {failed_path}"
        )

    # Save one-shot dataset; no incremental augmentation on future runs
    _save(fetched, raw_path, index=False)
    print(f"Saved raw fundamentals to {raw_path}")
    return fetched


def process_fundamentals(raw_df=None):
    """Process raw fundamentals into FF5 factor classifications."""
    _mk(DATA_PROC)

    # Load raw data if not provided
    if raw_df is None:
        raw_df = _read_csv(f"{DATA_RAW}/sp500_fundamentals_raw.csv")

    df = raw_df.copy().dropna(subset=["ME"])
    df["PB"] = df["PB"].replace(0, np.nan)
    df["BM_proxy"] = 1.0 / df["PB"]

    # Calculate thresholds
    med = lambda s: s.median(skipna=True)
    size_th, bm_th, roe_th, inv_th = (
        med(df["ME"]),
        med(df["BM_proxy"]),
        med(df["ROE"]),
        med(df["RevGrowth"]),
    )

    # Classify based on thresholds
    df["Size"] = np.where(df["ME"] <= size_th, "Small", "Big")
    df["Value"] = np.where(df["BM_proxy"] >= bm_th, "High", "Low")
    df["Profitability"] = np.where(df["ROE"] >= roe_th, "Robust", "Weak")
    df["Investment"] = np.where(df["RevGrowth"] <= inv_th, "Conservative", "Aggressive")

    df = df.sort_values("Ticker")
    output_path = f"{DATA_PROC}/sp500_ff5_classifications.csv"
    _save(df, output_path, index=False)
    return df


def classify_sp500_factors(
    tickers,
    force_recompute=False,
    output_path=f"{DATA_PROC}/sp500_ff5_classifications.csv",
    failed_path=f"{DATA_RAW}/fundamentals_failed.csv",
    blacklist=None,
    max_retries=4,
    backoff_base=2.0,
    polite_sleep=(0.05, 0.15),
):
    """Download (if needed) and process FF5-style classifications using yfinance fundamentals."""
    raw_df = download_fundamentals_raw(
        tickers,
        blacklist=blacklist,
        max_retries=max_retries,
        backoff_base=backoff_base,
        polite_sleep=polite_sleep,
    )
    return process_fundamentals(raw_df)


def build_factor_ml_dataset(
    returns_path=f"{DATA_PROC}/sp500_monthly_returns.csv",
    class_path=f"{DATA_PROC}/sp500_ff5_classifications.csv",
    factor_path=f"{DATA_PROC}/Fama_French.csv",
    out_path=f"{DATA_PROC}/factor_ml_dataset.csv",
):
    """Monthly dataset for predicting 1-month ahead (t+1) FF5 factor returns."""
    rets = _read_csv(returns_path, parse_dates=["Date"]).set_index("Date").sort_index()
    classes = _read_csv(class_path).set_index("Ticker")
    common = sorted(set(rets.columns) & set(classes.index))
    rets, classes = rets[common], classes.loc[common]

    feats = pd.DataFrame(index=rets.index)
    feats["ret_all_mean"] = rets.mean(axis=1)
    feats["ret_all_std"] = rets.std(axis=1)
    feats["ret_all_dispersion"] = rets.quantile(0.9, axis=1) - rets.quantile(
        0.1, axis=1
    )
    feats["ret_pos_ratio"] = (rets > 0).mean(axis=1)

    def grp(col, a, b, name):  # Helper to compute group returns and proxies
        A, B = classes.index[classes[col] == a], classes.index[classes[col] == b]
        if len(A) and len(B):
            feats[f"ret_{a.lower()}"] = rets[A].mean(axis=1)
            feats[f"ret_{b.lower()}"] = rets[B].mean(axis=1)
            feats[f"{name}_proxy"] = (
                feats[f"ret_{a.lower()}"] - feats[f"ret_{b.lower()}"]
            )

    grp("Size", "Small", "Big", "SMB")
    grp("Value", "High", "Low", "HML")
    grp("Profitability", "Robust", "Weak", "RMW")
    grp("Investment", "Conservative", "Aggressive", "CMA")

    ff = _read_csv(factor_path, parse_dates=["Date"]).set_index("Date").sort_index()
    ds = feats.join(ff, how="inner").sort_index()

    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    ds[factors] = ds[factors].shift(-1)  # Lag factors by 1 month for t+1 prediction
    ds = ds.dropna(subset=factors)
    _save(ds, out_path, index=True)
    return ds


def add_lagged_factors(dataset: pd.DataFrame, factor_cols, lags):
    """Add lagged factor features."""
    lagged = pd.DataFrame(index=dataset.index)
    for fac in factor_cols:
        if fac not in dataset.columns:
            continue
        s = dataset[fac]
        for lag in lags:
            lagged[f"{fac}_lag{lag}"] = s.shift(lag)
        lagged[f"{fac}_ma3"] = s.shift(1).rolling(3).mean()
        lagged[f"{fac}_vol6"] = s.shift(1).rolling(6).std()
        lagged[f"{fac}_mom12"] = s.shift(1).rolling(12).mean()
    return dataset.join(lagged, how="left")


def add_market_conditions(dataset: pd.DataFrame):
    """Add market regime indicators."""
    m, v, b, d = (
        dataset["ret_all_mean"],
        dataset["ret_all_std"],
        dataset["ret_pos_ratio"],
        dataset["ret_all_dispersion"],
    )
    dataset["market_trend_3m"], dataset["market_trend_6m"] = (
        m.rolling(3).mean(),
        m.rolling(6).mean(),
    )
    dataset["volatility_3m"], dataset["volatility_6m"] = (
        v.rolling(3).mean(),
        v.rolling(6).mean(),
    )
    dataset["breadth_ma3"], dataset["breadth_change"] = b.rolling(3).mean(), b.diff()
    dataset["dispersion_ma3"], dataset["dispersion_change"] = (
        d.rolling(3).mean(),
        d.pct_change(),
    )
    for p in ["SMB_proxy", "HML_proxy", "RMW_proxy", "CMA_proxy"]:
        if p in dataset.columns:
            dataset[f"{p}_ma3"] = dataset[p].rolling(3).mean()
    return dataset


def download_macro_raw(dataset):
    """
    Download raw macro data (VIX, Oil) from yfinance (cache in data/raw).

    Strict offline behavior once cached:
    - If data/raw/vix_raw.csv or data/raw/oil_raw.csv exist -> use them as-is.
    - If a cached file is malformed or contains no valid data -> raise RuntimeError
      (do NOT re-download or delete). User can manually remove file to force a
      fresh download on the next run.
    - If the file does not exist at all -> perform a one-time download and cache.
    """
    _mk(DATA_RAW)
    vix_path = f"{DATA_RAW}/vix_raw.csv"
    oil_path = f"{DATA_RAW}/oil_raw.csv"

    # still compute date range for first-time downloads only
    start, end = dataset.index.min(), dataset.index.max() + pd.offsets.Day(31)

    def fetch_series(ticker, cache_path):
        # 1) Pure cache path: if file exists, use it or fail fast.
        if os.path.exists(cache_path):
            print(f"Using cached {ticker} data from {cache_path}")
            try:
                # Read CSV - original format: 3 header rows, Date in first column.
                df = pd.read_csv(cache_path, skiprows=2, index_col=0)

                # Convert index to datetime
                df.index = pd.to_datetime(df.index)

                # Get the Close column (first data column)
                if len(df.columns) == 0:
                    raise RuntimeError(
                        f"Cached {ticker} file {cache_path} has no data columns."
                    )

                series = df.iloc[:, 0].dropna()
                if series.empty:
                    raise RuntimeError(
                        f"Cached {ticker} file {cache_path} has no valid (non-NaN) data."
                    )

                return series

            except Exception as e:
                # Do NOT delete or re-download; fail fast for strict offline behavior
                raise RuntimeError(
                    f"Error reading cached {ticker} data from {cache_path}: {e}"
                )

        # 2) First-time download only (no cache present)
        print(f"Downloading {ticker} data (no cache found at {cache_path})...")
        try:
            df = yf.download(
                ticker, start=start, end=end, interval="1mo", progress=False
            )
            if df.empty:
                raise RuntimeError(f"Downloaded {ticker} data is empty.")

            df = df[["Close"]].copy()
            df.to_csv(cache_path)
            print(f"Saved {ticker} data to {cache_path}")
            return df["Close"]

        except Exception as e:
            raise RuntimeError(
                f"Failed to download {ticker} data for initial cache creation: {e}"
            )

    vix_series = fetch_series("^VIX", vix_path)
    oil_series = fetch_series("CL=F", oil_path)

    return vix_series, oil_series


def add_macro_features(dataset: pd.DataFrame):
    """Add minimal macro features via yfinance: VIX + Oil."""

    def process_series(s, dataset_index):
        if s.empty:
            return pd.Series(np.nan, index=dataset_index), pd.Series(
                np.nan, index=dataset_index
            )

        # Ensure the series has a DatetimeIndex
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)

        s.index = s.index.to_period("M").to_timestamp()
        s = s.reindex(dataset_index, method="ffill")
        return s, s.pct_change()

    vix_series, oil_series = download_macro_raw(dataset)

    dataset["vix"], dataset["vix_change"] = process_series(vix_series, dataset.index)
    dataset["oil_price"], dataset["oil_change"] = process_series(
        oil_series, dataset.index
    )

    return dataset.ffill().bfill()


def build_enhanced_factor_ml_dataset(
    returns_path=f"{DATA_PROC}/sp500_monthly_returns.csv",
    class_path=f"{DATA_PROC}/sp500_ff5_classifications.csv",
    factor_path=f"{DATA_PROC}/Fama_French.csv",
    out_path=f"{DATA_PROC}/factor_ml_dataset_enhanced.csv",
    factor_lags=(1, 2, 3, 6, 12),
):
    """Enhanced dataset with lagged factors, market conditions, and macro features."""
    tmp = out_path.replace(".csv", "_temp.csv")
    build_factor_ml_dataset(
        returns_path=returns_path,
        class_path=class_path,
        factor_path=factor_path,
        out_path=tmp,
    )

    ds = _read_csv(tmp, parse_dates=["Date"]).set_index("Date").sort_index()
    targets = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    ff = _read_csv(factor_path, parse_dates=["Date"]).set_index("Date").sort_index()
    ff_hist = ff[targets].rename(columns={c: f"{c}_hist" for c in targets})
    ds = ds.join(ff_hist, how="left")
    ds = add_lagged_factors(ds, list(ff_hist.columns), factor_lags).drop(
        columns=list(ff_hist.columns), errors="ignore"
    )
    ds = add_market_conditions(ds)
    ds = add_macro_features(ds)
    ds = ds.dropna(subset=targets)

    _save(ds, out_path, index=True)
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass
    return ds
