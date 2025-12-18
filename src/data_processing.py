import os, time, zipfile, random
from io import BytesIO
import numpy as np
import pandas as pd
import requests, yfinance as yf
from tqdm import tqdm

DATA_RAW, DATA_PROC = "data/raw", "data/processed"
TICKER_FIX = (".", "-") # Replace dots with dashes in tickers
BAD_SYMBOLS_DEFAULT = {"WBA"} # Known bad symbols for yfinance
FF5_ZIP = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_CSV.zip") # Fama-French 5-factor data
SP500_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv" # S&P 500 companies


def _mk(*paths): [os.makedirs(p, exist_ok=True) for p in paths]
def _read_csv(path, **kw): return pd.read_csv(path, **kw)
def _save(df, path, **kw): _mk(os.path.dirname(path)); df.to_csv(path, **kw); return df


def load_sp500_companies():
    """Load S&P 500 company data from DataHub and export tickers + full table."""
    _mk(f"{DATA_RAW}", f"{DATA_PROC}")
    df = _read_csv(SP500_URL)
    if "Symbol" not in df.columns: raise ValueError("CSV is missing required columns: {'Symbol'}")
    df["Symbol"] = df["Symbol"].str.replace(*TICKER_FIX, regex=False)
    df["Symbol"].to_csv(f"{DATA_RAW}/sp500_tickers.csv", index=False, header=False)
    _save(df, f"{DATA_RAW}/sp500_companies.csv", index=False)
    print(f"Exported {len(df)} companies to {DATA_RAW}/sp500_companies.csv")
    return df


def load_rf():
    """Download Fama-French 5-factor monthly data; return Date, RF, factors, and Mkt."""
    _mk(DATA_RAW, DATA_PROC)
    r = requests.get(FF5_ZIP); r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(z.open(csv_name), skiprows=3)

    _save(df, f"{DATA_RAW}/French_Library_data.csv", index=False, float_format="%.6f")

    df = df.rename(columns={df.columns[0]: "Date"})
    df = df[pd.to_numeric(df["Date"], errors="coerce").notnull()]
    df["Date"] = df["Date"].astype(int).astype(str)
    df = df[df["Date"].str.len() == 6]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")

    cols = ["RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df["Mkt"] = df["Mkt-RF"] + df["RF"]

    rf = df[["Date", *cols, "Mkt"]].copy()
    _save(rf, f"{DATA_PROC}/Fama_French.csv", index=False, float_format="%.6f")
    print(f"Saved monthly RF + Mkt data to {DATA_PROC}/Fama_French.csv")
    return rf


def extract_monthly_close(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract monthly 'Close' for each ticker from yfinance output (MultiIndex columns)."""
    return pd.DataFrame({t: raw[t]["Close"] for t in raw.columns.levels[0] if (t in raw and "Close" in raw[t])})


def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly percentage returns from monthly prices."""
    return monthly_prices.pct_change(fill_method=None).dropna(how="all")


def _download_in_chunks(tickers, start, end, interval="1mo", chunk_size=50, timeout=60,
                        threads=False, max_retries=4, backoff_base=2.0):
    """Download yfinance data in chunks with retries + exponential backoff."""
    chunks = []
    for i in tqdm(range(0, len(tickers), chunk_size), desc="Downloading Yahoo data", unit="chunk"):
        batch, last_err = tickers[i:i + chunk_size], None
        for attempt in range(1, max_retries + 1):
            try:
                raw = yf.download(batch, start=start, end=end, interval=interval, auto_adjust=True,
                                  group_by="ticker", threads=threads, timeout=timeout, progress=False)
                if raw is not None and not raw.empty: chunks.append(raw); last_err = None; break
                last_err = RuntimeError("Empty response from Yahoo Finance")
            except Exception as e:
                last_err = e
            time.sleep(backoff_base ** (attempt - 1))
        if last_err is not None:
            print(f"\nWarning: chunk {i//chunk_size + 1} failed after {max_retries} retries: {last_err}")
    return pd.concat(chunks, axis=1) if chunks else pd.DataFrame()


def load_sp500_monthly_returns(start="1990-01-01", end="2025-12-01", force_download=False):
    """Cache-first: load computed returns or download in chunks with retries/backoff."""
    _mk(DATA_PROC)
    returns_path = f"{DATA_PROC}/sp500_monthly_returns.csv"
    prices_path = f"{DATA_PROC}/sp500_monthly_prices.csv"
    if (not force_download) and os.path.exists(returns_path):
        df = _read_csv(returns_path, parse_dates=["Date"]).set_index("Date").sort_index()
        print(f"Loaded cached monthly returns from {returns_path} (no Yahoo download).")
        return df

    tickers = _read_csv(f"{DATA_RAW}/sp500_tickers.csv", header=None)[0].tolist()
    tickers = [t for t in tickers if t not in BAD_SYMBOLS_DEFAULT]

    raw = _download_in_chunks(tickers, start=start, end=end, interval="1mo",
                              chunk_size=50, timeout=90, threads=False, max_retries=4, backoff_base=2.0)
    if raw.empty: raise RuntimeError("Yahoo download failed completely (no data returned).")

    prices = extract_monthly_close(raw)
    rets = compute_monthly_returns(prices).dropna(axis=1, how="all")
    prices.index.name = rets.index.name = "Date"
    _save(prices, prices_path); _save(rets, returns_path)
    print(f"Saved monthly S&P 500 prices and returns to {DATA_PROC}/")
    return rets


def classify_sp500_factors(tickers, force_recompute=False,
                           output_path=f"{DATA_PROC}/sp500_ff5_classifications.csv",
                           failed_path=f"{DATA_PROC}/fundamentals_failed.csv",
                           blacklist=None, max_retries=4, backoff_base=2.0, polite_sleep=(0.05, 0.15)):
    """Approx FF5-style classification using yfinance fundamentals with cache + retry/backoff to reduce load and failures."""
    _mk(DATA_PROC)
    blacklist = BAD_SYMBOLS_DEFAULT if blacklist is None else set(blacklist)
    tickers = [t for t in tickers if t not in blacklist]

    cached_df, cached_tickers = None, set()
    if (not force_recompute) and os.path.exists(output_path):
        try:
            cached_df = _read_csv(output_path)
            cached_tickers = set(cached_df["Ticker"].astype(str)) if "Ticker" in cached_df.columns else set()
        except Exception:
            cached_df, cached_tickers = None, set()

    to_fetch = [t for t in tickers if t not in cached_tickers]
    if cached_df is not None and not to_fetch: return cached_df

    def fetch_one(t): # Fetch fundamentals for one ticker with retries + backoff
        last = None
        for attempt in range(1, max_retries + 1):
            try:
                info = yf.Ticker(t).info
                return dict(Ticker=t,
                            ME=info.get("marketCap", np.nan),
                            PB=info.get("priceToBook", np.nan),
                            ROE=info.get("returnOnEquity", np.nan),
                            RevGrowth=info.get("revenueGrowth", np.nan),
                            error="")
            except Exception as e:
                last = str(e)
                time.sleep((backoff_base ** (attempt - 1)) + random.uniform(0, 0.25))
        return dict(Ticker=t, ME=np.nan, PB=np.nan, ROE=np.nan, RevGrowth=np.nan, error=last or "unknown")

    rows, failed = [], []
    for t in tqdm(to_fetch, desc="Fetching fundamentals", unit="ticker"): # Fetch each ticker
        r = fetch_one(t)
        (failed if r.get("error") else rows).append(r)
        time.sleep(random.uniform(*polite_sleep))

    fetched = pd.DataFrame(rows, columns=["Ticker", "ME", "PB", "ROE", "RevGrowth"]) 
    if failed:
        _save(pd.DataFrame(failed), failed_path, index=False)
        print(f"Warning: {len(failed)} tickers failed fundamentals fetch. See {failed_path}")

    merged = pd.concat([cached_df, fetched], ignore_index=True) if (cached_df is not None and not cached_df.empty) else fetched
    merged = merged.drop_duplicates(subset=["Ticker"], keep="first").dropna(subset=["ME"])
    merged["PB"] = merged["PB"].replace(0, np.nan) 
    merged["BM_proxy"] = 1.0 / merged["PB"]

    med = lambda s: s.median(skipna=True) # Helper to compute median while skipping NaNs
    size_th, bm_th, roe_th, inv_th = med(merged["ME"]), med(merged["BM_proxy"]), med(merged["ROE"]), med(merged["RevGrowth"]) # Thresholds
    # Classify based on thresholds
    merged["Size"] = np.where(merged["ME"] <= size_th, "Small", "Big") 
    merged["Value"] = np.where(merged["BM_proxy"] >= bm_th, "High", "Low")
    merged["Profitability"] = np.where(merged["ROE"] >= roe_th, "Robust", "Weak")
    merged["Investment"] = np.where(merged["RevGrowth"] <= inv_th, "Conservative", "Aggressive")

    merged = merged.sort_values("Ticker")
    _save(merged, output_path, index=False)
    return merged


def build_factor_ml_dataset(returns_path=f"{DATA_PROC}/sp500_monthly_returns.csv",
                            class_path=f"{DATA_PROC}/sp500_ff5_classifications.csv",
                            factor_path=f"{DATA_PROC}/Fama_French.csv",
                            out_path=f"{DATA_PROC}/factor_ml_dataset.csv"):
    """Monthly dataset for predicting 1-month ahead (t+1) FF5 factor returns."""
    rets = _read_csv(returns_path, parse_dates=["Date"]).set_index("Date").sort_index()
    classes = _read_csv(class_path).set_index("Ticker")
    common = sorted(set(rets.columns) & set(classes.index))
    rets, classes = rets[common], classes.loc[common]

    feats = pd.DataFrame(index=rets.index)
    feats["ret_all_mean"] = rets.mean(axis=1)
    feats["ret_all_std"] = rets.std(axis=1)
    feats["ret_all_dispersion"] = rets.quantile(0.9, axis=1) - rets.quantile(0.1, axis=1)
    feats["ret_pos_ratio"] = (rets > 0).mean(axis=1)

    def grp(col, a, b, name): # Helper to compute group returns and proxies
        A, B = classes.index[classes[col] == a], classes.index[classes[col] == b]
        if len(A) and len(B):
            feats[f"ret_{a.lower()}"] = rets[A].mean(axis=1)
            feats[f"ret_{b.lower()}"] = rets[B].mean(axis=1)
            feats[f"{name}_proxy"] = feats[f"ret_{a.lower()}"] - feats[f"ret_{b.lower()}"]

    grp("Size", "Small", "Big", "SMB")
    grp("Value", "High", "Low", "HML")
    grp("Profitability", "Robust", "Weak", "RMW")
    grp("Investment", "Conservative", "Aggressive", "CMA")

    ff = _read_csv(factor_path, parse_dates=["Date"]).set_index("Date").sort_index()
    ds = feats.join(ff, how="inner").sort_index()

    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    ds[factors] = ds[factors].shift(-1) # Lag factors by 1 month for t+1 prediction
    ds = ds.dropna(subset=factors)
    _save(ds, out_path, index=True)
    return ds


def add_lagged_factors(dataset: pd.DataFrame, factor_cols, lags):
    """Add lagged factor features."""
    lagged = pd.DataFrame(index=dataset.index)
    for fac in factor_cols:
        if fac not in dataset.columns: continue
        s = dataset[fac]
        for lag in lags: lagged[f"{fac}_lag{lag}"] = s.shift(lag)
        lagged[f"{fac}_ma3"] = s.shift(1).rolling(3).mean()
        lagged[f"{fac}_vol6"] = s.shift(1).rolling(6).std()
        lagged[f"{fac}_mom12"] = s.shift(1).rolling(12).mean()
    return dataset.join(lagged, how="left")


def add_market_conditions(dataset: pd.DataFrame):
    """Add market regime indicators."""
    m, v, b, d = dataset["ret_all_mean"], dataset["ret_all_std"], dataset["ret_pos_ratio"], dataset["ret_all_dispersion"]
    dataset["market_trend_3m"], dataset["market_trend_6m"] = m.rolling(3).mean(), m.rolling(6).mean()
    dataset["volatility_3m"], dataset["volatility_6m"] = v.rolling(3).mean(), v.rolling(6).mean()
    dataset["breadth_ma3"], dataset["breadth_change"] = b.rolling(3).mean(), b.diff()
    dataset["dispersion_ma3"], dataset["dispersion_change"] = d.rolling(3).mean(), d.pct_change()
    for p in ["SMB_proxy", "HML_proxy", "RMW_proxy", "CMA_proxy"]:
        if p in dataset.columns: dataset[f"{p}_ma3"] = dataset[p].rolling(3).mean()
    return dataset


def add_macro_features(dataset: pd.DataFrame):
    """Add minimal macro features via yfinance: VIX + Oil."""
    start, end = dataset.index.min(), dataset.index.max() + pd.offsets.Day(31)

    def _series(tkr, col, chg):
        try:
            s = yf.download(tkr, start=start, end=end, interval="1mo", progress=False)["Close"]
            if s.empty: return pd.Series(np.nan, index=dataset.index), pd.Series(np.nan, index=dataset.index)
            s.index = s.index.to_period("M").to_timestamp()
            s = s.reindex(dataset.index, method="ffill")
            return s, s.pct_change()
        except Exception:
            return pd.Series(np.nan, index=dataset.index), pd.Series(np.nan, index=dataset.index)

    dataset["vix"], dataset["vix_change"] = _series("^VIX", "vix", "vix_change")
    dataset["oil_price"], dataset["oil_change"] = _series("CL=F", "oil_price", "oil_change")
    return dataset.ffill().bfill()


def build_enhanced_factor_ml_dataset(returns_path=f"{DATA_PROC}/sp500_monthly_returns.csv",
                                     class_path=f"{DATA_PROC}/sp500_ff5_classifications.csv",
                                     factor_path=f"{DATA_PROC}/Fama_French.csv",
                                     out_path=f"{DATA_PROC}/factor_ml_dataset_enhanced.csv",
                                     factor_lags=(1, 2, 3, 6, 12)):
    """Enhanced dataset with lagged factors, market conditions, and macro features."""
    tmp = out_path.replace(".csv", "_temp.csv")
    build_factor_ml_dataset(returns_path=returns_path, class_path=class_path, factor_path=factor_path, out_path=tmp)

    ds = _read_csv(tmp, parse_dates=["Date"]).set_index("Date").sort_index()
    targets = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    ff = _read_csv(factor_path, parse_dates=["Date"]).set_index("Date").sort_index()
    ff_hist = ff[targets].rename(columns={c: f"{c}_hist" for c in targets})
    ds = ds.join(ff_hist, how="left")
    ds = add_lagged_factors(ds, list(ff_hist.columns), factor_lags).drop(columns=list(ff_hist.columns), errors="ignore")
    ds = add_market_conditions(ds)
    ds = add_macro_features(ds)
    ds = ds.dropna(subset=targets)

    _save(ds, out_path, index=True)
    try: os.remove(tmp)
    except FileNotFoundError: pass
    return ds
