import pandas as pd
import numpy as np
import os
import yfinance as yf


def build_factor_ml_dataset(
    returns_path="data/processed/sp500_monthly_returns.csv",
    class_path="data/processed/sp500_ff5_classifications.csv",
    factor_path="data/processed/Fama_French.csv",
    out_path="data/processed/factor_ml_dataset.csv",
):
    """
    Build a monthly, factor-prediction dataset by merging:
    - S&P 500 monthly returns
    - FF-style classifications (Size, Value, Profitability, Investment)
    - Fama–French factor data (RF, Mkt-RF, Mkt, ...)

    Output: one row per month, with cross-sectional features + factors.
    """

    # 1) Load monthly returns (Date index, tickers as columns)
    rets = pd.read_csv(returns_path, parse_dates=["Date"])
    rets = rets.set_index("Date").sort_index()

    # 2) Load classifications (one row per ticker)
    classes = pd.read_csv(class_path)
    classes = classes.set_index("Ticker")

    # 3) Align tickers between returns and classifications
    common = sorted(set(rets.columns) & set(classes.index))
    rets = rets[common]
    classes = classes.loc[common]

    # 4) Build cross-sectional features per month
    features = pd.DataFrame(index=rets.index)

    # Overall market-like features
    features["ret_all_mean"] = rets.mean(axis=1)
    features["ret_all_std"] = rets.std(axis=1)
    features["ret_all_dispersion"] = (
        rets.quantile(0.9, axis=1) - rets.quantile(0.1, axis=1)
    )
    features["ret_pos_ratio"] = (rets > 0).mean(axis=1)

    # Helper to get column list for each class
    def cols_for(mask):
        return classes.index[mask]

    # Size: Small vs Big
    small_cols = cols_for(classes["Size"] == "Small")
    big_cols = cols_for(classes["Size"] == "Big")

    features["ret_small"] = rets[small_cols].mean(axis=1)
    features["ret_big"] = rets[big_cols].mean(axis=1)
    features["SMB_proxy"] = features["ret_small"] - features["ret_big"]

    # Value: High vs Low (ignore Neutral if present)
    high_cols = cols_for(classes["Value"] == "High")
    low_cols = cols_for(classes["Value"] == "Low")

    if len(high_cols) > 0 and len(low_cols) > 0:
        features["ret_highBM"] = rets[high_cols].mean(axis=1)
        features["ret_lowBM"] = rets[low_cols].mean(axis=1)
        features["HML_proxy"] = features["ret_highBM"] - features["ret_lowBM"]

    # Profitability: Robust vs Weak
    robust_cols = cols_for(classes["Profitability"] == "Robust")
    weak_cols = cols_for(classes["Profitability"] == "Weak")

    if len(robust_cols) > 0 and len(weak_cols) > 0:
        features["ret_robust"] = rets[robust_cols].mean(axis=1)
        features["ret_weak"] = rets[weak_cols].mean(axis=1)
        features["RMW_proxy"] = features["ret_robust"] - features["ret_weak"]

    # Investment: Conservative vs Aggressive
    cons_cols = cols_for(classes["Investment"] == "Conservative")
    aggr_cols = cols_for(classes["Investment"] == "Aggressive")

    if len(cons_cols) > 0 and len(aggr_cols) > 0:
        features["ret_cons"] = rets[cons_cols].mean(axis=1)
        features["ret_aggr"] = rets[aggr_cols].mean(axis=1)
        features["CMA_proxy"] = features["ret_cons"] - features["ret_aggr"]

    # 5) Load factor data (RF, Mkt-RF, Mkt, …)
    ff = pd.read_csv(factor_path, parse_dates=["Date"])
    ff = ff.set_index("Date").sort_index()

    # 6) Merge features with factors on Date
    dataset = features.join(ff, how="inner")

    #shift factors to be "next month's" targets
    # X_t (features) -> Y_{t+1} (factors)
    dataset = dataset.sort_index()
    factors_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]  # adjust to what you have
    dataset[factors_cols] = dataset[factors_cols].shift(-1)

    # Drop any rows with missing data
    dataset = dataset.dropna()

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dataset.to_csv(out_path, index=True)
    print(f"Saved factor ML dataset to {out_path}")

    return dataset


def add_macro_features(dataset, fred_api_key=None):
    """
    Add macroeconomic indicators to the dataset.
    
    If fred_api_key is None, uses VIX and oil prices only.
    Otherwise fetches real data from FRED API.
    
    Parameters
    ----------
    dataset : pd.DataFrame
        Base dataset with Date index
    fred_api_key : str, optional
        FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
    
    Returns
    -------
    dataset : pd.DataFrame
        Dataset with added macro features
    """
    
    if fred_api_key:
        try:
            from fredapi import Fred
            print("Fetching macroeconomic data from FRED...")
            fred = Fred(api_key=fred_api_key)
            
            # Get date range
            start_date = dataset.index.min()
            end_date = dataset.index.max()
            
            # Fetch key macro indicators
            macro_df = pd.DataFrame(index=dataset.index)
            
            try:
                # GDP Growth (quarterly, resampled to monthly)
                gdp = fred.get_series('GDP', start_date, end_date)
                gdp_growth = gdp.pct_change(4)  # YoY growth
                gdp_growth = gdp_growth.resample('MS').ffill()
                macro_df['gdp_growth'] = gdp_growth.reindex(dataset.index, method='ffill')
            except: pass
            
            try:
                # Inflation (CPI YoY change)
                cpi = fred.get_series('CPIAUCSL', start_date, end_date)
                inflation = cpi.pct_change(12)
                macro_df['inflation'] = inflation.reindex(dataset.index, method='ffill')
            except: pass
            
            try:
                # Unemployment Rate
                unemployment = fred.get_series('UNRATE', start_date, end_date)
                macro_df['unemployment'] = unemployment.reindex(dataset.index, method='ffill')
            except: pass
            
            try:
                # 10-Year Treasury Yield
                treasury_10y = fred.get_series('DGS10', start_date, end_date)
                treasury_10y = treasury_10y.resample('MS').mean()
                macro_df['treasury_10y'] = treasury_10y.reindex(dataset.index, method='ffill')
            except: pass
            
            try:
                # 3-Month Treasury Yield
                treasury_3m = fred.get_series('DGS3MO', start_date, end_date)
                treasury_3m = treasury_3m.resample('MS').mean()
                macro_df['treasury_3m'] = treasury_3m.reindex(dataset.index, method='ffill')
                
                # Term Spread
                if 'treasury_10y' in macro_df.columns:
                    macro_df['term_spread'] = macro_df['treasury_10y'] - macro_df['treasury_3m']
            except: pass
            
            try:
                # Fed Funds Rate
                fed_funds = fred.get_series('FEDFUNDS', start_date, end_date)
                macro_df['fed_funds'] = fed_funds.reindex(dataset.index, method='ffill')
            except: pass
            
            try:
                # Credit Spread (BAA - AAA)
                baa = fred.get_series('DBAA', start_date, end_date).resample('MS').mean()
                aaa = fred.get_series('DAAA', start_date, end_date).resample('MS').mean()
                credit_spread = baa - aaa
                macro_df['credit_spread'] = credit_spread.reindex(dataset.index, method='ffill')
            except: pass
            
            try:
                # Industrial Production Growth
                indpro = fred.get_series('INDPRO', start_date, end_date)
                indpro_growth = indpro.pct_change(12)
                macro_df['indpro_growth'] = indpro_growth.reindex(dataset.index, method='ffill')
            except: pass
            
            dataset = dataset.join(macro_df, how='left')
            print(f"Successfully fetched {len(macro_df.columns)} macro indicators from FRED")
            
        except ImportError:
            print("fredapi not installed. Install with: pip install fredapi")
            print("Falling back to VIX and oil only...")
            fred_api_key = None
        except Exception as e:
            print(f"Warning: Could not fetch FRED data: {e}")
            print("Falling back to VIX and oil only...")
            fred_api_key = None
    
    if not fred_api_key:
        print("Using VIX and oil prices only (no FRED API key)...")
    
    # Add VIX (always available)
    print("Fetching VIX (volatility index)...")
    try:
        vix = yf.download('^VIX', start=dataset.index.min(), end=dataset.index.max(), 
                         interval='1mo', progress=False)['Close']
        vix.index = vix.index.to_period('M').to_timestamp()
        dataset['vix'] = vix.reindex(dataset.index, method='ffill')
        dataset['vix_change'] = dataset['vix'].pct_change()
        print("VIX data fetched successfully")
    except Exception as e:
        print(f"Warning: Could not fetch VIX: {e}")
    
    # Add oil prices
    print("Fetching oil prices...")
    try:
        oil = yf.download('CL=F', start=dataset.index.min(), end=dataset.index.max(),
                         interval='1mo', progress=False)['Close']
        oil.index = oil.index.to_period('M').to_timestamp()
        dataset['oil_price'] = oil.reindex(dataset.index, method='ffill')
        dataset['oil_change'] = dataset['oil_price'].pct_change()
        print("Oil price data fetched successfully")
    except Exception as e:
        print(f"Warning: Could not fetch oil prices: {e}")
    
    # Fill any remaining NaNs
    macro_cols = [c for c in dataset.columns if c not in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    for col in macro_cols:
        if dataset[col].isna().any():
            dataset[col] = dataset[col].ffill().bfill()
    
    return dataset


def add_lagged_factors(dataset, lags=[1, 2, 3, 6, 12]):
    """
    Add lagged factor returns as features.
    """
    print(f"Adding lagged factors: {lags} months...")
    
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    lagged_features = pd.DataFrame(index=dataset.index)
    
    for factor in factor_cols:
        if factor in dataset.columns:
            for lag in lags:
                col_name = f'{factor}_lag{lag}'
                lagged_features[col_name] = dataset[factor].shift(lag)
    
    # Add rolling statistics
    for factor in factor_cols:
        if factor in dataset.columns:
            lagged_features[f'{factor}_ma3'] = dataset[factor].shift(1).rolling(3).mean()
            lagged_features[f'{factor}_vol6'] = dataset[factor].shift(1).rolling(6).std()
            lagged_features[f'{factor}_mom12'] = dataset[factor].shift(1).rolling(12).mean()
    
    dataset = dataset.join(lagged_features, how='left')
    print(f"Added {len(lagged_features.columns)} lagged factor features")
    
    return dataset


def add_market_conditions(dataset):
    """
    Add market condition indicators.
    """
    print("Adding market condition features...")
    
    # Market regime indicators
    if 'ret_all_mean' in dataset.columns:
        dataset['market_trend_3m'] = dataset['ret_all_mean'].rolling(3).mean()
        dataset['market_trend_6m'] = dataset['ret_all_mean'].rolling(6).mean()
        dataset['volatility_3m'] = dataset['ret_all_std'].rolling(3).mean()
        dataset['volatility_6m'] = dataset['ret_all_std'].rolling(6).mean()
        dataset['volatility_change'] = dataset['ret_all_std'].pct_change()
    
    # Momentum indicators
    if 'ret_pos_ratio' in dataset.columns:
        dataset['breadth_ma3'] = dataset['ret_pos_ratio'].rolling(3).mean()
        dataset['breadth_change'] = dataset['ret_pos_ratio'].diff()
    
    # Dispersion regime
    if 'ret_all_dispersion' in dataset.columns:
        dataset['dispersion_ma3'] = dataset['ret_all_dispersion'].rolling(3).mean()
        dataset['dispersion_change'] = dataset['ret_all_dispersion'].pct_change()
    
    # Factor momentum
    for factor in ['SMB_proxy', 'HML_proxy', 'RMW_proxy', 'CMA_proxy']:
        if factor in dataset.columns:
            dataset[f'{factor}_ma3'] = dataset[factor].rolling(3).mean()
    
    print("Market condition features added")
    
    return dataset


def build_enhanced_factor_ml_dataset(
    returns_path="data/processed/sp500_monthly_returns.csv",
    class_path="data/processed/sp500_ff5_classifications.csv",
    factor_path="data/processed/Fama_French.csv",
    out_path="data/processed/factor_ml_dataset_enhanced.csv",
    fred_api_key=None,
    factor_lags=[1, 2, 3, 6, 12]
):
    """
    Build enhanced factor ML dataset with:
    - Original cross-sectional features
    - Lagged factor returns
    - Macroeconomic indicators
    - Market condition features
    
    Parameters
    ----------
    fred_api_key : str, optional
        FRED API key. Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html
    factor_lags : list of int
        Lags to use for factor features
    """
    
    print("="*80)
    print("BUILDING ENHANCED FACTOR ML DATASET")
    print("="*80)
    
    # Start with basic dataset
    dataset = build_factor_ml_dataset(
        returns_path=returns_path,
        class_path=class_path,
        factor_path=factor_path,
        out_path=out_path.replace('_enhanced', '_temp')
    )
    
    # Load it back to add features
    dataset = pd.read_csv(out_path.replace('_enhanced', '_temp'), parse_dates=['Date'])
    dataset = dataset.set_index('Date').sort_index()
    
    # Shift factors back temporarily to add lagged features
    factors_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    dataset[factors_cols] = dataset[factors_cols].shift(1)
    
    # Add lagged factors
    dataset = add_lagged_factors(dataset, lags=factor_lags)
    
    # Add market conditions
    dataset = add_market_conditions(dataset)
    
    # Add macro features
    dataset = add_macro_features(dataset, fred_api_key=fred_api_key)
    
    # Shift factors forward again for prediction
    dataset[factors_cols] = dataset[factors_cols].shift(-1)
    
    # Drop rows with missing data
    dataset = dataset.dropna()
    
    # Save
    dataset.to_csv(out_path, index=True)
    
    print("="*80)
    print(f"Enhanced dataset saved to {out_path}")
    print(f"Shape: {dataset.shape}")
    print(f"Features: {dataset.shape[1] - 5} (excluding 5 target factors)")
    print(f"Date range: {dataset.index.min()} to {dataset.index.max()}")
    print("="*80)
    
    # Clean up temp file
    try:
        os.remove(out_path.replace('_enhanced', '_temp'))
    except:
        pass
    
    return dataset