import pandas as pd
import os


def build_factor_ml_dataset(
    returns_path="data/processed/sp500_monthly_returns.csv",
    class_path="data/processed/sp500_ff5_classifications.csv",
    factor_path="data/processed/risk_free.csv",
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