import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.factor_predictor import FactorPredictor

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


# ======================================================================
# Utility: covariance regularization
# ======================================================================

def regularize_covariance(Sigma: np.ndarray, ridge_ratio: float = 1e-3) -> np.ndarray:
    """
    Symmetrize and ridge-regularize a covariance matrix.

    Parameters
    ----------
    Sigma : np.ndarray
        Input (approximate) covariance matrix.
    ridge_ratio : float
        Ridge size as a fraction of the average variance (trace / N).

    Returns
    -------
    Sigma_reg : np.ndarray
        Symmetric, regularized covariance matrix.
    """
    n = Sigma.shape[0]
    if n == 0:
        return Sigma

    # Symmetrize
    Sigma_sym = 0.5 * (Sigma + Sigma.T)

    # Compute average variance
    avg_var = np.trace(Sigma_sym) / n
    if not np.isfinite(avg_var) or avg_var <= 0:
        # Fallback ridge if something is off
        ridge = ridge_ratio
    else:
        ridge = ridge_ratio * avg_var

    Sigma_reg = Sigma_sym + ridge * np.eye(n)
    return Sigma_reg

MAX_WEIGHT = 0.10     # per-stock max |weight|
MAX_SHORT = 0.30      # max total short exposure (as fraction of NAV)


def apply_weight_constraints(weights: np.ndarray,
                             max_weight: float = MAX_WEIGHT,
                             max_short: float = MAX_SHORT) -> np.ndarray:
    """
    Post-process tangency weights to impose simple, realistic constraints:

    1) |w_i| <= max_weight
    2) Total short exposure <= max_short
    3) Portfolio sums to 1 (fully invested)

    Parameters
    ----------
    weights : np.ndarray
        Raw unconstrained weights (sum should be ~1, but may not be).
    max_weight : float
        Maximum absolute weight per stock.
    max_short : float
        Maximum total short exposure (sum of |w_i| for w_i < 0).

    Returns
    -------
    w_constrained : np.ndarray
        Constrained and renormalized weights.
    """

    w = weights.astype(float).copy()
    n = len(w)

    # 1) Clip each weight to [-max_weight, +max_weight]
    w = np.clip(w, -max_weight, max_weight)

    # If everything is zero after clipping, fallback to equal-weight
    if np.allclose(w, 0.0):
        return np.ones_like(w) / n

    # 2) Enforce max total short exposure
    neg_mask = w < 0
    pos_mask = w > 0

    short_exposure = -w[neg_mask].sum()  # positive number
    if short_exposure > max_short and short_exposure > 0:
        # Scale shorts so total short exposure = max_short
        scale_neg = max_short / short_exposure
        w[neg_mask] *= scale_neg

        # Recompute sums
        short_sum = w[neg_mask].sum()    # negative
        long_sum = w[pos_mask].sum()     # positive

        # If no long exposure left, fallback to equal-weight long-only
        if long_sum <= 0:
            w = np.zeros_like(w)
            w[~neg_mask] = 1.0 / (~neg_mask).sum()
            return w

        # Adjust long side so that total portfolio sums to 1
        # We keep the new short side fixed and rescale longs:
        target_long_sum = 1.0 - short_sum  # short_sum is negative
        scale_pos = target_long_sum / long_sum
        w[pos_mask] *= scale_pos
    else:
        # No short constraint binding: just normalize to sum to 1
        total = w.sum()
        if not np.allclose(total, 0.0):
            w /= total
        else:
            w = np.ones_like(w) / n

    # Final sanity checks
    if not np.isfinite(w).all():
        w = np.ones_like(w) / n

    # Enforce |w_i| <= max_weight again in case of tiny numerical drift
    w = np.clip(w, -max_weight, max_weight)

    # Renormalize one last time
    total = w.sum()
    if not np.allclose(total, 0.0):
        w /= total
    else:
        w = np.ones_like(w) / n

    return w

# ======================================================================
# FF5 beta estimation
# ======================================================================

def estimate_ff5_betas(
    returns_path: str = "data/processed/sp500_monthly_returns.csv",
    ff_path: str = "data/processed/Fama_French.csv",
    min_obs: int = 36,
    output_path: Optional[str] = "data/processed/sp500_ff5_betas.csv",
) -> pd.DataFrame:
    """
    Estimate Fama-French 5-factor betas for each stock via cross-sectional OLS.

    Parameters
    ----------
    returns_path : str
        Path to CSV with monthly stock returns (Date column, tickers as columns, decimal returns).
    ff_path : str
        Path to CSV with Fama-French data (Date, RF, Mkt-RF, SMB, HML, RMW, CMA, Mkt).
    min_obs : int
        Minimum number of overlapping observations required for a stock to get a beta estimate.
    output_path : str or None
        If provided, save the resulting beta table to this path.

    Returns
    -------
    betas_df : pd.DataFrame
        DataFrame with columns:
        - Ticker
        - Alpha
        - Beta_MKT
        - Beta_SMB
        - Beta_HML
        - Beta_RMW
        - Beta_CMA
        - R_squared
        - Adj_R_squared
        - ResidVar      (idiosyncratic variance)
        - N_obs
    """

    # 1) Load stock returns
    print("\n1. Loading stock returns...")
    returns_df = pd.read_csv(returns_path, parse_dates=["Date"], index_col="Date")
    print(f"   Loaded {len(returns_df)} months for {len(returns_df.columns)} stocks")

    # 2) Load Fama-French factors
    print("\n2. Loading Fama-French 5-factor data...")
    ff = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
    missing_cols = set(FACTOR_COLS + ["RF"]) - set(ff.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {ff_path}: {missing_cols}")

    # 3) Align by date and remove overlapping columns in returns_df
    print("\n3. Aligning stock returns and factor data...")
    overlap_cols = list(set(returns_df.columns) & set(FACTOR_COLS + ["RF"]))
    if overlap_cols:
        returns_df = returns_df.drop(columns=overlap_cols)

    data = returns_df.join(ff[FACTOR_COLS + ["RF"]], how="inner")
    print(f"   Overlapping period has {len(data)} months")

    # 4) Compute excess returns for all stocks
    print("\n4. Computing excess returns...")
    rf = data["RF"]
    stock_cols = [c for c in data.columns if c not in FACTOR_COLS + ["RF"]]
    stock_excess = data[stock_cols].sub(rf, axis=0)

    # Factor matrix X (no RF)
    X_factors = data[FACTOR_COLS].copy()
    X_factors["const"] = 1.0  # intercept

    X_cols = ["const"] + FACTOR_COLS

    betas = []
    for ticker in stock_cols:
        y = stock_excess[ticker]

        # Drop rows with any NaN in y or factors
        df_reg = pd.concat([y, X_factors], axis=1).dropna()
        if len(df_reg) < min_obs:
            betas.append(
                {
                    "Ticker": ticker,
                    "Alpha": np.nan,
                    "Beta_MKT": np.nan,
                    "Beta_SMB": np.nan,
                    "Beta_HML": np.nan,
                    "Beta_RMW": np.nan,
                    "Beta_CMA": np.nan,
                    "R_squared": np.nan,
                    "Adj_R_squared": np.nan,
                    "ResidVar": np.nan,
                    "N_obs": len(df_reg),
                }
            )
            continue

        y_reg = df_reg[ticker].values
        X_reg = df_reg[X_cols].values

        # OLS via lstsq
        beta_hat, residuals, rank, s = np.linalg.lstsq(X_reg, y_reg, rcond=None)
        # beta_hat: [alpha, beta_MKT, beta_SMB, beta_HML, beta_RMW, beta_CMA]
        alpha = beta_hat[0]
        beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma = beta_hat[1:]

        # Goodness of fit
        y_pred = X_reg @ beta_hat
        ss_res = np.sum((y_reg - y_pred) ** 2)
        ss_tot = np.sum((y_reg - y_reg.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        n = len(y_reg)
        k = len(FACTOR_COLS)  # number of factors (no intercept)
        if n > k + 1 and not np.isnan(r_squared):
            adj_r2 = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k - 1)
        else:
            adj_r2 = np.nan

        resid_var = ss_res / (n - k - 1) if n > k + 1 else np.nan

        betas.append(
            {
                "Ticker": ticker,
                "Alpha": alpha,
                "Beta_MKT": beta_mkt,
                "Beta_SMB": beta_smb,
                "Beta_HML": beta_hml,
                "Beta_RMW": beta_rmw,
                "Beta_CMA": beta_cma,
                "R_squared": r_squared,
                "Adj_R_squared": adj_r2,
                "ResidVar": resid_var,
                "N_obs": n,
            }
        )

    betas_df = pd.DataFrame(betas).set_index("Ticker").sort_index()

    print("\n5. Summary statistics for FF5 betas:")
    print("-" * 80)
    print(f"   Number of stocks:      {betas_df.shape[0]}")
    print(f"   Valid beta estimates:  {betas_df['Beta_MKT'].notna().sum()}")
    print(f"   Avg CAPM-like beta:    {betas_df['Beta_MKT'].mean():.3f}")
    print(f"   Avg HML beta:          {betas_df['Beta_HML'].mean():.3f}")
    print(f"   Avg R²:                {betas_df['R_squared'].mean():.3f}")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        betas_df.to_csv(output_path)
        print(f"\n   FF5 betas saved to: {output_path}")

    return betas_df


# ======================================================================
# Factor premia with ML-overlay on HML
# ======================================================================

def compute_factor_premia_with_hml_overlay(
    ff_path: str,
    factor_ml_dataset_path: str,
    best_model: FactorPredictor,
    lambda_hml: float = 0.2,
) -> pd.Series:
    """
    Compute expected factor premia E[f_{t+1}] using:
    - Historical means for all factors
    - ML overlay ONLY on HML, heavily shrunk toward historical mean

    Parameters
    ----------
    ff_path : str
        Path to Fama-French CSV.
    factor_ml_dataset_path : str
        Path to enhanced factor ML dataset (same as used by FactorPredictor).
    best_model : FactorPredictor
        Already-trained ML predictor from evaluate_all_models.
    lambda_hml : float
        Shrinkage parameter in [0, 1]. 0 = pure historical mean, 1 = pure ML HML.

    Returns
    -------
    mu_f : pd.Series
        Expected factor premia indexed by FACTOR_COLS.
    """

    print("\n" + "=" * 80)
    print("EXPECTED FACTOR PREMIA (HISTORICAL + ML HML OVERLAY)")
    print("=" * 80)

    # 1) Historical means from Fama-French data
    ff = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
    hist_means = ff[FACTOR_COLS].mean()
    print("\nHistorical factor means (monthly):")
    for fac in FACTOR_COLS:
        print(f"  {fac:7s}: μ = {hist_means[fac]:+.4f} ({hist_means[fac]*100:+.2f}%)")

    mu_f = hist_means.copy()

    # 2) ML prediction for HML using best_model on latest feature row
    print("\nComputing ML-based overlay for HML...")
    X_all, y_all, dates_all = best_model.prepare_data(factor_ml_dataset_path)

    current_features = X_all.iloc[-1]
    hml_pred = best_model.predict_next_month(current_features)["HML"]

    print(f"  ML HML forecast:        {hml_pred:+.4f} ({hml_pred*100:+.2f}%)")
    print(f"  Historical HML mean:    {hist_means['HML']:+.4f} ({hist_means['HML']*100:+.2f}%)")
    hml_shrunk = (1 - lambda_hml) * hist_means["HML"] + lambda_hml * hml_pred
    print(
        f"  Shrunk HML premium (λ={lambda_hml:.2f}): "
        f"{hml_shrunk:+.4f} ({hml_shrunk*100:+.2f}%)"
    )

    mu_f["HML"] = hml_shrunk

    return mu_f


# ======================================================================
# Factor covariance from FF data
# ======================================================================

def build_ff5_factor_model(
    betas_df: pd.DataFrame,
    ff_path: str = "data/processed/Fama_French.csv",
) -> np.ndarray:
    """
    Build factor covariance matrix Σ_f from Fama-French data.

    Parameters
    ----------
    betas_df : pd.DataFrame
        Output of estimate_ff5_betas(), indexed by Ticker.
        (Provided for potential future use; not used directly here.)
    ff_path : str
        Path to Fama-French CSV for estimating Σ_f.

    Returns
    -------
    Sigma_f : np.ndarray
        5x5 factor covariance matrix Σ_f (monthly).
    """

    ff = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
    factor_returns = ff[FACTOR_COLS].dropna()
    Sigma_f = factor_returns.cov().values
    return Sigma_f


# ======================================================================
# Static unconstrained FF5 tangency portfolio
# ======================================================================

def build_ff5_optimal_portfolio(
    returns_path: str,
    ff_path: str,
    factor_ml_dataset_path: str,
    best_model: FactorPredictor,
    lambda_hml: float = 0.2,
    min_obs: int = 36,
) -> pd.DataFrame:
    """
    Build the unconstrained FF5 tangency portfolio with an ML HML overlay.

    Steps
    -----
    1) Estimate FF5 betas and residual variances for each stock.
    2) Estimate factor covariance matrix Σ_f from history.
    3) Compute expected factor premia μ_f (historical means + shrunk ML HML).
    4) Compute asset expected excess returns μ_R = B μ_f.
    5) Compute asset covariance Σ_R = B Σ_f B' + Ω (regularized).
    6) Compute tangency weights: w* ∝ Σ_R^{-1} μ_R, normalized to sum to 1.

    Returns
    -------
    weights_df : pd.DataFrame
        DataFrame indexed by Ticker with columns:
        - Weight
        - Alpha, Beta_MKT, Beta_SMB, Beta_HML, Beta_RMW, Beta_CMA
        - ResidVar
    """

    # 1) Estimate FF5 betas
    betas_df = estimate_ff5_betas(
        returns_path=returns_path,
        ff_path=ff_path,
        min_obs=min_obs,
        output_path="data/processed/sp500_ff5_betas.csv",
    )

    # Keep only stocks with valid betas
    betas_valid = betas_df.dropna(
        subset=["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA"]
    )
    if betas_valid.empty:
        raise ValueError("No valid FF5 betas available for portfolio construction.")

    # 2) Factor covariance
    Sigma_f = build_ff5_factor_model(betas_valid, ff_path=ff_path)

    # 3) Expected factor premia μ_f (with HML overlay)
    mu_f = compute_factor_premia_with_hml_overlay(
        ff_path=ff_path,
        factor_ml_dataset_path=factor_ml_dataset_path,
        best_model=best_model,
        lambda_hml=lambda_hml,
    )

    # 4) Asset expected excess returns μ_R = B μ_f
    B = betas_valid[["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA"]].values
    mu_f_vec = mu_f[FACTOR_COLS].values.reshape(-1, 1)  # 5x1
    mu_R = (B @ mu_f_vec).reshape(-1)  # N

    # 5) Asset covariance Σ_R = B Σ_f B' + Ω (regularized)
    resid_var = betas_valid["ResidVar"].fillna(betas_valid["ResidVar"].median())
    Omega = np.diag(resid_var.values)  # N x N
    Sigma_R = B @ Sigma_f @ B.T + Omega
    Sigma_R = regularize_covariance(Sigma_R, ridge_ratio=1e-3)

    print("\n" + "=" * 80)
    print("FF5 TANGENCY PORTFOLIO (UNCONSTRAINED, EXCESS RETURN SPACE)")
    print("=" * 80)

    # 6) Tangency portfolio: w* ∝ Σ_R^{-1} μ_R
    inv_Sigma_R = np.linalg.pinv(Sigma_R)
    raw_w = inv_Sigma_R @ mu_R

    if np.allclose(raw_w.sum(), 0.0):
        raise ValueError("Sum of raw weights is zero; cannot normalize. Check inputs.")

    # Unconstrained tangency weights
    w_raw = raw_w / raw_w.sum()

    # Apply practical constraints
    w_star = apply_weight_constraints(w_raw)

    weights_df = betas_valid.copy()
    weights_df["Weight"] = w_star

    weights_df = weights_df.sort_values("Weight", ascending=False)

    # Portfolio-level betas
    total_weight = float(weights_df["Weight"].sum())
    avg_beta_mkt = float(np.sum(weights_df["Weight"] * weights_df["Beta_MKT"]))
    avg_beta_hml = float(np.sum(weights_df["Weight"] * weights_df["Beta_HML"]))

    # Portfolio risk & return
    mu_p = float(w_star @ mu_R)
    var_p = float(w_star @ (Sigma_R @ w_star))
    sigma_p = float(np.sqrt(max(var_p, 0.0)))
    sharpe_p = mu_p / sigma_p if sigma_p > 0 else np.nan

    # CAPM-style alpha vs Fama-French Mkt-RF mean
    ff_full = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
    mu_mkt = float(ff_full["Mkt-RF"].mean())
    alpha_capm = mu_p - avg_beta_mkt * mu_mkt

    print("\nTop 10 positions in FF5 tangency portfolio:")
    print("-" * 80)
    print(
        weights_df[["Weight", "Beta_MKT", "Beta_HML"]]
        .head(10)
        .to_string(float_format=lambda x: f"{x:+.4f}")
    )

    print("\nPortfolio summary:")
    print("-" * 80)
    print(f"  Sum of weights:         {total_weight:.4f}")
    print(f"  Portfolio MKT beta:     {avg_beta_mkt:.3f}")
    print(f"  Portfolio HML beta:     {avg_beta_hml:.3f}")
    print(f"  Expected excess return: {mu_p:+.4f} ({mu_p*100:+.2f}%)")
    print(f"  Volatility (stdev):     {sigma_p:.4f} ({sigma_p*100:.2f}%)")
    print(f"  Sharpe ratio:           {sharpe_p:.3f}")
    print(f"  CAPM alpha (monthly):   {alpha_capm:+.4f} ({alpha_capm*100:+.2f}%)")
    print(f"  Number of stocks:       {len(weights_df)}")

    os.makedirs("data/processed", exist_ok=True)
    weights_df.to_csv("data/processed/ff5_optimal_portfolio_weights.csv")
    print("\nWeights saved to: data/processed/ff5_optimal_portfolio_weights.csv")

    return weights_df


# ======================================================================
# Rolling expanding-window FF5 backtest
# ======================================================================

def backtest_ff5_tangency(
    returns_path: str,
    ff_path: str,
    min_train_months: int = 120,
    min_obs_per_stock: int = 36,
) -> pd.DataFrame:
    """
    Rolling expanding-window backtest of FF5 tangency portfolio.

    For each month t >= min_train_months:
        1) Use data up to t-1 to estimate FF5 betas and Σ_f.
        2) Compute μ_f as historical means (training window only).
        3) Build μ_R and Σ_R, compute portfolio weights .
        4) Apply weights to month t excess returns to get realized portfolio return.

    Returns
    -------
    results : pd.DataFrame
        Columns:
            Date
            Port_Excess_Return
            Mkt_RF
    """

    print(f"  min_train_months   = {min_train_months}")
    print(f"  min_obs_per_stock  = {min_obs_per_stock}")
    # Load data
    returns_df = pd.read_csv(returns_path, parse_dates=["Date"], index_col="Date")
    ff = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")

    # Align & drop overlapping factor/RF columns from returns_df
    overlap_cols = list(set(returns_df.columns) & set(FACTOR_COLS + ["RF"]))
    if overlap_cols:
        returns_df = returns_df.drop(columns=overlap_cols)

    # Join on date
    data = returns_df.join(ff[FACTOR_COLS + ["RF"]], how="inner")
    data = data.sort_index()

    stock_cols = [c for c in data.columns if c not in FACTOR_COLS + ["RF"]]

    # Fix universe to stocks with enough data overall
    valid_universe = [
        c for c in stock_cols if data[c].count() >= (min_train_months + min_obs_per_stock)
    ]
    data = data[valid_universe + FACTOR_COLS + ["RF"]]
    stock_cols = valid_universe

    print(f"\nBacktest universe size: {len(stock_cols)} stocks")
    print(f"Total months available: {len(data)}")

    dates = data.index
    n_months = len(dates)

    if n_months <= min_train_months + 1:
        raise ValueError("Not enough data for the requested training window.")

    port_rets = []
    mkt_rets = []
    out_dates = []

    for t_idx in tqdm(
        range(min_train_months, n_months),
        desc="Rolling FF5 Backtest",
        leave=True,
    ):
        # Training window: up to t_idx-1
        train_slice = data.iloc[:t_idx]
        test_row = data.iloc[t_idx]
        test_date = dates[t_idx]

        # 1) Compute excess returns in training window
        rf_train = train_slice["RF"]
        stock_excess_train = train_slice[stock_cols].sub(rf_train, axis=0)

        # 2) OLS FF5 betas for each stock in universe
        X_factors = train_slice[FACTOR_COLS].copy()
        X_factors["const"] = 1.0
        X_cols = ["const"] + FACTOR_COLS

        betas_list = []
        resid_vars = []

        for col in stock_cols:
            y = stock_excess_train[col]
            df_reg = pd.concat([y, X_factors], axis=1).dropna()
            if len(df_reg) < min_obs_per_stock:
                betas_list.append([np.nan] * 6)  # alpha + 5 betas
                resid_vars.append(np.nan)
                continue

            y_reg = df_reg[col].values
            X_reg = df_reg[X_cols].values

            beta_hat, residuals, rank, s = np.linalg.lstsq(X_reg, y_reg, rcond=None)
            alpha = beta_hat[0]
            betas = beta_hat[1:]

            # Residual variance
            k = len(FACTOR_COLS)
            n = len(y_reg)
            ss_res = np.sum((y_reg - X_reg @ beta_hat) ** 2)
            resid_var = ss_res / (n - k - 1) if n > k + 1 else np.nan

            betas_list.append([alpha] + list(betas))
            resid_vars.append(resid_var)

        betas_arr = np.array(betas_list)  # shape N x 6
        alpha_vec = betas_arr[:, 0]
        B = betas_arr[:, 1:]  # N x 5

        # 3) Factor moments from training window
        factor_returns_train = train_slice[FACTOR_COLS].dropna()
        mu_f = factor_returns_train.mean().values  # 5
        Sigma_f = factor_returns_train.cov().values  # 5x5

        # 4) Asset moments
        mu_R = B @ mu_f  # N
        resid_var_arr = np.array(resid_vars)
        # Replace missing residual vars with median of valid ones
        if np.all(np.isnan(resid_var_arr)):
            continue
        resid_var_filled = np.where(
            np.isnan(resid_var_arr),
            np.nanmedian(resid_var_arr),
            resid_var_arr,
        )
        Omega = np.diag(resid_var_filled)
        Sigma_R = B @ Sigma_f @ B.T + Omega
        Sigma_R = regularize_covariance(Sigma_R, ridge_ratio=1e-3)

        # 5) Compute weights
       
        try:
            inv_Sigma_R = np.linalg.pinv(Sigma_R)
            raw_w = inv_Sigma_R @ mu_R
            if np.allclose(raw_w.sum(), 0.0):
                # If the optimizer degenerates, skip this month
                continue

            # Unconstrained tangency weights
            w_raw = raw_w / raw_w.sum()

            # Apply practical constraints (max 10% per stock, max 30% shorts)
            w_star = apply_weight_constraints(w_raw)

        except np.linalg.LinAlgError:
            # Skip if Σ_R is too singular even after regularization
            continue


        # 6) Realized excess return at month t_idx
        rf_test = float(test_row["RF"])
        r_test = test_row[stock_cols].values.astype(float)
        ex_test = r_test - rf_test

        port_excess_ret = float(np.nansum(w_star * ex_test))
        mkt_excess_ret = float(test_row["Mkt-RF"]) if "Mkt-RF" in test_row.index else np.nan

        port_rets.append(port_excess_ret)
        mkt_rets.append(mkt_excess_ret)
        out_dates.append(test_date)

    results = pd.DataFrame(
        {
            "Date": out_dates,
            "Port_Excess_Return": port_rets,
            "Mkt_RF": mkt_rets,
        }
    ).set_index("Date")

    # Summary statistics
    if len(results) > 1:
        mu_port = results["Port_Excess_Return"].mean()
        sigma_port = results["Port_Excess_Return"].std(ddof=1)
        sharpe_port = mu_port / sigma_port if sigma_port > 0 else np.nan

        # CAPM beta & alpha vs Mkt-RF
        valid = results.dropna()
        if len(valid) > 1:
            cov_pm = np.cov(valid["Port_Excess_Return"], valid["Mkt_RF"])[0, 1]
            var_m = np.var(valid["Mkt_RF"], ddof=1)
            beta_capm = cov_pm / var_m if var_m > 0 else np.nan
            mu_mkt = valid["Mkt_RF"].mean()
            alpha_capm = mu_port - beta_capm * mu_mkt if not np.isnan(beta_capm) else np.nan
        else:
            beta_capm = np.nan
            alpha_capm = np.nan

        print("\nBacktest summary:")
        print("-" * 80)
        print(f"  Periods:                {len(results)}")
        print(f"  Mean excess return:     {mu_port:+.4f} ({mu_port*100:+.2f}%)")
        print(f"  Volatility:             {sigma_port:.4f} ({sigma_port*100:.2f}%)")
        print(f"  Sharpe ratio:           {sharpe_port:.3f}")
        print(f"  CAPM beta (vs Mkt-RF):  {beta_capm:.3f}")
        print(f"  CAPM alpha (monthly):   {alpha_capm:+.4f} ({alpha_capm*100:+.2f}%)")
    else:
        print("\nBacktest produced too few observations for summary statistics.")

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/ff5_backtest_unconstrained.csv"
    results.to_csv(out_path)
    print(f"\nBacktest results saved to: {out_path}")

    return results
