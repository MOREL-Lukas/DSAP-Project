import os
import numpy as np
import pandas as pd

from typing import Tuple, Optional

from src.factor_predictor import FactorPredictor
from src.monte_carlo import HistoricalMeanBaseline


FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


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

    print("\n" + "=" * 80)
    print("FF5 BETA ESTIMATION")
    print("=" * 80)

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

    # 3) Align by date (and remove any overlapping factor/RF columns from returns_df)
    print("\n3. Aligning stock returns and factor data...")

    # Some pipelines (like your CAPM step) may have already merged RF (or factors)
    # into the returns file. We want to treat the Fama-French file as canonical,
    # so drop any overlapping factor/RF columns from returns_df before joining.
    overlap_cols = list(set(returns_df.columns) & set(FACTOR_COLS + ["RF"]))
    if overlap_cols:
        print(f"   Dropping overlapping columns from returns_df to avoid duplication: {overlap_cols}")
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
    print(f"  Shrunk HML premium (λ={lambda_hml:.2f}): "
          f"{hml_shrunk:+.4f} ({hml_shrunk*100:+.2f}%)")

    mu_f["HML"] = hml_shrunk

    return mu_f


def build_ff5_factor_model(
    betas_df: pd.DataFrame,
    ff_path: str = "data/processed/Fama_French.csv",
) -> Tuple[pd.Series, np.ndarray]:
    """
    Build factor-based asset return moments (μ_R, Σ_R).

    Parameters
    ----------
    betas_df : pd.DataFrame
        Output of estimate_ff5_betas(), indexed by Ticker.
    ff_path : str
        Path to Fama-French CSV for estimating Σ_f.

    Returns
    -------
    factor_cov : np.ndarray
        5x5 factor covariance matrix Σ_f (in monthly terms).
    """

    ff = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
    factor_returns = ff[FACTOR_COLS].dropna()
    # Sample covariance of factors
    Sigma_f = factor_returns.cov().values

    return Sigma_f


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
    5) Compute asset covariance Σ_R = B Σ_f B' + Ω.
    6) Compute tangency weights: w* ∝ Σ_R^{-1} μ_R, normalized to sum to 1.

    Parameters
    ----------
    returns_path : str
        Path to stock returns CSV.
    ff_path : str
        Path to Fama-French factors CSV.
    factor_ml_dataset_path : str
        Path to enhanced factor ML dataset.
    best_model : FactorPredictor
        Trained ML predictor (from evaluate_all_models).
    lambda_hml : float
        Shrinkage weight on ML HML forecast.
    min_obs : int
        Min observations for FF5 beta estimation.

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
    betas_valid = betas_df.dropna(subset=["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA"])
    if betas_valid.empty:
        raise ValueError("No valid FF5 betas available for portfolio construction.")

    tickers = betas_valid.index.tolist()

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

    # 5) Asset covariance Σ_R = B Σ_f B' + Ω
    resid_var = betas_valid["ResidVar"].fillna(betas_valid["ResidVar"].median())
    Omega = np.diag(resid_var.values)  # N x N

    Sigma_R = B @ Sigma_f @ B.T + Omega

    # 6) Tangency portfolio: w* ∝ Σ_R^{-1} μ_R
    print("\n" + "=" * 80)
    print("FF5 TANGENCY PORTFOLIO (UNCONSTRAINED, EXCESS RETURN SPACE)")
    print("=" * 80)

    # Use pseudo-inverse for numerical safety
    inv_Sigma_R = np.linalg.pinv(Sigma_R)
    raw_w = inv_Sigma_R @ mu_R

    if np.allclose(raw_w.sum(), 0.0):
        raise ValueError("Sum of raw weights is zero; cannot normalize. Check inputs.")

    w_star = raw_w / raw_w.sum()

    weights_df = betas_valid.copy()
    weights_df["Weight"] = w_star

    # Sort by descending weight
    weights_df = weights_df.sort_values("Weight", ascending=False)

    print("\nTop 10 positions in FF5 tangency portfolio:")
    print("-" * 80)
    print(
        weights_df[["Weight", "Beta_MKT", "Beta_HML"]]
        .head(10)
        .to_string(float_format=lambda x: f"{x:+.4f}")
    )

    total_weight = weights_df["Weight"].sum()
    avg_beta_mkt = np.sum(weights_df["Weight"] * weights_df["Beta_MKT"])
    avg_beta_hml = np.sum(weights_df["Weight"] * weights_df["Beta_HML"])

    print("\nPortfolio summary:")
    print("-" * 80)
    print(f"  Sum of weights:         {total_weight:.4f}")
    print(f"  Portfolio MKT beta:     {avg_beta_mkt:.3f}")
    print(f"  Portfolio HML beta:     {avg_beta_hml:.3f}")
    print(f"  Number of stocks:       {len(weights_df)}")

    os.makedirs("data/processed", exist_ok=True)
    weights_df.to_csv("data/processed/ff5_optimal_portfolio_weights.csv")
    print("\nWeights saved to: data/processed/ff5_optimal_portfolio_weights.csv")

    return weights_df
