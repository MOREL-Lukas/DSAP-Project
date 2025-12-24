import os
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.ml_models import FactorPredictor

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
BETA_COLS = ["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA"]

MAX_WEIGHT = 0.10
MAX_SHORT = 0.0


# =============================================================================
# Math helpers
# =============================================================================


def _ensure_ff5_betas(
    betas_df: pd.DataFrame | None,
    returns_path: str,
    ff_path: str,
    min_obs: int,
    output_path: str | None,
) -> pd.DataFrame:
    """
    If betas_df is missing the FF5 beta columns, compute FF5 betas locally.
    This prevents passing CAPM betas (column 'Beta') from breaking the optimizer.
    """
    if betas_df is None:
        return estimate_ff5_betas(
            returns_path=returns_path,
            ff_path=ff_path,
            min_obs=min_obs,
            output_path=output_path,
        )

    required = set(BETA_COLS)
    if not required.issubset(set(betas_df.columns)):
        # CAPM betas (e.g., 'Beta') or other schema passed in -> recompute FF5 betas
        print(
            "\nNote: Provided betas_df is not FF5 betas. Recomputing FF5 betas for portfolio optimizer..."
        )
        return estimate_ff5_betas(
            returns_path=returns_path,
            ff_path=ff_path,
            min_obs=min_obs,
            output_path=output_path,
        )

    return betas_df


def regularize_covariance(Sigma: np.ndarray, ridge_ratio: float = 1e-3) -> np.ndarray:
    """Symmetrize + ridge-regularize covariance (ridge is ridge_ratio * avg variance)."""
    Sigma = np.asarray(Sigma, float)
    if Sigma.size == 0:
        return Sigma
    S = 0.5 * (Sigma + Sigma.T)
    n = S.shape[0]
    avg_var = float(np.trace(S) / n) if n > 0 else 0.0
    ridge = (
        ridge_ratio
        if (not np.isfinite(avg_var) or avg_var <= 0)
        else ridge_ratio * avg_var
    )
    return S + ridge * np.eye(n)


def apply_rmw_tilt(
    weights: np.ndarray,
    betas_rmw: np.ndarray,
    tilt_strength: float = 0.3,
) -> np.ndarray:
    """
    Tilt portfolio weights toward high-RMW stocks.

    Parameters
    ----------
    weights : array
        Original portfolio weights (pre-tilt)
    betas_rmw : array
        RMW betas for each stock
    tilt_strength : float
        Strength of tilt (0 = no tilt, 1 = extreme tilt)

    Returns
    -------
    tilted_weights : array
        Adjusted weights favoring high RMW exposure
    """
    w = np.asarray(weights, float).copy()
    rmw = np.asarray(betas_rmw, float).copy()

    if len(w) != len(rmw):
        raise ValueError(
            f"Dimension mismatch: weights ({len(w)}) vs RMW betas ({len(rmw)})"
        )

    # Normalize RMW betas to [0, 1] range for tilt calculation
    rmw_min, rmw_max = np.nanmin(rmw), np.nanmax(rmw)
    if np.isclose(rmw_max, rmw_min):
        # No variation in RMW betas
        return w

    rmw_normalized = (rmw - rmw_min) / (rmw_max - rmw_min)

    # Tilt factor: 1 + tilt_strength * (normalized_RMW - 0.5) * 2
    # This maps:
    #   - Low RMW (0.0) → tilt_factor = 1 - tilt_strength
    #   - Median RMW (0.5) → tilt_factor = 1.0
    #   - High RMW (1.0) → tilt_factor = 1 + tilt_strength
    tilt_factor = 1.0 + tilt_strength * (2.0 * rmw_normalized - 1.0)

    # Apply tilt
    w_tilted = w * tilt_factor

    # Handle edge cases
    if not np.isfinite(w_tilted).all():
        return w

    # Renormalize to sum to original total weight
    original_sum = float(np.sum(w))
    current_sum = float(np.sum(w_tilted))

    if not np.isclose(current_sum, 0.0):
        w_tilted = w_tilted * (original_sum / current_sum)
    else:
        return w

    return w_tilted


def apply_weight_constraints(
    weights: np.ndarray,
    max_weight: float = MAX_WEIGHT,
    max_short: float = MAX_SHORT,
) -> np.ndarray:
    """
    Enforce realistic position and leverage limits (per-name caps and
    short exposure bounds) while keeping the portfolio fully invested.
    This prevents unstable mean-variance solutions from dominating a
    few extreme positions.
    """
    w = np.asarray(weights, float).copy()
    n = len(w)
    if n == 0:
        return w

    w = np.clip(w, -max_weight, max_weight)
    if np.allclose(w, 0.0):
        return np.ones_like(w) / n

    neg = w < 0
    pos = w > 0
    short_exposure = float(-w[neg].sum())

    if short_exposure > max_short and short_exposure > 0:
        # scale shorts down to max_short
        w[neg] *= max_short / short_exposure

        short_sum = float(w[neg].sum())  # negative
        long_sum = float(w[pos].sum())  # positive

        if long_sum <= 0:
            w[:] = 0.0
            w[~neg] = 1.0 / max(int((~neg).sum()), 1)
            return w

        # renormalize longs to keep sum(w)=1
        w[pos] *= (1.0 - short_sum) / long_sum
    else:
        total = float(w.sum())
        w = (w / total) if (not np.isclose(total, 0.0)) else (np.ones_like(w) / n)

    if not np.isfinite(w).all():
        w = np.ones_like(w) / n

    w = np.clip(w, -max_weight, max_weight)
    total = float(w.sum())
    return (w / total) if (not np.isclose(total, 0.0)) else (np.ones_like(w) / n)


# =============================================================================
# Data + beta estimation
# =============================================================================


def _load_returns_and_factors(
    returns_path: str,
    ff_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    returns_df = pd.read_csv(returns_path, parse_dates=["Date"], index_col="Date")
    ff = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
    missing = set(FACTOR_COLS + ["RF"]) - set(ff.columns)
    if missing:
        raise ValueError(f"Missing columns in {ff_path}: {missing}")

    # Prevent name collisions (if any factor columns exist in returns_df)
    overlap = list(set(returns_df.columns) & set(FACTOR_COLS + ["RF"]))
    if overlap:
        returns_df = returns_df.drop(columns=overlap)

    return returns_df.sort_index(), ff.sort_index()


def estimate_ff5_betas(
    returns_path: str = "data/processed/sp500_monthly_returns.csv",
    ff_path: str = "data/processed/Fama_French.csv",
    min_obs: int = 36,
    output_path: Optional[str] = "data/processed/sp500_ff5_betas.csv",
) -> pd.DataFrame:
    """
    Estimate FF5 betas per stock by OLS on excess returns (monthly).

    Returns index=TICKER with columns:
      Alpha, Beta_MKT, Beta_SMB, Beta_HML, Beta_RMW, Beta_CMA,
      R_squared, Adj_R_squared, ResidVar, N_obs
    """
    print("\n1. Loading stock returns...")
    returns_df, ff = _load_returns_and_factors(returns_path, ff_path)
    print(f"   Loaded {len(returns_df)} months for {len(returns_df.columns)} stocks")
    print("\n2. Loading Fama-French 5-factor data...")
    print("\n3. Aligning stock returns and factor data...")

    data = returns_df.join(ff[FACTOR_COLS + ["RF"]], how="inner")
    print(f"   Overlapping period has {len(data)} months")
    print("\n4. Computing excess returns...")

    rf = data["RF"]
    stock_cols = [c for c in data.columns if c not in FACTOR_COLS + ["RF"]]
    stock_excess = data[stock_cols].sub(rf, axis=0)

    X = data[FACTOR_COLS].copy()
    X["const"] = 1.0
    X_cols = ["const"] + FACTOR_COLS

    out_rows: List[Dict] = []

    for ticker in stock_cols:
        y = stock_excess[ticker]
        df_reg = pd.concat([y, X], axis=1).dropna()
        n = len(df_reg)
        if n < min_obs:
            out_rows.append(
                dict(
                    Ticker=ticker,
                    Alpha=np.nan,
                    Beta_MKT=np.nan,
                    Beta_SMB=np.nan,
                    Beta_HML=np.nan,
                    Beta_RMW=np.nan,
                    Beta_CMA=np.nan,
                    R_squared=np.nan,
                    Adj_R_squared=np.nan,
                    ResidVar=np.nan,
                    N_obs=n,
                )
            )
            continue

        yv = df_reg[ticker].to_numpy(dtype=float)
        Xv = df_reg[X_cols].to_numpy(dtype=float)

        beta_hat, *_ = np.linalg.lstsq(Xv, yv, rcond=None)  # [alpha, betas...]

        y_hat = Xv @ beta_hat
        resid = yv - y_hat
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))

        r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        k = len(FACTOR_COLS)  # regressors excluding constant
        adj = (
            1.0 - (1.0 - r2_val) * (n - 1) / (n - k - 1)
            if (np.isfinite(r2_val) and (n - k - 1) > 0)
            else np.nan
        )
        resid_var = ss_res / (n - k - 1) if (n - k - 1) > 0 else np.nan

        out_rows.append(
            dict(
                Ticker=ticker,
                Alpha=float(beta_hat[0]),
                Beta_MKT=float(beta_hat[1]),
                Beta_SMB=float(beta_hat[2]),
                Beta_HML=float(beta_hat[3]),
                Beta_RMW=float(beta_hat[4]),
                Beta_CMA=float(beta_hat[5]),
                R_squared=float(r2_val) if np.isfinite(r2_val) else np.nan,
                Adj_R_squared=float(adj) if np.isfinite(adj) else np.nan,
                ResidVar=float(resid_var) if np.isfinite(resid_var) else np.nan,
                N_obs=int(n),
            )
        )

    betas_df = pd.DataFrame(out_rows).set_index("Ticker").sort_index()

    valid = betas_df.dropna(subset=["Beta_MKT"])
    print("\n5. Summary statistics for FF5 betas:\n" + "-" * 80)
    print(f"   Number of stocks:      {len(betas_df)}")
    print(f"   Valid beta estimates:  {len(valid)}")
    print(
        f"   Avg CAPM-like beta:    {valid['Beta_MKT'].mean():.3f}"
        if len(valid)
        else "   Avg CAPM-like beta:    n/a"
    )
    print(
        f"   Avg HML beta:          {valid['Beta_HML'].mean():.3f}"
        if len(valid)
        else "   Avg HML beta:          n/a"
    )
    print(
        f"   Avg RMW beta:          {valid['Beta_RMW'].mean():.3f}"
        if len(valid)
        else "   Avg RMW beta:          n/a"
    )
    print(
        f"   Avg R2:                {valid['R_squared'].mean():.3f}"
        if len(valid)
        else "   Avg R2:                n/a"
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        betas_df.to_csv(output_path)
        print(f"\nBetas saved to: {output_path}")

    return betas_df


# =============================================================================
# Factor premia overlay
# =============================================================================


def compute_factor_premia_with_ml_overlay(
    ff_path: str,
    factor_ml_dataset_path: str,
    default_model: FactorPredictor,
    overlay_factors: Optional[List[str]] = None,
    lambda_overlay: float = 0.2,
    per_factor_lambda: Optional[Dict[str, float]] = None,
    per_factor_model: Optional[Dict[str, FactorPredictor]] = None,
) -> pd.Series:
    """
    Historical monthly means of factors, optionally blended with ML forecasts for chosen factors.
    """
    overlay_factors = overlay_factors or []
    per_factor_lambda = per_factor_lambda or {}
    per_factor_model = per_factor_model or {}

    ff = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
    hist = ff[FACTOR_COLS].dropna().mean()
    mu_f = hist.copy()

    print("\n" + "=" * 80)
    print("EXPECTED FACTOR PREMIA (HISTORICAL + ML OVERLAY)")
    print("=" * 80)
    print("\nHistorical factor means (monthly):")
    for fac in FACTOR_COLS:
        v = float(hist[fac])
        print(f"  {fac:7s}: μ = {v:+.4f} ({v*100:+.2f}%)")

    if not overlay_factors:
        return mu_f

    # Default feature row
    X_all, _, _ = default_model.prepare_data(factor_ml_dataset_path)
    default_feat_row = X_all.iloc[-1]

    for fac in overlay_factors:
        if fac not in FACTOR_COLS:
            continue

        model = per_factor_model.get(fac, default_model)

        # ensure feature alignment with chosen model
        if model is default_model:
            feat_row = default_feat_row
        else:
            X_tmp, _, _ = model.prepare_data(factor_ml_dataset_path)
            feat_row = X_tmp.iloc[-1]

        pred = model.predict_next_month(feat_row)
        if fac not in pred:
            continue

        lam = float(per_factor_lambda.get(fac, lambda_overlay))
        lam = max(0.0, min(1.0, lam))

        ml_val = float(pred[fac])
        shrunk = (1.0 - lam) * float(hist[fac]) + lam * ml_val
        mu_f[fac] = shrunk

        print(f"\nOverlay for {fac}:")
        print(f"  ML forecast:           {ml_val:+.4f} ({ml_val*100:+.2f}%)")
        print(
            f"  Historical mean:       {float(hist[fac]):+.4f} ({float(hist[fac])*100:+.2f}%)"
        )
        print(f"  Shrunk (lambda={lam:.2f}): {shrunk:+.4f} ({shrunk*100:+.2f}%)")

    return mu_f


def build_ff5_factor_model(ff_path: str) -> np.ndarray:
    """Factor covariance Σ_f from historical FF factor returns (monthly)."""
    ff = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
    return ff[FACTOR_COLS].dropna().cov().to_numpy()


# =============================================================================
# Portfolio constructors
# =============================================================================


def build_ff5_optimal_portfolio(
    returns_path: str,
    ff_path: str,
    factor_ml_dataset_path: str,
    best_model: FactorPredictor,
    betas_df: Optional[pd.DataFrame] = None,
    min_obs: int = 36,
    overlay_factors: Optional[List[str]] = None,
    lambda_overlay: float = 0.2,
    per_factor_lambda: Optional[Dict[str, float]] = None,
    per_factor_model: Optional[Dict[str, FactorPredictor]] = None,
    save_path: str = "data/processed/ff5_optimal_portfolio_weights.csv",
    rmw_tilt_strength: float = 0.3,
) -> pd.DataFrame:
    """
    FF5 tangency portfolio with optional RMW tilt.

    Parameters
    ----------
    rmw_tilt_strength : float
        Strength of RMW tilt (0 = no tilt, 1 = strong tilt)
    """
    betas_df = _ensure_ff5_betas(
        betas_df=betas_df,
        returns_path=returns_path,
        ff_path=ff_path,
        min_obs=min_obs,
        output_path="data/processed/sp500_ff5_betas.csv",
    )

    betas_valid = betas_df.dropna(subset=BETA_COLS).copy()
    if betas_valid.empty:
        raise ValueError("No valid FF5 betas available for portfolio construction.")

    Sigma_f = build_ff5_factor_model(ff_path=ff_path)
    mu_f = compute_factor_premia_with_ml_overlay(
        ff_path=ff_path,
        factor_ml_dataset_path=factor_ml_dataset_path,
        default_model=best_model,
        overlay_factors=overlay_factors,
        lambda_overlay=lambda_overlay,
        per_factor_lambda=per_factor_lambda,
        per_factor_model=per_factor_model,
    )

    B = betas_valid[BETA_COLS].to_numpy(dtype=float)  # (N,5)
    mu_R = (B @ mu_f.to_numpy().reshape(-1, 1)).reshape(-1)  # (N,)

    resid = (
        betas_valid["ResidVar"].to_numpy(dtype=float)
        if "ResidVar" in betas_valid.columns
        else np.zeros(len(betas_valid))
    )
    fill = np.nanmedian(resid) if np.isfinite(np.nanmedian(resid)) else 0.0
    resid = np.where(np.isnan(resid), fill, resid)

    Sigma_R = regularize_covariance(
        B @ Sigma_f @ B.T + np.diag(resid), ridge_ratio=1e-3
    )

    raw_w = np.linalg.pinv(Sigma_R) @ mu_R
    if np.isclose(raw_w.sum(), 0.0):
        raw_w = np.ones_like(raw_w)

    # Normalize to sum=1 before tilt
    w_normalized = raw_w / raw_w.sum()

    # Apply RMW tilt BEFORE constraints
    if rmw_tilt_strength > 0:
        betas_rmw = betas_valid["Beta_RMW"].to_numpy(dtype=float)
        w_tilted = apply_rmw_tilt(
            w_normalized, betas_rmw, tilt_strength=rmw_tilt_strength
        )
        print(f"\n🎯 Applied RMW tilt (strength={rmw_tilt_strength:.2f})")
    else:
        w_tilted = w_normalized

    # Apply constraints after tilt
    w_star = apply_weight_constraints(
        w_tilted, max_weight=MAX_WEIGHT, max_short=MAX_SHORT
    )

    weights_df = betas_valid.copy()
    weights_df["Weight"] = w_star
    weights_df = weights_df.sort_values("Weight", ascending=False)
    weights_df = weights_df[np.abs(weights_df["Weight"]) >= 0.001]

    total_weight = float(weights_df["Weight"].sum())
    avg_beta_mkt = float((weights_df["Weight"] * weights_df["Beta_MKT"]).sum())
    avg_beta_hml = float((weights_df["Weight"] * weights_df["Beta_HML"]).sum())
    avg_beta_rmw = float((weights_df["Weight"] * weights_df["Beta_RMW"]).sum())

    mu_p = float(
        (
            weights_df["Weight"].to_numpy()
            * mu_R[weights_df.index.get_indexer(weights_df.index)]
        ).sum()
    )  # portfolio expected excess return
    sigma_p = float(
        np.sqrt(max(0.0, w_star @ (Sigma_R @ w_star)))
    )  # portfolio volatility
    sharpe_p = mu_p / sigma_p if sigma_p > 0 else np.nan  # Sharpe ratio
    mu_mkt = float(mu_f["Mkt-RF"])  # market risk premium
    alpha_capm = (
        mu_p - avg_beta_mkt * mu_mkt if np.isfinite(avg_beta_mkt) else np.nan
    )  # CAPM alpha

    print("\nPortfolio summary:\n" + "-" * 80)
    print(f"  Sum of weights:         {total_weight:.4f}")
    print(f"  Portfolio MKT beta:     {avg_beta_mkt:.3f}")
    print(f"  Portfolio HML beta:     {avg_beta_hml:.3f}")
    print(f"  Portfolio RMW beta:     {avg_beta_rmw:.3f}  ⬅ RMW tilt target")
    print(f"  Expected excess return: {mu_p:+.4f} ({mu_p*100:+.2f}%)")
    print(f"  Volatility (stdev):     {sigma_p:.4f} ({sigma_p*100:.2f}%)")
    print(f"  Sharpe ratio:           {sharpe_p:.3f}")
    print(f"  CAPM alpha (monthly):   {alpha_capm:+.4f} ({alpha_capm*100:+.2f}%)")
    print(f"  Number of stocks:       {len(weights_df)}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        weights_df.to_csv(save_path)
        print(f"\nWeights saved to: {save_path}")

    return weights_df


def _filter_by_sharpe(
    betas_df: pd.DataFrame,
    returns_path: str,
    ff_path: str,
    max_stocks: int,
    min_r_squared: float,
) -> pd.DataFrame:
    """Filter by min R2 then rank by historical annualized Sharpe of excess returns."""
    betas = betas_df[betas_df["R_squared"] >= float(min_r_squared)].copy()
    if len(betas) <= max_stocks:
        return betas

    returns_df, ff = _load_returns_and_factors(returns_path, ff_path)

    sharpe_vals = []
    for ticker in betas.index:
        if ticker not in returns_df.columns:
            sharpe_vals.append(np.nan)
            continue
        d = pd.concat([returns_df[ticker], ff["RF"]], axis=1).dropna()
        if len(d) < 24:
            sharpe_vals.append(np.nan)
            continue
        ex = d[ticker] - d["RF"]
        m, s = float(ex.mean()), float(ex.std(ddof=1))
        sharpe_vals.append((m / s) * np.sqrt(12) if s > 0 else np.nan)

    betas["Sharpe"] = sharpe_vals
    betas = betas.dropna(subset=["Sharpe"])
    if len(betas) <= max_stocks:
        return betas.drop(columns=["Sharpe"])
    return betas.nlargest(max_stocks, "Sharpe").drop(columns=["Sharpe"])


def _filter_by_r2(
    betas_df: pd.DataFrame, max_stocks: int, min_r_squared: float
) -> pd.DataFrame:
    """Filter by min R2 then keep highest R2 up to max_stocks."""
    betas = betas_df[betas_df["R_squared"] >= float(min_r_squared)].copy()
    return (
        betas if len(betas) <= max_stocks else betas.nlargest(max_stocks, "R_squared")
    )


def _filter_by_rmw(
    betas_df: pd.DataFrame, max_stocks: int, min_r_squared: float = 0.15
) -> pd.DataFrame:
    """Filter by min R2 then rank by RMW beta (highest RMW exposure)."""
    betas = betas_df[betas_df["R_squared"] >= float(min_r_squared)].copy()
    if len(betas) <= max_stocks:
        return betas

    # Sort by RMW beta (descending) - highest profitability exposure
    return betas.nlargest(max_stocks, "Beta_RMW")


def build_concentrated_portfolio(
    returns_path: str,
    ff_path: str,
    factor_ml_dataset_path: str,
    best_model: FactorPredictor,
    betas_df: Optional[pd.DataFrame] = None,
    min_obs: int = 36,
    max_stocks: int = 50,
    filter_method: str = "sharpe",
    min_r_squared: float = 0.15,
    overlay_factors: Optional[List[str]] = None,
    lambda_overlay: float = 0.2,
    per_factor_lambda: Optional[Dict[str, float]] = None,
    per_factor_model: Optional[Dict[str, FactorPredictor]] = None,
    rmw_tilt_strength: float = 0.0,  # NEW PARAMETER
) -> pd.DataFrame:
    """
    Concentrated FF5 portfolio: filter universe then build tangency weights.

    filter_method options: 'sharpe', 'r2', 'rmw'
    """
    betas_df = _ensure_ff5_betas(
        betas_df=betas_df,
        returns_path=returns_path,
        ff_path=ff_path,
        min_obs=min_obs,
        output_path=None,
    )

    mu_f = compute_factor_premia_with_ml_overlay(
        ff_path=ff_path,
        factor_ml_dataset_path=factor_ml_dataset_path,
        default_model=best_model,
        overlay_factors=overlay_factors,
        lambda_overlay=lambda_overlay,
        per_factor_lambda=per_factor_lambda,
        per_factor_model=per_factor_model,
    )

    betas_valid = betas_df.dropna(subset=BETA_COLS).copy()
    if betas_valid.empty:
        raise ValueError("No valid FF5 betas available for concentrated portfolio.")

    if filter_method.lower() == "sharpe":
        betas_filtered = _filter_by_sharpe(
            betas_valid, returns_path, ff_path, max_stocks, min_r_squared
        )
    elif filter_method.lower() == "rmw":
        betas_filtered = _filter_by_rmw(betas_valid, max_stocks, min_r_squared)
    else:
        betas_filtered = _filter_by_r2(betas_valid, max_stocks, min_r_squared)

    Sigma_f = build_ff5_factor_model(ff_path=ff_path)
    B = betas_filtered[BETA_COLS].to_numpy(dtype=float)
    mu_R = (B @ mu_f.to_numpy().reshape(-1, 1)).reshape(-1)

    resid = (
        betas_filtered["ResidVar"].to_numpy(dtype=float)
        if "ResidVar" in betas_filtered.columns
        else np.zeros(len(betas_filtered))
    )
    fill = np.nanmedian(resid) if np.isfinite(np.nanmedian(resid)) else 0.0
    resid = np.where(np.isnan(resid), fill, resid)

    Sigma_R = regularize_covariance(
        B @ Sigma_f @ B.T + np.diag(resid), ridge_ratio=1e-3
    )

    raw_w = np.linalg.pinv(Sigma_R) @ mu_R
    if np.isclose(raw_w.sum(), 0.0):
        raw_w = np.ones_like(raw_w)

    # Normalize then optionally apply RMW tilt
    w_normalized = raw_w / raw_w.sum()

    if rmw_tilt_strength > 0:
        betas_rmw = betas_filtered["Beta_RMW"].to_numpy(dtype=float)
        w_tilted = apply_rmw_tilt(
            w_normalized, betas_rmw, tilt_strength=rmw_tilt_strength
        )
    else:
        w_tilted = w_normalized

    w_star = apply_weight_constraints(
        w_tilted, max_weight=MAX_WEIGHT, max_short=MAX_SHORT
    )

    weights_df = betas_filtered.copy()
    weights_df["Weight"] = w_star
    weights_df = weights_df.sort_values("Weight", ascending=False)
    return weights_df[np.abs(weights_df["Weight"]) >= 0.001]


# =============================================================================
# Backtest (unchanged - kept for completeness)
# =============================================================================


def backtest_ff5_tangency(
    returns_path: str,
    ff_path: str,
    min_train_months: int = 120,
    min_obs_per_stock: int = 36,
    rmw_tilt_strength=1,
) -> pd.DataFrame:
    """Expanding-window FF5 tangency backtest (monthly, using in-sample moments)."""
    print(f"  min_train_months   = {min_train_months}")
    print(f"  min_obs_per_stock  = {min_obs_per_stock}")

    returns_df, ff = _load_returns_and_factors(returns_path, ff_path)
    data = returns_df.join(ff[FACTOR_COLS + ["RF"]], how="inner").sort_index()

    stock_cols = [c for c in data.columns if c not in FACTOR_COLS + ["RF"]]
    valid_universe = [
        c
        for c in stock_cols
        if data[c].count() >= (min_train_months + min_obs_per_stock)
    ]
    data = data[valid_universe + FACTOR_COLS + ["RF"]]
    stock_cols = valid_universe
    print(f"\nBacktest universe size: {len(stock_cols)} stocks")
    print(f"Total months available: {len(data)}")

    dates = data.index
    n_months = len(dates)
    if n_months <= min_train_months + 1:
        raise ValueError("Not enough data for the requested training window.")

    out_dates, port_rets, mkt_rets, ew_rets, tilt_rets = [], [], [], [], []

    for t_idx in tqdm(
        range(min_train_months, n_months), desc="Rolling FF5 Backtest", leave=True
    ):
        train = data.iloc[:t_idx]
        test = data.iloc[t_idx]
        test_date = dates[t_idx]

        rf_train = train["RF"]
        stock_excess_train = train[stock_cols].sub(rf_train, axis=0)

        X = train[FACTOR_COLS].copy()
        X["const"] = 1.0
        X_cols = ["const"] + FACTOR_COLS

        betas_list = []
        resid_vars = []

        for col in stock_cols:
            y = stock_excess_train[col]
            df_reg = pd.concat([y, X], axis=1).dropna()

            if len(df_reg) < min_obs_per_stock:
                betas_list.append([np.nan] * (1 + len(FACTOR_COLS)))
                resid_vars.append(np.nan)
                continue

            yv = df_reg[col].to_numpy(dtype=float)
            Xv = df_reg[X_cols].to_numpy(dtype=float)

            try:
                beta_hat, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
            except Exception:
                betas_list.append([np.nan] * (1 + len(FACTOR_COLS)))
                resid_vars.append(np.nan)
                continue

            ss_res = float(np.sum((yv - Xv @ beta_hat) ** 2))
            k = len(FACTOR_COLS)
            n = len(yv)
            resid_var = ss_res / (n - k - 1) if (n - k - 1) > 0 else np.nan

            betas_list.append(list(beta_hat))
            resid_vars.append(resid_var)

        B_full = np.asarray(betas_list, dtype=float)[:, 1:]
        resid_var_arr = np.asarray(resid_vars, dtype=float)

        valid_beta = np.isfinite(B_full).all(axis=1)
        valid_resid = np.isfinite(resid_var_arr)
        valid = valid_beta & valid_resid

        if valid.sum() < 10:
            continue

        B = B_full[valid]
        resid_var_arr = resid_var_arr[valid]

        factor_train = train[FACTOR_COLS].dropna()
        if len(factor_train) < 24:
            continue

        mu_f = factor_train.mean().to_numpy(dtype=float)
        Sigma_f = factor_train.cov().to_numpy(dtype=float)

        mu_R = B @ mu_f
        fill = float(np.median(resid_var_arr))
        if not np.isfinite(fill):
            continue

        resid_filled = resid_var_arr.copy()
        resid_filled[~np.isfinite(resid_filled)] = fill

        Sigma_R = regularize_covariance(
            B @ Sigma_f @ B.T + np.diag(resid_filled), ridge_ratio=1e-3
        )
        if not np.isfinite(Sigma_R).all():
            continue

        try:
            raw_w = np.linalg.pinv(Sigma_R) @ mu_R
        except np.linalg.LinAlgError:
            continue

        if not np.isfinite(raw_w).all() or np.isclose(raw_w.sum(), 0.0):
            continue

        w_star = apply_weight_constraints(raw_w / raw_w.sum())
        # Tilt toward high RMW exposure (same constraint regime as baseline)
        # FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] -> RMW index = 3
        betas_rmw_valid = B[:, 3]
        w_tilt = apply_rmw_tilt(
            w_star, betas_rmw_valid, tilt_strength=rmw_tilt_strength
        )
        w_tilt = apply_weight_constraints(w_tilt)  # keep feasible set identical

        rf_test = float(test["RF"])
        test_returns = test[stock_cols].to_numpy(dtype=float)
        ex_test_full = test_returns - rf_test

        ex_test = ex_test_full[valid]
        if len(ex_test) != len(w_star):
            continue

        port_excess = float(np.nansum(w_star * ex_test))
        tilt_excess = float(np.nansum(w_tilt * ex_test))
        finite = np.isfinite(ex_test)
        finite = np.isfinite(ex_test)
        # Use an equal-weight portfolio on the same investable subset to
        # provide a simple, implementation-agnostic benchmark each month.
        ew_excess = float(np.mean(ex_test[finite])) if finite.any() else np.nan
        mkt_excess = float(test["Mkt-RF"]) if "Mkt-RF" in test.index else np.nan

        out_dates.append(test_date)
        port_rets.append(port_excess)
        ew_rets.append(ew_excess)
        mkt_rets.append(mkt_excess)
        tilt_rets.append(tilt_excess)

    results = pd.DataFrame(
        {
            "Port_Excess_Return": port_rets,
            "Tilt_Excess_Return": tilt_rets,
            "EW_Excess_Return": ew_rets,
            "Mkt_RF": mkt_rets,
        },
        index=pd.to_datetime(out_dates),
    ).sort_index()

    if len(results) > 1:

        def _print_summary(label, excess, mkt):
            excess = excess.dropna()
            mkt = mkt.loc[excess.index]

            if len(excess) <= 1:
                print(f"\n{label} summary: insufficient data")
                return

            mu = excess.mean()
            sigma = excess.std(ddof=1)
            sharpe = mu / sigma if sigma > 0 else np.nan

            cov = np.cov(excess, mkt)[0, 1]
            beta = cov / np.var(mkt, ddof=1)
            alpha = mu - beta * mkt.mean()

            print(f"\n{label} summary:")
            print("-" * 80)
            print(f"  Periods:                {len(excess)}")
            print(f"  Mean excess return:     {mu:+.4f} ({mu*100:+.2f}%)")
            print(f"  Volatility:             {sigma:.4f} ({sigma*100:.2f}%)")
            print(f"  Sharpe ratio:           {sharpe:.3f}")
            print(f"  CAPM beta (vs Mkt-RF):  {beta:.3f}")
            print(f"  CAPM alpha (monthly):   {alpha:+.4f} ({alpha*100:+.2f}%)")

        # -------------------------------
        # Print all backtest summaries
        # -------------------------------
        _print_summary(
            "Equal-weight benchmark", results["EW_Excess_Return"], results["Mkt_RF"]
        )
        _print_summary(
            "Baseline (Tangency)", results["Port_Excess_Return"], results["Mkt_RF"]
        )
        _print_summary("RMW Tilt", results["Tilt_Excess_Return"], results["Mkt_RF"])

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/ff5_backtest_unconstrained.csv"
    results.to_csv(out_path)
    print(f"\nBacktest results saved to: {out_path}")

    return results


def build_equal_weight_portfolio(
    betas_df: pd.DataFrame,
    min_r_squared: float = 0.0,
    long_only: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build an equal-weight portfolio over the same investable universe as the beta table.

    Returns a weights_df compatible with calc_portfolio_stats(), i.e. it includes:
      - Weight
      - Beta_MKT ... Beta_CMA
    """
    if betas_df is None or len(betas_df) == 0:
        raise ValueError("betas_df is empty; cannot build equal-weight portfolio.")

    df = betas_df.copy()

    # Ensure required columns exist and are non-missing
    required = [c for c in BETA_COLS if c in df.columns]
    if len(required) != len(BETA_COLS):
        missing = set(BETA_COLS) - set(df.columns)
        raise ValueError(f"betas_df missing required beta columns: {missing}")

    df = df.dropna(subset=BETA_COLS)

    # Optional quality filter
    if "R_squared" in df.columns and min_r_squared > 0:
        df = df[df["R_squared"] >= float(min_r_squared)].copy()

    n = len(df)
    if n == 0:
        raise ValueError(
            "No assets left after filtering; cannot build equal-weight portfolio."
        )

    w = np.full(n, 1.0 / n, dtype=float)

    # Equal-weight is typically long-only; keep parameter to be explicit.
    if not long_only:
        raise ValueError(
            "Equal-weight benchmark should be long-only in this project context."
        )

    df["Weight"] = w
    df = df.sort_values("Weight", ascending=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=True)

    return df


# =============================================================================
# Reporting helpers
# =============================================================================


def compare_portfolio_strategies(
    full_portfolio: pd.DataFrame,
    concentrated_sharpe: pd.DataFrame,
    concentrated_r2: pd.DataFrame,
    ff_path: str,
) -> pd.DataFrame:
    """Compare basic portfolio diagnostics (n_stocks, avg R2, beta)."""
    rows = [
        calc_portfolio_stats(full_portfolio, ff_path, "Full"),
        calc_portfolio_stats(concentrated_sharpe, ff_path, "Sharpe-50"),
        calc_portfolio_stats(concentrated_r2, ff_path, "R2-50"),
    ]
    return pd.DataFrame(rows)


def calc_portfolio_stats(weights_df: pd.DataFrame, ff_path: str, name: str) -> Dict:
    """
    Portfolio diagnostics + factor-model implied moments.

    Expects weights_df with at least:
      - Weight
      - Beta_MKT ... Beta_CMA
    Optional:
      - ResidVar
      - R_squared
    """
    if weights_df is None or len(weights_df) == 0:
        return {"Strategy": name, "N_Stocks": 0}

    w = weights_df["Weight"].astype(float).to_numpy()
    w_sum = float(np.nansum(w))
    gross = float(np.nansum(np.abs(w)))
    long_exposure = float(np.nansum(w[w > 0]))
    short_exposure = float(np.nansum(np.abs(w[w < 0])))

    beta_mkt = (
        float(np.nansum(w * weights_df["Beta_MKT"]))
        if "Beta_MKT" in weights_df.columns
        else np.nan
    )
    beta_hml = (
        float(np.nansum(w * weights_df["Beta_HML"]))
        if "Beta_HML" in weights_df.columns
        else np.nan
    )
    beta_rmw = (
        float(np.nansum(w * weights_df["Beta_RMW"]))
        if "Beta_RMW" in weights_df.columns
        else np.nan
    )

    avg_r2 = np.nan
    if "R_squared" in weights_df.columns:
        avg_r2 = float(np.nansum(w * weights_df["R_squared"]))

    ff = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
    mu_f = ff[FACTOR_COLS].dropna().mean()
    Sigma_f = ff[FACTOR_COLS].dropna().cov().to_numpy()

    mu_excess = np.nan
    vol = np.nan
    sharpe = np.nan
    alpha_capm = np.nan

    if all(c in weights_df.columns for c in BETA_COLS):
        B = weights_df[BETA_COLS].to_numpy(dtype=float)
        mu_R = (B @ mu_f.to_numpy().reshape(-1, 1)).reshape(-1)

        if "ResidVar" in weights_df.columns:
            resid = weights_df["ResidVar"].astype(float).to_numpy()
            fill = np.nanmedian(resid) if np.isfinite(np.nanmedian(resid)) else 0.0
            resid = np.where(np.isnan(resid), fill, resid)
        else:
            resid = np.zeros(len(weights_df), dtype=float)

        Sigma_R = regularize_covariance(
            B @ Sigma_f @ B.T + np.diag(resid), ridge_ratio=1e-3
        )

        mu_excess = float(w @ mu_R)
        var = float(w @ (Sigma_R @ w))
        vol = float(np.sqrt(max(var, 0.0)))
        sharpe = (mu_excess / vol) if vol > 0 else np.nan

        mu_mkt = float(mu_f["Mkt-RF"])
        alpha_capm = (
            mu_excess - beta_mkt * mu_mkt
            if (np.isfinite(beta_mkt) and np.isfinite(mu_mkt))
            else np.nan
        )

    return {
        "Strategy": name,
        "N_Stocks": int(len(weights_df)),
        "Sum_Weights": w_sum,
        "Gross_Exposure": gross,
        "Long_Exposure": long_exposure,
        "Short_Exposure": short_exposure,
        "Portfolio_Beta_MKT": beta_mkt,
        "Portfolio_Beta_HML": beta_hml,
        "Portfolio_Beta_RMW": beta_rmw,
        "Avg_R2": avg_r2,
        "Expected_Excess_Return": mu_excess,
        "Volatility": vol,
        "Sharpe": sharpe,
        "CAPM_Alpha": alpha_capm,
    }
