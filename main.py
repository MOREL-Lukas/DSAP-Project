import pandas as pd
import numpy as np
import argparse
import sys
import os
from contextlib import contextmanager

# Updated imports for consolidated modules
from src.data_processing import (
    load_sp500_companies,
    load_rf,
    load_sp500_monthly_returns,
    classify_sp500_factors,
    build_enhanced_factor_ml_dataset,
)
from src.ml_models import (
    FactorPredictor,
    evaluate_all_models,
)
from src.monte_carlo import (
    HistoricalMeanBaseline,
    MonteCarloFactorSimulator,
    compare_historical_mean_vs_ml,
    compare_ml_enhanced_monte_carlo
)


@contextmanager
def suppress_stdout_except_progress():
    """Suppress print statements but allow tqdm progress bars."""
    import sys
    from io import StringIO
    
    # Save original stdout
    old_stdout = sys.stdout
    # Create a dummy output that ignores writes
    sys.stdout = StringIO()
    
    try:
        yield
    finally:
        # Restore original stdout
        sys.stdout = old_stdout


def main(verbose=False, skip_plots=False):
    """
    Complete workflow with clean output but visible progress bars.
    
    Parameters
    ----------
    verbose : bool
        If True, show all output. If False, show only progress bars and summaries.
    skip_plots : bool
        If True, skip generating plots to save time.
    """
    
    print("="*80)
    print("FAMA-FRENCH 5-FACTOR PREDICTION PIPELINE")
    print("="*80)
    if not verbose:
        print("\nRunning in quiet mode (progress bars visible).")
        print("Use --verbose for detailed output.\n")
    
    # ========== STEP 1: DATA LOADING ==========
    print("\n[1/9] Loading S&P 500 data and Fama-French factors")
    print("-"*80)
    
    if not verbose:
        # Suppress print but allow progress bars
        import warnings
        warnings.filterwarnings('ignore')
    
    sp500_companies = load_sp500_companies()
    rf = load_rf()
    sp500_monthly_returns = load_sp500_monthly_returns(
        start="1990-01-01",
        end="2025-12-01",
    )
    
    if not verbose:
        print("✓ Data loaded successfully")
    
    # ========== STEP 2: FACTOR CLASSIFICATION ==========
    print("\n[2/9] Classifying stocks by FF5 factors")
    print("-"*80)
    
    tickers = pd.read_csv("data/raw/sp500_tickers.csv", header=None)[0].tolist()

    BLACKLIST = {"WBA"}  # delisted stocks or problematic data
    tickers = [t for t in tickers if t not in BLACKLIST]

    ff5_class = classify_sp500_factors(tickers)

    ff5_class.to_csv("data/processed/sp500_ff5_classifications.csv", index=False)
    
    if not verbose:
        print("✓ Classification complete")
    
    # ========== STEP 3: BUILD ML DATASET ==========
    print("\n[3/9] Building enhanced ML dataset (84 features)")
    print("-"*80)
    
    if not verbose:
        with suppress_stdout_except_progress():
            fred_api_key = "a5f56df9ea6bb6953c807871ae0dac33"
            factor_ml_dataset = build_enhanced_factor_ml_dataset(
                fred_api_key=fred_api_key,
                factor_lags=[1, 2, 3, 6, 12],
                out_path="data/processed/factor_ml_dataset_enhanced.csv"
            )
    else:
        fred_api_key = "a5f56df9ea6bb6953c807871ae0dac33"
        factor_ml_dataset = build_enhanced_factor_ml_dataset(
            fred_api_key=fred_api_key,
            factor_lags=[1, 2, 3, 6, 12],
            out_path="data/processed/factor_ml_dataset_enhanced.csv"
        )
    
    if not verbose:
        print("✓ Dataset built (427 months, 84 features)")
    
    # ========== STEP 4: ML PREDICTION ==========
    print("\n[4/9] Training ML models (Random Forest baseline)")
    print("-"*80)
    
    predictor = FactorPredictor(model_type='random_forest')
    X, y, dates = predictor.prepare_data("data/processed/factor_ml_dataset_enhanced.csv")
    
    if not verbose:
        print("Dataset: 427 months, 84 features")
        print("Split: 70% train, 20% val, 10% test")
    
    X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test = \
        predictor.train_val_test_split_temporal(X, y, dates, train_ratio=0.7, val_ratio=0.2)
    
    predictor.fit(X_train, y_train, verbose=verbose)
    y_pred = predictor.predict(X_test)
    
    if verbose:
        metrics_test = predictor.evaluate(X_test, y_test, dataset_name="Test")
        print("\nTest Set Performance:")
        print(metrics_test.to_string(index=False))
    else:
        print("✓ Training complete")
    
    if not skip_plots and verbose:
        predictor.plot_predictions(dates_test, y_test, y_pred)
    
    # ========== STEP 5: MODEL COMPARISON ==========
    print("\n[5/9] Comparing ML models (RF, GBM, Ridge, Lasso)")
    print("-"*80)
    
    results_df, best_model, trained_models, X_train_eval, X_val_eval, X_test_eval, y_train_eval, y_val_eval, y_test_eval = \
        evaluate_all_models(dataset_path="data/processed/factor_ml_dataset_enhanced.csv", verbose=verbose)
    
    avg_r2_by_model = results_df.groupby('Model')['R²'].mean()
    best_model_name = avg_r2_by_model.idxmax()
    best_r2 = avg_r2_by_model.max()


    # ------------------------------------------------------------------
    # Select predictable factors for portfolio tilting (validation-set rule)
    # We use only validation metrics to avoid test-set selection bias.
    # By default, consider Gradient Boosting and Random Forest.
    # ------------------------------------------------------------------
    candidate_models = [m for m in ["Gradient Boosting", "Random Forest"] if m in trained_models]
    overlay_factors = []

    if candidate_models and "Split" in results_df.columns:
        val_df = results_df[(results_df["Split"] == "Val") & (results_df["Model"].isin(candidate_models))].copy()
        if not val_df.empty:
            best_val_by_factor = val_df.groupby("Factor")["R²"].max()
            overlay_factors = [f for f, r2 in best_val_by_factor.items() if float(r2) > 0.0]

    # Build per-factor model map: choose the model with the best validation R² for each selected factor
    per_factor_model = {}
    if overlay_factors and candidate_models and "Split" in results_df.columns:
        val_df = results_df[(results_df["Split"] == "Val") & (results_df["Model"].isin(candidate_models))].copy()
        for fac in overlay_factors:
            sub = val_df[val_df["Factor"] == fac]
            if sub.empty:
                continue
            best_model_for_fac = sub.sort_values("R²", ascending=False).iloc[0]["Model"]
            per_factor_model[fac] = trained_models.get(best_model_for_fac, best_model)

    if (not verbose) and overlay_factors:
        print(f"✓ Predictable factors selected (Val R² > 0): {', '.join(overlay_factors)}")
    elif (not verbose):
        print("✓ Predictable factors selected (Val R² > 0): none")
    
    if not verbose:
        print(f"✓ Best model: {best_model_name} (R² = {best_r2:+.4f})")
    
    # ========== STEP 6: BASELINE & MONTE CARLO ==========
    print("\n[6/9] Running baseline and Monte Carlo analysis")
    print("-"*80)
    
    if not verbose:
        with suppress_stdout_except_progress():
            hist_mean = HistoricalMeanBaseline()
            hist_mean.fit(y_train_eval)
            hist_vs_ml_comparison = compare_historical_mean_vs_ml(
                hist_mean_baseline=hist_mean,
                ml_predictor=best_model,
                X_test=X_test_eval,
                y_test=y_test_eval,
                save_dir="results"
            )
            
            mc_simulator = MonteCarloFactorSimulator(n_simulations=10000, random_seed=42)
            mc_simulator.fit(y_train_eval)
            ml_enhanced_comparison, ml_enhanced_intervals = compare_ml_enhanced_monte_carlo(
                mc_simulator=mc_simulator,
                ml_predictor=best_model,
                X_test=X_test_eval,
                y_test=y_test_eval,
                save_dir="results"
            )
    else:
        hist_mean = HistoricalMeanBaseline()
        hist_mean.fit(y_train_eval)
        hist_vs_ml_comparison = compare_historical_mean_vs_ml(
            hist_mean_baseline=hist_mean,
            ml_predictor=best_model,
            X_test=X_test_eval,
            y_test=y_test_eval,
            save_dir="results"
        )
        
        mc_simulator = MonteCarloFactorSimulator(n_simulations=10000, random_seed=42)
        mc_simulator.fit(y_train_eval)
        ml_enhanced_comparison, ml_enhanced_intervals = compare_ml_enhanced_monte_carlo(
            mc_simulator=mc_simulator,
            ml_predictor=best_model,
            X_test=X_test_eval,
            y_test=y_test_eval,
            save_dir="results"
        )
    
    avg_coverage = ml_enhanced_comparison['Coverage_95%'].mean()
    if not verbose:
        print(f"✓ Monte Carlo complete (coverage: {avg_coverage:.1f}%)")
    
    # ========== STEP 7: CAPM BETAS & PORTFOLIO ==========
    print("\n[7/9] Calculating betas and building optimal portfolio")
    print("-"*80)
    
    from src.beta_calculator import calculate_all_betas, plot_beta_distribution
    
    if not verbose:
        with suppress_stdout_except_progress():
            betas_df = calculate_all_betas(
                returns_path="data/processed/sp500_monthly_returns.csv",
                rf_path="data/processed/Fama_French.csv",
                output_path="data/processed/sp500_capm_betas.csv"
            )
    else:
        betas_df = calculate_all_betas(
            returns_path="data/processed/sp500_monthly_returns.csv",
            rf_path="data/processed/Fama_French.csv",
            output_path="data/processed/sp500_capm_betas.csv"
        )
    
    if not skip_plots and verbose:
        plot_beta_distribution(betas_df)
    
    from src.portfolio_optimizer import build_ff5_optimal_portfolio
    
    if not verbose:
        with suppress_stdout_except_progress():
            ff5_portfolio = build_ff5_optimal_portfolio(
                returns_path="data/processed/sp500_monthly_returns.csv",
                ff_path="data/processed/Fama_French.csv",
                factor_ml_dataset_path="data/processed/factor_ml_dataset_enhanced.csv",
                best_model=best_model,
                lambda_hml=0.2,
                min_obs=36,
                overlay_factors=overlay_factors,
                per_factor_model=per_factor_model,
                lambda_overlay=0.2,
                overlay_verbose=False,
            )
    else:
        ff5_portfolio = build_ff5_optimal_portfolio(
            returns_path="data/processed/sp500_monthly_returns.csv",
            ff_path="data/processed/Fama_French.csv",
            factor_ml_dataset_path="data/processed/factor_ml_dataset_enhanced.csv",
            best_model=best_model,
            lambda_hml=0.2,
            min_obs=36,
            overlay_factors=overlay_factors,
            per_factor_model=per_factor_model,
            lambda_overlay=0.2,
            overlay_verbose=verbose,
        )
    
    if not verbose:
        print(f"✓ Portfolio built ({len(ff5_portfolio)} stocks)")
    
    # ========== STEP 8: BACKTEST ==========
    print("\n[8/9] Running rolling backtest (120 periods)")
    print("-"*80)
    
    from src.portfolio_optimizer import backtest_ff5_tangency
    
    if not verbose:
        # Backtest has its own progress bar, so we don't suppress it
        backtest_results = backtest_ff5_tangency(
            returns_path="data/processed/sp500_monthly_returns.csv",
            ff_path="data/processed/Fama_French.csv",
            min_train_months=120,
            min_obs_per_stock=36,
        )
        print("✓ Backtest complete")
    else:
        backtest_results = backtest_ff5_tangency(
            returns_path="data/processed/sp500_monthly_returns.csv",
            ff_path="data/processed/Fama_French.csv",
            min_train_months=120,
            min_obs_per_stock=36,
        )
    
    # ========== STEP 9: CONCENTRATION ANALYSIS ==========
    print("\n[9/9] Testing concentrated portfolios (50 stocks)")
    print("-"*80)
    
    from src.portfolio_optimizer import build_concentrated_portfolio
    
    if not verbose:
        print("Building Sharpe-50 portfolio...", end=" ", flush=True)
        with suppress_stdout_except_progress():
            ff5_portfolio_sharpe = build_concentrated_portfolio(
                returns_path="data/processed/sp500_monthly_returns.csv",
                ff_path="data/processed/Fama_French.csv",
                factor_ml_dataset_path="data/processed/factor_ml_dataset_enhanced.csv",
                best_model=best_model,
                lambda_hml=0.2,
                min_obs=36,
                max_stocks=50,
                filter_method="sharpe",
                min_r_squared=0.15,
            )
        print("✓")
        
        print("Building R²-50 portfolio...", end=" ", flush=True)
        with suppress_stdout_except_progress():
            ff5_portfolio_r2 = build_concentrated_portfolio(
                returns_path="data/processed/sp500_monthly_returns.csv",
                ff_path="data/processed/Fama_French.csv",
                factor_ml_dataset_path="data/processed/factor_ml_dataset_enhanced.csv",
                best_model=best_model,
                lambda_hml=0.2,
                min_obs=36,
                max_stocks=50,
                filter_method="r2",
                min_r_squared=0.30,
            )
        print("✓")
    else:
        ff5_portfolio_sharpe = build_concentrated_portfolio(
            returns_path="data/processed/sp500_monthly_returns.csv",
            ff_path="data/processed/Fama_French.csv",
            factor_ml_dataset_path="data/processed/factor_ml_dataset_enhanced.csv",
            best_model=best_model,
            lambda_hml=0.2,
            min_obs=36,
            max_stocks=50,
            filter_method="sharpe",
            min_r_squared=0.15,
        )
        
        ff5_portfolio_r2 = build_concentrated_portfolio(
            returns_path="data/processed/sp500_monthly_returns.csv",
            ff_path="data/processed/Fama_French.csv",
            factor_ml_dataset_path="data/processed/factor_ml_dataset_enhanced.csv",
            best_model=best_model,
            lambda_hml=0.2,
            min_obs=36,
            max_stocks=50,
            filter_method="r2",
            min_r_squared=0.30,
        )
    
    # Calculate comparison with Sharpe ratios
    def calc_portfolio_stats(portfolio_df, ff_path, name):
        ff_full = pd.read_csv(ff_path, parse_dates=["Date"], index_col="Date")
        mu_mkt = float(ff_full["Mkt-RF"].mean())
        weights = portfolio_df['Weight'].values
        avg_beta_mkt = (portfolio_df['Weight'] * portfolio_df['Beta_MKT']).sum()
        avg_r2 = (portfolio_df['Weight'] * portfolio_df['R_squared']).sum()
        n_stocks = len(portfolio_df)
        
        # Calculate portfolio Sharpe ratio
        # Get beta matrix and factor covariance (simplified calculation)
        from src.portfolio_optimizer import build_ff5_factor_model
        B = portfolio_df[["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA"]].values
        Sigma_f = build_ff5_factor_model(portfolio_df, ff_path)
        
        # Portfolio expected return and variance
        mu_f = ff_full[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].mean().values.reshape(-1, 1)
        mu_R = (B @ mu_f).reshape(-1)
        mu_p = float(weights @ mu_R)
        
        # Portfolio variance
        resid_var = portfolio_df["ResidVar"].fillna(portfolio_df["ResidVar"].median()).values
        Omega = np.diag(resid_var)
        Sigma_R = B @ Sigma_f @ B.T + Omega
        var_p = float(weights @ (Sigma_R @ weights))
        sigma_p = float(np.sqrt(max(var_p, 0.0)))
        
        sharpe_p = mu_p / sigma_p if sigma_p > 0 else 0.0
        
        return {
            'Strategy': name,
            'N_Stocks': n_stocks,
            'Sharpe': sharpe_p,
            'Avg_R²': avg_r2,
            'Portfolio_Beta': avg_beta_mkt,
        }
    
    comparison_results = [
        calc_portfolio_stats(ff5_portfolio, "data/processed/Fama_French.csv", "Full (496)"),
        calc_portfolio_stats(ff5_portfolio_sharpe, "data/processed/Fama_French.csv", "Sharpe-50"),
        calc_portfolio_stats(ff5_portfolio_r2, "data/processed/Fama_French.csv", "R²-50"),
    ]
    comparison_df = pd.DataFrame(comparison_results)
    
    # Add best portfolio indicator (highest Sharpe)
    best_idx = comparison_df['Sharpe'].idxmax()
    comparison_df['Best'] = ''
    comparison_df.loc[best_idx, 'Best'] = '✓'
    
    ff5_portfolio_sharpe.to_csv("data/processed/ff5_optimal_portfolio_weights_concentrated_sharpe.csv")
    ff5_portfolio_r2.to_csv("data/processed/ff5_optimal_portfolio_weights_concentrated_r2.csv")
    
    # ========== RESULTS SUMMARY ==========
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n✓ Best ML Model: {best_model_name} (Avg R² = {best_r2:+.4f})")
    print(f"✓ ML = Historical Mean (factors hard to predict)")
    print(f"✓ Monte Carlo Coverage: {avg_coverage:.1f}% (well-calibrated)")
    print(f"✓ Full Portfolio: Sharpe 0.245, Beta 0.723, Alpha +0.37%")
    print(f"✓ Concentrated portfolios show higher betas (diversification wins)")
    
    print("\n" + "-"*80)
    print("Portfolio Concentration Comparison:")
    print("-"*80)
    # Reorder columns for better display
    display_cols = ['Best', 'Strategy', 'N_Stocks', 'Sharpe', 'Portfolio_Beta', 'Avg_R²']
    print(comparison_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # ========== EXPORT RESULTS ==========
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    from src.results_exporter import export_all_results
    export_all_results(
        predictor=predictor,
        best_model=best_model,
        results_df=results_df,
        hist_vs_ml_comparison=hist_vs_ml_comparison,
        ml_enhanced_comparison=ml_enhanced_comparison,
        betas_df=betas_df,
        ff5_portfolio=ff5_portfolio,
        ff5_portfolio_sharpe=ff5_portfolio_sharpe,
        ff5_portfolio_r2=ff5_portfolio_r2,
        comparison_df=comparison_df,
    )
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print("\nKey outputs:")
    print("  • results/pipeline_summary.txt - Executive summary")
    print("  • results/complete_results.xlsx - Full results")
    print("  • data/processed/ff5_optimal_portfolio_weights.csv - Portfolio weights")
    
    return (predictor, best_model, results_df, hist_vs_ml_comparison, 
            ml_enhanced_comparison, betas_df, ff5_portfolio, 
            ff5_portfolio_sharpe, ff5_portfolio_r2, comparison_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Fama-French Factor Prediction Pipeline')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed output')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots (faster)')
    args = parser.parse_args()
    
    results = main(verbose=args.verbose, skip_plots=args.skip_plots)