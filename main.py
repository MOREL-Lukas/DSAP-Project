import pandas as pd
import subprocess
import sys

from src.data_processing import (
    load_sp500_companies,
    load_rf,
    load_sp500_monthly_returns,
    classify_sp500_factors,
    build_enhanced_factor_ml_dataset,
)
from src.ml_models import FactorPredictor, evaluate_all_models
from src.monte_carlo import (
    HistoricalMeanBaseline,
    MonteCarloFactorSimulator,
    compare_historical_mean_vs_ml,
    compare_ml_enhanced_monte_carlo,
)
from src.beta_calculator import calculate_all_betas, plot_beta_distribution
from src.portfolio_optimizer import (
    estimate_ff5_betas,
    build_ff5_optimal_portfolio,
    backtest_ff5_tangency,
    build_concentrated_portfolio,
    calc_portfolio_stats,
    build_equal_weight_portfolio,
)
from src.results_exporter import export_all_results

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def step(i: int, title: str) -> None:
    print(f"\n[{i}/9] {title}\n" + "-" * 80)


def select_overlay_models(results_df: pd.DataFrame, trained_models: dict):
    """
    Select per-factor overlay models using validation split only.
    Returns (overlay_factors, per_factor_model, per_factor_lambda).
    """
    val_df = results_df[results_df["Split"] == "Val"].copy()

    r2_col = "R²" if "R²" in val_df.columns else "R2"
    if r2_col not in val_df.columns:
        raise KeyError(f"results_df missing R2 column. Found columns: {list(val_df.columns)}")

    overlay_factors = []
    per_factor_model = {}
    per_factor_lambda = {}

    for fac in FACTOR_COLS:
        sub = val_df[val_df["Factor"] == fac].sort_values(r2_col, ascending=False)
        if sub.empty:
            continue

        top_name = str(sub.iloc[0]["Model"])
        top_r2 = float(sub.iloc[0][r2_col])

        model_obj = trained_models.get(top_name)
        if model_obj is None or top_r2 <= 0:
            continue

        overlay_factors.append(fac)
        per_factor_model[fac] = model_obj

        # Factor-specific shrinkage (your original logic)
        per_factor_lambda[fac] = 0.4 if fac == "RMW" else 0.3 if fac == "Mkt-RF" else 0.2

    return overlay_factors, per_factor_model, per_factor_lambda


def run_tests():
    """Prompt user to run tests and execute if desired."""
    print("\n" + "=" * 80)
    print("TEST SUITE")
    print("=" * 80)
    print("\nWould you like to run the test suite? [Y/n]: ", end='', flush=True)
    
    try:
        response = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nSkipping tests.")
        return
    
    if response in ['n', 'no']:
        print("Tests skipped.")
        return
    
    # Default is yes (empty input or 'y' or 'yes')
    if response in ['', 'y', 'yes']:
        print("\n" + "-" * 80)
        print("Running: pytest tests/ -v")
        print("-" * 80 + "\n")
        
        try:
            result = subprocess.run(
                ['pytest', 'tests/', '-v'],
                cwd='.',
                check=False
            )
            
            print("\n" + "=" * 80)
            if result.returncode == 0:
                print("✅ All tests passed!")
            else:
                print(f"❌ Some tests failed (exit code: {result.returncode})")
            print("=" * 80)
            
        except FileNotFoundError:
            print("Error: pytest not found. Install with: pip install pytest pytest-cov")
        except Exception as e:
            print(f"Error running tests: {e}")
    else:
        print("Invalid input. Tests skipped.")


def main() -> None:
    print("=" * 80)
    print("FAMA-FRENCH 5-FACTOR PREDICTION PIPELINE")
    print("Testing: Baseline vs RMW Tilt vs Concentration")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1) Data acquisition
    # -------------------------------------------------------------------------
    step(1, "Loading S&P 500 data and Fama-French factors")
    load_sp500_companies()
    load_rf()
    load_sp500_monthly_returns("1990-01-01", "2025-12-01")

    # -------------------------------------------------------------------------
    # 2) Cross-sectional classifications
    # -------------------------------------------------------------------------
    step(2, "Classifying stocks by FF5 factors")
    tickers = [t for t in pd.read_csv("data/raw/sp500_tickers.csv", header=None)[0] if t != "WBA"]
    classify_sp500_factors(tickers).to_csv("data/processed/sp500_ff5_classifications.csv", index=False)

    # -------------------------------------------------------------------------
    # 3) Feature engineering
    # -------------------------------------------------------------------------
    step(3, "Building enhanced ML dataset")
    build_enhanced_factor_ml_dataset(
        fred_api_key="a5f56df9ea6bb6953c807871ae0dac33",
        factor_lags=[1, 2, 3, 6, 12],
        out_path="data/processed/factor_ml_dataset_enhanced.csv",
    )

    # -------------------------------------------------------------------------
    # 4) Baseline model (parity with reporting)
    # -------------------------------------------------------------------------
    step(4, "Training baseline ML model")
    predictor = FactorPredictor("random_forest")
    X, y, dates = predictor.prepare_data("data/processed/factor_ml_dataset_enhanced.csv")
    X_train, X_val, X_test, y_train, y_val, y_test, d_train, d_val, d_test = predictor.train_val_test_split_temporal(
        X, y, dates
    )
    predictor.fit(X_train, y_train)
    print("✓ Training complete")

    # -------------------------------------------------------------------------
    # 5) Model comparison + overlay selection
    # -------------------------------------------------------------------------
    step(5, "Model comparison and overlay selection")
    results_df, best_model, trained_models, *eval_sets = evaluate_all_models(
        "data/processed/factor_ml_dataset_enhanced.csv"
    )
    overlay_factors, per_factor_model, per_factor_lambda = select_overlay_models(results_df, trained_models)

    # -------------------------------------------------------------------------
    # 6) Baseline vs ML + ML-enhanced Monte Carlo
    # -------------------------------------------------------------------------
    step(6, "Baseline and Monte Carlo")
    hist_mean = HistoricalMeanBaseline().fit(eval_sets[3])
    hist_vs_ml_comparison = compare_historical_mean_vs_ml(
        hist_mean, best_model, eval_sets[2], eval_sets[5], "results"
    )

    mc = MonteCarloFactorSimulator(10_000, 42).fit(eval_sets[3])
    ml_enhanced_comparison, _ = compare_ml_enhanced_monte_carlo(
        mc, best_model, eval_sets[2], eval_sets[5], "results"
    )

    # -------------------------------------------------------------------------
    # 7) Betas + THREE portfolio strategies
    # -------------------------------------------------------------------------
    step(7, "Betas and portfolio construction (3 strategies)")

    # 7a) CAPM betas (reporting + plotting)
    capm_betas_df = calculate_all_betas(
        "data/processed/sp500_monthly_returns.csv",
        "data/processed/Fama_French.csv",
        "data/processed/sp500_capm_betas.csv",
    )
    plot_beta_distribution(capm_betas_df)

    # 7b) FF5 betas (portfolio optimizer input; reuse everywhere to avoid recomputation)
    ff5_betas_df = estimate_ff5_betas(
        returns_path="data/processed/sp500_monthly_returns.csv",
        ff_path="data/processed/Fama_French.csv",
        output_path="data/processed/sp500_ff5_betas.csv",
    )
    print("\n" + "=" * 80)
    print("BUILDING BENCHMARK: EQUAL-WEIGHT (same universe as beta estimates)")
    print("=" * 80)

    ff5_equal_weight = build_equal_weight_portfolio(
        betas_df=ff5_betas_df,
        min_r_squared=0.0,
        save_path="data/processed/ff5_equal_weight_weights.csv",
    )


    # 7c) Build THREE portfolios to test RMW tilt hypothesis
    
    print("\n" + "=" * 80)
    print("BUILDING PORTFOLIO 1/3: BASELINE (No RMW Tilt)")
    print("=" * 80)
    ff5_baseline = build_ff5_optimal_portfolio(
        "data/processed/sp500_monthly_returns.csv",
        "data/processed/Fama_French.csv",
        "data/processed/factor_ml_dataset_enhanced.csv",
        best_model,
        betas_df=ff5_betas_df,
        overlay_factors=overlay_factors,
        per_factor_model=per_factor_model,
        per_factor_lambda=per_factor_lambda,
        lambda_overlay=0.30,
        rmw_tilt_strength=0.0,  # NO TILT - Pure Markowitz
        save_path="data/processed/ff5_baseline_weights.csv",
    )

    print("\n" + "=" * 80)
    print("BUILDING PORTFOLIO 2/3: HIGH RMW TILT (strength=1)")
    print("=" * 80)
    ff5_tilt = build_ff5_optimal_portfolio(
        "data/processed/sp500_monthly_returns.csv",
        "data/processed/Fama_French.csv",
        "data/processed/factor_ml_dataset_enhanced.csv",
        best_model,
        betas_df=ff5_betas_df,
        overlay_factors=overlay_factors,
        per_factor_model=per_factor_model,
        per_factor_lambda=per_factor_lambda,
        lambda_overlay=0.30,
        rmw_tilt_strength=1,  # High RMW tilt
        save_path="data/processed/ff5_rmw_tilt_weights.csv",
    )

    print("\n" + "=" * 80)
    print("BUILDING PORTFOLIO 3/3: CONCENTRATED RMW (50 stocks, strength=1)")
    print("=" * 80)
    ff5_concentrated = build_concentrated_portfolio(
        "data/processed/sp500_monthly_returns.csv",
        "data/processed/Fama_French.csv",
        "data/processed/factor_ml_dataset_enhanced.csv",
        best_model,
        betas_df=ff5_betas_df,
        max_stocks=50,
        filter_method="rmw",
        min_r_squared=0.15,
        overlay_factors=overlay_factors,
        per_factor_model=per_factor_model,
        per_factor_lambda=per_factor_lambda,
        lambda_overlay=0.30,
        rmw_tilt_strength=1 # High RMW tilt on top of RMW filter
    )

    # -------------------------------------------------------------------------
    # 8) Rolling backtest (uses baseline - no tilt)
    # -------------------------------------------------------------------------
    step(8, "Rolling backtest (baseline strategy)")
    backtest_results = backtest_ff5_tangency(
        "data/processed/sp500_monthly_returns.csv",
        "data/processed/Fama_French.csv",
    )

    # -------------------------------------------------------------------------
    # 9) Compare all three strategies
    # -------------------------------------------------------------------------
    step(9, "Strategy comparison: Baseline vs Tilt vs Concentration")

    comparison_df = pd.DataFrame(
        [
            calc_portfolio_stats(ff5_equal_weight, "data/processed/Fama_French.csv", "Equal-Weight"),
            calc_portfolio_stats(ff5_baseline, "data/processed/Fama_French.csv", "Baseline (no tilt)"),
            calc_portfolio_stats(ff5_tilt, "data/processed/Fama_French.csv", "RMW Tilt (0.3)"),
            calc_portfolio_stats(ff5_concentrated, "data/processed/Fama_French.csv", "RMW-50 (concentrated)"),
        ]
    )


    # Print comparison table
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("\n" + "=" * 80)
    
    # Highlight key findings
    baseline_sharpe = comparison_df.loc[comparison_df["Strategy"] == "Baseline (no tilt)", "Sharpe"].values[0]
    tilt_sharpe = comparison_df.loc[comparison_df["Strategy"] == "RMW Tilt (0.3)", "Sharpe"].values[0]
    conc_sharpe = comparison_df.loc[comparison_df["Strategy"] == "RMW-50 (concentrated)", "Sharpe"].values[0]
    
    baseline_rmw = comparison_df.loc[comparison_df["Strategy"] == "Baseline (no tilt)", "Portfolio_Beta_RMW"].values[0]
    tilt_rmw = comparison_df.loc[comparison_df["Strategy"] == "RMW Tilt (0.3)", "Portfolio_Beta_RMW"].values[0]
    conc_rmw = comparison_df.loc[comparison_df["Strategy"] == "RMW-50 (concentrated)", "Portfolio_Beta_RMW"].values[0]
    
    print("\nKEY FINDINGS:")
    print("-" * 80)
    print(f"1. Does RMW tilt help diversified portfolio?")
    print(f"   Baseline: Sharpe {baseline_sharpe:.3f}, RMW beta {baseline_rmw:.3f}")
    print(f"   With Tilt: Sharpe {tilt_sharpe:.3f}, RMW beta {tilt_rmw:.3f}")
    if tilt_sharpe > baseline_sharpe:
        print(f"   → ✓ YES: Tilt improves Sharpe by {(tilt_sharpe/baseline_sharpe - 1)*100:+.1f}%")
    else:
        print(f"   → ✗ NO: Tilt reduces Sharpe by {(1 - tilt_sharpe/baseline_sharpe)*100:.1f}%")
    
    print(f"\n2. Does concentration help?")
    print(f"   Baseline: Sharpe {baseline_sharpe:.3f}")
    print(f"   Concentrated: Sharpe {conc_sharpe:.3f}, RMW beta {conc_rmw:.3f}")
    if conc_sharpe > baseline_sharpe:
        print(f"   → ✓ YES: Concentration improves Sharpe by {(conc_sharpe/baseline_sharpe - 1)*100:+.1f}%")
    else:
        print(f"   → ✗ NO: Concentration reduces Sharpe by {(1 - conc_sharpe/baseline_sharpe)*100:.1f}%")
    
    print(f"\n3. Overall ranking by Sharpe:")
    sorted_strats = comparison_df.sort_values("Sharpe", ascending=False)
    for i, (idx, row) in enumerate(sorted_strats.iterrows(), 1):
        print(f"   #{i}: {row['Strategy']:30s} Sharpe={row['Sharpe']:.3f}")
    
    print("=" * 80)

    export_all_results(
        predictor,
        best_model,
        results_df,
        hist_vs_ml_comparison,
        ml_enhanced_comparison,
        capm_betas_df,
        ff5_equal_weight,
        ff5_baseline,
        ff5_tilt,
        ff5_concentrated,
        comparison_df,
        backtest_results=backtest_results,
    )

    print("\n✅ PIPELINE COMPLETE")
    
    # -------------------------------------------------------------------------
    # PROMPT FOR TESTS
    # -------------------------------------------------------------------------
    run_tests()


if __name__ == "__main__":
    main()