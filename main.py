import pandas as pd
from src.data_loader import (
    load_sp500_companies,
    load_rf,
    load_sp500_monthly_returns,
    classify_sp500_factors,
)
from src.models import build_enhanced_factor_ml_dataset
from src.factor_predictor import FactorPredictor
from src.evaluation import evaluate_all_models
from src.monte_carlo import (
    HistoricalMeanBaseline,
    MonteCarloFactorSimulator,
    compare_historical_mean_vs_ml,
    compare_ml_enhanced_monte_carlo
)


def main():
    """
    Complete workflow: data loading, factor dataset creation, ML prediction,
    and comprehensive comparisons.
    """
    # ========== DATA LOADING ==========
    print("="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    sp500_companies = load_sp500_companies()
    rf = load_rf()
    sp500_monthly_returns = load_sp500_monthly_returns(
        start="1990-01-01",  # 35 years of data
        end="2025-12-01",    # Fixed end date
    )
    
    print("\n" + "="*80)
    print("S&P 500 COMPANIES (First 5 Rows)")
    print("="*80)
    print(sp500_companies.head())

    print("\n" + "="*80)
    print("FAMA–FRENCH RISK-FREE & MARKET DATA (First 5 Rows)")
    print("="*80)
    print(rf.head())

    print("\n" + "="*80)
    print("S&P 500 MONTHLY RETURNS (First 5 Rows)")
    print("="*80)
    print(sp500_monthly_returns.head())
    print("="*80 + "\n")

    # ========== FACTOR CLASSIFICATION ==========
    print("\n" + "="*80)
    print("STEP 2: CLASSIFYING STOCKS BY FACTORS")
    print("="*80)
    
    tickers = pd.read_csv("data/raw/sp500_tickers.csv", header=None)[0].tolist()
    ff5_class = classify_sp500_factors(tickers)
    ff5_class.to_csv("data/processed/sp500_ff5_classifications.csv", index=False)

    print("\n" + "="*80)
    print("FF5 APPROXIMATE CLASSIFICATIONS (First 5 Rows)")
    print("="*80)
    print(ff5_class.head())
    print("="*80 + "\n")

    # ========== BUILD ML DATASET ==========
    print("\n" + "="*80)
    print("STEP 3: BUILDING ENHANCED FACTOR ML DATASET")
    print("="*80)
    
    # Build enhanced dataset with lagged factors, macro variables, and market conditions
    fred_api_key = "a5f56df9ea6bb6953c807871ae0dac33"
    
    factor_ml_dataset = build_enhanced_factor_ml_dataset(
        fred_api_key=fred_api_key,
        factor_lags=[1, 2, 3, 6, 12],
        out_path="data/processed/factor_ml_dataset_enhanced.csv"
    )

    print("\n" + "="*80)
    print("FACTOR ML DATASET (First 5 Rows)")
    print("="*80)
    print(factor_ml_dataset.head())
    print("="*80 + "\n")

    # ========== MACHINE LEARNING PREDICTION ==========
    print("\n" + "="*80)
    print("STEP 4: FAMA-FRENCH FACTOR PREDICTION WITH MACHINE LEARNING")
    print("="*80)
    
    # Initialize predictor (Random Forest for initial demo)
    predictor = FactorPredictor(model_type='random_forest')
    
    # Load data (use enhanced dataset)
    X, y, dates = predictor.prepare_data(
        dataset_path="data/processed/factor_ml_dataset_enhanced.csv"
    )
    
    # Split data temporally: 70% train, 20% val, 10% test
    X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test = \
        predictor.train_val_test_split_temporal(X, y, dates, train_ratio=0.7, val_ratio=0.2)
    
    # Train models
    predictor.fit(X_train, y_train)
    
    # Make predictions
    y_pred = predictor.predict(X_test)
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE ON VALIDATION SET")
    print("=" * 80)
    metrics_val = predictor.evaluate(X_val, y_val, dataset_name="Validation")
    print(metrics_val.to_string(index=False))
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE ON TEST SET")
    print("=" * 80)
    metrics_test = predictor.evaluate(X_test, y_test, dataset_name="Test")
    print(metrics_test.to_string(index=False))
    
    # Compare validation vs test performance
    print("\n" + "=" * 80)
    print("VALIDATION vs TEST COMPARISON")
    print("=" * 80)
    comparison = pd.concat([metrics_val, metrics_test])
    comparison_pivot = comparison.pivot(index='Factor', columns='Dataset', values='R²')
    print("\nR² Scores:")
    print(comparison_pivot.to_string())
    
    # Feature importance
    importance = predictor.feature_importance(top_n=10)
    
    # Plot results
    predictor.plot_predictions(dates_test, y_test, y_pred)
    
    # Example: Predict next month using most recent features
    print("\n" + "=" * 80)
    print("PREDICTING NEXT MONTH")
    print("=" * 80)
    
    latest_features = X.iloc[-1]
    next_month_pred = predictor.predict_next_month(latest_features)
    
    print("\nPredicted factor returns for next month:")
    for factor, value in next_month_pred.items():
        print(f"  {factor:10s}: {value:+.4f} ({value*100:+.2f}%)")
    
    print("\n" + "="*80)
    print("INITIAL TRAINING COMPLETE!")
    print("="*80)
    
    # ========== ML MODELS COMPARISON ==========
    print("\n")
    results_df, best_model, X_train_eval, X_val_eval, X_test_eval, y_train_eval, y_val_eval, y_test_eval = evaluate_all_models(
        dataset_path="data/processed/factor_ml_dataset_enhanced.csv"
    )
    
    # ========== COMPARISON 1: HISTORICAL MEAN vs ML ==========
    print("\n")
    hist_mean = HistoricalMeanBaseline()
    hist_mean.fit(y_train_eval)
    
    hist_vs_ml_comparison = compare_historical_mean_vs_ml(
        hist_mean_baseline=hist_mean,
        ml_predictor=best_model,
        X_test=X_test_eval,
        y_test=y_test_eval,
        save_dir="results"
    )
    
    # ========== COMPARISON 2: ML-ENHANCED MONTE CARLO ==========
    print("\n")
    mc_simulator = MonteCarloFactorSimulator(n_simulations=10000, random_seed=42)
    mc_simulator.fit(y_train_eval)
    
    ml_enhanced_comparison, ml_enhanced_intervals = compare_ml_enhanced_monte_carlo(
        mc_simulator=mc_simulator,
        ml_predictor=best_model,
        X_test=X_test_eval,
        y_test=y_test_eval,
        save_dir="results"
    )
    
    # ========== CALCULATE CAPM BETAS ==========
    print("\n" + "="*80)
    print("STEP 6: CALCULATING CAPM BETAS")
    print("="*80)

    from src.beta_calculator import calculate_all_betas, plot_beta_distribution

    betas_df = calculate_all_betas(
        returns_path="data/processed/sp500_monthly_returns.csv",
        rf_path="data/processed/Fama_French.csv",
        output_path="data/processed/sp500_capm_betas.csv"
    )

    plot_beta_distribution(betas_df)

    # ========== FF5-BASED OPTIMAL PORTFOLIO ==========
    print("\n" + "="*80)
    print("STEP 7: BUILDING FF5 OPTIMAL PORTFOLIO (UNCONSTRAINED TANGENCY)")
    print("="*80)

    from src.portfolio_optimizer import build_ff5_optimal_portfolio

    ff5_portfolio = build_ff5_optimal_portfolio(
        returns_path="data/processed/sp500_monthly_returns.csv",
        ff_path="data/processed/Fama_French.csv",
        factor_ml_dataset_path="data/processed/factor_ml_dataset_enhanced.csv",
        best_model=best_model,
        lambda_hml=0.2,      # heavy shrinkage of ML HML signal
        min_obs=36
    )
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("COMPLETE PIPELINE SUMMARY")
    print("="*80)
    
    print("\n📊 Generated Visualizations:")
    print("  • results/factor_predictions.png - Initial RF model predictions")
    print("  • results/model_comparison.png - All ML models comparison")
    print("  • results/hist_mean_vs_ml_comparison.png - Historical Mean vs ML")
    print("  • results/hist_mean_vs_ml_timeseries.png - Time series comparison")
    print("  • results/ml_enhanced_mc_intervals.png - ML-Enhanced Monte Carlo intervals")
    print("  • results/beta_distribution.png - CAPM beta distribution")
    
    print("\n📁 Generated Data Files:")
    print("  • data/processed/sp500_capm_betas.csv - CAPM betas")
    print("  • results/model_comparison_detailed.csv - ML models metrics")
    print("  • results/hist_mean_vs_ml_comparison.csv - Baseline comparison")
    print("  • results/ml_enhanced_mc_comparison.csv - ML-Enhanced metrics")
    print("  • results/ml_enhanced_mc_intervals.csv - Prediction intervals")
    
    print("\n🎯 Key Findings:")
    print("-"*80)
    
    # Best ML model
    avg_r2_by_model = results_df.groupby('Model')['R²'].mean()
    best_model_name = avg_r2_by_model.idxmax()
    best_r2 = avg_r2_by_model.max()
    print(f"Best ML Model: {best_model_name} (Avg R² = {best_r2:+.4f})")
    
    # Historical Mean vs ML
    avg_hm_r2 = hist_vs_ml_comparison['Hist_Mean_R²'].mean()
    avg_ml_r2 = hist_vs_ml_comparison['ML_R²'].mean()
    ml_wins = (hist_vs_ml_comparison['R²_Improvement'] > 0).sum()
    
    print(f"\nHistorical Mean vs ML:")
    print(f"  Historical Mean: {avg_hm_r2:+.4f}")
    print(f"  Best ML Model:   {avg_ml_r2:+.4f}")
    print(f"  ML wins on {ml_wins}/5 factors")
    
    if avg_ml_r2 > avg_hm_r2:
        print(f"  ✓ ML adds value over simple baseline!")
    else:
        print(f"  → ML performs similarly to baseline (factors hard to predict)")
    
    # ML-Enhanced Monte Carlo
    avg_coverage = ml_enhanced_comparison['Coverage_95%'].mean()
    print(f"\nML-Enhanced Monte Carlo:")
    print(f"  Average 95% CI coverage: {avg_coverage:.1f}%")
    if 90 <= avg_coverage <= 100:
        print(f"  ✓ Well-calibrated prediction intervals!")
    
    # Most predictable factor
    best_factor_r2 = results_df.loc[results_df['R²'].idxmax()]
    print(f"\nMost Predictable Factor: {best_factor_r2['Factor']} with {best_factor_r2['Model']}")
    print(f"  R² = {best_factor_r2['R²']:+.4f}")
    
    # Beta statistics
    print(f"\nCAPM Beta Statistics:")
    print(f"  Average Beta: {betas_df['Beta'].mean():.3f}")
    print(f"  Median Beta:  {betas_df['Beta'].median():.3f}")
    print(f"  Most Volatile: {betas_df.nlargest(1, 'Beta')['Ticker'].values[0]} (β = {betas_df['Beta'].max():.2f})")
    print(f"  Most Defensive: {betas_df.nsmallest(1, 'Beta')['Ticker'].values[0]} (β = {betas_df['Beta'].min():.2f})")
    
    print("\n" + "="*80)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*80)
    
    return (predictor, best_model, metrics_val, metrics_test, results_df, 
            hist_vs_ml_comparison, ml_enhanced_comparison, betas_df, ff5_portfolio)


if __name__ == "__main__":
    results = main()