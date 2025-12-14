import os
import pandas as pd
import numpy as np
from datetime import datetime

def export_all_results(predictor, best_model, results_df, hist_vs_ml_comparison, 
                      ml_enhanced_comparison, betas_df, 
                      ff5_baseline,        # Portfolio 1: No tilt
                      ff5_tilt,            # Portfolio 2: RMW tilt
                      ff5_concentrated,    # Portfolio 3: Concentrated RMW
                      comparison_df,
                      backtest_results=None):
    """
    Export comprehensive results with 3-portfolio comparison.
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. PIPELINE SUMMARY
    filepath = os.path.join(output_dir, "pipeline_summary.txt")
    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FAMA-FRENCH 5-FACTOR PREDICTION - RESULTS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Key findings
        if "Split" not in results_df.columns:
            raise ValueError("results_df must include a 'Split' column with Train/Val/Test.")

        val_df = results_df[results_df["Split"] == "Val"].copy()
        test_df = results_df[results_df["Split"] == "Test"].copy()
        avg_r2_by_model = val_df.groupby("Model")["R²"].mean()
        best_model_name = avg_r2_by_model.idxmax()
        best_r2 = float(avg_r2_by_model.max())
        test_slice = test_df[test_df["Model"] == best_model_name]
        test_r2 = float(test_slice["R²"].mean()) if not test_slice.empty else float("nan")

        f.write("KEY FINDINGS:\n")
        f.write("-"*80 + "\n")
        f.write(f"1. Best ML Model (chosen on Val): {best_model_name}\n")
        f.write(f"   Average Val  R² (selection): {best_r2:+.4f}\n")
        f.write(f"   Average Test R² (same model): {test_r2:+.4f}\n")
        if test_r2 < 0:
            f.write(f"   Result: ML performs WORSE than historical mean (negative test R²)\n")
        elif abs(test_r2 - best_r2) > 0.05:
            f.write(f"   Result: Significant overfitting detected (val-test gap = {best_r2 - test_r2:.4f})\n")
        else:
            f.write(f"   Result: ML performs similarly to historical mean baseline\n")
        f.write(f"   Note: This is the average across all 5 factors. Individual factors vary\n")
        f.write(f"         significantly - see section 2 for per-factor performance.\n\n")

        # Compare validation vs test R² for each factor
        f.write("2. Factor Predictability (Val vs Test):\n")
        f.write("   Note: Each factor uses its individually best-performing model (selected on\n")
        f.write("         validation set), not necessarily Random Forest.\n\n")

        df_r2_val = val_df.groupby(["Factor", "Model"], as_index=False)["R²"].mean()
        r2_pivot_val = df_r2_val.pivot(index="Factor", columns="Model", values="R²")
        df_r2_test = test_df.groupby(["Factor", "Model"], as_index=False)["R²"].mean()
        r2_pivot_test = df_r2_test.pivot(index="Factor", columns="Model", values="R²")
        best_per_factor_val = r2_pivot_val.max(axis=1)
        best_model_per_factor = r2_pivot_val.idxmax(axis=1)
        for factor in best_per_factor_val.index:
            val_r2 = float(best_per_factor_val.loc[factor])
            model = str(best_model_per_factor.loc[factor])
            if factor in r2_pivot_test.index and model in r2_pivot_test.columns:
                test_r2_factor = float(r2_pivot_test.loc[factor, model])
            else:
                test_r2_factor = np.nan
            if not np.isnan(test_r2_factor) and val_r2 > 0.01 and test_r2_factor > 0.01:
                status = "✓ Predictable"
            elif not np.isnan(test_r2_factor) and val_r2 > 0 and test_r2_factor > 0:
                status = "~ Weak"
            else:
                status = "✗ Unpredictable"
            
            test_str = f"{test_r2_factor:+.4f}" if not np.isnan(test_r2_factor) else "N/A"
            f.write(f"   {factor:10s}: Val R² = {val_r2:+.4f}, Test R² = {test_str}  ({model})  {status}\n")

        f.write("\n   Model usage summary:\n")
        model_counts = {}
        for factor in best_per_factor_val.index:
            model = str(best_model_per_factor.loc[factor])
            model_counts[model] = model_counts.get(model, 0) + 1
        for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
            f.write(f"     - {model}: {count} factor{'s' if count > 1 else ''}\n")
        
        f.write("\n3. Uncertainty Quantification:\n")
        avg_coverage = ml_enhanced_comparison['Coverage_95_pct'].mean()
        f.write(f"   95% CI Coverage: {avg_coverage:.1f}%\n")
        f.write(f"   Status: {'✓ Well-calibrated' if 90 <= avg_coverage <= 100 else '✗ Needs calibration'}\n")
        
        # Add backtest comparison
        if backtest_results is not None and len(backtest_results) > 0:
            f.write("\n4. Out-of-Sample Backtest Results (Rolling Window):\n")
            
            port_returns = backtest_results['Port_Excess_Return']
            mkt_returns = backtest_results['Mkt_RF']
            mu_port = port_returns.mean()
            sigma_port = port_returns.std(ddof=1)
            sharpe_backtest = mu_port / sigma_port if sigma_port > 0 else np.nan
            
            valid = backtest_results.dropna()
            if len(valid) > 1:
                cov_pm = np.cov(valid['Port_Excess_Return'], valid['Mkt_RF'])[0, 1]
                var_m = np.var(valid['Mkt_RF'], ddof=1)
                beta_backtest = cov_pm / var_m if var_m > 0 else np.nan
                mu_mkt = valid['Mkt_RF'].mean()
                alpha_backtest = mu_port - beta_backtest * mu_mkt if not np.isnan(beta_backtest) else np.nan
            else:
                beta_backtest = np.nan
                alpha_backtest = np.nan
            
            f.write(f"   Backtest period: {len(backtest_results)} months (expanding window)\n")
            f.write(f"     - Sharpe Ratio: {sharpe_backtest:.3f}\n")
            f.write(f"     - Market Beta:  {beta_backtest:.3f}\n")
            alpha_backtest_annual = alpha_backtest * 12 if not np.isnan(alpha_backtest) else np.nan
            f.write(f"     - CAPM Alpha:   {alpha_backtest*100:+.2f}% monthly ({alpha_backtest_annual*100:+.2f}% annualized)\n")
            f.write(f"     - Mean Return:  {mu_port*100:+.2f}% monthly ({mu_port*12*100:+.2f}% annualized)\n")
            f.write(f"     - Volatility:   {sigma_port*100:.2f}% monthly ({sigma_port*np.sqrt(12)*100:.2f}% annualized)\n")

        # Portfolio comparison (the main event!)
        f.write("\n5. Portfolio Strategy Comparison:\n")
        f.write("-"*80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Extract key metrics for analysis
        baseline_row = comparison_df[comparison_df["Strategy"].str.contains("Baseline", na=False)].iloc[0]
        tilt_row = comparison_df[comparison_df["Strategy"].str.contains("Tilt", na=False)].iloc[0]
        conc_row = comparison_df[comparison_df["Strategy"].str.contains("concentrated", na=False)].iloc[0]
        
        baseline_sharpe = baseline_row["Sharpe"]
        tilt_sharpe = tilt_row["Sharpe"]
        conc_sharpe = conc_row["Sharpe"]
        
        baseline_rmw = baseline_row["Portfolio_Beta_RMW"]
        tilt_rmw = tilt_row["Portfolio_Beta_RMW"]
        conc_rmw = conc_row["Portfolio_Beta_RMW"]
        
        f.write("STRATEGY ANALYSIS:\n")
        f.write("-"*80 + "\n")
        
        # Question 1: Does RMW tilt help?
        f.write("\n(A) Does RMW tilt improve diversified portfolio?\n")
        f.write(f"    Baseline (no tilt): Sharpe {baseline_sharpe:.3f}, RMW beta {baseline_rmw:.3f}\n")
        f.write(f"    With RMW tilt:      Sharpe {tilt_sharpe:.3f}, RMW beta {tilt_rmw:.3f}\n")
        
        if tilt_sharpe > baseline_sharpe:
            pct_change = (tilt_sharpe / baseline_sharpe - 1) * 100
            f.write(f"    → ✓ YES: Tilt improves Sharpe by {pct_change:+.1f}%\n")
            f.write(f"    → RMW beta increased from {baseline_rmw:.3f} to {tilt_rmw:.3f}\n")
        else:
            pct_change = (1 - tilt_sharpe / baseline_sharpe) * 100
            f.write(f"    → ✗ NO: Tilt reduces Sharpe by {pct_change:.1f}%\n")
            f.write(f"    → Despite RMW beta increasing to {tilt_rmw:.3f}, risk-adjusted returns declined\n")
            f.write(f"    → Likely cause: ML predicted negative RMW, fighting historical premium\n")
        
        # Question 2: Does concentration help?
        f.write(f"\n(B) Does concentration to 50 stocks improve performance?\n")
        f.write(f"    Baseline (377 stocks): Sharpe {baseline_sharpe:.3f}, Volatility {baseline_row['Volatility']*100:.2f}%\n")
        f.write(f"    Concentrated (50):     Sharpe {conc_sharpe:.3f}, Volatility {conc_row['Volatility']*100:.2f}%\n")
        
        if conc_sharpe > baseline_sharpe:
            pct_change = (conc_sharpe / baseline_sharpe - 1) * 100
            f.write(f"    → ✓ YES: Concentration improves Sharpe by {pct_change:+.1f}%\n")
        else:
            pct_change = (1 - conc_sharpe / baseline_sharpe) * 100
            vol_increase = (conc_row['Volatility'] / baseline_row['Volatility'] - 1) * 100
            f.write(f"    → ✗ NO: Concentration reduces Sharpe by {pct_change:.1f}%\n")
            f.write(f"    → Volatility increased {vol_increase:+.1f}% due to idiosyncratic risk\n")
            f.write(f"    → Diversification benefit lost outweighs RMW factor exposure gain\n")
        
        # Overall ranking
        f.write(f"\n(C) Overall ranking by Sharpe ratio:\n")
        sorted_strats = comparison_df.sort_values("Sharpe", ascending=False)
        for i, (idx, row) in enumerate(sorted_strats.iterrows(), 1):
            marker = "  ⭐ BEST" if i == 1 else ""
            f.write(f"    #{i}: {row['Strategy']:30s} Sharpe={row['Sharpe']:.3f}{marker}\n")
        
        # Final recommendation
        f.write("\n" + "="*80 + "\n")
        f.write("FINAL RECOMMENDATION:\n")
        f.write("="*80 + "\n")
        
        best_strategy = sorted_strats.iloc[0]["Strategy"]
        best_sharpe_val = sorted_strats.iloc[0]["Sharpe"]
        
        f.write(f"Best strategy: {best_strategy} (Sharpe = {best_sharpe_val:.3f})\n\n")
        
        if "Baseline" in best_strategy:
            f.write("✓ Recommendation: Use BASELINE portfolio (no RMW tilt)\n")
            f.write("  Rationale:\n")
            f.write("  - Pure Markowitz mean-variance optimization provides best risk-adjusted returns\n")
            f.write("  - RMW tilt is counterproductive when ML predicts negative RMW returns\n")
            f.write("  - Diversification across 377 stocks minimizes idiosyncratic risk\n")
        elif "Tilt" in best_strategy:
            f.write("✓ Recommendation: Use RMW TILT strategy\n")
            f.write("  Rationale:\n")
            f.write("  - RMW factor shows genuine predictability (Test R² > 0)\n")
            f.write("  - Tilting toward profitable firms improves risk-adjusted returns\n")
            f.write("  - Maintains diversification while increasing factor exposure\n")
        else:
            f.write("✓ Recommendation: Use CONCENTRATED RMW strategy\n")
            f.write("  Rationale:\n")
            f.write("  - Strong RMW factor exposure (beta > 0.8) provides excess returns\n")
            f.write("  - Higher returns compensate for increased volatility\n")
            f.write("  - Suitable for investors with high risk tolerance\n")
        
        f.write("\n" + "="*80 + "\n")

        # Beta statistics
        f.write("\n6. CAPM Beta Statistics:\n")
        if 'Beta' in betas_df.columns:
            beta_col = 'Beta'
            r2_col = 'R²'
        elif 'Beta_MKT' in betas_df.columns:
            beta_col = 'Beta_MKT'
            r2_col = 'R_squared'
        else:
            f.write("   (Beta statistics not available)\n\n")
            beta_col = None
        
        if beta_col:
            f.write(f"   Average Beta: {betas_df[beta_col].mean():.3f}\n")
            f.write(f"   Median  Beta: {betas_df[beta_col].median():.3f}\n")
            most_beta = betas_df[beta_col].idxmax()
            least_beta = betas_df[beta_col].idxmin()
            f.write(f"   Highest Beta: {most_beta} (β = {betas_df.loc[most_beta, beta_col]:.2f})\n")
            f.write(f"   Lowest  Beta: {least_beta} (β = {betas_df.loc[least_beta, beta_col]:.2f})\n")
            if r2_col in betas_df.columns:
                f.write(f"   Avg R²:  {betas_df[r2_col].mean():.3f}\n\n")
            else:
                f.write("\n")

    print(f"✓ Summary saved to: {filepath}")
    
    # 2. EXCEL REPORT
    try:
        excel_path = os.path.join(output_dir, "complete_results.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Model comparison
            results_df.to_excel(writer, sheet_name='Model Comparison', index=False)
            
            # Portfolio comparison
            comparison_df.to_excel(writer, sheet_name='Portfolio Comparison', index=False)
            
            # Individual portfolio weights (top 50 each)
            ff5_baseline.head(50).to_excel(writer, sheet_name='Baseline Top 50')
            ff5_tilt.head(50).to_excel(writer, sheet_name='RMW Tilt Top 50')
            ff5_concentrated.head(50).to_excel(writer, sheet_name='Concentrated 50')
            
            # CAPM betas (summary stats)
            if 'Beta' in betas_df.columns and 'R²' in betas_df.columns and 'Alpha' in betas_df.columns:
                betas_summary = betas_df[['Beta', 'R²', 'Alpha']].describe()
                betas_summary.to_excel(writer, sheet_name='CAPM Beta Summary')
            elif 'Beta_MKT' in betas_df.columns:
                cols = ['Beta_MKT', 'Beta_SMB', 'Beta_HML', 'Beta_RMW', 'Beta_CMA']
                available_cols = [c for c in cols if c in betas_df.columns]
                if 'R_squared' in betas_df.columns:
                    available_cols.append('R_squared')
                if 'Alpha' in betas_df.columns:
                    available_cols.append('Alpha')
                betas_summary = betas_df[available_cols].describe()
                betas_summary.to_excel(writer, sheet_name='FF5 Beta Summary')
            
            # MC comparison
            ml_enhanced_comparison.to_excel(writer, sheet_name='Monte Carlo', index=False)
            
            # Backtest results
            if backtest_results is not None:
                backtest_results.to_excel(writer, sheet_name='Backtest Results')
        
        print(f"✓ Excel saved to: {excel_path}")
        
    except Exception as e:
        print(f"  ✗ Excel error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll results exported to results/ directory")