import os
import numpy as np
import pandas as pd
from datetime import datetime


def _compute_backtest_stats(excess: pd.Series, mkt: pd.Series):
    """
    Compute Sharpe, CAPM beta, and CAPM alpha for a backtest series.
    Returns dict with keys: sharpe, beta, alpha_monthly, mean, vol.
    """
    excess = pd.to_numeric(excess, errors="coerce")
    mkt = pd.to_numeric(mkt, errors="coerce")

    mu = float(excess.mean())
    sigma = float(excess.std(ddof=1))
    sharpe = mu / sigma if sigma > 0 else np.nan

    df = pd.concat([excess.rename("ex"), mkt.rename("mkt")], axis=1).dropna()
    if len(df) > 1:
        cov = float(
            np.cov(df["ex"], df["mkt"])[0, 1]
        )  # covariance between excess returns and market
        var_m = float(np.var(df["mkt"], ddof=1))  # variance of market returns
        beta = cov / var_m if var_m > 0 else np.nan  # CAPM beta
        mu_m = float(df["mkt"].mean())  # mean market excess return
        alpha = mu - beta * mu_m if not np.isnan(beta) else np.nan  # CAPM alpha
    else:
        beta = np.nan
        alpha = np.nan

    return {
        "mean": mu,
        "vol": sigma,
        "sharpe": sharpe,
        "beta": beta,
        "alpha_monthly": alpha,
    }


def export_all_results(
    predictor,
    best_model,
    results_df,
    hist_vs_ml_comparison,
    ml_enhanced_comparison,
    betas_df,
    ff5_equal_weight,
    ff5_baseline,
    ff5_tilt,
    ff5_concentrated,
    comparison_df,
    backtest_results=None,
):
    """
    Export results:
      - pipeline_summary.txt
      - complete_results.xlsx
    Supports backtests containing:
      - Port_Excess_Return (baseline)
      - Tilt_Excess_Return (rmw tilt)
      - EW_Excess_Return (equal weight)
      - Mkt_RF
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) PIPELINE SUMMARY (TXT)
    # ---------------------------------------------------------------------
    filepath = os.path.join(output_dir, "pipeline_summary.txt")
    with open(filepath, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FAMA-FRENCH 5-FACTOR PREDICTION - RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # -------------------------
        # Model selection summary
        # -------------------------
        if "Split" not in results_df.columns:
            raise ValueError(
                "results_df must include a 'Split' column with Train/Val/Test."
            )

        val_df = results_df[results_df["Split"] == "Val"].copy()
        test_df = results_df[results_df["Split"] == "Test"].copy()

        avg_r2_by_model = val_df.groupby("Model")["R²"].mean()
        best_model_name = str(avg_r2_by_model.idxmax())
        best_r2 = float(avg_r2_by_model.max())

        test_slice = test_df[test_df["Model"] == best_model_name]
        test_r2 = (
            float(test_slice["R²"].mean()) if not test_slice.empty else float("nan")
        )

        f.write("KEY FINDINGS: (Static / Single-Period Comparison)\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. Best ML Model (chosen on Val): {best_model_name}\n")
        f.write(f"   Average Val  R² (selection): {best_r2:+.4f}\n")
        f.write(f"   Average Test R² (same model): {test_r2:+.4f}\n")
        if np.isnan(test_r2):
            f.write("   Result: Test R² not available for selected model.\n")
        elif test_r2 < 0:
            f.write(
                "   Result: ML performs WORSE than historical mean (negative test R²).\n"
            )
        else:
            f.write(
                "   Result: ML performs similar to / better than historical mean on average.\n"
            )
        f.write(
            "   Note: This is the average across all 5 factors; factor-level results vary.\n\n"
        )

        # -------------------------
        # Per-factor Val vs Test
        # -------------------------
        f.write("2. Factor Predictability (Val vs Test):\n")
        f.write("   Note: Model selection is based on validation performance.\n\n")

        df_r2_val = val_df.groupby(["Factor", "Model"], as_index=False)["R²"].mean()
        r2_pivot_val = df_r2_val.pivot(index="Factor", columns="Model", values="R²")

        df_r2_test = test_df.groupby(["Factor", "Model"], as_index=False)["R²"].mean()
        r2_pivot_test = df_r2_test.pivot(index="Factor", columns="Model", values="R²")

        best_val = r2_pivot_val.max(axis=1)
        best_model_per_factor = r2_pivot_val.idxmax(axis=1)

        for factor in best_val.index:
            val_r2 = float(best_val.loc[factor])
            model = str(best_model_per_factor.loc[factor])
            test_r2_factor = np.nan
            if factor in r2_pivot_test.index and model in r2_pivot_test.columns:
                test_r2_factor = float(r2_pivot_test.loc[factor, model])

            if not np.isnan(test_r2_factor) and val_r2 > 0.01 and test_r2_factor > 0.01:
                status = "✓ Predictable"
            elif not np.isnan(test_r2_factor) and val_r2 > 0 and test_r2_factor > 0:
                status = "~ Weak"
            else:
                status = "✗ Unpredictable"

            test_str = (
                f"{test_r2_factor:+.4f}" if not np.isnan(test_r2_factor) else "N/A"
            )
            f.write(
                f"   {factor:10s}: Val R² = {val_r2:+.4f}, Test R² = {test_str} ({model}) {status}\n"
            )

        f.write("\n3. Uncertainty Quantification:\n")
        avg_coverage = float(ml_enhanced_comparison["Coverage_95_pct"].mean())
        f.write(f"   Avg 95% interval coverage: {avg_coverage:.1f}%\n")
        f.write(
            "   Note: coverage varies by factor; interpret 'well-calibrated' with caution.\n"
        )

        # -------------------------
        # Backtest section (multi-strategy)
        # -------------------------
        if backtest_results is not None and len(backtest_results) > 0:
            f.write("\n4. Out-of-Sample Backtest Results (Rolling Window):\n")
            f.write("-" * 80 + "\n")

            if "Mkt_RF" not in backtest_results.columns:
                f.write(
                    "   Backtest results missing 'Mkt_RF'; cannot compute beta/alpha.\n"
                )
            else:
                mkt = backtest_results["Mkt_RF"]

                series_map = [
                    ("Baseline", "Port_Excess_Return"),
                    ("RMW Tilt", "Tilt_Excess_Return"),
                    ("Equal-Weight", "EW_Excess_Return"),
                ]

                for label, col in series_map:
                    if col in backtest_results.columns:
                        stats = _compute_backtest_stats(backtest_results[col], mkt)
                        alpha_ann = (
                            stats["alpha_monthly"] * 12
                            if not np.isnan(stats["alpha_monthly"])
                            else np.nan
                        )
                        f.write(f"   {label}:\n")
                        f.write(
                            f"     - Periods:      {int(backtest_results[col].dropna().shape[0])}\n"
                        )
                        f.write(f"     - Sharpe:       {stats['sharpe']:.3f}\n")
                        f.write(
                            f"     - Mean:         {stats['mean']*100:+.2f}% monthly ({stats['mean']*12*100:+.2f}% ann.)\n"
                        )
                        f.write(
                            f"     - Volatility:   {stats['vol']*100:.2f}% monthly ({stats['vol']*np.sqrt(12)*100:.2f}% ann.)\n"
                        )
                        f.write(f"     - CAPM beta:    {stats['beta']:.3f}\n")
                        f.write(
                            f"     - CAPM alpha:   {stats['alpha_monthly']*100:+.2f}% monthly ({alpha_ann*100:+.2f}% ann.)\n"
                        )
                    else:
                        f.write(
                            f"   {label}: column '{col}' not found in backtest_results.\n"
                        )

        # -------------------------
        # Cross-sectional strategy comparison table
        # -------------------------
        f.write("\n5. Portfolio Strategy Comparison (Cross-Sectional Snapshot):\n")
        f.write("-" * 80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")

        # Strategy analysis with tolerance to avoid misleading "NO" on rounding noise
        f.write("STRATEGY ANALYSIS:\n")
        f.write("-" * 80 + "\n")

        baseline_row = comparison_df[
            comparison_df["Strategy"].str.contains("Baseline", na=False)
        ].iloc[0]
        tilt_row = comparison_df[
            comparison_df["Strategy"].str.contains("Tilt", na=False)
        ].iloc[0]
        conc_row = comparison_df[
            comparison_df["Strategy"].str.contains("concentrated", case=False, na=False)
        ].iloc[0]
        ew_row = comparison_df[
            comparison_df["Strategy"].str.contains("Equal-Weight", na=False)
        ].iloc[0]

        baseline_sharpe = float(baseline_row["Sharpe"])
        tilt_sharpe = float(tilt_row["Sharpe"])
        conc_sharpe = float(conc_row["Sharpe"])
        ew_sharpe = float(ew_row["Sharpe"])

        baseline_rmw = float(baseline_row["Portfolio_Beta_RMW"])
        tilt_rmw = float(tilt_row["Portfolio_Beta_RMW"])
        conc_rmw = float(conc_row["Portfolio_Beta_RMW"])

        tol = (
            0.002  # ~0.2% absolute Sharpe tolerance, avoids overinterpreting tiny diffs
        )

        f.write("\n(A) RMW tilt vs baseline (snapshot):\n")
        f.write(
            f"    Baseline: Sharpe {baseline_sharpe:.3f}, RMW beta {baseline_rmw:.3f}\n"
        )
        f.write(f"    Tilt:     Sharpe {tilt_sharpe:.3f}, RMW beta {tilt_rmw:.3f}\n")
        if abs(tilt_sharpe - baseline_sharpe) <= tol:
            f.write(
                "    → Result: Sharpe essentially unchanged; tilt increases RMW exposure.\n"
            )
        elif tilt_sharpe > baseline_sharpe:
            pct_change = (tilt_sharpe / baseline_sharpe - 1) * 100
            f.write(f"    → Result: Tilt improves Sharpe by {pct_change:+.2f}%.\n")
        else:
            pct_change = (1 - tilt_sharpe / baseline_sharpe) * 100
            f.write(f"    → Result: Tilt reduces Sharpe by {pct_change:.2f}%.\n")

        f.write("\n(B) Concentration (50) vs baseline (snapshot):\n")
        f.write(
            f"    Baseline Sharpe {baseline_sharpe:.3f} vs Concentrated Sharpe {conc_sharpe:.3f}\n"
        )
        f.write(
            f"    RMW beta increases to {conc_rmw:.3f} but Sharpe declines, consistent with diversification loss.\n"
        )

        f.write("\n(C) Equal-weight vs optimized baseline (snapshot):\n")
        f.write(
            f"    Equal-Weight Sharpe {ew_sharpe:.3f} vs Baseline Sharpe {baseline_sharpe:.3f}\n"
        )

        f.write("\n" + "=" * 80 + "\n")

        # Beta statistics
        f.write("\n6. Beta Statistics:\n")
        if "Beta" in betas_df.columns:
            beta_col, r2_col = "Beta", "R²"
        elif "Beta_MKT" in betas_df.columns:
            beta_col, r2_col = "Beta_MKT", "R_squared"
        else:
            beta_col, r2_col = None, None
            f.write("   (Beta statistics not available)\n")

        if beta_col is not None:
            f.write(f"   Average {beta_col}: {betas_df[beta_col].mean():.3f}\n")
            f.write(f"   Median  {beta_col}: {betas_df[beta_col].median():.3f}\n")
            most_beta = betas_df[beta_col].idxmax()
            least_beta = betas_df[beta_col].idxmin()
            f.write(
                f"   Highest: {most_beta} (β = {betas_df.loc[most_beta, beta_col]:.2f})\n"
            )
            f.write(
                f"   Lowest:  {least_beta} (β = {betas_df.loc[least_beta, beta_col]:.2f})\n"
            )
            if r2_col is not None and r2_col in betas_df.columns:
                f.write(f"   Avg {r2_col}: {betas_df[r2_col].mean():.3f}\n")

    print(f"✓ Summary saved to: {filepath}")

    # ---------------------------------------------------------------------
    # 2) EXCEL REPORT
    # ---------------------------------------------------------------------
    try:
        excel_path = os.path.join(output_dir, "complete_results.xlsx")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Model Comparison", index=False)
            comparison_df.to_excel(
                writer, sheet_name="Portfolio Comparison", index=False
            )
            ml_enhanced_comparison.to_excel(
                writer, sheet_name="Monte Carlo", index=False
            )

            # Portfolio weights (Top 50 for readability)
            ff5_equal_weight.head(50).to_excel(writer, sheet_name="Equal-Weight Top 50")
            ff5_baseline.head(50).to_excel(writer, sheet_name="Baseline Top 50")
            ff5_tilt.head(50).to_excel(writer, sheet_name="RMW Tilt Top 50")
            ff5_concentrated.head(50).to_excel(writer, sheet_name="Concentrated 50")

            # Beta summary
            if (
                "Beta" in betas_df.columns
                and "R²" in betas_df.columns
                and "Alpha" in betas_df.columns
            ):
                betas_df[["Beta", "R²", "Alpha"]].describe().to_excel(
                    writer, sheet_name="CAPM Beta Summary"
                )
            elif "Beta_MKT" in betas_df.columns:
                cols = ["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA"]
                available = [c for c in cols if c in betas_df.columns]
                if "R_squared" in betas_df.columns:
                    available.append("R_squared")
                if "Alpha" in betas_df.columns:
                    available.append("Alpha")
                betas_df[available].describe().to_excel(
                    writer, sheet_name="FF5 Beta Summary"
                )

            # Backtest
            if backtest_results is not None:
                backtest_results.to_excel(writer, sheet_name="Backtest Results")

        print(f"✓ Excel saved to: {excel_path}")

    except Exception as e:
        print(f"  ✗ Excel error: {e}")
        import traceback

        traceback.print_exc()

    print("\nAll results exported to results/ directory")
