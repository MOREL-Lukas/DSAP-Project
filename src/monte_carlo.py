# src/monte_carlo.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error as mse, r2_score as r2, mean_absolute_error as mae

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


class HistoricalMeanBaseline:
    def __init__(self, factor_names=FACTOR_NAMES):
        self.factor_names = list(factor_names)
        self.means = {}

    def fit(self, y_train: pd.DataFrame):
        self.means = {f: y_train[f].mean() for f in self.factor_names}

        print("\n" + "=" * 80)
        print("HISTORICAL MEAN BASELINE - FITTED PARAMETERS")
        print("=" * 80)
        print("\nHistorical Mean (from training data):\n" + "-" * 80)
        for f in self.factor_names:
            m = self.means[f]
            print(f"{f:10s}: μ = {m:+.4f} ({m * 100:+.2f}%)")

        return self

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if not self.means:
            raise ValueError("HistoricalMeanBaseline is not fitted. Call .fit(y_train) first.")
        return pd.DataFrame({f: self.means[f] for f in self.factor_names}, index=X_test.index)


class MonteCarloFactorSimulator:
    def __init__(self, n_simulations=10000, random_seed=42, factor_names=FACTOR_NAMES):
        self.n_simulations = int(n_simulations)
        self.random_seed = int(random_seed)
        self.factor_names = list(factor_names)

        self.means = {}
        self.stds = {}
        self.correlations = None
        self.is_ml_enhanced = False

    def fit(self, y_train: pd.DataFrame):
        y = y_train[self.factor_names]
        self.means = y.mean().to_dict()
        self.stds = y.std().to_dict()
        self.correlations = y.corr()

        print("\n" + "=" * 80)
        print("MONTE CARLO SIMULATOR - FITTED PARAMETERS")
        print("=" * 80)
        print("\nHistorical Statistics (from training data):\n" + "-" * 80)
        for f in self.factor_names:
            mu, sd = self.means[f], self.stds[f]
            print(f"{f:10s}: μ = {mu:+.4f} ({mu * 100:+.2f}%), σ = {sd:.4f} ({sd * 100:.2f}%)")
        print("\n" + "-" * 80 + "\nCorrelation Matrix:\n" + "-" * 80)
        print(self.correlations.to_string())

        return self

    def _chol(self) -> np.ndarray:
        if self.correlations is None:
            raise ValueError("MonteCarloFactorSimulator is not fitted. Call .fit(y_train) first.")

        try:
            return np.linalg.cholesky(self.correlations.values)
        except np.linalg.LinAlgError:
            print("Warning: Correlation matrix not positive definite. Using uncorrelated simulations.")
            return np.eye(len(self.factor_names))

    def simulate(self, n_periods: int, ml_predictions: pd.DataFrame | None = None) -> dict:
        """
        Returns dict factor -> simulations array of shape (S, T).
        If ml_predictions provided, its per-period values are used as time-varying means.
        """
        if not self.means or not self.stds or self.correlations is None:
            raise ValueError("MonteCarloFactorSimulator is not fitted. Call .fit(y_train) first.")

        np.random.seed(self.random_seed)
        k = len(self.factor_names)

        # (S, T, K)
        z = np.random.randn(self.n_simulations, n_periods, k)
        # correlate: (S, T, K)
        x = np.einsum("stj,ij->sti", z, self._chol())

        self.is_ml_enhanced = ml_predictions is not None

        sims = {}
        for i, f in enumerate(self.factor_names):
            if self.is_ml_enhanced:
                if f not in ml_predictions.columns:
                    raise KeyError(f"ml_predictions missing required column: '{f}'")
                mu = ml_predictions[f].to_numpy()[None, :]  # (1, T)
            else:
                mu = self.means[f]  # scalar
            sims[f] = mu + self.stds[f] * x[:, :, i]  # broadcast to (S, T)

        return sims

    def predict(self, X_test: pd.DataFrame, ml_predictions: pd.DataFrame | None = None) -> pd.DataFrame:
        sims = self.simulate(len(X_test), ml_predictions)
        return pd.DataFrame({f: sims[f].mean(axis=0) for f in self.factor_names}, index=X_test.index)

    def get_prediction_intervals(
        self,
        X_test: pd.DataFrame,
        ml_predictions: pd.DataFrame | None = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        sims = self.simulate(len(X_test), ml_predictions)

        alpha = 1.0 - float(confidence_level)
        lo_pct, hi_pct = (alpha / 2.0) * 100.0, (1.0 - alpha / 2.0) * 100.0

        out = pd.DataFrame(index=X_test.index)
        for f in self.factor_names:
            out[f"{f}_mean"] = sims[f].mean(axis=0)
            out[f"{f}_lower"] = np.percentile(sims[f], lo_pct, axis=0)
            out[f"{f}_upper"] = np.percentile(sims[f], hi_pct, axis=0)

        return out

    def plot_simulation_distribution(
        self,
        factor: str,
        n_periods: int,
        ml_predictions: pd.DataFrame | None = None,
        save_path: str = "results/monte_carlo_distribution.png",
    ):
        sims = self.simulate(int(n_periods), ml_predictions)[factor]
        mode = "ML-Enhanced" if self.is_ml_enhanced else "Historical"
        final = sims[:, -1]

        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax = ax.ravel()

        ax[0].hist(final, bins=50, alpha=0.7, edgecolor="black")
        m, med = final.mean(), np.median(final)
        ax[0].axvline(m, linestyle="--", linewidth=2, label=f"Mean: {m * 100:.2f}%")
        ax[0].axvline(med, linestyle="--", linewidth=2, label=f"Median: {med * 100:.2f}%")
        ax[0].set(
            title=f"{factor} - Distribution (Final Period) [{mode}]",
            xlabel="Return",
            ylabel="Frequency",
        )
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        n_paths = min(100, self.n_simulations)
        ax[1].plot(np.arange(n_periods), sims[:n_paths].T, alpha=0.1)
        ax[1].plot(np.arange(n_periods), sims.mean(axis=0), linewidth=2, label="Mean Path")
        ax[1].axhline(0, linestyle="--", alpha=0.3)
        ax[1].set(
            title=f"{factor} - Sample Paths ({n_paths} of {self.n_simulations}) [{mode}]",
            xlabel="Period",
            ylabel="Return",
        )
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        pcts = [5, 25, 50, 75, 95]
        pv = np.percentile(sims, pcts, axis=0)
        labels = ["5th", "25th", "50th (Median)", "75th", "95th"]
        for series, lab in zip(pv, labels):
            ax[2].plot(np.arange(n_periods), series, linewidth=2, label=f"{lab} percentile")
        ax[2].fill_between(np.arange(n_periods), pv[0], pv[-1], alpha=0.2, label="5th-95th range")
        ax[2].axhline(0, linestyle="--", alpha=0.3)
        ax[2].set(title=f"{factor} - Confidence Bands [{mode}]", xlabel="Period", ylabel="Return")
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)

        stats.probplot(final, dist="norm", plot=ax[3])
        ax[3].set(title=f"{factor} - Q-Q Plot (Normality Check)")
        ax[3].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nMonte Carlo distribution plot saved to: {save_path}")
        plt.close()


def compare_historical_mean_vs_ml(hist_mean_baseline, ml_predictor, X_test, y_test, save_dir="results"):
    print("\n" + "=" * 80)
    print("COMPARISON: HISTORICAL MEAN vs ML PREDICTIONS")
    print("=" * 80)

    hm_pred = hist_mean_baseline.predict(X_test)
    ml_pred = ml_predictor.predict(X_test)

    model_type = getattr(ml_predictor, "model_type", type(ml_predictor).__name__)
    feat_names = getattr(ml_predictor, "feature_names", [])
    print("\nHistorical Mean: predicting constant values (training average)")
    print(f"ML Model: using {model_type} with {len(feat_names)} features")

    rows = []
    for f in FACTOR_NAMES:
        hm_rmse = float(np.sqrt(mse(y_test[f], hm_pred[f])))
        ml_rmse = float(np.sqrt(mse(y_test[f], ml_pred[f])))

        hm_r2 = float(r2(y_test[f], hm_pred[f]))
        ml_r2 = float(r2(y_test[f], ml_pred[f]))

        rows.append(
            dict(
                Factor=f,
                Hist_Mean_RMSE=hm_rmse,
                ML_RMSE=ml_rmse,
                RMSE_Improvement_pct=((hm_rmse - ml_rmse) / hm_rmse) * 100.0 if hm_rmse != 0 else np.nan,
                Hist_Mean_R2=hm_r2,
                ML_R2=ml_r2,
                R2_Improvement=ml_r2 - hm_r2,
                Hist_Mean_MAE=float(mae(y_test[f], hm_pred[f])),
                ML_MAE=float(mae(y_test[f], ml_pred[f])),
            )
        )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80 + "\nPERFORMANCE COMPARISON\n" + "=" * 80)

    print("\nR2 Scores (higher is better):\n" + "-" * 80)
    r2_tbl = df[["Factor", "Hist_Mean_R2", "ML_R2", "R2_Improvement"]].copy()
    r2_tbl["Winner"] = r2_tbl["R2_Improvement"].apply(lambda x: "ML" if x > 0 else ("Hist" if x < 0 else "Tie"))
    print(r2_tbl.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n" + "-" * 80 + "\nRMSE Scores (lower is better):\n" + "-" * 80)
    print(
        df[["Factor", "Hist_Mean_RMSE", "ML_RMSE", "RMSE_Improvement_pct"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n" + "=" * 80 + "\nSUMMARY STATISTICS\n" + "=" * 80)
    avg_hm, avg_ml = df["Hist_Mean_R2"].mean(), df["ML_R2"].mean()
    print(
        f"\nAverage R2 across all factors:\n"
        f"  Historical Mean: {avg_hm:+.4f}\n"
        f"  ML Model:        {avg_ml:+.4f}\n"
        f"  Improvement:     {avg_ml - avg_hm:+.4f}"
    )

    ml_w = int((df["R2_Improvement"] > 0).sum())
    hm_w = int((df["R2_Improvement"] < 0).sum())
    ties = int((df["R2_Improvement"] == 0).sum())

    print(f"\nWin/Loss Record (by R2):\n  ML Model wins:        {ml_w}/5\n  Historical Mean wins: {hm_w}/5\n  Ties:                 {ties}/5")
    print("\n  ✓ ML model provides value over simple historical mean!" if ml_w > hm_w
          else ("\n  → ML and Historical Mean perform similarly" if ml_w == hm_w
                else "\n  ⚠ Historical Mean outperforms ML (possible overfitting)"))

    _plot_hist_vs_ml_comparison(df, y_test, hm_pred, ml_pred, X_test.index, save_dir)

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/hist_mean_vs_ml_comparison.csv", index=False)
    print(f"\nDetailed comparison saved to: {save_dir}/hist_mean_vs_ml_comparison.csv")

    return df


def _plot_hist_vs_ml_comparison(df, y_test, hm_pred, ml_pred, dates, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    x = np.arange(len(df))
    w = 0.35

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax = ax.ravel()

    ax[0].bar(x - w / 2, df["Hist_Mean_R2"], w, label="Historical Mean", alpha=0.8)
    ax[0].bar(x + w / 2, df["ML_R2"], w, label="ML Model", alpha=0.8)
    ax[0].axhline(0, linestyle="--", linewidth=1, alpha=0.3)
    ax[0].set(title="R2 Comparison: Historical Mean vs ML", xlabel="Factor", ylabel="R2 Score")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(df["Factor"])
    ax[0].legend()
    ax[0].grid(True, alpha=0.3, axis="y")

    ax[1].bar(x - w / 2, df["Hist_Mean_RMSE"], w, label="Historical Mean", alpha=0.8)
    ax[1].bar(x + w / 2, df["ML_RMSE"], w, label="ML Model", alpha=0.8)
    ax[1].set(title="RMSE Comparison (lower is better)", xlabel="Factor", ylabel="RMSE")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(df["Factor"])
    ax[1].legend()
    ax[1].grid(True, alpha=0.3, axis="y")

    ax[2].bar(x, df["R2_Improvement"], alpha=0.7, edgecolor="black")
    ax[2].axhline(0, linestyle="-", linewidth=1)
    ax[2].set(title="ML Improvement over Historical Mean", xlabel="Factor", ylabel="R2 Improvement (ML - Hist Mean)")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(df["Factor"])
    ax[2].grid(True, alpha=0.3, axis="y")

    ml_w, hm_w = int((df["R2_Improvement"] > 0).sum()), int((df["R2_Improvement"] < 0).sum())
    if ml_w + hm_w:
        sizes = [ml_w, hm_w]
        labels = [f"ML Model\n({ml_w}/5)", f"Historical Mean\n({hm_w}/5)"]
        wedges, texts, autotexts = ax[3].pie(
            sizes, labels=labels, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 12}
        )
        for t in autotexts:
            t.set_color("white")
            t.set_fontweight("bold")
            t.set_fontsize(14)
    else:
        ax[3].text(0.5, 0.5, "Perfect Tie\n(All factors identical)", ha="center", va="center", fontsize=14, fontweight="bold")
        ax[3].set_xlim(0, 1)
        ax[3].set_ylim(0, 1)
    ax[3].set_title("Win Rate by R2 Score", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/hist_mean_vs_ml_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to: {save_dir}/hist_mean_vs_ml_comparison.png")
    plt.close()

    _plot_timeseries(y_test, hm_pred, ml_pred, dates, save_dir, "hist_mean_vs_ml_timeseries.png")


def _plot_timeseries(y_test, baseline_pred, ml_pred, dates, save_dir, filename):
    fig, ax = plt.subplots(3, 2, figsize=(16, 14))
    ax = ax.ravel()

    for i, f in enumerate(FACTOR_NAMES):
        a = ax[i]
        a.plot(dates, y_test[f], "o-", label="Actual", linewidth=2, markersize=4, alpha=0.7)
        a.axhline(baseline_pred[f].iloc[0], label="Historical Mean", linestyle="--", linewidth=2, alpha=0.8)
        a.plot(dates, ml_pred[f], "s--", label="ML Model", linewidth=2, markersize=4, alpha=0.8)
        a.axhline(0, linestyle="-", linewidth=0.5, alpha=0.3)
        a.set(title=f"{f}", xlabel="Date", ylabel="Return")
        a.legend(loc="best")
        a.grid(True, alpha=0.3)

        a.text(
            0.02,
            0.98,
            f"R2 (HM): {r2(y_test[f], baseline_pred[f]):.3f}\nR2 (ML): {r2(y_test[f], ml_pred[f]):.3f}",
            transform=a.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    ax[5].axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches="tight")
    print(f"Time series comparison saved to: {save_dir}/{filename}")
    plt.close()


def compare_ml_enhanced_monte_carlo(mc_simulator, ml_predictor, X_test, y_test, save_dir="results"):
    print("\n" + "=" * 80)
    print("ML-ENHANCED MONTE CARLO ANALYSIS")
    print("=" * 80)

    ml_pred = ml_predictor.predict(X_test)

    print("\nGenerating ML-Enhanced Monte Carlo predictions...")
    print("  • Using ML predictions as means")
    print("  • Adding historical volatility and correlations")
    print(f"  • Running {mc_simulator.n_simulations:,} simulations")

    intervals = mc_simulator.get_prediction_intervals(X_test, ml_predictions=ml_pred, confidence_level=0.95)

    rows = []
    for f in FACTOR_NAMES:
        ml_r2 = float(r2(y_test[f], ml_pred[f]))
        ml_rmse = float(np.sqrt(mse(y_test[f], ml_pred[f])))

        enh = intervals[f"{f}_mean"]
        enh_r2 = float(r2(y_test[f], enh))
        enh_rmse = float(np.sqrt(mse(y_test[f], enh)))

        lo, hi = intervals[f"{f}_lower"], intervals[f"{f}_upper"]
        within = ((y_test[f] >= lo) & (y_test[f] <= hi))

        rows.append(
            dict(
                Factor=f,
                ML_R2=ml_r2,
                ML_Enhanced_R2=enh_r2,
                R2_Difference=enh_r2 - ml_r2,
                ML_RMSE=ml_rmse,
                ML_Enhanced_RMSE=enh_rmse,
                Coverage_95_pct=float(within.mean() * 100.0),
                Avg_Interval_Width=float((hi - lo).mean()),
            )
        )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80 + "\nML vs ML-ENHANCED MONTE CARLO\n" + "=" * 80)

    print("\nPrediction Accuracy:\n" + "-" * 80)
    print(df[["Factor", "ML_R2", "ML_Enhanced_R2", "R2_Difference"]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n" + "-" * 80 + "\nUncertainty Quantification (ML-Enhanced only):\n" + "-" * 80)
    print(df[["Factor", "Coverage_95_pct", "Avg_Interval_Width"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n" + "=" * 80 + "\nKEY INSIGHTS\n" + "=" * 80)
    avg_cov = float(df["Coverage_95_pct"].mean())
    print(f"\n1. PREDICTION INTERVALS:\n   • Average 95% coverage: {avg_cov:.1f}%")
    print("   ✓ Well-calibrated intervals!" if 90.0 <= avg_cov <= 100.0 else
          ("   ⚠ Intervals too narrow (underestimating uncertainty)" if avg_cov < 90.0
           else "   ⚠ Intervals too wide (overestimating uncertainty)"))

    avg_r2d = float(df["R2_Difference"].mean())
    print("\n2. ML-ENHANCED vs PURE ML:")
    print("   • ML-Enhanced uses ML predictions + historical volatility")
    print("   • Provides uncertainty quantification")
    print("   • Point predictions should be very similar to ML")
    print(f"   • Average R2 difference: {avg_r2d:+.6f}")
    if abs(avg_r2d) < 0.001:
        print("   ✓ Point predictions nearly identical (as expected)")

    _plot_ml_enhanced_intervals(y_test, ml_pred, intervals, X_test.index, save_dir)

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/ml_enhanced_mc_comparison.csv", index=False)
    intervals.to_csv(f"{save_dir}/ml_enhanced_mc_intervals.csv")
    print("\nResults saved to:")
    print(f"  • {save_dir}/ml_enhanced_mc_comparison.csv")
    print(f"  • {save_dir}/ml_enhanced_mc_intervals.csv")

    return df, intervals


def _plot_ml_enhanced_intervals(y_test, ml_pred, intervals, dates, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(3, 2, figsize=(16, 14))
    ax = ax.ravel()

    for i, f in enumerate(FACTOR_NAMES):
        a = ax[i]
        a.plot(dates, y_test[f], "o-", label="Actual", linewidth=2, markersize=5, alpha=0.8, zorder=3)
        a.plot(dates, ml_pred[f], "s-", label="ML Prediction", linewidth=2, markersize=4, alpha=0.7, zorder=2)

        lo, hi = intervals[f"{f}_lower"], intervals[f"{f}_upper"]
        a.fill_between(dates, lo, hi, alpha=0.3, label="95% Confidence Interval", zorder=1)

        a.axhline(0, linestyle="--", linewidth=0.5, alpha=0.5)
        a.set(title=f"{f} - ML-Enhanced Monte Carlo", xlabel="Date", ylabel="Return")
        a.legend(loc="best", fontsize=9)
        a.grid(True, alpha=0.3)

        cov = float((((y_test[f] >= lo) & (y_test[f] <= hi)).mean() * 100.0))
        face = "lightgreen" if abs(cov - 95.0) < 10.0 else "lightyellow"
        a.text(
            0.02,
            0.98,
            f"Coverage: {cov:.1f}%\nTarget: 95.0%",
            transform=a.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor=face, alpha=0.7),
            fontsize=9,
        )

    ax[5].axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ml_enhanced_mc_intervals.png", dpi=300, bbox_inches="tight")
    print(f"ML-Enhanced intervals plot saved to: {save_dir}/ml_enhanced_mc_intervals.png")
    plt.close()


if __name__ == "__main__":
    print("Monte Carlo Factor Simulator with ML Enhancement")
    print("\nFeatures:")
    print("  1. Historical Mean Baseline")
    print("  2. ML Predictions")
    print("  3. ML-Enhanced Monte Carlo (ML + uncertainty quantification)")
