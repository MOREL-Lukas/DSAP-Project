import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class FactorPredictor:
    """
    ML wrapper that predicts Fama-French factors in a way designed to be
    out-of-sample stable, so its forecasts can be safely used as a modest
    overlay on top of classical factor models.
    """

    def __init__(self, model_type="random_forest"):
        """
        Initialize the predictor.

        Parameters
        ----------
        model_type : str
            One of: 'random_forest', 'gradient_boosting', 'ridge', 'lasso'
        """
        self.model_type = model_type
        self.models = {}  # Separate model per factor
        self.scaler = StandardScaler()  # Feature scaling
        self.feature_names = None
        self.target_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    def _create_model(self):
        """Create a fresh model instance with conservative, OOS-stable defaults."""

        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=600,  # variance reduction
                max_depth=3,  # strong regularization
                min_samples_leaf=30,  # critical for macro data
                min_samples_split=40,
                max_features=0.5,  # decorrelate trees
                random_state=42,  # reproducibility
                n_jobs=-1,
            )

        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.03,  # slow learning = stability
                max_depth=2,
                min_samples_leaf=20,
                subsample=0.7,  # stochastic boosting
                random_state=42,  # reproducibility
            )

        elif self.model_type == "ridge":
            return Ridge(
                alpha=10.0, random_state=42  # stronger shrinkage  # reproducibility
            )

        elif self.model_type == "lasso":
            return Lasso(
                alpha=0.001,  # avoid instability
                max_iter=20000,
                random_state=42,  # reproducibility
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def prepare_data(self, dataset_path="data/processed/factor_ml_dataset.csv"):
        """
        Load the engineered factor dataset and enforce a consistent split
        between features and targets so different model types can be
        compared on an identical information set.
        """
        df = pd.read_csv(dataset_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()

        # Separate features from targets
        feature_cols = [col for col in df.columns if col not in self.target_names]
        X = df[feature_cols].copy()
        y = df[self.target_names].copy()
        self.feature_names = X.columns.tolist()

        print(f"Dataset shape: {df.shape}")
        print(f"Features: {X.shape[1]}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total months: {len(df)}")

        return X, y, df.index

    def train_val_test_split_temporal(
        self, X, y, dates, train_ratio=0.7, val_ratio=0.2
    ):
        """
        Perform a strictly chronological train/val/test split to avoid
        look-ahead bias and mimic a realistic forecasting workflow.
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        dates_train = dates[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        dates_val = dates[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]
        dates_test = dates[val_end:]

        print(
            f"\nTrain period: {dates_train[0]} to {dates_train[-1]} ({len(X_train)} months, {len(X_train)/n*100:.1f}%)"
        )
        print(
            f"Val period:   {dates_val[0]} to {dates_val[-1]} ({len(X_val)} months, {len(X_val)/n*100:.1f}%)"
        )
        print(
            f"Test period:  {dates_test[0]} to {dates_test[-1]} ({len(X_test)} months, {len(X_test)/n*100:.1f}%)"
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            dates_train,
            dates_val,
            dates_test,
        )

    def fit(self, X_train, y_train):
        """
        Fit one model per factor so each risk premium can have its own
        mapping from predictors, avoiding cross-factor interference in a
        multi-output setup.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"\nTraining {self.model_type} models...")

        for factor in self.target_names:
            model = self._create_model()
            model.fit(X_train_scaled, y_train[factor])
            self.models[factor] = model

    def predict(self, X):
        """Predict all factors for new data."""
        X_scaled = self.scaler.transform(X)
        predictions = pd.DataFrame(index=X.index)
        for factor in self.target_names:
            predictions[factor] = self.models[factor].predict(X_scaled)
        return predictions

    def evaluate(self, X_test, y_test, dataset_name="Test"):
        """
        Report factor-wise error metrics so we can see which premia are
        genuinely predictable and which behave like noise.
        """
        y_pred = self.predict(X_test)
        metrics = []
        for factor in self.target_names:
            mse = mean_squared_error(
                y_test[factor], y_pred[factor]
            )  # Mean Squared Error
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            mae = mean_absolute_error(
                y_test[factor], y_pred[factor]
            )  # Mean Absolute Error
            r2 = r2_score(
                y_test[factor], y_pred[factor]
            )  # Coefficient of determination
            metrics.append({"Factor": factor, "RMSE": rmse, "MAE": mae, "R²": r2})

        df_metrics = pd.DataFrame(metrics)
        df_metrics["Dataset"] = dataset_name

        return df_metrics

    def feature_importance(self, top_n=10):
        """Get feature importance for tree-based models."""
        if self.model_type not in ["random_forest", "gradient_boosting"]:
            print("Feature importance only available for tree-based models")
            return None

        importance_data = {}
        for factor in self.target_names:
            importance = self.models[factor].feature_importances_
            importance_data[factor] = importance

        importance_df = pd.DataFrame(importance_data, index=self.feature_names)

        print(f"\nTop {top_n} Features by Factor:")
        print("=" * 80)

        for factor in self.target_names:
            top_features = importance_df[factor].nlargest(top_n)
            print(f"\n{factor}:")
            for feat, imp in top_features.items():
                print(f"  {feat:30s} {imp:.4f}")

        return importance_df

    def plot_predictions(self, dates_test, y_test, y_pred):
        """Plot actual vs predicted values."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        for i, factor in enumerate(self.target_names):
            ax = axes[i]
            ax.plot(dates_test, y_test[factor], label="Actual", linewidth=2)
            ax.plot(
                dates_test, y_pred[factor], label="Predicted", linewidth=2, alpha=0.7
            )
            ax.set_title(
                f"{factor} - Actual vs Predicted", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Return")
            ax.legend()
            ax.grid(True, alpha=0.3)
            corr = np.corrcoef(y_test[factor], y_pred[factor])[0, 1]
            ax.text(
                0.02,
                0.98,
                f"Corr: {corr:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        axes[5].axis("off")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/factor_predictions.png", dpi=300, bbox_inches="tight")
        print("\nPlot saved to: results/factor_predictions.png")
        plt.show()

    def predict_next_month(self, current_features):
        """
        Generate a single-step-ahead factor forecast using the latest
        available feature row, which then feeds into Monte Carlo and
        portfolio construction as a forward-looking overlay.
        """
        if isinstance(current_features, dict):
            current_features = pd.Series(current_features)
        missing = set(self.feature_names) - set(current_features.index)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = pd.DataFrame([current_features[self.feature_names]])
        predictions = self.predict(X)

        return predictions.iloc[0]


def evaluate_model(model, X, y, model_name="Model", split="Test"):
    """Evaluate a single model."""
    y_pred = model.predict(X)
    metrics = []
    target_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    for factor in target_names:
        mse = mean_squared_error(y[factor], y_pred[factor])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y[factor], y_pred[factor])
        r2 = r2_score(y[factor], y_pred[factor])
        metrics.append(
            {
                "Model": model_name,
                "Split": split,
                "Factor": factor,
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2,
            }
        )

    return pd.DataFrame(metrics)


def compare_all_models(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    dataset_path="data/processed/factor_ml_dataset_enhanced.csv",
):
    """Compare all ML models and select the best model on Validation R² (mean across factors)."""

    print("=" * 80)
    print("ML MODELS COMPARISON")
    print("=" * 80)

    models_to_test = {
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "Ridge Regression": "ridge",
        "Lasso Regression": "lasso",
    }

    all_results = []
    trained_models = {}

    for i, (model_name, model_type) in enumerate(models_to_test.items(), start=1):
        print(f"\n[{i}/{len(models_to_test)}] Training and evaluating {model_name}...")

        try:
            predictor = FactorPredictor(model_type=model_type)

            # Ensure scaler + model are fitted consistently
            predictor.feature_names = X_train.columns.tolist()
            predictor.scaler.fit(X_train)
            predictor.fit(X_train, y_train)

            metrics_val = evaluate_model(
                predictor, X_val, y_val, model_name=model_name, split="Val"
            )
            metrics_test = evaluate_model(
                predictor, X_test, y_test, model_name=model_name, split="Test"
            )

            all_results.append(
                pd.concat([metrics_val, metrics_test], ignore_index=True)
            )
            trained_models[model_name] = predictor

            val_r2 = metrics_val["R²"].mean()
            test_r2 = metrics_test["R²"].mean()
            print(f"  ✓ OK | mean Val R²={val_r2:+.4f} | mean Test R²={test_r2:+.4f}")

        except Exception as e:
            print(f"Warning: {model_name} failed: {e}")
            continue

    if not all_results or not trained_models:
        raise RuntimeError(
            "compare_all_models(): No models were successfully trained/evaluated. "
            "Check the warnings above for the root cause (data shape, NaNs, scaler, etc.)."
        )

    results_df = pd.concat(all_results, ignore_index=True)

    # ---- Select best on Validation only ----
    if "Split" not in results_df.columns:
        raise RuntimeError(
            "compare_all_models(): results_df missing required column 'Split'."
        )

    val_df = results_df[results_df["Split"] == "Val"].copy()
    if val_df.empty:
        raise RuntimeError(
            "compare_all_models(): Validation split produced no rows; cannot select best model."
        )

    avg_r2_by_model = val_df.groupby("Model")["R²"].mean()

    # Keep only models that actually exist in trained_models (defensive)
    avg_r2_by_model = avg_r2_by_model[avg_r2_by_model.index.isin(trained_models.keys())]
    if avg_r2_by_model.empty:
        raise RuntimeError(
            "compare_all_models(): No validation R² entries matched trained_models keys. "
            "Possible mismatch between 'Model' labels and trained_models dict keys."
        )

    best_model_name = avg_r2_by_model.idxmax()
    best_model = trained_models[best_model_name]
    print(
        f"\n✓ Best model on average (Val): {best_model_name} (Avg Val R² = {avg_r2_by_model[best_model_name]:+.4f})"
    )

    return results_df, best_model, trained_models


def print_comparison_table(results_df, split="Test"):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print(f"ML MODELS COMPARISON - R² SCORES ON {split.upper()} SET")
    print("=" * 80)

    df = results_df.copy()
    if "Split" in df.columns:
        df = df[df["Split"] == split].copy()
    r2_pivot = df.pivot(index="Factor", columns="Model", values="R²")

    print("\nR² Scores (higher is better):")
    print(r2_pivot.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\n" + "-" * 80)
    print("BEST MODEL FOR EACH FACTOR:")
    print("-" * 80)

    for factor in r2_pivot.index:
        best_model = r2_pivot.loc[factor].idxmax()
        best_r2 = r2_pivot.loc[factor].max()
        print(f"{factor:10s}: {best_model:30s} (R² = {best_r2:+.4f})")

    print("\n" + "-" * 80)
    print("AVERAGE PERFORMANCE ACROSS ALL FACTORS:")
    print("-" * 80)
    avg_r2 = r2_pivot.mean().sort_values(ascending=False)
    for model, score in avg_r2.items():
        print(f"{model:30s}: {score:+.4f}")


def plot_model_comparison(
    results_df, split="Test", save_path="results/model_comparison.png"
):
    """Create visualization comparing all models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = results_df.copy()
    if "Split" in df.columns:
        df = df[df["Split"] == split].copy()

    r2_pivot = df.pivot(index="Factor", columns="Model", values="R²")

    if "Monte Carlo Simulation" in r2_pivot.columns:
        cols = ["Monte Carlo Simulation"] + [
            c for c in r2_pivot.columns if c != "Monte Carlo Simulation"
        ]
        r2_pivot = r2_pivot[cols]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1 = axes[0]
    r2_pivot.plot(kind="bar", ax=ax1, width=0.8)
    ax1.set_title(
        f"R² Scores by Factor and Model ({split})", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Factor", fontsize=12)
    ax1.set_ylabel("R² Score", fontsize=12)
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.3)
    ax1.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

    # Heatmap
    ax2 = axes[1]
    sns.heatmap(
        r2_pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "R² Score"},
        ax=ax2,
        linewidths=0.5,
    )
    ax2.set_title(f"R² Scores Heatmap ({split})", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Factor", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n{split} set comparison plot saved to: {save_path}")
    plt.close()


def save_detailed_results(
    results_df, save_path="results/model_comparison_detailed.csv"
):
    """Save detailed comparison results to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"Detailed results saved to: {save_path}")


def evaluate_all_models(dataset_path="data/processed/factor_ml_dataset_enhanced.csv"):
    """
    Run the full model-comparison workflow so different algorithms are
    evaluated on a common temporal split and can be ranked consistently
    for downstream selection.
    """
    temp_predictor = FactorPredictor()
    X, y, dates = temp_predictor.prepare_data(dataset_path)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        dates_train,
        dates_val,
        dates_test,
    ) = temp_predictor.train_val_test_split_temporal(
        X, y, dates, train_ratio=0.7, val_ratio=0.2
    )
    results_df, best_model, trained_models = compare_all_models(
        X_train, X_val, X_test, y_train, y_val, y_test, dataset_path
    )
    print_comparison_table(results_df)

    plot_model_comparison(
        results_df, split="Test", save_path="results/model_comparison.png"
    )
    plot_model_comparison(
        results_df, split="Val", save_path="results/model_comparison_val.png"
    )
    save_detailed_results(results_df)
    return (
        results_df,
        best_model,
        trained_models,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def compare_mc_to_best_ml(results_df, split="Test"):
    """Detailed comparison of Monte Carlo vs best ML model."""
    print("\n" + "=" * 80)
    print(f"MONTE CARLO vs BEST ML MODEL - DETAILED COMPARISON ({split.upper()})")
    print("=" * 80)
    df = results_df.copy()
    if "Split" in df.columns:
        df = df[df["Split"] == split].copy()

    r2_pivot = df.pivot(index="Factor", columns="Model", values="R²")

    if "Monte Carlo Simulation" not in r2_pivot.columns:
        print("Monte Carlo Simulation not found in results; skipping MC comparison.")
        return

    ml_models = [c for c in r2_pivot.columns if c != "Monte Carlo Simulation"]
    best_models = r2_pivot[ml_models].idxmax(axis=1)

    print("\nBest ML model for each factor:")
    print("-" * 80)
    for factor, model in best_models.items():
        mc_score = float(r2_pivot.loc[factor, "Monte Carlo Simulation"])
        ml_score = float(r2_pivot.loc[factor, model])
        improvement = ml_score - mc_score
        print(
            f"{factor:10s}: {model:25s} | MC: {mc_score:+.4f} → ML: {ml_score:+.4f} (Δ = {improvement:+.4f})"
        )

    print("\n" + "-" * 80)
    print("Overall Performance:")
    print("-" * 80)
    mc_avg = float(r2_pivot["Monte Carlo Simulation"].mean())
    ml_avg = float(r2_pivot[ml_models].max(axis=1).mean())
    print(f"Monte Carlo average R²:  {mc_avg:+.4f}")
    print(f"Best ML average R²:      {ml_avg:+.4f}")
    print(f"Average improvement:     {ml_avg - mc_avg:+.4f}")
    wins = int(
        (r2_pivot[ml_models].max(axis=1) > r2_pivot["Monte Carlo Simulation"]).sum()
    )
    print(f"\nML wins on {wins}/5 factors ({wins/5*100:.0f}%)")
