import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class HistoricalMeanBaseline:
    """
    Simple baseline that predicts using historical mean of each factor.
    """
    
    def __init__(self):
        self.means = {}
        self.factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    def fit(self, y_train):
        """Calculate historical mean for each factor."""
        for factor in self.factor_names:
            self.means[factor] = y_train[factor].mean()
        
        print("\n" + "="*80)
        print("HISTORICAL MEAN BASELINE - FITTED PARAMETERS")
        print("="*80)
        print("\nHistorical Mean (from training data):")
        print("-"*80)
        for factor in self.factor_names:
            print(f"{factor:10s}: μ = {self.means[factor]:+.4f} ({self.means[factor]*100:+.2f}%)")
    
    def predict(self, X_test):
        """Predict using historical means."""
        predictions = pd.DataFrame(index=X_test.index)
        for factor in self.factor_names:
            predictions[factor] = self.means[factor]
        return predictions


class MonteCarloFactorSimulator:
    """
    Monte Carlo simulation for Fama-French factors.
    Can operate in two modes:
    1. Historical baseline: Uses historical statistics (μ, σ, ρ)
    2. ML-enhanced: Uses ML predictions as means with historical volatility
    """
    
    def __init__(self, n_simulations=10000, random_seed=42):
        """
        Initialize Monte Carlo simulator.
        
        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo paths to simulate
        random_seed : int
            Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        self.means = {}
        self.stds = {}
        self.correlations = None
        self.is_ml_enhanced = False
        
    def fit(self, y_train):
        """
        Estimate historical statistics from training data.
        
        Parameters
        ----------
        y_train : pd.DataFrame
            Training factor returns
        """
        # Calculate mean and std for each factor
        for factor in self.factor_names:
            self.means[factor] = y_train[factor].mean()
            self.stds[factor] = y_train[factor].std()
        
        # Calculate correlation matrix
        self.correlations = y_train[self.factor_names].corr()
        
        print("\n" + "="*80)
        print("MONTE CARLO SIMULATOR - FITTED PARAMETERS")
        print("="*80)
        print("\nHistorical Statistics (from training data):")
        print("-"*80)
        for factor in self.factor_names:
            print(f"{factor:10s}: μ = {self.means[factor]:+.4f} ({self.means[factor]*100:+.2f}%), "
                  f"σ = {self.stds[factor]:.4f} ({self.stds[factor]*100:.2f}%)")
        
        print("\n" + "-"*80)
        print("Correlation Matrix:")
        print("-"*80)
        print(self.correlations.to_string())
    
    def simulate(self, n_periods, ml_predictions=None):
        """
        Generate Monte Carlo simulations.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate
        ml_predictions : pd.DataFrame, optional
            ML model predictions to use as means (ML-enhanced mode)
            If None, uses historical means (baseline mode)
        
        Returns
        -------
        simulations : dict
            Dictionary with simulated returns for each factor
        """
        np.random.seed(self.random_seed)
        
        # Use Cholesky decomposition to preserve correlations
        try:
            chol = np.linalg.cholesky(self.correlations.values)
        except np.linalg.LinAlgError:
            print("Warning: Correlation matrix not positive definite. Using uncorrelated simulations.")
            chol = np.eye(len(self.factor_names))
        
        # Generate independent standard normal samples
        # Shape: (n_simulations, n_periods, n_factors)
        independent_samples = np.random.randn(self.n_simulations, n_periods, len(self.factor_names))
        
        # Apply correlation structure
        correlated_samples = np.zeros_like(independent_samples)
        for sim in range(self.n_simulations):
            for period in range(n_periods):
                correlated_samples[sim, period, :] = chol @ independent_samples[sim, period, :]
        
        # Scale by std and add mean for each factor
        simulations = {}
        
        if ml_predictions is not None:
            # ML-enhanced mode: use ML predictions as means
            self.is_ml_enhanced = True
            for i, factor in enumerate(self.factor_names):
                # Use ML predictions as time-varying means
                ml_means = ml_predictions[factor].values.reshape(1, -1)  # Shape: (1, n_periods)
                
                simulations[factor] = (
                    ml_means +  # ML-predicted mean for each period
                    self.stds[factor] * correlated_samples[:, :, i]  # Add uncertainty
                )
        else:
            # Baseline mode: use historical means
            self.is_ml_enhanced = False
            for i, factor in enumerate(self.factor_names):
                simulations[factor] = (
                    self.means[factor] + 
                    self.stds[factor] * correlated_samples[:, :, i]
                )
        
        return simulations
    
    def predict(self, X_test, ml_predictions=None):
        """
        Generate predictions for test set.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features (only used for index alignment)
        ml_predictions : pd.DataFrame, optional
            ML model predictions to enhance (ML-enhanced mode)
            If None, uses historical means (baseline mode)
        
        Returns
        -------
        predictions : pd.DataFrame
            Mean of Monte Carlo simulations
        """
        n_periods = len(X_test)
        simulations = self.simulate(n_periods, ml_predictions)
        
        # Take mean across simulations
        predictions = pd.DataFrame(index=X_test.index)
        for factor in self.factor_names:
            predictions[factor] = np.mean(simulations[factor], axis=0)
        
        return predictions
    
    def get_prediction_intervals(self, X_test, ml_predictions=None, confidence_level=0.95):
        """
        Generate prediction intervals from Monte Carlo simulations.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        ml_predictions : pd.DataFrame, optional
            ML predictions to use as means
        confidence_level : float
            Confidence level (default: 0.95 for 95% CI)
        
        Returns
        -------
        intervals : pd.DataFrame
            Lower and upper bounds for each factor
        """
        n_periods = len(X_test)
        simulations = self.simulate(n_periods, ml_predictions)
        
        alpha = 1 - confidence_level
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100
        
        intervals = pd.DataFrame(index=X_test.index)
        
        for factor in self.factor_names:
            intervals[f'{factor}_mean'] = np.mean(simulations[factor], axis=0)
            intervals[f'{factor}_lower'] = np.percentile(simulations[factor], lower_pct, axis=0)
            intervals[f'{factor}_upper'] = np.percentile(simulations[factor], upper_pct, axis=0)
        
        return intervals
    
    def plot_simulation_distribution(self, factor, n_periods, ml_predictions=None,
                                     save_path="results/monte_carlo_distribution.png"):
        """
        Plot distribution of simulated factor returns.
        """
        simulations = self.simulate(n_periods, ml_predictions)
        factor_sims = simulations[factor]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        mode_str = "ML-Enhanced" if self.is_ml_enhanced else "Historical"
        
        # 1. Histogram of final period returns
        ax1 = axes[0, 0]
        final_returns = factor_sims[:, -1]
        ax1.hist(final_returns, bins=50, alpha=0.7, edgecolor='black')
        mean_val = np.mean(final_returns)
        ax1.axvline(mean_val, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_val*100:.2f}%')
        ax1.axvline(np.median(final_returns), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(final_returns)*100:.2f}%')
        ax1.set_xlabel('Return', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'{factor} - Distribution (Final Period) [{mode_str}]', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sample paths
        ax2 = axes[0, 1]
        n_paths = 100
        for i in range(n_paths):
            ax2.plot(range(n_periods), factor_sims[i, :], alpha=0.1, color='blue')
        mean_path = np.mean(factor_sims, axis=0)
        ax2.plot(range(n_periods), mean_path, color='red', linewidth=2, label='Mean Path')
        ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Period', fontsize=11)
        ax2.set_ylabel('Return', fontsize=11)
        ax2.set_title(f'{factor} - Sample Paths ({n_paths} of {self.n_simulations}) [{mode_str}]', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution evolution over time
        ax3 = axes[1, 0]
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(factor_sims, percentiles, axis=0)
        colors = ['red', 'orange', 'green', 'orange', 'red']
        labels = ['5th', '25th', '50th (Median)', '75th', '95th']
        
        for i, (pct, color, label) in enumerate(zip(percentile_values, colors, labels)):
            ax3.plot(range(n_periods), pct, color=color, label=f'{label} percentile', linewidth=2)
        
        ax3.fill_between(range(n_periods), percentile_values[0], percentile_values[-1], 
                        alpha=0.2, color='gray', label='5th-95th range')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Period', fontsize=11)
        ax3.set_ylabel('Return', fontsize=11)
        ax3.set_title(f'{factor} - Confidence Bands [{mode_str}]', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q plot (test normality assumption)
        ax4 = axes[1, 1]
        stats.probplot(final_returns, dist="norm", plot=ax4)
        ax4.set_title(f'{factor} - Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Monte Carlo distribution plot saved to: {save_path}")
        plt.close()


def compare_historical_mean_vs_ml(hist_mean_baseline, ml_predictor, X_test, y_test, save_dir="results"):
    """
    Compare Historical Mean baseline with ML predictions.
    
    Parameters
    ----------
    hist_mean_baseline : HistoricalMeanBaseline
        Fitted historical mean baseline
    ml_predictor : FactorPredictor
        Fitted ML model
    X_test : pd.DataFrame
        Test features
    y_test : pd.DataFrame
        Test targets
    save_dir : str
        Directory to save results
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison metrics
    """
    
    print("\n" + "="*80)
    print("COMPARISON: HISTORICAL MEAN vs ML PREDICTIONS")
    print("="*80)
    
    # Generate predictions
    hist_mean_pred = hist_mean_baseline.predict(X_test)
    ml_pred = ml_predictor.predict(X_test)
    
    print(f"\nHistorical Mean: predicting constant values (training average)")
    print(f"ML Model: using {ml_predictor.model_type} with {len(ml_predictor.feature_names)} features")
    
    # Calculate metrics
    results = []
    factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    for factor in factor_names:
        # Historical Mean metrics
        hm_mse = mean_squared_error(y_test[factor], hist_mean_pred[factor])
        hm_rmse = np.sqrt(hm_mse)
        hm_mae = mean_absolute_error(y_test[factor], hist_mean_pred[factor])
        hm_r2 = r2_score(y_test[factor], hist_mean_pred[factor])
        
        # ML metrics
        ml_mse = mean_squared_error(y_test[factor], ml_pred[factor])
        ml_rmse = np.sqrt(ml_mse)
        ml_mae = mean_absolute_error(y_test[factor], ml_pred[factor])
        ml_r2 = r2_score(y_test[factor], ml_pred[factor])
        
        # Calculate differences
        rmse_improvement = ((hm_rmse - ml_rmse) / hm_rmse) * 100
        r2_improvement = ml_r2 - hm_r2
        
        results.append({
            'Factor': factor,
            'Hist_Mean_RMSE': hm_rmse,
            'ML_RMSE': ml_rmse,
            'RMSE_Improvement_%': rmse_improvement,
            'Hist_Mean_R²': hm_r2,
            'ML_R²': ml_r2,
            'R²_Improvement': r2_improvement,
            'Hist_Mean_MAE': hm_mae,
            'ML_MAE': ml_mae
        })
    
    comparison_df = pd.DataFrame(results)
    
    # Print comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    print("\nR² Scores (higher is better):")
    print("-"*80)
    r2_table = comparison_df[['Factor', 'Hist_Mean_R²', 'ML_R²', 'R²_Improvement']].copy()
    r2_table['Winner'] = r2_table['R²_Improvement'].apply(lambda x: 'ML' if x > 0 else 'Hist' if x < 0 else 'Tie')
    print(r2_table.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
    
    print("\n" + "-"*80)
    print("RMSE Scores (lower is better):")
    print("-"*80)
    rmse_table = comparison_df[['Factor', 'Hist_Mean_RMSE', 'ML_RMSE', 'RMSE_Improvement_%']].copy()
    print(rmse_table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    avg_hm_r2 = comparison_df['Hist_Mean_R²'].mean()
    avg_ml_r2 = comparison_df['ML_R²'].mean()
    
    print(f"\nAverage R² across all factors:")
    print(f"  Historical Mean: {avg_hm_r2:+.4f}")
    print(f"  ML Model:        {avg_ml_r2:+.4f}")
    print(f"  Improvement:     {avg_ml_r2 - avg_hm_r2:+.4f}")
    
    ml_wins = (comparison_df['R²_Improvement'] > 0).sum()
    hm_wins = (comparison_df['R²_Improvement'] < 0).sum()
    ties = (comparison_df['R²_Improvement'] == 0).sum()
    
    print(f"\nWin/Loss Record (by R²):")
    print(f"  ML Model wins:        {ml_wins}/5")
    print(f"  Historical Mean wins: {hm_wins}/5")
    print(f"  Ties:                 {ties}/5")
    
    if ml_wins > hm_wins:
        print(f"\n  ✓ ML model provides value over simple historical mean!")
    elif ml_wins == hm_wins:
        print(f"\n  → ML and Historical Mean perform similarly")
    else:
        print(f"\n  ⚠ Historical Mean outperforms ML (possible overfitting)")
    
    # Create visualizations
    _plot_hist_vs_ml_comparison(comparison_df, y_test, hist_mean_pred, ml_pred, 
                                X_test.index, save_dir)
    
    # Save detailed results
    os.makedirs(save_dir, exist_ok=True)
    comparison_df.to_csv(f"{save_dir}/hist_mean_vs_ml_comparison.csv", index=False)
    print(f"\nDetailed comparison saved to: {save_dir}/hist_mean_vs_ml_comparison.csv")
    
    return comparison_df


def _plot_hist_vs_ml_comparison(comparison_df, y_test, hist_mean_pred, ml_pred, dates, save_dir):
    """Create comparison visualizations."""
    
    # Plot 1: Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² comparison
    ax1 = axes[0, 0]
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax1.bar(x - width/2, comparison_df['Hist_Mean_R²'], width, 
           label='Historical Mean', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, comparison_df['ML_R²'], width,
           label='ML Model', alpha=0.8, color='coral')
    
    ax1.set_xlabel('Factor', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² Comparison: Historical Mean vs ML', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['Factor'])
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # RMSE comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, comparison_df['Hist_Mean_RMSE'], width,
           label='Historical Mean', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, comparison_df['ML_RMSE'], width,
           label='ML Model', alpha=0.8, color='coral')
    
    ax2.set_xlabel('Factor', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('RMSE Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_df['Factor'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # R² Improvement
    ax3 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in comparison_df['R²_Improvement']]
    ax3.bar(x, comparison_df['R²_Improvement'], color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Factor', fontsize=12)
    ax3.set_ylabel('R² Improvement (ML - Hist Mean)', fontsize=12)
    ax3.set_title('ML Improvement over Historical Mean', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(comparison_df['Factor'])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Win rate summary
    ax4 = axes[1, 1]
    ml_wins = (comparison_df['R²_Improvement'] > 0).sum()
    hm_wins = (comparison_df['R²_Improvement'] < 0).sum()
    
    if ml_wins + hm_wins > 0:
        colors = ['coral', 'steelblue']
        sizes = [ml_wins, hm_wins]
        labels = [f'ML Model\n({ml_wins}/5)', f'Historical Mean\n({hm_wins}/5)']
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                            startangle=90, textprops={'fontsize': 12})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)
    else:
        ax4.text(0.5, 0.5, 'Perfect Tie\n(All factors identical)', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
    
    ax4.set_title('Win Rate by R² Score', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/hist_mean_vs_ml_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_dir}/hist_mean_vs_ml_comparison.png")
    plt.close()
    
    # Plot 2: Time series comparison
    _plot_timeseries(y_test, hist_mean_pred, ml_pred, dates, 
                    save_dir, "hist_mean_vs_ml_timeseries.png")


def _plot_timeseries(y_test, baseline_pred, ml_pred, dates, save_dir, filename):
    """Plot time series comparison."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    for i, factor in enumerate(factor_names):
        ax = axes[i]
        
        ax.plot(dates, y_test[factor], 'o-', label='Actual', 
               color='black', linewidth=2, markersize=4, alpha=0.7)
        ax.axhline(baseline_pred[factor].iloc[0], 
                  label='Historical Mean', color='steelblue', 
                  linestyle='--', linewidth=2, alpha=0.8)
        ax.plot(dates, ml_pred[factor], 's--', label='ML Model', 
               color='coral', linewidth=2, markersize=4, alpha=0.8)
        
        ax.set_title(f'{factor}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        hm_r2 = r2_score(y_test[factor], baseline_pred[factor])
        ml_r2 = r2_score(y_test[factor], ml_pred[factor])
        
        textstr = f'R² (HM): {hm_r2:.3f}\nR² (ML): {ml_r2:.3f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
    
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    print(f"Time series comparison saved to: {save_dir}/{filename}")
    plt.close()


def compare_ml_enhanced_monte_carlo(mc_simulator, ml_predictor, X_test, y_test, save_dir="results"):
    """
    Compare ML-enhanced Monte Carlo with regular ML predictions.
    
    Parameters
    ----------
    mc_simulator : MonteCarloFactorSimulator
        Fitted Monte Carlo simulator
    ml_predictor : FactorPredictor
        Fitted ML model
    X_test : pd.DataFrame
        Test features
    y_test : pd.DataFrame
        Test targets
    save_dir : str
        Directory to save results
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison metrics
    ml_enhanced_intervals : pd.DataFrame
        Prediction intervals from ML-enhanced MC
    """
    
    print("\n" + "="*80)
    print("ML-ENHANCED MONTE CARLO ANALYSIS")
    print("="*80)
    
    # Get ML predictions
    ml_pred = ml_predictor.predict(X_test)
    
    print("\nGenerating ML-Enhanced Monte Carlo predictions...")
    print(f"  • Using ML predictions as means")
    print(f"  • Adding historical volatility and correlations")
    print(f"  • Running {mc_simulator.n_simulations:,} simulations")
    
    # Get ML-enhanced predictions with intervals
    ml_enhanced_intervals = mc_simulator.get_prediction_intervals(
        X_test, 
        ml_predictions=ml_pred,
        confidence_level=0.95
    )
    
    # Compare metrics
    results = []
    factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    for factor in factor_names:
        # ML predictions (point estimates)
        ml_r2 = r2_score(y_test[factor], ml_pred[factor])
        ml_rmse = np.sqrt(mean_squared_error(y_test[factor], ml_pred[factor]))
        
        # ML-enhanced MC predictions (mean of simulations)
        ml_enh_pred = ml_enhanced_intervals[f'{factor}_mean']
        ml_enh_r2 = r2_score(y_test[factor], ml_enh_pred)
        ml_enh_rmse = np.sqrt(mean_squared_error(y_test[factor], ml_enh_pred))
        
        # Coverage: % of actuals within prediction interval
        lower = ml_enhanced_intervals[f'{factor}_lower']
        upper = ml_enhanced_intervals[f'{factor}_upper']
        within_interval = ((y_test[factor] >= lower) & (y_test[factor] <= upper))
        coverage = within_interval.mean() * 100
        
        # Average interval width
        avg_width = (upper - lower).mean()
        
        results.append({
            'Factor': factor,
            'ML_R²': ml_r2,
            'ML_Enhanced_R²': ml_enh_r2,
            'R²_Difference': ml_enh_r2 - ml_r2,
            'ML_RMSE': ml_rmse,
            'ML_Enhanced_RMSE': ml_enh_rmse,
            'Coverage_95%': coverage,
            'Avg_Interval_Width': avg_width
        })
    
    comparison_df = pd.DataFrame(results)
    
    # Print results
    print("\n" + "="*80)
    print("ML vs ML-ENHANCED MONTE CARLO")
    print("="*80)
    
    print("\nPrediction Accuracy:")
    print("-"*80)
    print(comparison_df[['Factor', 'ML_R²', 'ML_Enhanced_R²', 'R²_Difference']].to_string(
        index=False, float_format=lambda x: f'{x:+.4f}'))
    
    print("\n" + "-"*80)
    print("Uncertainty Quantification (ML-Enhanced only):")
    print("-"*80)
    print(comparison_df[['Factor', 'Coverage_95%', 'Avg_Interval_Width']].to_string(
        index=False, float_format=lambda x: f'{x:.2f}'))
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    avg_coverage = comparison_df['Coverage_95%'].mean()
    print(f"\n1. PREDICTION INTERVALS:")
    print(f"   • Average 95% coverage: {avg_coverage:.1f}%")
    if avg_coverage >= 90 and avg_coverage <= 100:
        print(f"   ✓ Well-calibrated intervals!")
    elif avg_coverage < 90:
        print(f"   ⚠ Intervals too narrow (underestimating uncertainty)")
    else:
        print(f"   ⚠ Intervals too wide (overestimating uncertainty)")
    
    print(f"\n2. ML-ENHANCED vs PURE ML:")
    print(f"   • ML-Enhanced uses ML predictions + historical volatility")
    print(f"   • Provides uncertainty quantification")
    print(f"   • Point predictions should be very similar to ML")
    
    avg_r2_diff = comparison_df['R²_Difference'].mean()
    print(f"   • Average R² difference: {avg_r2_diff:+.6f}")
    if abs(avg_r2_diff) < 0.001:
        print(f"   ✓ Point predictions nearly identical (as expected)")
    
    # Create visualizations
    _plot_ml_enhanced_intervals(y_test, ml_pred, ml_enhanced_intervals, X_test.index, save_dir)
    
    # Save results
    comparison_df.to_csv(f"{save_dir}/ml_enhanced_mc_comparison.csv", index=False)
    ml_enhanced_intervals.to_csv(f"{save_dir}/ml_enhanced_mc_intervals.csv")
    print(f"\nResults saved to:")
    print(f"  • {save_dir}/ml_enhanced_mc_comparison.csv")
    print(f"  • {save_dir}/ml_enhanced_mc_intervals.csv")
    
    return comparison_df, ml_enhanced_intervals


def _plot_ml_enhanced_intervals(y_test, ml_pred, ml_enhanced_intervals, dates, save_dir):
    """Plot ML-enhanced Monte Carlo prediction intervals."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    for i, factor in enumerate(factor_names):
        ax = axes[i]
        
        # Plot actual values
        ax.plot(dates, y_test[factor], 'o-', label='Actual', 
               color='black', linewidth=2, markersize=5, alpha=0.8, zorder=3)
        
        # Plot ML prediction (point estimate)
        ax.plot(dates, ml_pred[factor], 's-', label='ML Prediction', 
               color='coral', linewidth=2, markersize=4, alpha=0.7, zorder=2)
        
        # Plot prediction interval
        lower = ml_enhanced_intervals[f'{factor}_lower']
        upper = ml_enhanced_intervals[f'{factor}_upper']
        
        ax.fill_between(dates, lower, upper,
                       alpha=0.3, color='coral', label='95% Confidence Interval', zorder=1)
        
        ax.set_title(f'{factor} - ML-Enhanced Monte Carlo', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Calculate and display coverage
        within_interval = ((y_test[factor] >= lower) & (y_test[factor] <= upper))
        coverage = within_interval.mean() * 100
        
        textstr = f'Coverage: {coverage:.1f}%\n'
        textstr += f'Target: 95.0%'
        
        color = 'lightgreen' if abs(coverage - 95) < 10 else 'lightyellow'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
               fontsize=9)
    
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ml_enhanced_mc_intervals.png", dpi=300, bbox_inches='tight')
    print(f"ML-Enhanced intervals plot saved to: {save_dir}/ml_enhanced_mc_intervals.png")
    plt.close()


if __name__ == "__main__":
    print("Monte Carlo Factor Simulator with ML Enhancement")
    print("\nFeatures:")
    print("  1. Historical Mean Baseline")
    print("  2. ML Predictions")
    print("  3. ML-Enhanced Monte Carlo (ML + uncertainty quantification)")