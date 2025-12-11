import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from src.factor_predictor import FactorPredictor
from src.monte_carlo import MonteCarloFactorSimulator


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a single model.
    
    Parameters
    ----------
    model : FactorPredictor or MonteCarloFactorSimulator
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.DataFrame
        True target values
    model_name : str
        Name for display
    
    Returns
    -------
    metrics : pd.DataFrame
        Performance metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = []
    target_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    for factor in target_names:
        mse = mean_squared_error(y_test[factor], y_pred[factor])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[factor], y_pred[factor])
        r2 = r2_score(y_test[factor], y_pred[factor])
        
        metrics.append({
            'Model': model_name,
            'Factor': factor,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
    
    return pd.DataFrame(metrics)


def compare_all_models(X_train, X_val, X_test, y_train, y_val, y_test, 
                       dataset_path="data/processed/factor_ml_dataset_enhanced.csv"):
    """
    Compare all ML models (no baseline comparison here).
    
    Returns
    -------
    results : pd.DataFrame
        Comparison results for all models
    best_model : FactorPredictor
        Best performing model
    """
    
    print("="*80)
    print("ML MODELS COMPARISON")
    print("="*80)
    
    models_to_test = {
        'Random Forest': 'random_forest',
        'Gradient Boosting': 'gradient_boosting',
        'Ridge Regression': 'ridge',
        'Lasso Regression': 'lasso'
    }
    
    all_results = []
    trained_models = {}
    
    # Evaluate ML Models
    for i, (model_name, model_type) in enumerate(models_to_test.items(), start=1):
        print(f"\n[{i}/4] Training and evaluating {model_name}...")
        
        predictor = FactorPredictor(model_type=model_type)
        
        # Use our already-split data
        predictor.feature_names = X_train.columns.tolist()
        predictor.scaler.fit(X_train)
        
        # Fit model
        predictor.fit(X_train, y_train)
        
        # Evaluate on test set
        metrics = evaluate_model(predictor, X_test, y_test, model_name)
        all_results.append(metrics)
        
        # Store trained model
        trained_models[model_name] = predictor
    
    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Find best model
    avg_r2_by_model = results_df.groupby('Model')['R²'].mean()
    best_model_name = avg_r2_by_model.idxmax()
    best_model = trained_models[best_model_name]
    
    print(f"\n✓ Best model: {best_model_name} (Avg R² = {avg_r2_by_model[best_model_name]:+.4f})")
    
    return results_df, best_model


def print_comparison_table(results_df):
    """
    Print formatted comparison table.
    """
    print("\n" + "="*80)
    print("ML MODELS COMPARISON - R² SCORES ON TEST SET")
    print("="*80)
    
    # Pivot to show R² for each model-factor combination
    r2_pivot = results_df.pivot(index='Factor', columns='Model', values='R²')
    
    print("\nR² Scores (higher is better):")
    print(r2_pivot.to_string(float_format=lambda x: f'{x:+.4f}'))
    
    # Show best model for each factor
    print("\n" + "-"*80)
    print("BEST MODEL FOR EACH FACTOR:")
    print("-"*80)
    
    for factor in r2_pivot.index:
        best_model = r2_pivot.loc[factor].idxmax()
        best_r2 = r2_pivot.loc[factor].max()
        print(f"{factor:10s}: {best_model:30s} (R² = {best_r2:+.4f})")
    
    # Overall statistics
    print("\n" + "-"*80)
    print("AVERAGE PERFORMANCE ACROSS ALL FACTORS:")
    print("-"*80)
    avg_r2 = r2_pivot.mean()
    avg_r2_sorted = avg_r2.sort_values(ascending=False)
    for model, score in avg_r2_sorted.items():
        print(f"{model:30s}: {score:+.4f}")


def plot_model_comparison(results_df, save_path="results/model_comparison.png"):
    """
    Create visualization comparing all models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare data
    r2_pivot = results_df.pivot(index='Factor', columns='Model', values='R²')
    
    # Reorder columns
    if 'Monte Carlo Simulation' in r2_pivot.columns:
        cols = ['Monte Carlo Simulation'] + [c for c in r2_pivot.columns if c != 'Monte Carlo Simulation']
        r2_pivot = r2_pivot[cols]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Grouped bar chart
    ax1 = axes[0]
    r2_pivot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('R² Scores by Factor and Model', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Factor', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    
    # Plot 2: Heatmap
    ax2 = axes[1]
    sns.heatmap(r2_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'R² Score'}, ax=ax2, linewidths=0.5)
    ax2.set_title('R² Scores Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Factor', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def plot_mc_prediction_intervals(mc_simulator, X_test, y_test, 
                                 save_path="results/mc_prediction_intervals.png"):
    """
    Plot Monte Carlo prediction intervals alongside actual values.
    """
    print("\nGenerating Monte Carlo prediction intervals...")
    intervals = mc_simulator.get_prediction_intervals(X_test, confidence_level=0.95)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, factor in enumerate(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']):
        ax = axes[i]
        
        dates = X_test.index
        
        # Plot actual values
        ax.plot(dates, y_test[factor], 'o-', label='Actual', 
               color='black', linewidth=2, markersize=4)
        
        # Plot mean prediction
        ax.plot(dates, intervals[f'{factor}_mean'], '--', 
               label='MC Mean', color='blue', linewidth=2, alpha=0.7)
        
        # Plot prediction interval
        ax.fill_between(dates, 
                       intervals[f'{factor}_lower'],
                       intervals[f'{factor}_upper'],
                       alpha=0.3, color='blue', label='95% CI')
        
        ax.set_title(f'{factor} - Monte Carlo Predictions', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        
        # Calculate coverage (% of actual values within prediction interval)
        within_interval = ((y_test[factor] >= intervals[f'{factor}_lower']) & 
                          (y_test[factor] <= intervals[f'{factor}_upper']))
        coverage = within_interval.mean() * 100
        
        ax.text(0.02, 0.98, f'Coverage: {coverage:.1f}%', 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Monte Carlo prediction intervals saved to: {save_path}")
    plt.close()


def save_detailed_results(results_df, save_path="results/model_comparison_detailed.csv"):
    """
    Save detailed comparison results to CSV.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"Detailed results saved to: {save_path}")


def evaluate_all_models(dataset_path="data/processed/factor_ml_dataset_enhanced.csv"):
    """
    Complete evaluation pipeline: compare all ML models.
    
    Parameters
    ----------
    dataset_path : str
        Path to the enhanced dataset
    
    Returns
    -------
    results_df : pd.DataFrame
        Full comparison results
    best_model : FactorPredictor
        Best performing ML model
    X_train, X_val, X_test, y_train, y_val, y_test : DataFrames
        Split data for further analysis
    """
    
    print("\n" + "="*80)
    print("STEP 5: ML MODELS EVALUATION")
    print("="*80)
    
    # Load data using FactorPredictor
    temp_predictor = FactorPredictor()
    X, y, dates = temp_predictor.prepare_data(dataset_path)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test = \
        temp_predictor.train_val_test_split_temporal(X, y, dates, train_ratio=0.7, val_ratio=0.2)
    
    # Compare all models
    results_df, best_model = compare_all_models(X_train, X_val, X_test, y_train, y_val, y_test, dataset_path)
    
    # Print comparison table
    print_comparison_table(results_df)
    
    # Create visualizations
    plot_model_comparison(results_df)
    
    # Save detailed results
    save_detailed_results(results_df)
    
    return results_df, best_model, X_train, X_val, X_test, y_train, y_val, y_test


def compare_mc_to_best_ml(results_df, mc_simulator, X_test, y_test):
    """
    Detailed comparison of Monte Carlo vs best ML model.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from evaluate_all_models
    mc_simulator : MonteCarloFactorSimulator
        Fitted Monte Carlo simulator
    X_test : pd.DataFrame
        Test features
    y_test : pd.DataFrame
        Test targets
    """
    
    print("\n" + "="*80)
    print("MONTE CARLO vs BEST ML MODEL - DETAILED COMPARISON")
    print("="*80)
    
    # Find best model for each factor
    r2_pivot = results_df.pivot(index='Factor', columns='Model', values='R²')
    
    # Exclude Monte Carlo from "best model" consideration
    ml_models = [c for c in r2_pivot.columns if c != 'Monte Carlo Simulation']
    best_models = r2_pivot[ml_models].idxmax(axis=1)
    
    print("\nBest ML model for each factor:")
    print("-"*80)
    for factor, model in best_models.items():
        mc_score = r2_pivot.loc[factor, 'Monte Carlo Simulation']
        ml_score = r2_pivot.loc[factor, model]
        improvement = ml_score - mc_score
        print(f"{factor:10s}: {model:25s} | MC: {mc_score:+.4f} → ML: {ml_score:+.4f} "
              f"(Δ = {improvement:+.4f})")
    
    # Overall comparison
    print("\n" + "-"*80)
    print("Overall Performance:")
    print("-"*80)
    
    mc_avg = r2_pivot['Monte Carlo Simulation'].mean()
    ml_avg = r2_pivot[ml_models].max(axis=1).mean()
    
    print(f"Monte Carlo average R²:  {mc_avg:+.4f}")
    print(f"Best ML average R²:      {ml_avg:+.4f}")
    print(f"Average improvement:     {ml_avg - mc_avg:+.4f}")
    
    # Win rate
    wins = (r2_pivot[ml_models].max(axis=1) > r2_pivot['Monte Carlo Simulation']).sum()
    print(f"\nML wins on {wins}/5 factors ({wins/5*100:.0f}%)")


if __name__ == "__main__":
    # Run complete evaluation
    results_df, mc_simulator = evaluate_all_models()