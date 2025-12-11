import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class FactorPredictor:
    """
    Machine Learning model to predict Fama-French 5 factors.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor.
        
        Parameters
        ----------
        model_type : str
            One of: 'random_forest', 'gradient_boosting', 'ridge', 'lasso'
        """
        self.model_type = model_type
        self.models = {}  # One model per factor
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        
    def _create_model(self):
        """Create a fresh model instance."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'lasso':
            return Lasso(alpha=0.01, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def prepare_data(self, dataset_path="data/processed/factor_ml_dataset.csv"):
        """
        Load and prepare the dataset.
        
        Returns
        -------
        X : pd.DataFrame
            Feature matrix
        y : pd.DataFrame
            Target factors
        dates : pd.DatetimeIndex
            Date index
        """
        df = pd.read_csv(dataset_path, parse_dates=['Date'])
        df = df.set_index('Date').sort_index()
        
        # Separate features from targets
        feature_cols = [col for col in df.columns if col not in self.target_names]
        
        X = df[feature_cols].copy()
        y = df[self.target_names].copy()
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {X.shape[1]}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total months: {len(df)}")
        
        return X, y, df.index
    
    def train_val_test_split_temporal(self, X, y, dates, train_ratio=0.7, val_ratio=0.2):
        """
        Split data temporally into train/validation/test (no shuffling to preserve time series structure).
        
        Parameters
        ----------
        train_ratio : float
            Proportion of data to use for training (default 0.7)
        val_ratio : float
            Proportion of data to use for validation (default 0.2)
            Test ratio is automatically 1 - train_ratio - val_ratio
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Training set
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        dates_train = dates[:train_end]
        
        # Validation set
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        dates_val = dates[train_end:val_end]
        
        # Test set
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]
        dates_test = dates[val_end:]
        
        print(f"\nTrain period: {dates_train[0]} to {dates_train[-1]} ({len(X_train)} months, {len(X_train)/n*100:.1f}%)")
        print(f"Val period:   {dates_val[0]} to {dates_val[-1]} ({len(X_val)} months, {len(X_val)/n*100:.1f}%)")
        print(f"Test period:  {dates_test[0]} to {dates_test[-1]} ({len(X_test)} months, {len(X_test)/n*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test
    
    def fit(self, X_train, y_train):
        """
        Train separate models for each factor.
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"\nTraining {self.model_type} models...")
        
        for factor in self.target_names:
            print(f"  Training {factor}...", end=' ')
            
            model = self._create_model()
            model.fit(X_train_scaled, y_train[factor])
            self.models[factor] = model
            
            print("✓")
        
        print("Training complete!")
    
    def predict(self, X):
        """
        Predict all factors for new data.
        
        Returns
        -------
        predictions : pd.DataFrame
            Predicted factor values
        """
        X_scaled = self.scaler.transform(X)
        
        predictions = pd.DataFrame(index=X.index)
        
        for factor in self.target_names:
            predictions[factor] = self.models[factor].predict(X_scaled)
        
        return predictions
    
    def evaluate(self, X_test, y_test, dataset_name="Test"):
        """
        Evaluate model performance on a dataset.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Features
        y_test : pd.DataFrame
            True target values
        dataset_name : str
            Name of dataset for display (e.g., "Validation", "Test")
        
        Returns
        -------
        metrics : pd.DataFrame
            Performance metrics for each factor
        """
        y_pred = self.predict(X_test)
        
        metrics = []
        
        for factor in self.target_names:
            mse = mean_squared_error(y_test[factor], y_pred[factor])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[factor], y_pred[factor])
            r2 = r2_score(y_test[factor], y_pred[factor])
            
            metrics.append({
                'Factor': factor,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            })
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics['Dataset'] = dataset_name
        
        return df_metrics
    
    def feature_importance(self, top_n=10):
        """
        Get feature importance for tree-based models.
        
        Returns
        -------
        importance_df : pd.DataFrame
            Feature importance for each factor
        """
        if self.model_type not in ['random_forest', 'gradient_boosting']:
            print("Feature importance only available for tree-based models")
            return None
        
        importance_data = {}
        
        for factor in self.target_names:
            importance = self.models[factor].feature_importances_
            importance_data[factor] = importance
        
        importance_df = pd.DataFrame(
            importance_data,
            index=self.feature_names
        )
        
        # Get top features for each factor
        print(f"\nTop {top_n} Features by Factor:")
        print("=" * 80)
        
        for factor in self.target_names:
            top_features = importance_df[factor].nlargest(top_n)
            print(f"\n{factor}:")
            for feat, imp in top_features.items():
                print(f"  {feat:30s} {imp:.4f}")
        
        return importance_df
    
    def plot_predictions(self, dates_test, y_test, y_pred):
        """
        Plot actual vs predicted values for all factors.
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, factor in enumerate(self.target_names):
            ax = axes[i]
            
            ax.plot(dates_test, y_test[factor], label='Actual', linewidth=2)
            ax.plot(dates_test, y_pred[factor], label='Predicted', 
                   linewidth=2, alpha=0.7)
            
            ax.set_title(f'{factor} - Actual vs Predicted', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add correlation in title
            corr = np.corrcoef(y_test[factor], y_pred[factor])[0, 1]
            ax.text(0.02, 0.98, f'Corr: {corr:.3f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide the 6th subplot
        axes[5].axis('off')
        
        plt.tight_layout()
        
        # Save to results folder
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/factor_predictions.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved to: results/factor_predictions.png")
        plt.show()
    
    def predict_next_month(self, current_features):
        """
        Predict factors for the next month given current features.
        
        Parameters
        ----------
        current_features : pd.Series or dict
            Current month's feature values
        
        Returns
        -------
        predictions : pd.Series
            Predicted factor values for next month
        """
        if isinstance(current_features, dict):
            current_features = pd.Series(current_features)
        
        # Ensure all features are present
        missing = set(self.feature_names) - set(current_features.index)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        X = pd.DataFrame([current_features[self.feature_names]])
        predictions = self.predict(X)
        
        return predictions.iloc[0]