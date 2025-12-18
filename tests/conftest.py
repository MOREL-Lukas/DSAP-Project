"""
Pytest fixtures and test utilities for DSAP-Project.

This module provides reusable test fixtures for creating synthetic data,
mock objects, and common test utilities.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


# =============================================================================
# Synthetic Data Generators
# =============================================================================

@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """Generate synthetic monthly stock returns for testing."""
    np.random.seed(42)
    n_months = 120
    n_stocks = 50
    
    dates = pd.date_range(start='2015-01-01', periods=n_months, freq='MS')
    tickers = [f'STOCK{i:03d}' for i in range(n_stocks)]
    
    # Generate returns with realistic properties
    returns = np.random.randn(n_months, n_stocks) * 0.05 + 0.01
    
    df = pd.DataFrame(returns, index=dates, columns=tickers)
    df.index.name = 'Date'
    
    return df


@pytest.fixture
def synthetic_factors() -> pd.DataFrame:
    """Generate synthetic Fama-French 5 factors for testing."""
    np.random.seed(42)
    n_months = 120
    
    dates = pd.date_range(start='2015-01-01', periods=n_months, freq='MS')
    
    data = {
        'Mkt-RF': np.random.randn(n_months) * 0.04 + 0.008,
        'SMB': np.random.randn(n_months) * 0.025 + 0.002,
        'HML': np.random.randn(n_months) * 0.025 + 0.003,
        'RMW': np.random.randn(n_months) * 0.02 + 0.004,
        'CMA': np.random.randn(n_months) * 0.02 + 0.001,
        'RF': np.ones(n_months) * 0.002,  # Risk-free rate
    }
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    
    return df


@pytest.fixture
def synthetic_betas() -> pd.DataFrame:
    """Generate synthetic FF5 betas for testing."""
    np.random.seed(42)
    n_stocks = 50
    
    tickers = [f'STOCK{i:03d}' for i in range(n_stocks)]
    
    data = {
        'Ticker': tickers,
        'Alpha': np.random.randn(n_stocks) * 0.002,
        'Beta_MKT': np.random.randn(n_stocks) * 0.3 + 1.0,
        'Beta_SMB': np.random.randn(n_stocks) * 0.5,
        'Beta_HML': np.random.randn(n_stocks) * 0.5,
        'Beta_RMW': np.random.randn(n_stocks) * 0.4,
        'Beta_CMA': np.random.randn(n_stocks) * 0.4,
        'R_squared': np.random.uniform(0.3, 0.8, n_stocks),
        'Adj_R_squared': np.random.uniform(0.25, 0.75, n_stocks),
        'ResidVar': np.random.uniform(0.001, 0.01, n_stocks),
        'N_obs': np.full(n_stocks, 120),
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('Ticker')
    
    return df


@pytest.fixture
def synthetic_ml_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Generate synthetic ML dataset with features and targets."""
    np.random.seed(42)
    n_months = 100
    n_features = 20
    
    dates = pd.date_range(start='2015-01-01', periods=n_months, freq='MS')
    
    # Features
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(
        np.random.randn(n_months, n_features),
        index=dates,
        columns=feature_names
    )
    
    # Targets (5 factors)
    y = pd.DataFrame(
        np.random.randn(n_months, 5) * 0.03,
        index=dates,
        columns=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    )
    
    return X, y, dates


@pytest.fixture
def synthetic_weights() -> np.ndarray:
    """Generate synthetic portfolio weights."""
    np.random.seed(42)
    n_stocks = 50
    
    # Random weights that sum to 1
    raw_weights = np.random.uniform(0, 1, n_stocks)
    weights = raw_weights / raw_weights.sum()
    
    return weights


# =============================================================================
# Covariance Matrices
# =============================================================================

@pytest.fixture
def positive_definite_cov() -> np.ndarray:
    """Generate a guaranteed positive definite covariance matrix."""
    np.random.seed(42)
    n = 10
    
    # Generate random matrix and make it positive definite
    A = np.random.randn(n, n)
    cov = A @ A.T + 0.1 * np.eye(n)  # Add ridge for numerical stability
    
    return cov


@pytest.fixture
def singular_cov() -> np.ndarray:
    """Generate a singular (non-invertible) covariance matrix."""
    np.random.seed(42)
    n = 10
    
    # Create rank-deficient matrix
    A = np.random.randn(n, 3)  # Only rank 3
    cov = A @ A.T
    
    return cov


@pytest.fixture
def ill_conditioned_cov() -> np.ndarray:
    """Generate an ill-conditioned covariance matrix."""
    np.random.seed(42)
    n = 10
    
    # Create matrix with very different eigenvalues
    eigvals = np.logspace(-6, 0, n)  # From 1e-6 to 1
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    cov = Q @ np.diag(eigvals) @ Q.T
    
    return cov


# =============================================================================
# Test Utilities
# =============================================================================

def assert_valid_weights(weights: np.ndarray, 
                         long_only: bool = True,
                         tolerance: float = 1e-6):
    """Assert that portfolio weights are valid."""
    assert isinstance(weights, np.ndarray), "Weights must be numpy array"
    assert np.isfinite(weights).all(), "Weights must be finite"
    assert np.isclose(weights.sum(), 1.0, atol=tolerance), \
        f"Weights must sum to 1, got {weights.sum()}"
    
    if long_only:
        assert (weights >= -tolerance).all(), "Weights must be non-negative"


def assert_valid_covariance(cov: np.ndarray, 
                           check_positive_definite: bool = True,
                           tolerance: float = 1e-10):
    """Assert that covariance matrix is valid."""
    assert isinstance(cov, np.ndarray), "Covariance must be numpy array"
    assert cov.ndim == 2, "Covariance must be 2D"
    assert cov.shape[0] == cov.shape[1], "Covariance must be square"
    assert np.isfinite(cov).all(), "Covariance must be finite"
    assert np.allclose(cov, cov.T, atol=tolerance), \
        "Covariance must be symmetric"
    
    if check_positive_definite:
        eigvals = np.linalg.eigvals(cov)
        assert (eigvals >= -tolerance).all(), \
            f"Covariance must be positive semi-definite, got min eigval: {eigvals.min()}"


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield