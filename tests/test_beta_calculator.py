"""
Unit tests for beta_calculator.py - FIXED VERSION
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.beta_calculator import (
    calculate_excess_returns,
    calculate_market_excess_return,
    calculate_capm_beta,
)


class TestCalculateExcessReturns:
    """Tests for excess return calculation."""

    def test_basic_excess_returns(self, synthetic_returns):
        """Test basic excess return calculation."""
        rf_series = pd.Series(0.002, index=synthetic_returns.index)
        excess = calculate_excess_returns(synthetic_returns, rf_series)

        assert excess.shape == synthetic_returns.shape
        expected = synthetic_returns.sub(rf_series, axis=0)
        assert np.allclose(excess, expected, atol=1e-10)

    def test_excess_returns_with_varying_rf(self, synthetic_returns):
        """Test with time-varying risk-free rate."""
        rf_series = pd.Series(
            np.linspace(0.003, 0.001, len(synthetic_returns)),
            index=synthetic_returns.index,
        )
        excess = calculate_excess_returns(synthetic_returns, rf_series)

        for i, (idx, row) in enumerate(synthetic_returns.iterrows()):
            expected_row = row - rf_series.iloc[i]
            assert np.allclose(excess.loc[idx], expected_row, atol=1e-10)


class TestCalculateMarketExcessReturn:
    """Tests for market excess return calculation."""

    def test_basic_market_excess(self):
        """Test basic market excess return."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        market_return = pd.Series(np.random.randn(12) * 0.04 + 0.008, index=dates)
        rf_series = pd.Series(0.002, index=dates)

        market_excess = calculate_market_excess_return(market_return, rf_series)

        expected = market_return - rf_series
        assert np.allclose(market_excess, expected, atol=1e-10)


class TestCalculateCAPMBeta:
    """Tests for CAPM beta calculation."""

    def test_perfect_correlation_gives_beta_one(self):
        """Test that perfect correlation gives beta=1."""
        dates = pd.date_range("2020-01-01", periods=50, freq="MS")
        market_excess = pd.Series(np.random.randn(50) * 0.04, index=dates)
        stock_excess = market_excess.copy()

        result = calculate_capm_beta(stock_excess, market_excess, min_obs=24)

        assert np.isclose(result["beta"], 1.0, atol=0.01)
        assert np.isclose(result["alpha"], 0.0, atol=0.01)
        assert result["r_squared"] > 0.99

    def test_high_beta_stock(self):
        """Test high-beta stock (2x market volatility)."""
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        market_excess = pd.Series(np.random.randn(60) * 0.04, index=dates)
        stock_excess = 2.0 * market_excess + np.random.randn(60) * 0.01

        result = calculate_capm_beta(stock_excess, market_excess, min_obs=24)

        assert 1.8 < result["beta"] < 2.2
        assert result["r_squared"] > 0.8

    def test_defensive_stock(self):
        """Test defensive stock (0.5x market volatility)."""
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        market_excess = pd.Series(np.random.randn(60) * 0.04, index=dates)
        stock_excess = 0.5 * market_excess + np.random.randn(60) * 0.01

        result = calculate_capm_beta(stock_excess, market_excess, min_obs=24)

        assert 0.3 < result["beta"] < 0.7

    def test_insufficient_data_returns_nan(self):
        """Test that insufficient observations return NaN."""
        dates = pd.date_range("2020-01-01", periods=20, freq="MS")
        market_excess = pd.Series(np.random.randn(20) * 0.04, index=dates)
        stock_excess = pd.Series(np.random.randn(20) * 0.05, index=dates)

        result = calculate_capm_beta(stock_excess, market_excess, min_obs=24)

        assert np.isnan(result["beta"])
        assert np.isnan(result["alpha"])
        assert result["n_obs"] == 20

    def test_handles_missing_values(self):
        """Test handling of NaN values in data."""
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        market_excess = pd.Series(np.random.randn(60) * 0.04, index=dates)
        stock_excess = pd.Series(np.random.randn(60) * 0.05, index=dates)

        stock_excess.iloc[10:15] = np.nan
        market_excess.iloc[20:22] = np.nan

        result = calculate_capm_beta(stock_excess, market_excess, min_obs=24)

        assert result["n_obs"] < 60

    def test_constant_market_returns(self):
        """Test handling of constant market returns (zero variance edge case)."""
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        market_excess = pd.Series(0.005, index=dates)  # All the same
        stock_excess = pd.Series(np.random.randn(60) * 0.05, index=dates)

        # This should handle the edge case gracefully
        # Either return NaN or raise a caught exception
        try:
            result = calculate_capm_beta(stock_excess, market_excess, min_obs=24)
            # If it doesn't raise an error, beta should be NaN
            assert np.isnan(
                result["beta"]
            ), "Beta should be NaN when market has zero variance"
        except ValueError as e:
            # This is also acceptable - function detects zero variance
            assert "identical" in str(e).lower() or "variance" in str(e).lower()
