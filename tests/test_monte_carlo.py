"""
Unit tests for monte_carlo.py
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.monte_carlo import (
    HistoricalMeanBaseline,
    MonteCarloFactorSimulator,
)

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


class TestHistoricalMeanBaseline:
    """Tests for historical mean baseline predictor."""

    def test_fit_stores_means(self, synthetic_factors):
        """Test that fit stores correct means."""
        baseline = HistoricalMeanBaseline()
        baseline.fit(synthetic_factors[FACTOR_NAMES])

        for factor in FACTOR_NAMES:
            expected_mean = synthetic_factors[factor].mean()
            assert np.isclose(baseline.means[factor], expected_mean)

    def test_predict_returns_constant(self, synthetic_factors):
        """Test that predict returns constant values."""
        baseline = HistoricalMeanBaseline()
        baseline.fit(synthetic_factors[FACTOR_NAMES])

        test_index = pd.date_range("2020-01-01", periods=10, freq="MS")
        X_test = pd.DataFrame(index=test_index)
        predictions = baseline.predict(X_test)

        for factor in FACTOR_NAMES:
            assert predictions[factor].nunique() == 1

    def test_predict_before_fit_raises_error(self):
        """Test that predict before fit raises error."""
        baseline = HistoricalMeanBaseline()
        X_test = pd.DataFrame(index=pd.date_range("2020-01-01", periods=10, freq="MS"))

        with pytest.raises(ValueError):
            baseline.predict(X_test)


class TestMonteCarloFactorSimulator:
    """Tests for Monte Carlo factor simulator."""

    def test_fit_stores_parameters(self, synthetic_factors):
        """Test that fit stores means, stds, and correlations."""
        simulator = MonteCarloFactorSimulator(n_simulations=1000)
        simulator.fit(synthetic_factors[FACTOR_NAMES])

        for factor in FACTOR_NAMES:
            expected_mean = synthetic_factors[factor].mean()
            assert np.isclose(simulator.means[factor], expected_mean, atol=1e-6)

        assert simulator.correlations.shape == (5, 5)
        assert np.allclose(np.diag(simulator.correlations), 1.0)

    def test_simulate_returns_correct_shape(self, synthetic_factors):
        """Test that simulations have correct shape."""
        n_sims = 1000
        n_periods = 24
        simulator = MonteCarloFactorSimulator(n_simulations=n_sims)
        simulator.fit(synthetic_factors[FACTOR_NAMES])

        sims = simulator.simulate(n_periods)

        for factor in FACTOR_NAMES:
            assert sims[factor].shape == (n_sims, n_periods)

    def test_predict_returns_mean_of_simulations(self, synthetic_factors):
        """Test that predict returns mean across simulations."""
        n_periods = 10
        simulator = MonteCarloFactorSimulator(n_simulations=1000)
        simulator.fit(synthetic_factors[FACTOR_NAMES])

        X_test = pd.DataFrame(
            index=pd.date_range("2020-01-01", periods=n_periods, freq="MS")
        )
        predictions = simulator.predict(X_test)

        for factor in FACTOR_NAMES:
            assert np.allclose(
                predictions[factor].mean(), simulator.means[factor], atol=0.01
            )

    def test_get_prediction_intervals(self, synthetic_factors):
        """Test that prediction intervals are generated correctly."""
        n_periods = 10
        simulator = MonteCarloFactorSimulator(n_simulations=5000)
        simulator.fit(synthetic_factors[FACTOR_NAMES])

        X_test = pd.DataFrame(
            index=pd.date_range("2020-01-01", periods=n_periods, freq="MS")
        )
        intervals = simulator.get_prediction_intervals(X_test, confidence_level=0.95)

        for factor in FACTOR_NAMES:
            assert f"{factor}_mean" in intervals.columns
            assert f"{factor}_lower" in intervals.columns
            assert f"{factor}_upper" in intervals.columns

            mean = intervals[f"{factor}_mean"]
            lower = intervals[f"{factor}_lower"]
            upper = intervals[f"{factor}_upper"]

            assert (lower <= mean).all()
            assert (mean <= upper).all()

    def test_reproducibility_with_seed(self, synthetic_factors):
        """Test that same seed gives same results."""
        sim1 = MonteCarloFactorSimulator(n_simulations=100, random_seed=42)
        sim2 = MonteCarloFactorSimulator(n_simulations=100, random_seed=42)

        sim1.fit(synthetic_factors[FACTOR_NAMES])
        sim2.fit(synthetic_factors[FACTOR_NAMES])

        sims1 = sim1.simulate(10)
        sims2 = sim2.simulate(10)

        for factor in FACTOR_NAMES:
            assert np.allclose(sims1[factor], sims2[factor])
