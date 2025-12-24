"""
Unit tests for portfolio_optimizer.py - FINAL VERSION
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path so 'src' imports work
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.portfolio_optimizer import (
    regularize_covariance,
    apply_rmw_tilt,
    apply_weight_constraints,
)


class TestRegularizeCovariance:
    """Tests for covariance matrix regularization."""

    def test_symmetrizes_matrix(self):
        """Test that asymmetric matrix becomes symmetric."""
        cov = np.array([[1.0, 0.5], [0.6, 1.0]])
        reg_cov = regularize_covariance(cov)
        assert np.allclose(reg_cov, reg_cov.T)

    def test_adds_ridge(self):
        """Test that ridge is added to diagonal."""
        cov = np.eye(5)
        reg_cov = regularize_covariance(cov, ridge_ratio=0.1)
        assert np.all(np.diag(reg_cov) >= np.diag(cov))

    def test_handles_empty_matrix(self):
        """Test handling of empty matrix."""
        cov = np.array([])
        reg_cov = regularize_covariance(cov)
        assert reg_cov.size == 0


class TestApplyRMWTilt:
    """Tests for RMW tilting function."""

    def test_preserves_sum(self):
        """Test that weights sum is preserved."""
        weights = np.array([0.2, 0.3, 0.5])
        betas_rmw = np.array([0.5, 1.0, 1.5])
        tilted = apply_rmw_tilt(weights, betas_rmw, tilt_strength=0.3)
        assert np.isclose(tilted.sum(), 1.0, atol=1e-6)

    def test_increases_high_rmw_weights(self):
        """Test that high RMW stocks get higher weights."""
        weights = np.array([0.2, 0.3, 0.5])
        betas_rmw = np.array([0.5, 1.0, 1.5])
        tilted = apply_rmw_tilt(weights, betas_rmw, tilt_strength=0.3)
        assert tilted[2] / weights[2] > tilted[0] / weights[0]

    def test_zero_tilt_returns_original(self):
        """Test that zero tilt returns original weights."""
        weights = np.array([0.2, 0.3, 0.5])
        betas_rmw = np.array([0.5, 1.0, 1.5])
        tilted = apply_rmw_tilt(weights, betas_rmw, tilt_strength=0.0)
        assert np.allclose(tilted, weights)

    def test_dimension_mismatch_raises_error(self):
        """Test that mismatched dimensions raise error."""
        weights = np.array([0.3, 0.7])
        betas_rmw = np.array([0.5, 1.0, 1.5])
        with pytest.raises(ValueError):
            apply_rmw_tilt(weights, betas_rmw, tilt_strength=0.3)


class TestApplyWeightConstraints:
    """Tests for weight constraint application."""

    def test_renormalizes_to_one(self):
        """Test that weights sum to 1 after constraints."""
        weights = np.array([0.5, 0.3, 0.2])
        constrained = apply_weight_constraints(weights, max_weight=0.10)
        assert np.isclose(constrained.sum(), 1.0, atol=1e-6)

    def test_enforces_long_only(self):
        """Test that negative weights are handled."""
        weights = np.array([0.6, 0.5, -0.1])
        constrained = apply_weight_constraints(weights, max_short=0.0)
        assert (constrained >= -1e-6).all()
        assert np.isclose(constrained.sum(), 1.0, atol=1e-6)

    def test_handles_zero_weights(self):
        """Test handling of all-zero weights."""
        weights = np.zeros(10)
        constrained = apply_weight_constraints(weights, max_weight=0.10)
        assert np.isfinite(constrained).all()
        assert np.isclose(constrained.sum(), 1.0, atol=1e-6)

    def test_weight_constraint_basic(self):
        """Test basic weight constraint behavior."""
        # When input has reasonable weights that need adjustment
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        constrained = apply_weight_constraints(weights, max_weight=0.10)

        # Should still sum to 1
        assert np.isclose(constrained.sum(), 1.0, atol=1e-6)
        # All should be non-negative
        assert (constrained >= -1e-6).all()
        # All should be finite
        assert np.isfinite(constrained).all()
