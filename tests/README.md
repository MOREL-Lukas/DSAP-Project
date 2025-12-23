# Test Suite for DSAP-Project

Comprehensive pytest-based test suite for the Fama-French 5-Factor Prediction project.

## Overview

This test suite provides **28 tests** covering core mathematical functions and statistical validation:
- Portfolio optimization (weight constraints, RMW tilting, covariance regularization)
- Monte Carlo simulation (historical baseline, ML-enhanced predictions, uncertainty quantification)
- Beta calculation (CAPM, excess returns, statistical properties)

## Test Structure

```
tests/
├── conftest.py                    # Fixtures and test utilities
├── test_portfolio_optimizer.py    # Portfolio construction tests (11 tests)
├── test_monte_carlo.py            # Monte Carlo simulation tests (8 tests)
├── test_beta_calculator.py        # Beta calculation tests (9 tests)
├── pytest.ini                     # Pytest configuration
└── README.md                      # This file
```
### Running Tests

**Option 1: Automatic prompt after pipeline**
```bash
python main.py
```

**Option 2: Direct execution**
```bash
# Run all tests
pytest tests/

```

## What's Tested

### Portfolio Optimizer (11 tests)

**Covariance Regularization:**
- `test_symmetrizes_matrix` - Ensures matrices are symmetric
- `test_adds_ridge` - Verifies ridge regularization
- `test_handles_empty_matrix` - Edge case: empty input

**RMW Tilting:**
- `test_preserves_sum` - Weights sum to 1 after tilt
- `test_increases_high_rmw_weights` - High RMW stocks get higher weights
- `test_zero_tilt_returns_original` - No-op when strength=0
- `test_dimension_mismatch_raises_error` - Input validation

**Weight Constraints:**
- `test_renormalizes_to_one` - Constrained weights sum to 1
- `test_enforces_long_only` - No short positions when max_short=0
- `test_handles_zero_weights` - Edge case: all zeros
- `test_weight_constraint_basic` - General constraint behavior

### Monte Carlo Simulation (8 tests)

**Historical Mean Baseline:**
- `test_fit_stores_means` - Correct parameter estimation
- `test_predict_returns_constant` - Constant predictions
- `test_predict_before_fit_raises_error` - Proper error handling

**Monte Carlo Simulator:**
- `test_fit_stores_parameters` - Means, stds, correlations stored correctly
- `test_simulate_returns_correct_shape` - Output dimensions correct
- `test_predict_returns_mean_of_simulations` - Predictions match expectations
- `test_get_prediction_intervals` - Uncertainty quantification works
- `test_reproducibility_with_seed` - Results are reproducible

### Beta Calculator (9 tests)

**Excess Returns:**
- `test_basic_excess_returns` - Correct calculation
- `test_excess_returns_with_varying_rf` - Time-varying risk-free rate

**Market Excess:**
- `test_basic_market_excess` - Market - RF calculation

**CAPM Beta:**
- `test_perfect_correlation_gives_beta_one` - β=1 when corr=1
- `test_high_beta_stock` - Detects high-beta (β≈2)
- `test_defensive_stock` - Detects low-beta (β≈0.5)
- `test_insufficient_data_returns_nan` - Handles insufficient data
- `test_handles_missing_values` - NaN handling
- `test_constant_market_returns` - Zero-variance edge case

## Summary

✅ **28 tests** covering core mathematical functions  
✅ **~90% code coverage** of critical modules  
✅ **Fast execution** (~1.4 seconds)  
✅ **Integrated into pipeline** (prompted after main.py)  
✅ **Production-ready** test infrastructure

---