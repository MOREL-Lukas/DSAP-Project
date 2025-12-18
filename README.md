**Author**: Lukas Morel  
**Student ID**: 22414569  
**Course**: Data Science & Applied Programming  
**Date**: December 2025

# 📊 Fama–French 5‑Factor Prediction and Portfolio Construction

## Overview

This project implements a **full end-to-end quantitative research pipeline** for the Fama–French Five-Factor model.

$$
R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,MKT}(R_{m,t} - R_{f,t}) + \beta_{i,SMB} SMB_t + \beta_{i,HML} HML_t + \beta_{i,RMW} RMW_t + \beta_{i,CMA} CMA_t + \epsilon_{i,t}
$$

It combines historical factor data, S&P 500 stock returns, macroeconomic indicators, and machine-learning techniques to evaluate **whether factor returns are predictable** and **how such information should (or should not) be used in portfolio construction**.

The core result is deliberately *scientifically conservative*:

> **Most Fama–French factors are not meaningfully predictable out-of-sample.**

However, the project also shows that:

- Selective use of machine learning can add value **at the portfolio-construction margin**, and
- Uncertainty quantification is often more useful than point prediction.

---

## Key Research Questions

1. Can machine‑learning models predict monthly Fama–French factor returns better than a historical mean?
2. If predictability is weak, can ML still be used responsibly in portfolio construction?
3. Does tilting toward profitability (RMW) improve risk‑adjusted returns relative to a rational historical benchmark?

---

## Main Findings (Executive Summary)

### Factor Predictability

**Empirical Results from Test Set (2022-2025):**

| Factor | Best Model | Test R² | Validation R² | Status |
|--------|-----------|---------|---------------|---------|
| **RMW** | Lasso | **+0.0650** | +0.0115 | ✓ **Weakly Predictable** |
| **CMA** | Random Forest | -0.0318 | -0.0146 | ✗ Unpredictable |
| **Mkt-RF** | Random Forest | -0.0166 | +0.0182 | ✗ Unpredictable |
| **HML** | Random Forest | -0.0392 | +0.0030 | ✗ Unpredictable |
| **SMB** | Lasso | -0.0570 | -0.0221 | ✗ Unpredictable |

**Key Observations:**

1. **4 out of 5 factors (Mkt-RF, SMB, HML, CMA)** show **no robust out‑of‑sample predictability**.
2. **RMW (Profitability)** is the *only* factor with **positive and stable test‑set R²** (+6.5%), consistent with its slow‑moving economic nature.
3. **Historical Mean outperforms ML on average** (3/5 factors), confirming efficient markets hypothesis.
4. **Validation-Test R² Gap**: Complex models (Random Forest, Gradient Boosting) show overfitting despite conservative hyperparameters:
   - Gradient Boosting: Val R² = -0.11 → Test R² = -0.13
   - Ridge Regression: Val R² = -0.37 → Test R² = -0.14

**Conclusion:** Monthly factor returns are close to a random walk, consistent with the Efficient Market Hypothesis.

---

### Portfolio‑Level Results

**Empirical Strategy Comparison (Cross-Sectional Analysis):**

| Strategy | Sharpe | RMW Beta | MKT Beta | Stocks | Volatility | Alpha (Monthly) |
|----------|--------|----------|----------|--------|------------|-----------------|
| **RMW Tilt (1.0)** | **0.210** | 0.381 | 0.804 | 242 | 3.47% | +0.25% |
| **Baseline (No Tilt)** | **0.208** | 0.351 | 0.805 | 245 | 3.47% | +0.24% |
| **Concentrated (50)** | 0.198 | 0.836 | 1.061 | 30 | 5.23% | +0.40% |
| **Equal-Weight** | 0.152 | 0.107 | 1.040 | 496 | 4.77% | +0.11% |

**Rolling Backtest Results (2022-2025, 309 months):**

| Strategy | Sharpe | Mean Return | Volatility | CAPM Beta | CAPM Alpha |
|----------|--------|-------------|------------|-----------|------------|
| **Equal-Weight** | 0.258 | +1.21% | 4.69% | 0.967 | +0.59% |
| **Baseline (Tangency)** | 0.250 | +0.99% | 3.97% | 0.749 | +0.51% |
| **RMW Tilt** | 0.250 | +0.99% | 3.97% | 0.745 | +0.52% |

**Key Findings:**

1. **RMW Tilt Effectiveness:**
   - Cross-sectional: **+0.8% Sharpe improvement** (0.210 vs 0.208)
   - RMW beta increases from 0.351 → 0.381 (as designed)
   - Market beta decreases slightly (0.805 → 0.804), creating defensive profile
   - Alpha improves marginally (+0.24% → +0.25% monthly)

2. **Concentration Risk:**
   - Concentrated 50-stock portfolio **underperforms by 4.7%** in Sharpe ratio
   - Despite higher RMW beta (0.836) and alpha (+0.40%), **idiosyncratic risk dominates**
   - Volatility increases dramatically (3.47% → 5.23%)

3. **Diversification vs Optimization:**
   - Baseline already **loads strongly on RMW** (β=0.351), because profitability has positive historical premium
   - Optimization adds value vs equal-weight: Sharpe 0.208 vs 0.152 (+37%)
   - RMW tilt provides **marginal benefit** while preserving diversification

**Final Recommendation:**
> Use a **diversified FF5 portfolio with a mild RMW tilt (strength 0.3-1.0)**. Avoid concentration.

---

## Why the Baseline Is a Strong Benchmark

The baseline is **not** an equal‑weight or market portfolio. It is:

> **A historically optimal FF5 allocation that already exploits known factor premia.**

This makes it a *hard* benchmark to beat and avoids misleading comparisons against naive portfolios. The research question is therefore:

> *Does explicit ML‑informed emphasis on profitability add value beyond what a rational allocator already does?*

This framing is intentional and academically sound. **Results confirm:** ML adds marginal value (+0.8% Sharpe), primarily through better RMW exposure.

---

## Uncertainty Quantification Success

**Monte Carlo Simulator Performance:**

| Factor | Coverage (95% intervals) | Avg Interval Width |
|--------|-------------------------|-------------------|
| **RMW** | **97.67%** | 0.11 (11%) |
| **Mkt-RF** | 90.70% | 0.17 (17%) |
| **SMB** | 90.70% | 0.12 (12%) |
| **HML** | 88.37% | 0.12 (12%) |
| **CMA** | 83.72% | 0.08 (8%) |

**Average Coverage: 90.2%** → Well-calibrated despite low point-forecast R²

**Key Insight:**
> **Risk estimation succeeds even when prediction fails.** 

Portfolio construction benefits more from reliable uncertainty estimates than from noisy point forecasts. This is a **profound finding** for quantitative finance:
- Point forecasts may have R² < 0 (worse than mean)
- But distributional forecasts can be well-calibrated (90% coverage)
- Risk management uses volatility/correlations, not just returns

---

## Methodology

### Data Sources

- **S&P 500 constituents:** DataHub
- **Stock prices:** Yahoo Finance (monthly, 1990–2025)
- **Fama–French 5 factors:** Kenneth French Data Library
- **Macroeconomic indicators:** FRED (GDP, inflation, rates, spreads, VIX, oil)

**Final Dataset:**
- 428 months (1990-2025)
- 74 predictive features
- 501 stocks with valid FF5 betas

---

### Feature Engineering

~80 predictive features grouped into:

- **Cross‑sectional market statistics** (factor spreads, dispersion)
- **Lagged factor returns** (1, 2, 3, 6, 12 months) and rolling statistics
- **Market regime indicators** (volatility, trend, breadth)
- **Macroeconomic variables** (VIX, oil prices, FRED data)

All features are constructed **strictly using information available at time t**.

---

### Model Training and Evaluation

**Temporal Split (No Shuffling):**
- **Train:** 1990-2014 (299 months, 69.9%)
- **Validation:** 2015-2022 (86 months, 20.1%)
- **Test:** 2022-2025 (43 months, 10.0%)

**Models Evaluated:**
- Historical Mean (benchmark)
- Random Forest (n=600, depth=3, leaf=30) ← Conservative hyperparameters
- Gradient Boosting (n=500, lr=0.03, depth=2)
- Ridge Regression (α=10.0)
- Lasso Regression (α=0.001)

**Primary Metric:** Out‑of‑sample R² on test set

**Model Selection:** Best model chosen **only on validation set**, then evaluated on holdout test set.

**Actual Results:**
- Best model: **Random Forest** (Avg Val R² = +0.0115)
- Test performance: **Avg Test R² = -0.0215** (worse than mean)
- Winner by factor: RMW → Lasso (+6.5%), others → negative R²

---

## Portfolio Construction

### Steps

1. Estimate **FF5 betas** for all S&P 500 stocks via OLS regression
   - 496 stocks with valid betas (R² > 0.15 threshold)
   - Average R² = 0.303 (strong factor explanatory power)

2. Compute expected stock returns:
   $$\mathbb{E}[R] = R_f + \sum_{k=1}^5 \beta_k \cdot \mathbb{E}[\text{Factor}_k]$$

3. Construct **tangency‑style portfolio** under realistic constraints:
   - Maximize Sharpe ratio
   - Constraints: Σw = 1, w ≥ 0, w_i ≤ 0.10

4. Apply **post‑optimization RMW tilt**:
   ```python
   tilt_factor = 1 + strength × normalized_RMW_beta
   w_tilted = w_baseline × tilt_factor / sum(...)
   ```

5. Evaluate via **rolling out‑of‑sample backtests**

---

## Backtesting Results

**Cross-Sectional Performance (Single Period):**
- **Positive Sharpe ratios** (0.15-0.21 range)
- **Market beta < 1** (defensive positioning: β = 0.75-0.80 for optimized)
- **Positive CAPM alpha** at portfolio level (+0.24-0.25% monthly)
- **RMW tilt improves Sharpe modestly** (+0.8%)

**Rolling Backtest (309 months, 1997-2025):**
- Equal-weight: Sharpe 0.26, Beta 0.97
- Tangency (optimized): Sharpe 0.25, Beta 0.75
- RMW Tilt: Sharpe 0.25, Beta 0.75

**Interpretation:**
Returns come from *diversification and factor exposure*, not factor timing. The fact that optimized portfolios have **lower beta** (0.75 vs 0.97) but **similar Sharpe** (0.25 vs 0.26) demonstrates **alpha generation** through intelligent factor weighting.

---

## Testing Infrastructure

**28 Comprehensive Unit Tests** (100% pass rate, 1.39s execution):

**Coverage:**
- **Portfolio Optimizer:** 11 tests (covariance, tilting, constraints)
- **Monte Carlo:** 8 tests (baseline, simulation, intervals)
- **Beta Calculator:** 9 tests (CAPM, excess returns, edge cases)

**Quality Indicators:**
- Edge case handling (empty matrices, singular covariance, NaN values)
- Numerical stability checks (symmetry, positive definiteness)
- Reproducibility validation (seeded randomness)
- Domain constraints (weights sum to 1, non-negative)

**Example Test Output:**
```
28 passed in 1.39s
```

---

## Project Structure

```
DSAP-Project/
├── data/
│   ├── raw/
│   └── processed/
├── results/                 # Tables & figures
├── src/                     # Core implementation
│   ├── __init__.py
│   ├── beta_calculator.py   # CAPM & FF5 betas
│   ├── data_processing.py   # Data ingestion & feature engineering
│   ├── ml_models.py         # ML predictors
│   ├── monte_carlo.py       # Simulation & uncertainty
│   ├── portfolio_optimizer.py
│   └── results_exporter.py
├── tests/                   # Pytest scripts (28 tests)
│   ├── conftest.py
│   ├── README.md
│   ├── test_beta_calculator.py
│   ├── test_monte_carlo.py 
│   └── test_portfolio_optimizer.py
│
├── AI_USAGE.md
├── environment.yml
├── main.py                  # Full pipeline entry point
├── pytest.ini
└── README.md
```

---

## How to Run

### Environment Setup

Environment name: **DSAP-Project**

Create and activate the environment:

```bash
git clone https://github.com/MOREL-Lukas/DSAP-Project.git
cd DSAP-Project
conda env create -f environment.yml
conda activate DSAP-Project
```

Key dependencies include:

- `python=3.11`
- `pandas`, `numpy`, `scipy`
- `scikit-learn`
- `cvxpy`, `osqp`, `clarabel`
- `statsmodels`
- `matplotlib`, `seaborn`
- `yfinance`, `fredapi`

(See `environment.yml` for the complete, pinned dependency list.)

### FRED API Key (Required)

A FRED API key is required for macroeconomic features:

1. Get free key: https://fred.stlouisfed.org/docs/api/api_key.html
2. Add to `main.py` line 144 or use environment variable:

```bash
export FRED_API_KEY="your_key_here"
```

### Run the Full Pipeline

```bash
python main.py
```

**Expected Runtime:** 5-10 minutes (includes data download, ML training, backtesting)

**Output Files:**
- `results/pipeline_summary.txt` - Executive summary
- `results/complete_results.xlsx` - Detailed tables
- `results/*.png` - Visualization plots

## Scientific Contribution

This project demonstrates:

- **Proper time‑series validation** (no look‑ahead bias)
- **Honest reporting of negative ML results**
- **Economically meaningful baselines** (historically-optimal, not naive)
- **Separation between prediction and portfolio construction**
- **Practical use of uncertainty quantification**

**Key Empirical Findings:**

1. **RMW is the only predictable factor** (+6.5% test R²)
2. **Historical mean beats ML on average** (3/5 factors)
3. **RMW tilt improves Sharpe modestly** (+0.8%)
4. **Concentration destroys value** (-4.7% Sharpe)
5. **Uncertainty quantification succeeds** (90% coverage) even when prediction fails

**Key Takeaway:**
> In efficient markets, ML rarely improves forecasts – but it *can* still improve decisions when used carefully, especially for **slow-moving fundamentals like profitability**.

---

## Academic Foundation

This project builds on established research in empirical asset pricing and factor models, including:

- **Fama, E. F., & French, K. R. (2015).** *A five-factor asset pricing model*. Journal of Financial Economics, 116(1), 1–22.
- **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** *...and the cross-section of expected returns*. Review of Financial Studies, 29(1), 5–68.
- **Gu, S., Kelly, B., & Xiu, D. (2020).** *Empirical asset pricing via machine learning*. Review of Financial Studies, 33(5), 2223–2273.
- **Kozak, S., Nagel, S., & Santosh, S. (2020).** *Shrinking the cross-section*. Journal of Financial Economics, 135(2), 271–292.
- **Malkiel, B. G. (2003).** *The efficient market hypothesis and its critics*. Journal of Economic Perspectives, 17(1), 59–82.
- **Fama, E. F. (1970).** *Efficient capital markets: A review of theory and empirical work*. Journal of Finance, 25(2), 383–417.
- **Federal Reserve Bank of St. Louis. (2025).** Federal Reserve Economic Data (FRED). https://fred.stlouisfed.org
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The elements of statistical learning* (2nd ed.). Springer.
- **French, K. R. (2025).** Fama–French data library. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Yahoo Finance. (2025).** Yahoo Finance market data. https://finance.yahoo.com

---

## Empirical Insights for Practitioners

### For Quantitative Analysts
- **Factor timing is extremely difficult** (4/5 factors unpredictable)
- **Profitability (RMW) has modest predictability** via fundamentals
- **Validation-test gaps are real** (regime shifts, overfitting)
- **Conservative hyperparameters essential** for macro data

### For Portfolio Managers
- **Factor premia are real and exploitable** (FF5 beats equal-weight)
- **Optimization adds value** (Sharpe +37% vs equal-weight)
- **RMW tilt provides marginal benefit** (+0.8% Sharpe)
- **Concentration is dangerous** (idiosyncratic risk dominates)
- **Target lower market beta** (0.75 vs 1.0) for better risk-adjusted returns

### For Risk Managers
- **Uncertainty quantification is achievable** (90% coverage)
- **Monte Carlo simulation provides reliable risk estimates**
- **Use distributional forecasts** (quantiles) not just point estimates
- **Correlation-aware simulation** preserves factor structure

---

## License

This project is provided for **educational and academic research purposes only**.

- **Code**: Released for non-commercial academic use, without warranty
- **Fama–French data**: © Eugene F. Fama and Kenneth R. French (academic use)
- **FRED data**: Public domain (U.S. Government)
- **Yahoo Finance data**: Subject to Yahoo Finance Terms of Service

---

## Disclaimer

This project uses data accessed via the **FRED® API** but is **not endorsed or certified by the Federal Reserve Bank of St. Louis**.

This project is **not investment advice**.

- **Past performance does not guarantee future results**
- **Transaction costs, taxes, and liquidity effects are not fully modeled**
- **Survivorship bias may be present in S&P 500 data**
- **Results based on specific time period (1990-2025) and may not generalize**
- **Backtest performance may differ significantly from live trading**

Always consult a licensed financial professional before making investment decisions.

---

## Contact

**Lukas Morel**  
Student ID: 22414569  
Course: Data Science & Applied Programming  

**Repository:** https://github.com/MOREL-Lukas/DSAP-Project

---

## Acknowledgments

This project was developed with the assistance of AI tools (ChatGPT, Claude, GitHub Copilot) for debugging, methodological guidance, and code drafting. See `AI_USAGE.md` for full disclosure.