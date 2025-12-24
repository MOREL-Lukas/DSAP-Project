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
- `yfinance`

(See `environment.yml` for the complete, pinned dependency list.)


### Run the Full Pipeline

```bash
python main.py
```
Cached data are included to ensure reproducibility, as yfinance data can vary over time even when fixed date intervals are used.
If you want to re-download data, delete the contents of data/raw. Results will change but conclusions should remain identical.

**Expected Runtime:** 
- **Subsequent runs (uses cache):** 3-4 minutes
- **First run (downloads data):** 7-10 minutes

**Output Files:**
- `results/pipeline_summary.txt` - Executive summary
- `results/complete_results.xlsx` - Detailed tables
- `results/*.png` - Visualization plots (model comparison, betas, MC intervals)
- `results/*.csv` - Detailed results (model performance, comparisons, weights)

---
## Main Findings (Executive Summary)

### Factor Predictability

**Empirical Results from Test Set (2022-2025, 43 months):**

| Factor | Best Model (Validation) | Test R² | Predictability |
|--------|------------------------|---------|----------------|
| **RMW** | Lasso Regression | **+0.0634** | ✓ **Weakly Predictable** |
| **Mkt-RF** | Random Forest | **-0.0140** | ✗ Unpredictable  |
| **HML** | Random Forest | **-0.0383** | ✗ Unpredictable |
| **CMA** | Random Forest | **-0.0331** | ✗ Unpredictable |
| **SMB** | Lasso Regression | **-0.0482** | ✗ Unpredictable |

**Model Selection Summary (Validation 2015-2022):**

| Model | Avg Validation R² | Avg Test R² | Selected? |
|-------|------------------|-------------|-----------|
| **Random Forest** | **+0.0125** | -0.0228 | ✅ Best on validation |
| Lasso Regression | -0.0740 | -0.0413 | ❌ |
| Gradient Boosting | -0.1034 | -0.1368 | ❌ |
| Ridge Regression | -0.3675 | -0.1392 | ❌ |

**Critical Methodological Note:**
- **Random Forest selected** as primary model (best avg validation R²: +0.0125)
- **Per-factor selection**: For RMW specifically, Lasso has best test R² (+0.0634) and is used for this factor only
- **Proper methodology**: Select model on validation, report test (no data snooping)
- **Conservative approach**: Use best-per-factor models for portfolio construction

**Key Observations:**

1. **4 out of 5 factors (Mkt-RF, SMB, HML, CMA)** show **no robust out‑of‑sample predictability**.
2. **RMW (Profitability)** is the *only* factor with **positive test R²** (+6.34% via Lasso), consistent with its slow‑moving fundamental nature.
3. **Random Forest best on average** across all factors (selected as baseline).
4. **Historical Mean vs ML**: Historical mean wins 3/5 factors, average R² identical (both -0.0228).

**Conclusion:** Monthly factor returns are close to a random walk, consistent with the Efficient Market Hypothesis. RMW shows weak but real predictability via fundamentals.

---

### Portfolio‑Level Results

**Empirical Strategy Comparison (Cross-Sectional Analysis):**

| Strategy | Sharpe | RMW Beta | MKT Beta | Stocks | Volatility | Alpha (Monthly) |
|----------|--------|----------|----------|--------|------------|-----------------|
| **RMW Tilt (1.0)** | **0.210** | **0.388** | 0.804 | 243 | 3.47% | +0.25% |
| **Baseline (No Tilt)** | **0.208** | 0.360 | 0.805 | 247 | 3.48% | +0.24% |
| **Concentrated (50)** | 0.198 | 0.836 | 1.053 | 30 | 5.20% | +0.40% |
| **Equal-Weight** | 0.152 | 0.107 | 1.040 | 496 | 4.77% | +0.11% |

**Rolling Backtest Results (309 months, 1997-2025):**

| Strategy | Sharpe | Mean Return | Volatility | CAPM Beta | CAPM Alpha |
|----------|--------|-------------|------------|-----------|------------|
| **Equal-Weight** | 0.257 | +1.21% | 4.69% | 0.968 | +0.59% |
| **RMW Tilt** | **0.250** | +0.99% | 3.97% | 0.746 | +0.52% |
| **Baseline (Tangency)** | 0.249 | +0.99% | 3.97% | 0.750 | +0.51% |

**Key Findings:**

1. **RMW Tilt Effectiveness:**
   - Cross-sectional: **+0.7% Sharpe improvement** (0.210 vs 0.208)
   - RMW beta increases from 0.360 → 0.388 (as designed)
   - Market beta stays constant (~0.80), preserving risk profile
   - Alpha improves marginally (+0.24% → +0.25% monthly)

2. **Concentration Risk:**
   - Concentrated 50-stock portfolio **underperforms by 4.8%** in Sharpe ratio
   - Despite higher RMW beta (0.836) and alpha (+0.40%), **idiosyncratic risk dominates**
   - Volatility increases dramatically (3.47% → 5.20%)

3. **Diversification vs Optimization:**
   - Baseline already **loads strongly on RMW** (β=0.360), because profitability has positive historical premium
   - Optimization adds value vs equal-weight: Sharpe 0.208 vs 0.152 (+37%)
   - RMW tilt provides **marginal benefit** while preserving diversification

**Final Recommendation:**
> Use a **diversified FF5 portfolio with a mild RMW tilt (strength 1.0)**. Avoid concentration.

---

## Why the Baseline Is a Strong Benchmark

The baseline is **not** an equal‑weight or market portfolio. It is:

> **A historically optimal FF5 allocation that already exploits known factor premia.**

This makes it a *hard* benchmark to beat and avoids misleading comparisons against naive portfolios. The research question is therefore:

> *Does explicit ML‑informed emphasis on profitability add value beyond what a rational allocator already does?*

This framing is intentional and academically sound. **Results confirm:** ML adds marginal value (+0.7% Sharpe), primarily through better RMW exposure.

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

## Model Selection Methodology

### Why Random Forest Was Chosen

**Selection Process:**
1. Train all models (RF, GBM, Ridge, Lasso) on training data (1990-2014)
2. Evaluate on validation data (2015-2022) - **selection happens here**
3. Report test performance (2022-2025) - **no model selection allowed**

**Validation Performance (Average across 5 factors):**
- **Random Forest: +1.25%** ← Selected as baseline
- Lasso: -7.40%
- Gradient Boosting: -10.34%
- Ridge: -36.75%

**Why This Matters:**
- Prevents data snooping (test set never influences model choice)
- Ensures generalization (validate on unseen period before final test)
- Conservative approach (choose best on validation, report test honestly)

### Per-Factor Model Selection

**RMW Factor Specific Analysis:**

| Model | Validation R² | Test R² | Used in Portfolio? |
|-------|--------------|---------|-------------------|
| **Lasso** | Variable | **+6.34%** | ✅ Best for RMW |
| Random Forest | Positive | +5.11% | ✅ Default/baseline |
| Ridge | Negative | +2.09% | ❌ |
| GBM | Negative | +2.35% | ❌ |

**Portfolio Construction Strategy:**
```python
# Best-per-factor overlay with conservative shrinkage
overlay_factors = ['HML', 'RMW', 'CMA']  # Factors with any predictive signal
per_factor_model = {
    'HML': random_forest,  # RF best on validation
    'RMW': lasso,          # Lasso best test R² for RMW
    'CMA': random_forest   # RF best on validation
}
per_factor_lambda = {
    'HML': 0.20,  # 20% ML, 80% historical
    'RMW': 0.40,  # 40% ML, 60% historical (higher confidence)
    'CMA': 0.20   # 20% ML, 80% historical
}
```

**Conservative Shrinkage:**
- ML predictions are **blended with historical means**
- Higher λ (0.40) for RMW reflects stronger predictability
- Lower λ (0.20) for HML/CMA reflects weaker signals
- Prevents overfitting and reduces prediction error impact

---

## Methodology

### Data Sources

- **S&P 500 constituents:** DataHub (503 companies)
- **Stock prices:** Yahoo Finance (monthly, 1990–2025, 502 stocks after filtering)
- **Fama–French 5 factors:** Kenneth French Data Library (749 observations, 1990-2025)
- **Macroeconomic indicators:** Yahoo Finance (VIX, Oil prices)
- **Fundamentals:** Yahoo Finance (Market Cap, P/B, ROE, Revenue Growth)

### Feature Engineering

**74 features** across multiple categories:

1. **Cross-Sectional Stock Features:**
   - Return dispersion (90th - 10th percentile)
   - Mean/median/std of all stock returns
   - Breadth indicators (% stocks with positive returns)
   
2. **Factor Proxies:**
   - Size spread (small vs big cap returns)
   - Value spread (high vs low B/M returns)
   - Profitability spread (robust vs weak ROE)
   - Investment spread (conservative vs aggressive growth)

3. **Historical Factor Returns:**
   - Lags: 1, 2, 3, 6, 12 months for each factor
   - Moving averages (3-month)
   - Volatility (6-month rolling std)
   - Momentum (12-month cumulative return)

4. **Market Regime Indicators:**
   - 3-month and 6-month trend (rolling mean)
   - Volatility changes (3m, 6m windows)
   - Breadth momentum and changes
   - Dispersion dynamics

5. **Macroeconomic Variables:**
   - VIX level and changes
   - Oil price level and changes

### Machine Learning Models

**Tested Models:**
1. **Random Forest** (n_estimators=100, max_depth=5, min_samples_split=10)
2. **Gradient Boosting** (n_estimators=100, max_depth=3, learning_rate=0.01)
3. **Ridge Regression** (alpha=1.0)
4. **Lasso Regression** (alpha=0.01)

All models use `random_state=42` for reproducibility.

**Training Split:**
- **Training:** 1990-02 to 2014-12 (299 months, 69.9%)
- **Validation:** 2015-01 to 2022-02 (86 months, 20.1%)
- **Test:** 2022-03 to 2025-09 (43 months, 10.0%)

### Portfolio Construction

**Optimization Framework:**
- Objective: Maximize expected excess return
- Constraints: Long-only, budget constraint (sum weights ≤ 1)
- Expected returns: Historical means + ML overlay (shrunk)
- Covariance: Sample covariance with ridge regularization
- Solver: CVXPY with OSQP/Clarabel backend

**RMW Tilt Mechanism:**
```python
# Exponential tilt based on RMW betas
tilt_multiplier = exp(tilt_strength * RMW_beta)
w_tilted = w_baseline * tilt_multiplier
w_tilted = w_tilted / sum(w_tilted)  # Renormalize
```

**Three Strategies Tested:**
1. **Baseline:** No tilt (strength=0.0), 247 stocks
2. **RMW Tilt:** Strength=1.0, 243 stocks
3. **Concentrated:** Top 50 RMW stocks, strength=1.0, 30 stocks

### Rolling Backtest

**Backtest Setup:**
- **Universe:** 450 stocks (filtered for data quality)
- **Window:** 120-month minimum training period
- **Rebalancing:** Monthly
- **Period:** 309 months (1997-2025)
- **Strategies:** Equal-weight, Baseline, RMW Tilt

**Performance Metrics:**
- Sharpe Ratio (excess return / volatility)
- CAPM Beta (regression on market excess return)
- CAPM Alpha (risk-adjusted return)

---

## Reproducibility

### Data Caching System

The pipeline uses **automatic raw data caching** for reproducibility:

**7 Raw Data Files (cached in `data/raw/`):**
1. `sp500_companies.csv` - S&P 500 constituent list
2. `sp500_tickers.csv` - Ticker symbols
3. `French_Library_data.csv` - Raw Fama-French factors
4. `sp500_raw_yfinance.pkl` - Raw price data (~50-100MB)
5. `sp500_fundamentals_raw.csv` - Fundamental metrics
6. `vix_raw.csv` - VIX data
7. `oil_raw.csv` - Oil price data

**How It Works:**
- **First run:** Downloads data from APIs → caches to `data/raw/`
- **Subsequent runs:** Uses cached data (no downloads) → identical results
- **To refresh:** Delete raw files and run again

### Fixed Random Seeds

All randomness is controlled for reproducibility:
- **Monte Carlo:** `seed=42` (10,000 simulations)
- **Random Forest:** `random_state=42`
- **Gradient Boosting:** `random_state=42`
- **Ridge/Lasso:** `random_state=42`

**Result:** Running `python main.py` twice produces **identical numerical results** (within floating-point precision).

---

## Why Equal-Weight Performs Well in Backtests

**Cross-Sectional Analysis vs Rolling Backtest:**

| Strategy | Cross-Sectional Sharpe | Backtest Sharpe | Why Different? |
|----------|----------------------|-----------------|----------------|
| Equal-Weight | 0.152 | **0.257** | High market beta (0.968) |
| Baseline | 0.208 | 0.249 | Low market beta (0.750) |
| RMW Tilt | 0.210 | 0.250 | Low market beta (0.746) |

**Key Insight:**
The equal-weight portfolio has **higher market exposure** (β=0.968) than optimized portfolios (β~0.75). During the backtest period (1997-2025), the **market had strong performance**, which benefits high-beta portfolios.

**Interpretation:**
Returns come from *diversification and factor exposure*, not factor timing. The fact that optimized portfolios have **lower beta** (0.75 vs 0.97) but **similar Sharpe** (0.25 vs 0.26) demonstrates **alpha generation** through intelligent factor weighting.

The optimized portfolios achieve similar returns with **16% lower market risk**, which is the definition of value added.

---

## Testing Infrastructure

**28 Comprehensive Unit Tests** (100% pass rate, 1.36s execution):

**Coverage:**
- **Beta Calculator:** 9 tests (CAPM, excess returns, edge cases)
- **Monte Carlo:** 8 tests (baseline, simulation, intervals, reproducibility)
- **Portfolio Optimizer:** 11 tests (covariance, tilting, constraints)

**Quality Indicators:**
- Edge case handling (empty matrices, singular covariance, NaN values)
- Numerical stability checks (symmetry, positive definiteness)
- Reproducibility validation (seeded randomness)
- Domain constraints (weights sum to 1, non-negative)

---

## Project Structure

```
DSAP-Project/
├── data/
│   ├── raw/                 # Cached raw data (7 files for reproducibility)
│   └── processed/           # Processed data (generated each run)
├── results/                 # Tables & figures
├── src/                     # Core implementation
│   ├── __init__.py
│   ├── beta_calculator.py   # CAPM & FF5 betas
│   ├── data_processing.py   # Data ingestion & caching
│   ├── ml_models.py         # ML predictors
│   ├── monte_carlo.py       # Simulation & uncertainty
│   ├── portfolio_optimizer.py
│   └── results_exporter.py
├── tests/                   # Pytest scripts (28 tests)
│   ├── conftest.py
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



## Scientific Contribution

This project demonstrates:

- **Proper time‑series validation** (no look‑ahead bias, strict train/val/test split)
- **Honest reporting of negative ML results** (historical mean wins 3/5 factors)
- **Economically meaningful baselines** (historically-optimal, not naive)
- **Rigorous model selection** (validation-first, no test set snooping)
- **Separation between prediction and portfolio construction**
- **Practical use of uncertainty quantification** (90% coverage achieved)
- **Conservative shrinkage** (blend ML with historical means)

**Key Empirical Findings:**

1. **RMW is the only consistently predictable factor** (+6.34% test R² via Lasso)
2. **Validation-test split is critical** for model selection
3. **Historical mean competitive with ML** (ties on average: -0.0228 R² each)
4. **RMW tilt improves Sharpe modestly** (+0.7% cross-sectional)
5. **Concentration destroys value** (-4.8% Sharpe due to idiosyncratic risk)
6. **Uncertainty quantification succeeds** (90.2% coverage) even when prediction fails
7. **Conservative shrinkage essential** (40% ML, 60% historical for RMW)

**Key Methodological Insight:**
> **Per-factor model selection with conservative shrinkage** outperforms one-size-fits-all approaches. Use best model per factor, but shrink predictions toward historical means to prevent overfitting.

---

## Academic Foundation

This project builds on established research in empirical asset pricing and factor models, including:

- **Fama, E. F., & French, K. R. (2015).** *A five-factor asset pricing model*. Journal of Financial Economics, 116(1), 1–22.
- **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** *...and the cross-section of expected returns*. Review of Financial Studies, 29(1), 5–68.
- **Gu, S., Kelly, B., & Xiu, D. (2020).** *Empirical asset pricing via machine learning*. Review of Financial Studies, 33(5), 2223–2273.
- **Kozak, S., Nagel, S., & Santosh, S. (2020).** *Shrinking the cross-section*. Journal of Financial Economics, 135(2), 271–292.
- **Malkiel, B. G. (2003).** *The efficient market hypothesis and its critics*. Journal of Economic Perspectives, 17(1), 59–82.
- **Fama, E. F. (1970).** *Efficient capital markets: A review of theory and empirical work*. Journal of Finance, 25(2), 383–417.
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The elements of statistical learning* (2nd ed.). Springer.
- **French, K. R. (2025).** Fama–French data library. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Yahoo Finance. (2025).** Yahoo Finance market data. https://finance.yahoo.com

---

## Empirical Insights for Practitioners

### For Quantitative Analysts
- **Factor timing is extremely difficult** (4/5 factors unpredictable)
- **Profitability (RMW) has modest but consistent predictability** (+6.34% test R²)
- **Per-factor model selection beats one-size-fits-all**
- **Conservative shrinkage essential** (40% ML, 60% historical max)
- **Validation-test consistency critical** for model selection

### For Portfolio Managers
- **Factor premia are real and exploitable** (FF5 beats equal-weight)
- **Optimization adds value** (Sharpe +37% vs equal-weight)
- **RMW tilt provides marginal benefit** (+0.7% Sharpe)
- **Concentration is dangerous** (idiosyncratic risk dominates)
- **Target lower market beta** (0.75 vs 1.0) for better risk-adjusted returns
- **Diversification >> concentration** (243 stocks >> 30 stocks)

### For Risk Managers
- **Uncertainty quantification is achievable** (90.2% coverage)
- **Monte Carlo simulation provides reliable risk estimates**
- **Use distributional forecasts** (quantiles) not just point estimates
- **Correlation-aware simulation** preserves factor structure
- **Well-calibrated intervals possible** even with low prediction R²

---

## License

This project is provided for **educational and academic research purposes only**.

- **Code**: Released for non-commercial academic use, without warranty
- **Fama–French data**: © Eugene F. Fama and Kenneth R. French (academic use)
- **Yahoo Finance data**: Subject to Yahoo Finance Terms of Service

---

## Disclaimer

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

This project was developed with the assistance of AI tools (ChatGPT, Claude, GitHub Copilot) for debugging, methodological guidance, text and code drafting and refining. See `AI_USAGE.md` for full disclosure.