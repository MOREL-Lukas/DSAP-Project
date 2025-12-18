**Author**: Lukas Morel  
**Student ID**: 22414569  
**Course**: Data Science & Applied Programming  
**Date**: December 2025

# 📊 Fama–French 5‑Factor Prediction and Portfolio Construction

## Overview

This project implements a **full end-to-end quantitative research pipeline** for the Fama–French Five-Factor model. It combines historical factor data, S&P 500 stock returns, macroeconomic indicators, and machine-learning techniques to evaluate **whether factor returns are predictable** and **how such information should (or should not) be used in portfolio construction**.

The core result is deliberately *scientifically conservative*:

> **Most Fama–French factors are not meaningfully predictable out-of-sample.**

However, the project also shows that:

- selective use of machine learning can add value **at the portfolio-construction margin**, and
- uncertainty quantification is often more useful than point prediction.

---

## Key Research Questions

1. Can machine‑learning models predict monthly Fama–French factor returns better than a historical mean?
2. If predictability is weak, can ML still be used responsibly in portfolio construction?
3. Does tilting toward profitability (RMW) improve risk‑adjusted returns relative to a rational historical benchmark?

---

## Main Findings (Executive Summary)

### Factor Predictability

- **4 out of 5 factors (Mkt‑RF, SMB, HML, CMA)** show **no robust out‑of‑sample predictability**.
- **RMW (Profitability)** is the *only* factor with **positive and stable test‑set R²**, consistent with its slow‑moving economic nature.
- Complex models (Random Forest, Gradient Boosting) exhibit **validation–test collapse**, highlighting regime sensitivity and overfitting risk.
- A profitability factor tilted portfolio can preserve alpha and sharpe ratio while decreasing market beta
**Conclusion:** Monthly factor returns are close to a random walk, consistent with the Efficient Market Hypothesis.

---

### Portfolio‑Level Results

Three portfolio strategies are evaluated:

1. **Baseline (No Explicit Tilt)**  
   A *mean–variance–optimal FF5 portfolio* built using **historical factor premia**.

2. **RMW Tilt (0.3)**  
   The same portfolio with a **moderate profitability tilt**, applied *after* optimization.

3. **RMW‑50 (Concentrated)**  
   A deliberately concentrated portfolio emphasizing high‑RMW stocks.

**Results:**

- The **Baseline portfolio already loads strongly on RMW**, because profitability has a positive historical premium.
- The **RMW Tilt portfolio slightly improves Sharpe** while preserving diversification.
- The **concentrated strategy underperforms**, as idiosyncratic risk dominates factor exposure.

**Final Recommendation:**
> Use a **diversified FF5 portfolio with a mild RMW tilt**. Avoid concentration.

---

## Why the Baseline Is a Strong Benchmark

The baseline is **not** an equal‑weight or market portfolio. It is:

> **A historically optimal FF5 allocation that already exploits known factor premia.**

This makes it a *hard* benchmark to beat and avoids misleading comparisons against naive portfolios. The research question is therefore:

> *Does explicit ML‑informed emphasis on profitability add value beyond what a rational allocator already does?*

This framing is intentional and academically sound.

---

## Methodology



### Data Sources

- **S&P 500 constituents:** DataHub
- **Stock prices:** Yahoo Finance (monthly, 1990–2025)
- **Fama–French 5 factors:** Kenneth French Data Library
- **Macroeconomic indicators:** FRED (GDP, inflation, rates, spreads, VIX, oil)

---

### Feature Engineering

~80 predictive features grouped into:

- Cross‑sectional market statistics (factor spreads, dispersion)
- Lagged factor returns and rolling statistics
- Market regime indicators (volatility, trend, breadth)
- Macroeconomic variables

All features are constructed **strictly using information available at time t**.

---

### Model Training and Evaluation

- **Temporal split (no shuffling):**
  - Train: ~70%
  - Validation: ~20%
  - Test: ~10%

- **Models evaluated:**
  - Historical Mean (benchmark)
  - Random Forest
  - Gradient Boosting
  - Ridge Regression
  - Lasso Regression

- **Primary metric:** Out‑of‑sample R²

Model selection is performed **only on the validation set**.

---

### Uncertainty Quantification

A **correlation‑aware Monte Carlo simulator** generates predictive distributions:

- Historical means and covariances are estimated from training data
- ML predictions (where applicable) are used as time‑varying conditional means

**Result:**

- ~90% empirical coverage for 95% prediction intervals
- Well‑calibrated uncertainty despite weak point forecasts

This demonstrates that **risk estimation can succeed even when prediction fails**.

---

## Portfolio Construction

### Steps

1. Estimate **FF5 betas** for all S&P 500 stocks via OLS
2. Compute expected stock returns as:
   
   $$
   \mathbb{E}[R] = B\,\mu_f
   $$

3. Construct a **tangency‑style portfolio** under realistic constraints
4. Apply a **post‑optimization RMW tilt**
5. Evaluate via rolling out‑of‑sample backtests

---

## Backtesting Results

- Positive but modest Sharpe ratios
- Market beta < 1 (defensive positioning)
- Positive CAPM alpha at the portfolio level
- Performance degrades gracefully out‑of‑sample

**Interpretation:**
Returns come from *diversification and factor exposure*, not factor timing.

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
├── environment.yml
├── main.py                  # Full pipeline entry point
└── README.md
```

---

## How to Run

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

### Run the Full Pipeline

```bash
python main.py
```

A FRED API key is required for macroeconomic features and must be provided in `main.py` as fred_api_key.

---

## Scientific Contribution

This project demonstrates:

- Proper time‑series validation (no look‑ahead bias)
- Honest reporting of negative ML results
- Economically meaningful baselines
- Separation between prediction and portfolio construction
- Practical use of uncertainty quantification

**Key takeaway:**
> In efficient markets, ML rarely improves forecasts — but it *can* still improve decisions when used carefully.

---

## Academic Foundation

This project builds on established research in empirical asset pricing and factor models, including:

- Fama, E. F., & French, K. R. (2015). *A five-factor asset pricing model*. Journal of Financial Economics, 116(1), 1–22.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). *… and the cross-section of expected returns*. Review of Financial Studies, 29(1), 5–68.
- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical asset pricing via machine learning*. Review of Financial Studies, 33(5), 2223–2273.
- Kozak, S., Nagel, S., & Santosh, S. (2020). *Shrinking the cross-section*. Journal of Financial Economics, 135(2), 271–292.
- Malkiel, B. G. (2003). *The efficient market hypothesis and its critics*. Journal of Economic Perspectives, 17(1), 59–82.
-Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. Journal of Finance, 25(2), 383–417.
-Federal Reserve Bank of St. Louis. (2025). Federal Reserve Economic Data (FRED). https://fred.stlouisfed.org
-Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning (2nd ed.). Springer.
-French, K. R. (2025). Fama–French data library. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
-Yahoo Finance. (2025). Yahoo Finance market data. https://finance.yahoo.com

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

- Past performance does not guarantee future results
- Transaction costs, taxes, and liquidity effects are not fully modeled
- Survivorship bias may be present in S&P 500 data

Always consult a licensed financial professional before making investment decisions.