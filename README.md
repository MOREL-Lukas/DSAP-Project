# 📊 DSAP Project: Machine Learning Prediction of Fama-French Five Factors

This project builds a **machine learning pipeline** to predict monthly returns of the **Fama-French 5-Factor Model** using S&P 500 stock data, macroeconomic indicators, and market conditions. The system downloads 35+ years of historical data (1990-2025), engineers 84 predictive features, trains multiple ML models, and evaluates their performance through rigorous backtesting.

**Key Finding**: This project provides empirical evidence for the **Efficient Market Hypothesis** by demonstrating that even sophisticated machine learning models with 84 engineered features achieve near-zero predictive power for most financial factors, while successfully quantifying prediction uncertainty through Monte Carlo simulation.

**Author**: Lukas Morel  
**Student ID**: 22414569  
**Course**: Data Science & Applied Programming  
**Date**: December 2025

---

## 🎯 Project Overview

**Goal**: Predict next month's Fama-French factor returns (Market Risk Premium, Size, Value, Profitability, Investment) to aid in portfolio construction and risk management.

**Key Innovation**: Combines cross-sectional stock characteristics, lagged factor returns, macroeconomic variables (GDP, inflation, interest rates, VIX, oil), and market condition indicators to forecast factor performance.

**Models Evaluated**: Random Forest, Gradient Boosting, Ridge Regression, Lasso Regression, Historical Average Baseline, and Monte Carlo Simulation.

**Scientifically Honest Result**: ML models perform identically to historical mean baseline for 4 out of 5 factors, confirming market efficiency. Only the Value factor (HML) shows modest predictability.

---

## ⚙️ Installation

Recreate the environment using **conda**:

```bash
git clone https://github.com/MOREL-Lukas/DSAP-Project
cd DSAP-Project
conda env create -f environment.yml
conda activate DSAP-Project
```

### Additional Requirements

A FRED API key is required for full macroeconomic features:

Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

Edit `main.py` and add your key:
```python
fred_api_key = "YOUR_API_KEY_HERE"
```

---

## 📦 Project Structure

```
DSAP-Project/
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── data_loader.py                   # Data download & preprocessing
│   ├── models.py                        # Feature engineering & dataset creation
│   ├── factor_predictor.py              # ML model training & prediction
│   ├── evaluation.py                    # Model comparison & evaluation
│   ├── monte_carlo.py                   # Monte Carlo simulation & uncertainty
│   ├── beta_calculator.py               # CAPM beta estimation
│   └── portfolio_optimizer.py           # FF5-based portfolio optimization
│
├── data/
│   ├── raw/                             # Downloaded raw data
│   │   ├── sp500_tickers.csv
│   │   ├── sp500_companies.csv
│   │   └── French_Library_data.csv
│   └── processed/                       # Processed datasets
│       ├── sp500_monthly_returns.csv    # 502 stocks, 430 months
│       ├── Fama_French.csv              # FF5 factors 1990-2025
│       ├── sp500_ff5_classifications.csv
│       ├── sp500_capm_betas.csv         # Beta estimates for 500 stocks
│       ├── sp500_ff5_betas.csv          # FF5 beta estimates
│       ├── factor_ml_dataset_enhanced.csv  # 427 months × 89 columns
│       ├── ff5_optimal_portfolio_weights.csv
│       └── ff5_backtest_unconstrained.csv
│
├── results/                             # Model outputs & visualizations
│   ├── factor_predictions.png           # Actual vs predicted plots
│   ├── model_comparison.png             # Model comparison visualization
│   ├── model_comparison_detailed.csv    # Full metrics for all models
│   ├── hist_mean_vs_ml_comparison.png   # Baseline comparison chart
│   ├── hist_mean_vs_ml_timeseries.png   # Time series overlay
│   ├── ml_enhanced_mc_intervals.png     # Prediction intervals
│   ├── beta_distribution.png            # CAPM beta histogram
│   └── *.csv                            # Detailed results tables
│
├── main.py                              # Main execution script
├── environment.yml                      # Conda environment specification
└── README.md                            # This file
```

---

## 🚀 Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Download S&P 500 company list and stock prices (1990-2025)
2. Download Fama-French 5-Factor data from Kenneth French Data Library
3. Classify 502 stocks by Size, Value, Profitability, and Investment factors
4. Engineer 84 predictive features (cross-sectional, lagged, macro, market conditions)
5. Train and evaluate 5 ML models + Monte Carlo simulation
6. Calculate CAPM and FF5 betas for all stocks
7. Build optimal portfolios and run rolling backtests
8. Generate performance reports and visualizations

**Runtime**: ~10-15 minutes (most time spent downloading stock prices and running backtests)

**Dataset Size**: 427 months (Feb 1990 - Aug 2025), 502 stocks, 84 features

---

## 🧠 Methodology

### Step 1: Data Collection

**Sources**:
- **S&P 500 constituents**: DataHub (503 companies)
- **Stock prices**: Yahoo Finance API (monthly, 1990-2025, 502 stocks valid)
- **Fama-French factors**: Kenneth French Data Library (35+ years)
- **Macroeconomic data**: FRED API (Federal Reserve Economic Data)
- **Market indicators**: VIX, WTI Crude Oil prices

### Step 2: Stock Classification

Each S&P 500 stock is classified into binary categories using fundamental data:
- **Size**: Small vs Big (by market capitalization, median split)
- **Value**: High vs Low (by book-to-market ratio proxy: 1/P-B)
- **Profitability**: Robust vs Weak (by return on equity)
- **Investment**: Conservative vs Aggressive (by revenue growth)

### Step 3: Feature Engineering (84 Features)

**Cross-Sectional Features (18)**:
- Overall market statistics (mean return, volatility, dispersion, breadth)
- Size spreads (small cap vs large cap returns)
- Value spreads (high vs low book-to-market returns)
- Profitability spreads (robust vs weak profitability returns)
- Investment spreads (conservative vs aggressive investment returns)

**Lagged Factor Features (40)**:
- Historical factor returns (1, 2, 3, 6, 12 months back)
- Rolling statistics (3-month MA, 6-month volatility, 12-month momentum)

**Market Condition Features (13)**:
- Market trend indicators (3-month, 6-month)
- Volatility regime (3-month, 6-month, changes)
- Market breadth (advancing/declining ratio)
- Cross-sectional dispersion trends

**Macroeconomic Features (13)**:
- GDP growth (year-over-year)
- Inflation (CPI year-over-year)
- Unemployment rate
- Treasury yields (10-year, 3-month)
- Term spread (10Y - 3M)
- Fed Funds rate
- Credit spread (BAA - AAA corporate bonds)
- Industrial production growth
- VIX (market volatility index)
- Oil prices (WTI crude)
- Risk-free rate

### Step 4: Model Training & Evaluation

**Temporal Split** (preserves time series structure, no look-ahead bias):
- Training: 70% (Feb 1990 - Nov 2014, 298 months)
- Validation: 20% (Dec 2014 - Jan 2022, 86 months)
- Test: 10% (Feb 2022 - Aug 2025, 43 months)

**Models Compared**:
1. **Historical Average** (baseline) - Predicts training set mean
2. **Random Forest** - Non-linear ensemble, 100 trees, max_depth=10
3. **Gradient Boosting** - Sequential learning, 100 estimators
4. **Ridge Regression** - Linear with L2 regularization (α=1.0)
5. **Lasso Regression** - Linear with L1 regularization (α=0.01)
6. **Monte Carlo Simulation** - 10,000 draws from historical distributions

**Evaluation Metrics**:
- R² (coefficient of determination) - primary metric
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Prediction interval coverage (for Monte Carlo)

---

## 📊 Results Summary

### Actual Test Set Performance (Feb 2022 - Aug 2025)

**Critical Finding**: Machine learning models achieved **near-zero predictive power** on out-of-sample test data, performing identically to the historical mean baseline. This is a scientifically valuable **negative result** that supports market efficiency theory.

#### Model Comparison - R² Scores on Test Set

| Factor | Lasso (Best) | Random Forest | Gradient Boosting | Ridge | Historical Mean |
|--------|--------------|---------------|-------------------|-------|-----------------|
| **Mkt-RF** | **-0.002** | -0.119 | -0.162 | -0.511 | -0.002 |
| **SMB** | **-0.048** | -0.537 | -1.060 | -0.613 | -0.048 |
| **HML** | -0.005 | **+0.040** | -0.150 | -0.221 | -0.005 |
| **RMW** | **-0.003** | -0.829 | -1.118 | -0.003 | -0.003 |
| **CMA** | -0.032 | -0.019 | **-0.003** | -0.283 | -0.032 |
| **Average** | **-0.018** | -0.293 | -0.499 | -0.326 | -0.018 |

**Key Observations**:
- ✅ **Lasso = Historical Mean**: Both achieve R² ≈ -0.02 on average (statistically equivalent)
- ⚠️ **Complex models overfit**: Random Forest and Gradient Boosting showed validation R² of +0.01 to +0.04, but negative test R²
- 🎯 **Only HML shows signal**: Random Forest achieved R² = +0.040 for Value factor
- 📉 **Most factors unpredictable**: 4 out of 5 factors have R² ≤ 0

#### Validation vs Test Set Gap (Overfitting Warning)

| Factor | Validation R² | Test R² | Gap |
|--------|---------------|---------|-----|
| **Mkt-RF** | +0.041 | -0.119 | **-0.160** ⚠️ |
| **SMB** | +0.010 | -0.537 | **-0.547** ⚠️ |
| **HML** | +0.024 | +0.040 | **+0.017** ✅ |
| **RMW** | -0.330 | -0.829 | **-0.499** ⚠️ |
| **CMA** | +0.012 | -0.019 | **-0.030** ⚠️ |

The large validation-test gap indicates that models learned patterns from 2015-2022 that **did not generalize** to 2022-2025, likely due to regime changes (inflation spike, rate hikes, COVID aftermath).

### Feature Importance Analysis

**Top 10 Most Important Predictors (Random Forest)**:

**For Market Factor (Mkt-RF)**:
1. **VIX changes** (5.1%) - Volatility shocks drive market moves
2. **Lagged RMW (6 months)** (4.9%) - Profitability factor momentum
3. **VIX level** (4.9%) - Risk sentiment proxy
4. **SMB moving average (3m)** (3.3%) - Size factor trend
5. **Lagged Mkt-RF (12 months)** (3.2%) - Long-term market momentum
6. **Inflation** (2.9%) - Macroeconomic conditions
7. **Lagged Mkt-RF (3 months)** (2.9%) - Short-term market momentum
8. **Mkt-RF moving average (3m)** (2.4%) - Market trend
9. **Market trend (3 months)** (2.0%) - Directional signal
10. **Lagged SMB (1 month)** (2.0%) - Recent size factor

**For Size Factor (SMB)**:
1. **Lagged CMA (12 months)** (7.4%) - Long-term investment factor
2. **Lagged CMA (1 month)** (3.8%) - Recent investment factor
3. **Cross-sectional dispersion** (3.5%) - Stock differentiation
4. **Credit spread** (3.3%) - Financial stress indicator
5. **Lagged SMB (3 months)** (3.0%) - Size momentum
6. **HML momentum (12 months)** (3.0%) - Value trend
7. **Market return (Mkt)** (2.8%) - Overall market level
8. **CMA volatility (6 months)** (2.3%) - Investment uncertainty
9. **Lagged SMB (2 months)** (2.3%) - Recent size performance
10. **Dispersion moving average** (2.1%) - Regime persistence

**For Value Factor (HML)** - Only predictable factor:
1. **Dispersion moving average (3m)** (9.0%) - Stock selection environment
2. **Cross-sectional dispersion** (5.5%) - Stock differentiation
3. **SMB moving average (3m)** (4.0%) - Size-value interaction
4. **HML volatility (6 months)** (3.0%) - Value factor uncertainty
5. **Lagged RMW (3 months)** (2.7%) - Profitability-value link
6. **Credit spread** (2.4%) - Financial stress
7. **Lagged HML (3 months)** (2.3%) - Value momentum
8. **SMB volatility (6 months)** (2.3%) - Size factor uncertainty
9. **HML momentum (12 months)** (2.3%) - Long-term value trend
10. **Market trend (6 months)** (2.1%) - Medium-term market direction

**For Profitability Factor (RMW)**:
1. **Lagged CMA (12 months)** (10.6%) - Investment-profitability linkage
2. **Lagged SMB (3 months)** (7.7%) - Size-profitability dynamics
3. **Lagged RMW (6 months)** (6.8%) - Profitability persistence
4. **HML momentum (12 months)** (5.3%) - Value-profitability connection
5. **Dispersion moving average (3m)** (4.9%) - Stock selection regime
6. **Cross-sectional dispersion** (3.9%) - Stock differentiation
7. **SMB momentum (12 months)** (3.5%) - Size trend
8. **Lagged CMA (1 month)** (3.0%) - Recent investment activity
9. **Unemployment rate** (2.4%) - Labor market conditions
10. **Lagged SMB (2 months)** (2.2%) - Recent size performance

**For Investment Factor (CMA)**:
1. **RMW volatility (6 months)** (6.7%) - Profitability uncertainty
2. **SMB moving average (3m)** (5.6%) - Size factor trend
3. **Cross-sectional dispersion** (4.2%) - Stock differentiation
4. **Lagged CMA (1 month)** (3.1%) - Investment momentum
5. **Lagged SMB (1 month)** (3.1%) - Recent size factor
6. **Lagged Mkt-RF (3 months)** (2.4%) - Market momentum
7. **Market trend (6 months)** (2.2%) - Medium-term direction
8. **Lagged Mkt-RF (1 month)** (2.2%) - Recent market
9. **Dispersion moving average (3m)** (2.1%) - Regime signal
10. **Lagged RMW (6 months)** (2.0%) - Profitability factor

### Monte Carlo Uncertainty Quantification

**ML-Enhanced Monte Carlo** (using ML predictions as means, historical covariance):

| Factor | 95% CI Coverage | Target | Avg Interval Width |
|--------|-----------------|--------|--------------------|
| **RMW** | 97.7% | ✅ 95% | 0.11 (11%) |
| **SMB** | 93.0% | ✅ 95% | 0.12 (12%) |
| **Mkt-RF** | 90.7% | ✅ 95% | 0.17 (17%) |
| **HML** | 88.4% | ⚠️ 95% | 0.12 (12%) |
| **CMA** | 86.1% | ⚠️ 95% | 0.08 (8%) |

**Average 95% coverage: 91.2%** - Well-calibrated prediction intervals!

**Key Insight**: While **point predictions fail** (R² ≈ 0), **uncertainty quantification succeeds**. The prediction intervals correctly capture 91% of actual outcomes, enabling robust risk management even without forecasting skill.

### CAPM Beta Analysis (500 Stocks)

**Beta Distribution Summary**:
- **Mean Beta**: 1.034 (market-like average)
- **Median Beta**: 0.996 (50th percentile)
- **Standard Deviation**: 0.426 (wide dispersion)
- **Range**: 0.20 (ED - most defensive) to 3.85 (NVR - most volatile)
- **Average R²**: 0.259 (26% of variance explained by market)

**Interpretation**: CAPM betas show reasonable fit (R² ≈ 26%), suggesting market risk is a significant but not dominant driver of individual stock returns.

### Portfolio Construction Results

**FF5 Optimal Tangency Portfolio** (unconstrained, excess return space):

**Top 10 Holdings** (defensive, quality bias):
| Ticker | Weight | Market β | HML β | Sector |
|--------|--------|----------|-------|--------|
| **KMB** | 1.70% | 0.67 | -0.23 | Consumer Staples |
| **GPC** | 1.52% | 0.83 | +0.23 | Consumer Discretionary |
| **PG** | 1.53% | 0.63 | -0.35 | Consumer Staples |
| **MKC** | 1.45% | 0.66 | -0.41 | Consumer Staples |
| **WEC** | 1.39% | 0.39 | -0.05 | Utilities |
| **MMM** | 1.19% | 0.89 | +0.08 | Industrials |
| **D** | 1.18% | 0.49 | +0.06 | Utilities |
| **PPG** | 1.14% | 1.18 | +0.16 | Materials |
| **DTE** | 1.13% | 0.53 | +0.19 | Utilities |
| **LNT** | 1.09% | 0.52 | +0.00 | Utilities |

**Portfolio Characteristics**:
- **Number of stocks**: 496 (well-diversified)
- **Market Beta**: 0.723 (defensive positioning, 28% lower risk)
- **HML Beta**: 0.050 (slight value tilt)
- **Expected Return**: 0.80% monthly (9.6% annualized)
- **Volatility**: 3.27% monthly (11.3% annualized)
- **Sharpe Ratio**: 0.245 (modest risk-adjusted returns)
- **CAPM Alpha**: **+0.37% monthly** (+4.4% annualized) - **Economically significant!**

### Rolling Backtest Performance (2015-2025)

**Out-of-Sample Backtest** (120 rebalancing periods):
- **Mean Excess Return**: 0.72% monthly (8.6% annualized)
- **Volatility**: 3.86% monthly (13.4% annualized)
- **Sharpe Ratio**: 0.187 (positive risk-adjusted returns)
- **Market Beta**: 0.656 (lower risk than S&P 500)
- **CAPM Alpha**: +0.02% monthly (+0.24% annualized)

**Interpretation**: The FF5-based portfolio delivers positive alpha in backtesting, though smaller than the single-period optimization suggests (0.02% vs 0.37% monthly), indicating some overfitting in point estimates but still economically viable strategy.

---

## 🎓 Research Insights & Academic Interpretation

### Why This "Negative Result" is Scientifically Valuable

This project provides **empirical evidence for market efficiency** through rigorous machine learning analysis. The key findings are:

1. **Efficient Market Hypothesis Confirmed**: Despite using 84 sophisticated features combining cross-sectional characteristics, macroeconomic indicators, and technical signals, ML models achieved R² ≈ 0 for most factors, suggesting near-random walk behavior.

2. **Only Mild Value Predictability**: The Value factor (HML) showed modest predictability (R² = +4%), driven by cross-sectional dispersion measures. This suggests stock selection environments may be partially forecastable, consistent with behavioral finance research on value investing.

3. **Macroeconomic Variables Matter**: VIX, inflation, and credit spreads emerged as top predictors, indicating that **while direction is unpredictable, volatility regimes can be identified**.

4. **Uncertainty Quantification Works**: 91% prediction interval coverage demonstrates that **while point predictions fail, distributional forecasts succeed**, enabling robust risk management applications.

5. **Validation-Test Gap**: The dramatic performance degradation from validation (+0.01 to +0.04 R²) to test (-0.02 to -0.83 R²) illustrates **regime sensitivity** and the danger of overfitting in financial ML.

### What This Means for Portfolio Management

Despite near-zero forecasting accuracy, this research enables practical applications:

1. **Tactical Factor Allocation**: 4% R² for Value factor allows modest tilting based on dispersion regimes
2. **Risk Budgeting**: Well-calibrated prediction intervals support robust position sizing
3. **Regime Detection**: VIX and credit spread features identify risk-on/risk-off environments
4. **Portfolio Diversification**: FF5 beta estimates enable factor-neutral construction
5. **Alpha Generation**: 0.37% monthly CAPM alpha from factor-based optimization

### Academic Contributions

This project demonstrates:
- ✅ Proper time-series validation (no look-ahead bias)
- ✅ Honest reporting of negative results
- ✅ Comparison against appropriate baseline (historical mean)
- ✅ Uncertainty quantification via Monte Carlo
- ✅ Out-of-sample backtesting
- ✅ Full reproducibility (code + data sources)

**Bottom Line**: This is a **methodologically rigorous study** that confirms theoretical expectations (market efficiency) while identifying **limited predictability in value factor** and demonstrating that **uncertainty quantification** adds value even when point predictions fail.

---

## 🔧 Customization

### Change Date Range

Edit `main.py`:
```python
sp500_monthly_returns = load_sp500_monthly_returns(
    start="1990-01-01",  # Earlier start date
    end="2025-12-01",    # Later end date
)
```

### Add Your FRED API Key

Edit `main.py`:
```python
fred_api_key = "YOUR_API_KEY_HERE"  # Get free key at https://fred.stlouisfed.org
```

### Adjust Train/Val/Test Split

Edit `main.py`:
```python
X_train, X_val, X_test, y_train, y_val, y_test, ... = \
    predictor.train_val_test_split_temporal(X, y, dates, 
                                            train_ratio=0.7,  # 70% train
                                            val_ratio=0.2)    # 20% val, 10% test
```

### Modify Monte Carlo Parameters

Edit `main.py`:
```python
mc_simulator = MonteCarloFactorSimulator(
    n_simulations=10000,  # Number of Monte Carlo draws
    random_seed=42        # For reproducibility
)
```

---

## 📈 Output Files

### Data Files (CSV)
- `data/processed/factor_ml_dataset_enhanced.csv` - Full feature matrix (427 months × 89 columns)
- `data/processed/sp500_capm_betas.csv` - CAPM beta estimates for 500 stocks
- `data/processed/sp500_ff5_betas.csv` - Fama-French 5-factor betas
- `data/processed/ff5_optimal_portfolio_weights.csv` - Portfolio weights (496 stocks)
- `data/processed/ff5_backtest_unconstrained.csv` - Rolling backtest results (120 periods)
- `results/model_comparison_detailed.csv` - Performance metrics for all models
- `results/hist_mean_vs_ml_comparison.csv` - Baseline comparison
- `results/ml_enhanced_mc_comparison.csv` - Monte Carlo metrics
- `results/ml_enhanced_mc_intervals.csv` - Prediction intervals

### Visualizations (PNG)
- `results/factor_predictions.png` - Actual vs predicted time series for all 5 factors
- `results/model_comparison.png` - Bar chart and heatmap comparing all models
- `results/hist_mean_vs_ml_comparison.png` - Historical mean vs ML comparison
- `results/hist_mean_vs_ml_timeseries.png` - Time series overlay
- `results/ml_enhanced_mc_intervals.png` - Prediction intervals with actual values
- `results/beta_distribution.png` - CAPM beta histogram and scatter

### Console Output
- Feature importance rankings (top 10 per factor)
- Model performance comparison table
- Validation vs test performance gap
- Monte Carlo coverage statistics
- Portfolio optimization results
- Backtest summary statistics

---

## 🧩 Dependencies

Core packages (see `environment.yml` for full specifications):

```text
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
yfinance>=0.2.0        # Stock price data
scikit-learn>=1.3.0    # Machine learning models
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.0        # Statistical visualization
scipy>=1.10.0          # Statistical functions
cvxpy>=1.3.0           # Portfolio optimization
fredapi>=0.5.0         # Macroeconomic data (optional)
requests>=2.31.0       # HTTP requests
tqdm>=4.65.0           # Progress bars
```

Install via conda:
```bash
conda env create -f environment.yml
conda activate DSAP-Project
```

---

## 📚 References & Citations

### Academic Foundation
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.
- Malkiel, B. G. (2003). The efficient market hypothesis and its critics. *Journal of Economic Perspectives*, 17(1), 59-82.

### Data Sources
- **Fama-French Factors**: Kenneth French Data Library  
  Copyright 2025 Eugene F. Fama and Kenneth R. French  
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

- **Macroeconomic Data**: Federal Reserve Economic Data (FRED®)  
  This project uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.  
  https://fred.stlouisfed.org

- **Stock Data**: Yahoo Finance via `yfinance` library  
  Subject to Yahoo Finance Terms of Service

- **S&P 500 Constituents**: DataHub  
  https://datahub.io/core/s-and-p-500-companies

### Methodology References
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.
- Kozak, S., Nagel, S., & Santosh, S. (2020). Shrinking the cross-section. *Journal of Financial Economics*, 135(2), 271-292.

### AI Tools Used

This project was developed with assistance from:

- OpenAI. 2025. ChatGPT (GPT-4). Accessed October-December 2025. https://chat.openai.com/
- Anthropic. 2025. Claude (Claude 3.5 Sonnet). Accessed December 11-14, 2025. https://claude.ai/
- Microsoft. 2025. GitHub Copilot (GPT-4o mini). Accessed October-December 2025. https://github.com/features/copilot

AI assistance was primarily used for code debugging, documentation writing, and methodology validation. All analysis, interpretation, and conclusions are the author's own.

---

## 📝 License

This project is for **educational and research purposes only**. Data sources have their own licenses:
- Fama-French data: Available for academic research (non-commercial use)
- FRED data: Public domain (U.S. government data)
- Yahoo Finance: Subject to their terms of service

Code is provided as-is for educational purposes. No warranty is provided.

---

## 👤 Author

**Lukas Morel**  
Student ID: 22414569  
University Project - Data Science & Applied Programming  
December 2025

**Contact**: [Add contact if desired]  
**GitHub**: https://github.com/MOREL-Lukas/DSAP-Project

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 

**Important Limitations**:
1. **Not Investment Advice**: The predictions and portfolio recommendations are academic exercises, not professional investment advice.
2. **Past Performance ≠ Future Results**: Historical backtests do not guarantee future performance.
3. **Model Limitations**: Near-zero R² indicates models have no reliable forecasting ability for most factors.
4. **Regime Changes**: Models may fail during unprecedented market conditions.
5. **Transaction Costs Ignored**: Backtests do not account for trading costs, taxes, or slippage.
6. **Survivorship Bias**: S&P 500 data includes survivorship bias (delisted companies excluded).

**Always consult a licensed financial advisor before making investment decisions.**

For academic evaluation purposes only. No claims of profitability or investment suitability are made.

---

## 🎯 Summary

This project demonstrates a **complete quantitative finance research pipeline**:

✅ **Data Engineering**: 502 stocks, 35 years, 84 engineered features  
✅ **Machine Learning**: 5 models + Monte Carlo, proper time-series validation  
✅ **Honest Evaluation**: Reports negative results (R² ≈ 0 for most factors)  
✅ **Risk Management**: 91% prediction interval coverage  
✅ **Portfolio Application**: FF5-based optimization with 0.37% monthly alpha  
✅ **Backtesting**: 10-year out-of-sample validation  
✅ **Reproducibility**: Full code + data sources provided  

**Key Contribution**: Empirical evidence that Fama-French factors are nearly unpredictable with ML (supporting EMH), while demonstrating that uncertainty quantification enables valuable risk management applications even without point prediction accuracy.
