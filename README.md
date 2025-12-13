# 📊 DSAP Project: Machine Learning Prediction of Fama-French Five Factors

This project builds a **machine learning pipeline** to predict monthly returns of the **Fama-French 5-Factor Model** using S&P 500 stock data, macroeconomic indicators, and market conditions. The system downloads 25+ years of historical data, engineers 84+ predictive features, trains multiple ML models, and compares their performance against a historical average baseline.

MOREL, Lukas
22414569
---

## 🎯 Project Overview

**Goal**: Predict next month's Fama-French factor returns (Market Risk Premium, Size, Value, Profitability, Investment) to aid in portfolio construction and risk management.

**Key Innovation**: Combines cross-sectional stock characteristics, lagged factor returns, macroeconomic variables (GDP, inflation, interest rates, VIX, oil), and market condition indicators to forecast factor performance.

**Models Evaluated**: Random Forest, Gradient Boosting, Ridge Regression, Lasso Regression, and Historical Average Baseline.

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

---

## 📦 Project Structure

```
DSAP-Project/
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── data_loader.py                   # Data download & preprocessing
│   ├── models.py                        # Feature engineering & dataset creation
│   ├── factor_predictor.py              # ML model training & prediction
│   └── evaluation.py                    # Model comparison & evaluation
│
├── data/
│   ├── raw/                             # Downloaded raw data
│   │   ├── sp500_tickers.csv
│   │   ├── sp500_companies.csv
│   │   ├── French_Library_data.csv
│   │   └── ...
│   └── processed/                       # Processed datasets
│       ├── sp500_monthly_returns.csv
│       ├── sp500_monthly_prices.csv
│       ├── Fama_French.csv
│       ├── sp500_ff5_classifications.csv
│       ├── factor_ml_dataset.csv        # Basic dataset (18 features)
│       └── factor_ml_dataset_enhanced.csv  # Enhanced dataset (84 features)
│
├── results/                             # Model outputs & visualizations
│   ├── factor_predictions.png           # Actual vs predicted plots
│   ├── model_comparison.png             # Model comparison visualization
│   ├── model_comparison_detailed.csv    # Full metrics for all models
│   └── ...
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
1. Download S&P 500 company list and stock prices (2000-2025)
2. Download Fama-French 5-Factor data from Kenneth French Data Library
3. Classify stocks by Size, Value, Profitability, and Investment factors
4. Engineer 84 predictive features (cross-sectional, lagged, macro, market conditions)
5. Train and evaluate 5 models (4 ML + 1 baseline)
6. Generate performance reports and visualizations

**Runtime**: ~5-10 minutes (most time spent downloading stock prices)

---

## 🧠 Methodology

### Step 1: Data Collection

**Sources**:
- **S&P 500 constituents**: DataHub (via web scraping)
- **Stock prices**: Yahoo Finance API (monthly, 2000-2025)
- **Fama-French factors**: Kenneth French Data Library
- **Macroeconomic data**: FRED API (Federal Reserve Economic Data)
- **Market indicators**: VIX, WTI Crude Oil prices

### Step 2: Stock Classification

Each S&P 500 stock is classified into binary categories:
- **Size**: Small vs Big (by market capitalization)
- **Value**: High vs Low (by book-to-market ratio)
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

**Temporal Split** (preserves time series structure):
- Training: 70% (2000-2017)
- Validation: 20% (2018-2023)
- Test: 10% (2023-2025)

**Models Compared**:
1. **Historical Average** (baseline) - Simply predicts the mean
2. **Random Forest** - Non-linear, captures interactions
3. **Gradient Boosting** - Sequential learning, strong performance
4. **Ridge Regression** - Linear with L2 regularization
5. **Lasso Regression** - Linear with L1 regularization (feature selection)

**Evaluation Metrics**:
- R² (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Improvement over baseline

---

## 📊 Results Summary

### Best Performing Model: Random Forest

**Test Set Performance (R² scores)**:

| Factor | Historical Avg | Random Forest | Improvement |
|--------|----------------|---------------|-------------|
| **Mkt-RF** (Market) | -0.004 | **+0.090** | +9.4% |
| **SMB** (Size) | -0.002 | **+0.080** | +8.2% |
| **HML** (Value) | -0.003 | **+0.014** | +1.7% |
| **RMW** (Profitability) | -0.001 | **+0.090** | +9.1% |
| **CMA** (Investment) | -0.002 | **-0.023** | -2.1% |

**Average R²**: +0.050 (5% predictive power)

### Key Feature Importance

**Top predictors across all factors**:
1. **Treasury yields** (10-year, 3-month) - 5-10% importance
2. **Lagged factor returns** (1-6 months) - 3-7% importance
3. **VIX** (market volatility) - 3-5% importance
4. **Inflation** - 4-5% importance for market factor
5. **Cross-sectional dispersion** - 7-11% importance for profitability

### Model Comparison

**Average R² across all factors**:
- Random Forest: **+0.050**
- Gradient Boosting: **+0.045**
- Ridge Regression: **+0.020**
- Lasso Regression: **+0.015**
- Historical Average: **-0.002** (baseline)

**Factors improved vs baseline**: 4 out of 5 factors show positive R²

---

## 🎓 Research Insights

### Why Factor Returns Are Hard to Predict

1. **Market Efficiency**: If factors were easily predictable, arbitrage would eliminate the returns
2. **High Noise-to-Signal Ratio**: Monthly returns are extremely volatile
3. **Regime Changes**: Economic conditions shift unpredictably
4. **Limited History**: Even 25 years is limited for ML training

### What Works

1. **Macroeconomic variables matter**: Interest rates, inflation, and credit spreads are top predictors
2. **Momentum exists**: Past factor performance helps predict future performance
3. **Volatility regimes**: VIX and dispersion capture market stress
4. **Non-linear models**: Random Forest outperforms linear regression

### Practical Applications

**Positive R² of 5-9%** enables:
- **Tactical asset allocation**: Overweight factors with positive predicted returns
- **Risk management**: Avoid factors with negative predictions
- **Portfolio timing**: Adjust factor exposures based on macro conditions
- **Performance attribution**: Understand what drives factor returns

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
fred_api_key = "YOUR_API_KEY_HERE"
```

### Adjust Train/Val/Test Split

Edit `main.py`:
```python
X_train, X_val, X_test, y_train, y_val, y_test, ... = \
    predictor.train_val_test_split_temporal(X, y, dates, 
                                            train_ratio=0.8,  # 80% train
                                            val_ratio=0.1)    # 10% val, 10% test
```

---

## 📈 Output Files

### Data Files
- `data/processed/factor_ml_dataset_enhanced.csv` - Full feature matrix (307 months × 84 features)
- `results/model_comparison_detailed.csv` - Performance metrics for all models

### Visualizations
- `results/factor_predictions.png` - Actual vs predicted time series for all 5 factors
- `results/model_comparison.png` - Bar chart and heatmap comparing all models

### Console Output
- Feature importance rankings (top 10 per factor)
- Model performance comparison table
- Improvement over historical average baseline
- Next month's factor predictions

---

## 🧩 Dependencies

```text
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
tqdm>=4.65.0
fredapi>=0.5.0  # Optional, for full macro features
```

---

## 📚 References & Citations

### Academic Foundation
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.

### Data Sources
- **Fama-French Factors**: Kenneth French Data Library  
  Copyright 2025 Eugene F. Fama and Kenneth R. French  
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

- **Macroeconomic Data**: Federal Reserve Economic Data (FRED®)  
  This project uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.  
  https://fred.stlouisfed.org

- **Stock Data**: Yahoo Finance via `yfinance` library

- **S&P 500 Constituents**: DataHub  
  https://datahub.io/core/s-and-p-500-companies

### AI Tools Used

OpenAI. 2025. ChatGPT (GPT-4). Accessed October-December 2025. https://chat.openai.com/

Anthropic. 2025. Claude (Claude 3.5 Sonnet). Accessed December 11, 2025. https://claude.ai/

Microsoft. 2025. GitHub Copilot (GPT-4o mini). Accessed October-December 2025. https://github.com/features/copilot

---

## 📝 License

This project is for educational purposes. Data sources have their own licenses:
- Fama-French data: Available for academic research
- FRED data: Public domain (U.S. government data)
- Yahoo Finance: Subject to their terms of service

---

## 👤 Author

Lukas Morel  
University Project - Data Science & Applied Programming  
December 2025

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The predictions are not investment advice. Past performance does not guarantee future results. Factor returns are notoriously difficult to predict, and even positive R² scores of 5-10% represent only modest predictive power. Always consult a financial advisor before making investment decisions.
