# 📊 DSAP Project: Fama–French 5-Factor & Beta-Neutral Portfolio Optimization

This project performs an end-to-end **data acquisition, multi-factor modeling, and portfolio optimization** process using the **Fama–French 5-Factor Model (FF5)**.  
It estimates β coefficients using machine learning, constructs a **beta-neutral portfolio** maximizing expected excess return, simulates its 2025 performance using **Monte Carlo** simulation , and backtests by comparing simulated vs. realized results.

---

## ⚙️ Installation

Recreate the environment using **conda**.

```bash
git clone https://github.com/MOREL-Lukas/DSAP-Project
cd DSAP-Project
conda env create -f environment.yml
conda activate DSAP-Project
```
---

## 📦 Project Structure Before Running Scripts

```
DSAP-Project/Before_Running
├── Data/
│   ├── 2025/                          # 2025 realized market data folder
│
├── Output/
│
├── requirements.txt
│
├── s01_Fetch_SP500_list.py            # Fetch S&P 500 symbols from Wikipedia
├── s02_main.py                        # Compute CAPM betas/alphas (2020–2024)
├── s03_main_ff5.py                    # Compute Fama–French 5-Factor betas
├── s04_Fetch_SP500_sectors.py         # Download sector mapping for each ticker
├── s05_compare_betas_yfinance.py      # Compare computed vs Yahoo Finance betas
├── s06_plot_betas.py                  # Plot CAPM vs Yahoo beta correlation
├── s07_beta_neutral.py                # Optimize FF5 beta-neutral long/short portfolio
├── s08_simulation.py                  # Monte Carlo simulation of expected 2025 returns
├── s09_2025data.py                    # Fetch realized S&P 500 & RF data for 2025
└── s10_comparaison.py                 # Compare simulated vs realized 2025 performance
```

---

## 📦 Project Structure After Running Scripts

```
DSAP-Project/After_Running
├── Data/
│   ├── 2025/                            # 2025 realized market data
│   │   ├── market_adj_close_2025.csv
│   │   ├── rf_irx_daily_2025.csv
│   │   ├── rf_irx_monthly_2025.csv
│   │   └── sp500_adj_close.csv          #
│   ├── fama_french_5factors_monthly.csv # 
│   ├── market_adj_close.csv             #  
│   ├── rf_irx_daily.csv                 # 
│   ├── rf_irx_monthly.csv               # Risk-free rate (3M T-Bill)
│   ├── rf_irx_yield_monthly.csv         # 
│   ├── sp500_adj_close.csv              # 
│   ├── sp500_sectors.csv                # Sector classification
│   └── yahoo_betas.csv                  #
├── Output/
│   ├── beta_comparison_scatter.png
│   ├── beta_comparison.csv 
│   ├── montecarlo_2025_distribution.png
│   ├── opt_with_rf_backtest.csv
│   ├── opt_with_rf_backtest.png
│   ├── opt_with_rf_best_portfolio.csv
│   ├── opt_with_rf_results.csv
│   ├── portfolio_excess_vs_market_2025_9M.png
│   ├── portfolio_realized_2025_9M_monthly.csv
│   ├── portfolio_realized_2025_9M_summary.csv
│   ├── portfolio_realized_2025_9M_.png
│   ├── sp500_betas_alphas_2020_2024_monthly_excess.csv 
│   ├── sp500_excess_logreturns_monthly_2020_2024.csv
│   ├── sp500_ff5_betas_2020_2024.csv
│   ├── sp500_logreturns_monthly_2020_2024.csv
│   └── sp500_prices_monthly_2020_2024.csv
│
├── montecarlo_params.py
├── requirements.txt
│
├── s01_Fetch_SP500_list.py              # Fetch S&P 500 symbols from Wikipedia
├── s02_main.py                          # Compute CAPM betas/alphas (2020–2024)
├── s03_main_ff5.py                      # Compute Fama–French 5-Factor betas
├── s04_Fetch_SP500_sectors.py           # Download sector mapping for each ticker
├── s05_compare_betas_yfinance.py        # Compare computed vs Yahoo Finance betas
├── s06_plot_betas.py                    # Plot CAPM vs Yahoo beta correlation
├── s07_beta_neutral.py                  # Optimize FF5 beta-neutral long/short
├── s08_simulation.py                    # Monte Carlo simulation
├── s09_2025data.py                      # Fetch realized S&P 500 & RF data5
├── s10_comparaison.py                   # Compare simulated vs realized 2025 
│
└── sp500_symbols.py                     # Output from s01
```

---

## 🚀 Execution Order

Run scripts **sequentially from `s01` to `s10`**:

```bash
python s01_Fetch_SP500_list.py
python s02_main.py
python s03_main_ff5.py
python s04_Fetch_SP500_sectors.py
python s05_compare_betas_yfinance.py
python s06_plot_betas.py
python s07_beta_neutral.py
python s08_simulation.py
python s09_2025data.py
python s10_comparaison.py
```

Each script is **idempotent** — it uses cached data if available.

---

## 🧠 Workflow Overview

| Step | Script | Purpose |
|------|---------|----------|
| **1** | `s01_Fetch_SP500_list.py` | Retrieve current S&P 500 constituents from Wikipedia. |
| **2** | `s02_main.py` | Estimate CAPM betas & alphas using 2020–2024 monthly data. |
| **3** | `s03_main_ff5.py` | Estimate **Fama–French 5-Factor** betas (MKT, SMB, HML, RMW, CMA). |
| **4** | `s04_Fetch_SP500_sectors.py` | Add sector constraints to portfolio optimization. |
| **5** | `s05_compare_betas_yfinance.py` | Validate computed betas vs Yahoo Finance. |
| **6** | `s06_plot_betas.py` | Visualize our betas vs Yahoo betas (correlation, RMSE, R²). |
| **7** | `s07_beta_neutral.py` | Optimize **beta-neutral** portfolio  |
| **8** | `s08_simulation.py` | Simulate 2025 returns via Monte Carlo using μ, σ from optimizer. |
| **9** | `s09_2025data.py` | Collect realized 2025 prices & market returns. |
| **10** | `s10_comparaison.py` | Compare simulated vs realized 2025 portfolio performance. |

---

## 📊 Example Results (Jan–Sep 2025)

| Metric | Value |
|--------|--------|
| Portfolio Cumulative Return | **+10.9 %** |
| Market (S&P 500) Return | **+13.7 %** |
| Excess Return | **−2.8 %** |
| Portfolio Beta | **≈ 0.00 (Neutral)** |
| Correlation vs Market | **0.18** |
| Tracking Error | **6.1 %** |
| Information Ratio | **−0.46** |
| Sharpe Ratio | **1.95** |

---

## 🧩 Dependencies

```text
pandas
numpy
yfinance
matplotlib
tqdm
requests
lxml
scipy
```

Install via:
```bash
pip install -r requirements.txt
```

---

## 🧾 Citation

OpenAI. 2025. ChatGPT (GPT-5). Accessed October 2025. https://chat.openai.com/
