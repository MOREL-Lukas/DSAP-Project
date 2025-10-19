# 📊 DSAP Project: CAPM & Beta-Neutral Portfolio Optimization

This project performs end-to-end **data acquisition, factor modeling, and portfolio optimization** based on CAPM (Capital Asset Pricing Model).  
It fetches S&P 500 data, computes betas, compares them with Yahoo Finance, constructs a beta-neutral portfolio, and evaluates its realized performance in 2025.

---

## ⚙️ Installation

Make sure you have Python **3.10+** and `pip` installed.

Clone the repository (if not already done):

```bash
git clone https://github.com/MOREL-Lukas/DSAP-Project
cd DSAP-Project
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Project Structure

```
DSAP-Project/
├── Data/                               # Cached and downloaded market data
├── Output/                             # Generated results and plots
├── sp500_symbols.py                    # List of S&P 500 tickers (auto-fetched)
├── requirements.txt                    # Dependencies list
├── README.md                           # This file
├── s01_fetch_sp500_symbols.py          # Fetch S&P 500 symbols from Wikipedia
├── s02_main.py                         # Compute CAPM betas and alphas (2020–2024)
├── s03_make_sp500_sectors.py           # Fetch sector data for S&P 500 tickers
├── s04_compare_betas_yfinance.py       # Compare CAPM betas with Yahoo Finance
├── s05_plot_beta_comparison.py         # Scatter plot of Yahoo vs CAPM betas
├── s06_optimize_portfolio_with_rf.py   # Optimize beta-neutral portfolio
├── s07_2025data.py                     # Download and cache 2025 YTD market data
├── s08_portfolio_realized_2025_9M.py   # Evaluate portfolio performance (2025)
└── s09_montecarlo_simulation.py        # Simulate 2025 portfolio return distribution
```

---

## 🚀 Execution Order

Run the scripts sequentially **from s01 to s09**.

```bash
python s01_fetch_sp500_symbols.py
python s02_main.py
python s03_make_sp500_sectors.py
python s04_compare_betas_yfinance.py
python s05_plot_beta_comparison.py
python s06_optimize_portfolio_with_rf.py
python s07_2025data.py
python s08_portfolio_realized_2025_9M.py
python s09_montecarlo_simulation.py
```

Each script is **idempotent** — it uses cached data if available and only downloads missing files.

---

## 🧠 Overview of Scripts

| Script | Purpose |
|--------|----------|
| **s01_Fetch_SP500_list.py** | Fetches the current list of S&P 500 tickers from Wikipedia and saves to `sp500_symbols.py`. |
| **s02_main.py** | Downloads market and stock data, computes CAPM betas, alphas, and returns, and saves clean CSVs. |
| **s03_Fetch_SP500_sectors.py** | Fetches each ticker’s sector info from Yahoo Finance and stores it in `Data/sp500_sectors.csv`. |
| **s04_compare_betas_yfinance.py** | Compares computed betas to Yahoo Finance betas, with caching and progress bar. |
| **s05_plot_betas.py** | Generates scatter plot comparing CAPM vs Yahoo betas and computes R², RMSE, and correlation. |
| **s06_beta_neutral.py** | Constructs a **beta-neutral long/short portfolio** using CAPM signals, sector constraints, and risk-free returns. |
| **s07_simulation.py** | Runs a Monte Carlo simulation projecting the distribution of 2025 portfolio returns. |
| **s08_2025data.py** | Downloads 2025 YTD market and stock data for realized backtesting. |
| **s09_comparaison.py** | Evaluates portfolio performance from Jan–Sep 2025 vs the S&P 500 benchmark. |

---

## 📈 Expected Outputs

After successful execution:

```
Data/
├── market_adj_close.csv
├── sp500_adj_close.csv
├── rf_irx_monthly.csv
├── sp500_sectors.csv
└── 2025/
    ├── market_adj_close_2025.csv
    ├── rf_irx_monthly_2025.csv
    └── sp500_adj_close_2025.csv

Output/
├── sp500_betas_alphas_2020_2024_monthly_excess.csv
├── beta_comparison.csv
├── beta_comparison_scatter.png
├── opt_with_rf_best_portfolio.csv
├── opt_with_rf_backtest.png
├── portfolio_realized_2025_9M.png
├── portfolio_excess_vs_market_2025_9M.png
└── montecarlo_2025_distribution.png
```

---

## 📊 Example Results (Jan–Sep 2025)

| Metric | Value |
|--------|--------|
| Portfolio Cumulative Return | **+10.9%** |
| Market (S&P 500) Return | **+13.7%** |
| Excess Return | **−2.8%** |
| Portfolio Beta | **1.12** |
| Correlation vs Market | **0.91** |
| Tracking Error | **5.36%** |
| Information Ratio | **−0.67** |

---

## 🧩 Dependencies

The project requires:

```text
pandas
numpy
yfinance
matplotlib
tqdm
requests
lxml
```

(Installed automatically via `requirements.txt`.)

---

## 💬 Notes

- Yahoo Finance betas are rolling **5-year weekly estimates**, while this project computes **5-year monthly CAPM betas (2020–2024)**.
- Results may slightly differ but should be directionally consistent.
- All scripts are optimized for reproducibility, caching, and batch-safe execution.

---

## 🧾 Usage

OpenAI. 2025. ChatGPT (GPT-5). Accessed October, 2025. https://chat.openai.com/
