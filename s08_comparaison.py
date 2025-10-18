# s08_portfolio_realized_2025_9M.py
"""
Computes monthly realized performance (Jan–Sep 2025)
for a given portfolio using locally downloaded CAPM data.
Generates summary CSV, benchmark comparison, and plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========================= CONFIG =========================
PORTFOLIO_FILE = "Output/opt_with_rf_best_portfolio.csv"
PRICE_FILE = "Data/2025/sp500_adj_close_2025.csv"
MARKET_FILE = "Data/2025/market_adj_close_2025.csv"
OUTPUT_RETURNS = "Output/portfolio_realized_2025_9M_monthly.csv"
OUTPUT_SUMMARY = "Output/portfolio_realized_2025_9M_summary.csv"
OUTPUT_PLOT = "Output/portfolio_realized_2025_9M.png"
OUTPUT_EXCESS_PLOT = "Output/portfolio_excess_vs_market_2025_9M.png"

os.makedirs("Output", exist_ok=True)

# ========================= LOAD PORTFOLIO =========================
if not os.path.exists(PORTFOLIO_FILE):
    raise FileNotFoundError(f"❌ Portfolio file not found: {PORTFOLIO_FILE}")

port = pd.read_csv(PORTFOLIO_FILE)
port = port[["Symbol", "Weight", "Side", "Beta"]]
port["Weight"] = pd.to_numeric(port["Weight"], errors="coerce")
port = port.dropna(subset=["Symbol", "Weight"])

# Signed weight for short positions
# Signed weight for short and cash positions
port["Side"] = port["Side"].str.capitalize().fillna("Long")

port["SignedWeight"] = np.select(
    [
        port["Side"] == "Short",
        port["Side"] == "Cash",
    ],
    [
        -port["Weight"],   # short positions negative
        port["Weight"],    # cash stays positive
    ],
    default=port["Weight"],  # default = long
)

# Drop cash from equity tickers list (not in S&P 500 prices)
tickers = [t for t in port["Symbol"].unique().tolist() if t != "RF"]
print(f"✅ Loaded portfolio with {len(tickers)} tickers.")

# ========================= LOAD PRICE DATA =========================
if not os.path.exists(PRICE_FILE):
    raise FileNotFoundError(f"❌ Price data file not found: {PRICE_FILE}")

prices = pd.read_csv(PRICE_FILE, index_col=0)
prices.index = pd.to_datetime(prices.index, format="%Y-%m-%d", errors="coerce")
prices = prices[tickers] if all(t in prices.columns for t in tickers) else prices.loc[:, prices.columns.isin(tickers)]
prices = prices.dropna(how="all")
print(f"✅ Loaded price data ({len(prices)} monthly observations).")

# ========================= COMPUTE RETURNS =========================
rets = prices.pct_change().dropna(how="all")  # monthly simple returns
rets.index.name = "Date"

weights = port.set_index("Symbol")["SignedWeight"].reindex(rets.columns).fillna(0.0)
rets["Portfolio"] = (rets * weights).sum(axis=1)

# ========================= MARKET RETURNS =========================
if os.path.exists(MARKET_FILE):
    market = pd.read_csv(MARKET_FILE, index_col=0)
    market.index = pd.to_datetime(market.index, format="%Y-%m-%d", errors="coerce")
    market = market.dropna(subset=["MKT"])
    mkt_rets = market["MKT"].pct_change().dropna()
    mkt_rets.name = "Market"
else:
    mkt_rets = None

# ========================= SUMMARY METRICS =========================
cum_ret = (1 + rets["Portfolio"]).prod() - 1
ann_equiv = (1 + cum_ret) ** (12 / len(rets)) - 1
avg_beta = (port["Beta"] * port["SignedWeight"]).sum()

summary = {
    "Start": rets.index.min().date(),
    "End": rets.index.max().date(),
    "Months": len(rets),
    "Cumulative_Return": cum_ret,
    "Annualized_Equivalent": ann_equiv,
    "Average_Beta": avg_beta,
    "Long_Exposure": port.loc[port["Side"] == "Long", "Weight"].sum(),
    "Short_Exposure": port.loc[port["Side"] == "Short", "Weight"].sum(),
}

# ========================= BENCHMARK COMPARISON =========================
if mkt_rets is not None:
    aligned_port, aligned_mkt = rets["Portfolio"].align(mkt_rets, join="inner")
    excess_rets = aligned_port - aligned_mkt

    # Core metrics
    summary["Market_Cumulative_Return"] = (1 + aligned_mkt).prod() - 1
    summary["Excess_Cumulative_Return"] = (1 + excess_rets).prod() - 1
    summary["Tracking_Error"] = excess_rets.std() * np.sqrt(12)

    # ✅ Standard CFA-style Information Ratio:
    # (Mean monthly excess return * sqrt(12)) / (std monthly excess * sqrt(12)) = mean/std
    mean_excess = excess_rets.mean() * 12
    std_excess = excess_rets.std() * np.sqrt(12)
    summary["Information_Ratio"] = (
        mean_excess / std_excess if std_excess != 0 else np.nan
    )

    summary["Correlation_vs_Market"] = aligned_port.corr(aligned_mkt)

# ========================= PLOT — CUMULATIVE PERFORMANCE =========================
plt.figure(figsize=(10, 6))
(1 + rets["Portfolio"]).cumprod().plot(label="Portfolio", lw=2)
if mkt_rets is not None:
    (1 + mkt_rets).cumprod().plot(label="S&P 500 (^GSPC)", lw=2, ls="--")

plt.title("Portfolio vs Market — Jan–Sep 2025 (Monthly)")
plt.ylabel("Cumulative Growth (×)")
plt.xlabel("Date")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()

# ========================= PLOT — EXCESS RETURN =========================
if mkt_rets is not None:
    plt.figure(figsize=(10, 4))
    excess_rets.cumsum().plot(label="Cumulative Excess Return (Portfolio − Market)", lw=2, color="purple")
    plt.axhline(0, color="black", lw=1, ls="--")
    plt.title("Cumulative Excess Return vs Market (Jan–Sep 2025)")
    plt.ylabel("Cumulative Excess Return")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_EXCESS_PLOT, dpi=300)
    plt.close()

# ========================= SAVE RESULTS =========================
rets.to_csv(OUTPUT_RETURNS)
pd.DataFrame([summary]).to_csv(OUTPUT_SUMMARY, index=False)

# ========================= PRINT SUMMARY =========================
print("\n✅ Portfolio realized performance (Jan–Sep 2025):")
print(f"   Period: {summary['Start']} → {summary['End']} ({summary['Months']} months)")
print(f"   Cumulative return: {summary['Cumulative_Return']:.2%}")
print(f"   Annualized equivalent: {summary['Annualized_Equivalent']:.2%}")
if "Market_Cumulative_Return" in summary:
    print(f"   Market cumulative return: {summary['Market_Cumulative_Return']:.2%}")
    print(f"   Excess cumulative return: {summary['Excess_Cumulative_Return']:.2%}")
print(f"   Avg. portfolio beta: {summary['Average_Beta']:.2f}")
if "Correlation_vs_Market" in summary:
    print(f"   Corr. vs S&P 500: {summary['Correlation_vs_Market']:.2f}")
if "Information_Ratio" in summary:
    print(f"   Tracking error: {summary['Tracking_Error']:.2%}")
    print(f"   Information ratio: {summary['Information_Ratio']:.2f}")
print(f"   Long exposure:  {summary['Long_Exposure']:.2%}")
print(f"   Short exposure: {summary['Short_Exposure']:.2%}")

print(f"\n📁 Saved outputs:")
print(f"   - {OUTPUT_RETURNS}")
print(f"   - {OUTPUT_SUMMARY}")
print(f"   - {OUTPUT_PLOT}")
if mkt_rets is not None:
    print(f"   - {OUTPUT_EXCESS_PLOT}")
