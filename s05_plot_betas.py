# plot_beta_comparison yfinance betas data are on a rolling basis and calculated using weekly 5y historic data 
# -> the data range doesn't match exactly our calculated betas of 5y monthly data from 2020-2024
# ============================================================
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE  = "Output/beta_comparison.csv"
OUTPUT_PNG  = "Output/beta_comparison_scatter.png"

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find: {INPUT_FILE}")

    # Load and coerce to numeric
    df = pd.read_csv(INPUT_FILE)
    for col in ["Our_Beta", "Yahoo_Beta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows with both betas present and finite
    data = df.dropna(subset=["Our_Beta", "Yahoo_Beta"]).copy()
    data = data[np.isfinite(data["Our_Beta"]) & np.isfinite(data["Yahoo_Beta"])]

    if data.empty:
        raise ValueError("No valid beta pairs to plot after cleaning.")

    x = data["Our_Beta"].values  # predictor
    y = data["Yahoo_Beta"].values  # response

    # Quick stats
    n = len(data)
    corr = float(np.corrcoef(x, y)[0, 1])
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    mae  = float(np.mean(np.abs(x - y)))
    bias = float(np.mean(x - y))  # Our - Yahoo

    # --- OLS regression: y = a + b x ---
    # Using numpy least squares (same as OLS without standard errors)
    b, a = np.polyfit(x, y, 1)   # slope, intercept (note: np.polyfit returns [slope, intercept])
    y_hat = a + b * x
    residuals = y - y_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    print(f"Pairs: {n}")
    print(f"Pearson r: {corr:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"Bias (Our - Yahoo): {bias:.3f}")
    print(f"OLS: Yahoo = {a:.3f} + {b:.3f} * Our   (R² = {r2:.3f})")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=18, alpha=0.7, edgecolors="none", label="Tickers")

    # 45-degree reference line
    xymin = -0.5
    xymax = 3
    plt.plot([xymin, xymax], [xymin, xymax], linestyle="--", label="y = x")

    # Regression line over plotting range
    xs = np.linspace(xymin, xymax, 100)
    ys = a + b * xs
    plt.plot(xs, ys, linewidth=2, label=f"OLS: y = {a:.2f} + {b:.2f}x\nR² = {r2:.3f}")

    # Labels & styling
    plt.title("Beta Comparison: Our CAPM vs Yahoo Finance")
    plt.xlabel("Our Beta (2020–2024, monthly CAPM)")
    plt.ylabel("Yahoo Beta")
    plt.xlim(xymin, xymax)
    plt.ylim(xymin, xymax)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend(loc="best")

    # Annotate stats block
    annotation = f"n={n}\nr={corr:.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}\nBias={bias:.3f}"
    plt.gcf().text(0.02, 0.98, annotation, va="top", ha="left", fontsize=9)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
