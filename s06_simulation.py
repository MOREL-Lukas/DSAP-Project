import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import montecarlo_params # Import the parameters calculated and exported in 5.beta_neutral.py
    if not hasattr(montecarlo_params, "mu_sigma"):
        raise AttributeError("The file 'montecarlo_params.py' must define a variable named 'mu_sigma'.")
except ModuleNotFoundError:
    sys.exit("❌ Error: 'montecarlo_params.py' not found. Please create it and define mu_sigma = [mu, sigma].")
except AttributeError as e:
    sys.exit(f"❌ Error: {e}")
import os
os.makedirs("Output", exist_ok=True)
output_path = "Output/montecarlo_2025_distribution.png"

mu, sigma = montecarlo_params.mu_sigma[0], montecarlo_params.mu_sigma[1]  # from the portfolio
n_months, n_paths = 12, 10000

# Monte Carlo simulation for 2025
simulated_returns = np.random.normal(mu, sigma, (n_months, n_paths))
cum_returns = (1 + simulated_returns).cumprod(axis=0) - 1

# Summary statistics
expected_2025 = cum_returns[-1]
summary = pd.Series(expected_2025).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
summary_pct = summary.copy()

# Convert all except 'count' to percent
summary_pct.loc[summary_pct.index != "count"] *= 100

# Format output with percent sign
formatted_summary = pd.Series(
    [
        f"{int(v)}" if i == "count" else f"{v:.2f}%"
        for i, v in summary_pct.items()
    ],
    index=summary_pct.index,
)


print("\n📊 Monte Carlo Summary (2025 projected cumulative return)\n")
print(formatted_summary)

# Create the histogram
plt.figure(figsize=(9, 5))
plt.hist(expected_2025 * 100, bins=50, alpha=0.7, color="skyblue", edgecolor="gray")
plt.xlabel("Cumulative Return (2025, %)")
plt.title("Monte Carlo Simulation: 2025 Cumulative Portfolio Return")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)

# Median line
median_val = summary["50%"] * 100
plt.axvline(median_val, color="red", linestyle="--", linewidth=2, label=f"Median = {median_val:.2f}%")
plt.legend(loc="upper right", frameon=True)

# Prepare text box content
summary_text = "\n".join([f"{i}: {v}" for i, v in formatted_summary.items()])

# Add text box to plot (top-left corner)
plt.gcf().text(
    0.1, 0.9, summary_text,
    fontsize=9,
    va="top", ha="left",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
)

plt.tight_layout()

# Save plot
plt.savefig(output_path, dpi=300)
print(f"\n✅ Saved Monte Carlo distribution plot → {output_path}")
