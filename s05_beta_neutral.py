import os
import pandas as pd
import numpy as np
from itertools import product
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Inputs ----------
BETA_FILE   = "Output/sp500_betas_alphas_2020_2024_monthly_excess.csv"
PRICE_FILE  = "Data/sp500_adj_close.csv"
RF_FILE     = "Data/rf_irx_monthly.csv"
OUT_RESULTS = "Output/opt_with_rf_results.csv"
OUT_BEST    = "Output/opt_with_rf_best_portfolio.csv"
OUT_BACKTEST = "Output/opt_with_rf_backtest.csv"

# ---------- Parameters ----------
WIN_MONTHS            = 59
N_LONG_RANGE          = list(range(10, 101))
N_SHORT_RANGE         = list(range(10, 101))
GROSS_EXPOSURE        = 1.0
MIN_UNIVERSE_REQUIRED = 120
MAX_ABS_W             = 0.1
LAMBDA_CASH           = 100.0
cashcap               = 0.05 # +/- max weight in rf
PEN_BETA              = 0.10
PEN_VOL               = 1.0
EXCLUDE_MOST_RECENT   = False
CASH_SYMBOL           = "RF"

pd.set_option("future.no_silent_downcasting", True)

# ------------------------------------------------------------
def load_data():
    if not all(os.path.exists(f) for f in [BETA_FILE, PRICE_FILE, RF_FILE]):
        raise FileNotFoundError("Missing one or more required data files.")

    betas = pd.read_csv(BETA_FILE)
    betas["Beta"] = pd.to_numeric(betas["Beta"], errors="coerce")
    betas = betas.dropna(subset=["Symbol", "Beta"])[["Symbol", "Beta"]].drop_duplicates("Symbol")

    prices = pd.read_csv(PRICE_FILE)
    prices.iloc[:, 0] = pd.to_datetime(prices.iloc[:, 0], errors="coerce")
    prices = prices.dropna(subset=[prices.columns[0]]).set_index(prices.columns[0]).sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    rf_raw = pd.read_csv(RF_FILE)
    rf_raw.iloc[:, 0] = pd.to_datetime(rf_raw.iloc[:, 0], errors="coerce")
    rf_raw = rf_raw.dropna(subset=[rf_raw.columns[0]]).set_index(rf_raw.columns[0]).sort_index()
    rf_raw.iloc[:, 0] = pd.to_numeric(rf_raw.iloc[:, 0], errors="coerce")
    rf_raw = rf_raw.dropna()

    # Detect whether input is annualized (%) or already monthly effective (decimal)
    median_val = rf_raw.iloc[:, 0].median()
    if median_val > 1:  # likely % annualized
        print("⚙️ Detected annualized yields in percent — converting to monthly effective rate.")
        rf_yield = rf_raw.resample("ME").last().iloc[:, 0] / 100.0
        rf_monthly = (1 + rf_yield) ** (1/12) - 1.0
    else:
        print("⚙️ Detected monthly effective rate — using as-is.")
        rf_monthly = rf_raw.resample("ME").last().iloc[:, 0]

    rf = rf_monthly.to_frame(name=CASH_SYMBOL)


    return betas, prices, rf


def compute_returns(prices):
    logrets = np.log(prices / prices.shift(1))
    logrets.index = logrets.index.to_period("M").to_timestamp("M")
    simplrets = np.expm1(logrets)
    return logrets, simplrets


def expected_return_signal(logrets, rf):
    rf = rf.copy()
    rf.index = rf.index.to_period("M").to_timestamp("M")
    idx = logrets.index.intersection(rf.index)
    logrets, rf_aligned = logrets.loc[idx], rf.loc[idx]
    excess = logrets.sub(rf_aligned[CASH_SYMBOL], axis=0)
    valid_cols = excess.columns[excess.count() >= WIN_MONTHS]
    excess = excess[valid_cols]

    window = excess.iloc[-(WIN_MONTHS+1):-1] if EXCLUDE_MOST_RECENT else excess.tail(WIN_MONTHS)
    if len(window) < WIN_MONTHS:
        raise ValueError(f"Not enough rows to build a {WIN_MONTHS}-month signal (have {len(window)}).")

    exp_ret, last_ret = window.mean(axis=0), excess.iloc[-1]
    uni = pd.concat([exp_ret.rename("ExpRet"), last_ret.rename("LastRet")], axis=1).dropna()
    uni.index.name = "Symbol"
    return uni, rf_aligned


def construct_portfolio(uni, betas, n_long, n_short, simplrets, rf_monthly_last):
    df = uni.reset_index().merge(betas, on="Symbol", how="inner").dropna(subset=["ExpRet", "Beta"])
    if len(df) < max(MIN_UNIVERSE_REQUIRED, n_long + n_short):
        return None
    df = df.sort_values("ExpRet", ascending=False, ignore_index=True)
    longs, shorts = df.head(n_long).copy(), df.tail(n_short).copy()
    if longs.empty or shorts.empty:
        return None
    longs["Weight"], shorts["Weight"] = 1.0 / len(longs), -1.0 / len(shorts)
    beta_long, beta_short = (longs["Weight"] * longs["Beta"]).sum(), (shorts["Weight"] * shorts["Beta"]).sum()
    if abs(beta_short) > 1e-10:
        shorts["Weight"] *= abs(beta_long / beta_short)
    port = pd.concat([longs.assign(Side="Long"), shorts.assign(Side="Short")], ignore_index=True)

    gross = float(port["Weight"].abs().sum())
    if gross <= 1e-10:
        return None
    port["Weight"] *= (GROSS_EXPOSURE / gross)

    # Cap weights
    w_capped = port["Weight"].clip(-MAX_ABS_W, MAX_ABS_W)
    slack = GROSS_EXPOSURE - w_capped.abs().sum()
    if slack > 1e-12:
        residual = port["Weight"] - w_capped
        denom = residual.abs().sum()
        port["Weight"] = w_capped + (residual * (slack / denom) if denom > 0 else 0)
    else:
        port["Weight"] = w_capped

    # Cash leg
    cash_weight = np.clip(1.0 - port["Weight"].sum(), -cashcap, cashcap)
    if abs(cash_weight) > 1e-12:
        rf_row = pd.DataFrame([[CASH_SYMBOL, "Cash", 0.0, 0.0, 0.0, cash_weight, 0.0]],
                              columns=["Symbol","Side","Beta","ExpRet","LastRet","Weight","Beta_Contribution"])
        port = pd.concat([port, rf_row], ignore_index=True)

    port = port.drop_duplicates(subset=["Symbol"], keep="last")
    port["Beta_Contribution"] = port["Weight"] * port["Beta"]
    beta_p = float(port["Beta_Contribution"].sum())
    exp_excess = float((port["Weight"] * port["ExpRet"].fillna(0)).sum())
    last_excess = float((port["Weight"] * port["LastRet"].fillna(0)).sum())
    rf_value = float(rf_monthly_last.iloc[0]) if isinstance(rf_monthly_last, pd.Series) else float(rf_monthly_last)
    exp_nominal = exp_excess + rf_value


    port_syms = [s for s in port["Symbol"] if s in simplrets.columns]
    if len(port_syms) > 0:
        w = port.set_index("Symbol").loc[port_syms, "Weight"]
        port_ret_series = simplrets[port_syms].mul(w, axis=1).sum(axis=1)
        vol_m = float(port_ret_series.std(ddof=1))
    else:
        vol_m = np.nan

    sharpe = (exp_excess / vol_m * np.sqrt(12)) if (vol_m == vol_m and vol_m > 0) else np.nan
    return port.sort_values("Weight", ascending=False), exp_excess, exp_nominal, last_excess, beta_p, vol_m, sharpe, float(cash_weight)


def objective(exp_excess, beta_p, cash_abs, last_excess=None, vol_m=None):
    score = exp_excess - PEN_BETA * abs(beta_p) - LAMBDA_CASH * abs(cash_abs)
    if vol_m is not None and np.isfinite(vol_m):
        score -= PEN_VOL * vol_m
    if last_excess is not None and np.isfinite(last_excess):
        score += 0.02 * last_excess
    return score


def bootstrap_portfolio(uni, betas, rets, rf_monthly_last, n_long, n_short, n_boot=200, random_state=42):
    np.random.seed(random_state)
    months = rets.index
    results = []
    for _ in tqdm(range(n_boot), desc="Bootstrapping", ncols=80):
        sample_idx = np.random.choice(len(months), size=len(months), replace=True)
        sampled_rets = rets.iloc[sample_idx]
        exp_ret, last_ret = sampled_rets.mean(axis=0), sampled_rets.iloc[-1]
        boot_uni = pd.concat([exp_ret.rename("ExpRet"), last_ret.rename("LastRet")], axis=1).dropna()
        boot_uni.index.name = "Symbol"
        res = construct_portfolio(boot_uni, betas, n_long, n_short, rets, rf_monthly_last)
        if res is None:
            continue
        port, exp_excess, exp_nominal, last_excess, beta_p, vol_m, sharpe, cash_w = res
        results.append({"ExpExcess": exp_excess, "ExpNominal": exp_nominal,
                        "Beta": beta_p, "VolMonthly": vol_m,
                        "Sharpe": sharpe, "CashWeight": cash_w})
    if not results:
        raise RuntimeError("No valid bootstrap samples.")
    df = pd.DataFrame(results)
    summary = df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    return df, summary


def backtest_portfolio(port, simplrets, rf):
    """Backtest monthly returns of final portfolio."""
    port_syms = [s for s in port["Symbol"] if s in simplrets.columns]
    w = port.set_index("Symbol").loc[port_syms, "Weight"]

    # Compute portfolio returns
    port_rets = simplrets[port_syms].mul(w, axis=1).sum(axis=1)

    # Align risk-free series properly (handles both DataFrame and Series)
    if isinstance(rf, pd.DataFrame):
        rf_series = rf["RF"]
    else:
        rf_series = rf
    rf_series = rf_series.reindex(port_rets.index).fillna(method="ffill").fillna(0)

    # Compute excess returns and cumulative P&L
    excess_rets = port_rets - rf_series
    cum_excess = (1 + excess_rets).cumprod() - 1

    stats = {
        "Annualized Return": port_rets.mean() * 12,
        "Annualized Vol": port_rets.std(ddof=1) * np.sqrt(12),
        "Sharpe": port_rets.mean() / port_rets.std(ddof=1) * np.sqrt(12),
        "Max Drawdown": (cum_excess / cum_excess.cummax() - 1).min()
    }

    # Save and plot
    out = pd.DataFrame({"Port_Ret": port_rets, "Excess_Ret": excess_rets, "Cum_Excess": cum_excess})
    out.to_csv(OUT_BACKTEST)

    plt.figure(figsize=(9, 5))
    plt.plot(out.index, (1 + out["Port_Ret"]).cumprod() - 1, label="Nominal Portfolio Return", lw=1.8)
    plt.plot(out.index, out["Cum_Excess"], label="Excess Return vs RF", lw=1.8)
    plt.title("Portfolio Backtest: Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Output/opt_with_rf_backtest.png")

    return stats

# ------------------------------------------------------------
def main():
    betas, prices, rf = load_data()
    logrets, simplrets = compute_returns(prices)
    uni, rf_aligned = expected_return_signal(logrets, rf)
    rf_monthly_last = float(rf_aligned.iloc[-1, 0])
    print(f"Universe after merging: {len(uni)} tickers")

    results, best = [], {"score": -np.inf}

    tqdm_kwargs = dict(
        desc="Grid Search",
        ncols=80,
        leave=True,
        file=sys.stdout,       # ensures proper carriage return
        dynamic_ncols=True,    # adapts to terminal width
        mininterval=0.3,       # refresh rate (avoid flicker/spam)
        ascii=True             # simpler bar for non-UTF terminals
    )

    # ✅ properly indented inside main()
    for nl, ns in tqdm(list(product(N_LONG_RANGE, N_SHORT_RANGE)), **tqdm_kwargs):
        res = construct_portfolio(uni, betas, nl, ns, simplrets, rf_monthly_last)
        if res is None:
            continue
        port, exp_exc, exp_nom, last_exc, beta_p, vol_m, sharpe, cash_w = res
        score = objective(exp_exc, beta_p, abs(cash_w), last_exc, vol_m)
        results.append({
            "N_LONG": nl, "N_SHORT": ns, "Score": score,
            "ExpExcess": exp_exc, "ExpNominal": exp_nom,
            "Beta": beta_p, "VolMonthly": vol_m,
            "SharpeAnn": sharpe, "CashAbs": abs(cash_w)
        })
        if score > best["score"]:
            best.update({
                "score": score, "N_LONG": nl, "N_SHORT": ns, "port": port,
                "ExpExcess": exp_exc, "ExpNominal": exp_nom,
                "Beta": beta_p, "VolMonthly": vol_m, "SharpeAnn": sharpe
            })

    os.makedirs("Output", exist_ok=True)
    pd.DataFrame(results).sort_values("Score", ascending=False).to_csv(OUT_RESULTS, index=False)
    best["port"].to_csv(OUT_BEST, index=False)
    print("\n✅ Optimization complete")
    print(f"Best (N_LONG, N_SHORT) = ({best['N_LONG']}, {best['N_SHORT']})")
    print(f"Expected monthly excess return: {best['ExpExcess']:.4%}")
    print(f"Portfolio beta: {best['Beta']:.4f} | Vol (m): {best['VolMonthly']:.2%} | Sharpe (ann): {best['SharpeAnn']:.2f}")
    return best, uni, betas, simplrets, rf_aligned


# ------------------------------------------------------------
if __name__ == "__main__":
    best, uni, betas, simplrets, rf_aligned = main()

    boot_df, boot_summary = bootstrap_portfolio(
        uni, betas, simplrets, rf_aligned.iloc[-1],
        n_long=best["N_LONG"], n_short=best["N_SHORT"], n_boot=200
    )
    print("\n📊 Bootstrap summary (200 resamples)")
    print(boot_summary[["mean", "std", "5%", "50%", "95%"]])

    # Backtest the final portfolio
    print("\n📈 Running backtest...")
    backtest_stats = backtest_portfolio(best["port"], simplrets, rf_aligned)
    print("\nBacktest performance:")
    for k, v in backtest_stats.items():
        print(f"{k:<20} : {v:>8.2%}")
    print(f"\nSaved backtest results → {OUT_BACKTEST}")
    print("Saved performance chart → Output/opt_with_rf_backtest.png")
    # --- Export portfolio parameters (mu, sigma) for Monte Carlo simulation ---
    mu = best["ExpExcess"]          # mean monthly excess return
    sigma = best["VolMonthly"]      # monthly volatility

    params = [mu, sigma]
    with open("montecarlo_params.py", "w") as f:
        f.write(f"mu_sigma = {params}\n")

    print(f"\n📦 Exported Monte Carlo parameters → montecarlo_params.py")
    print(f"mu = {mu:.6f}, sigma = {sigma:.6f}")
