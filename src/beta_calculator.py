import pandas as pd
import numpy as np
from scipy import stats
import os


def calculate_excess_returns(returns_df, rf_series):
    """
    Calculate excess returns for all stocks.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Stock returns (Date index, tickers as columns)
    rf_series : pd.Series
        Risk-free rate (Date index)
    
    Returns
    -------
    excess_returns : pd.DataFrame
        Excess returns (returns - risk-free rate)
    """
    # Align risk-free rate with returns
    rf_aligned = rf_series.reindex(returns_df.index)
    
    # Calculate excess returns: R_i - R_f
    excess_returns = returns_df.sub(rf_aligned, axis=0)
    
    return excess_returns


def calculate_market_excess_return(market_return, rf_series):
    """
    Calculate market excess return (market risk premium).
    
    Parameters
    ----------
    market_return : pd.Series
        Market returns (e.g., S&P 500)
    rf_series : pd.Series
        Risk-free rate
    
    Returns
    -------
    market_excess : pd.Series
        Market excess returns (R_m - R_f)
    """
    rf_aligned = rf_series.reindex(market_return.index)
    market_excess = market_return - rf_aligned
    
    return market_excess


def calculate_capm_beta(stock_excess_returns, market_excess_returns, min_obs=24):
    """
    Calculate CAPM beta using OLS regression.
    
    Beta = Cov(R_i - R_f, R_m - R_f) / Var(R_m - R_f)
    
    Parameters
    ----------
    stock_excess_returns : pd.Series
        Stock excess returns
    market_excess_returns : pd.Series
        Market excess returns
    min_obs : int
        Minimum number of observations required
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - beta: CAPM beta
        - alpha: Jensen's alpha
        - r_squared: R²
        - std_error: Standard error of beta
        - t_stat: t-statistic for beta
        - p_value: p-value for beta
        - n_obs: Number of observations
    """
    # Align data and drop NaN
    data = pd.DataFrame({
        'stock': stock_excess_returns,
        'market': market_excess_returns
    }).dropna()
    
    if len(data) < min_obs:
        return {
            'beta': np.nan,
            'alpha': np.nan,
            'r_squared': np.nan,
            'adj_r_squared': np.nan,
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'f_stat': np.nan,
            'f_pvalue': np.nan,
            'n_obs': len(data)
        }
    
    # OLS regression: R_i - R_f = alpha + beta * (R_m - R_f) + epsilon
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data['market'], 
        data['stock']
    )
    
    # Calculate both regular and adjusted R²
    n = len(data)
    k = 1  # Number of predictors (just market return for CAPM)
    r_squared = r_value**2
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
    
    # Calculate F-statistic for overall model significance
    # F = (R² / k) / ((1 - R²) / (n - k - 1))
    if r_squared < 1.0:  # Avoid division by zero
        f_statistic = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
        f_pvalue = stats.f.sf(f_statistic, k, n - k - 1)
    else:
        f_statistic = np.nan
        f_pvalue = np.nan
    
    return {
        'beta': slope,
        'alpha': intercept,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'std_error': std_err,
        't_stat': slope / std_err if std_err > 0 else np.nan,
        'p_value': p_value,
        'f_stat': f_statistic,
        'f_pvalue': f_pvalue,
        'n_obs': len(data)
    }


def calculate_all_betas(returns_path="data/processed/sp500_monthly_returns.csv",
                       rf_path="data/processed/Fama_French.csv",
                       market_ticker="^GSPC",
                       output_path="data/processed/sp500_capm_betas.csv"):
    """
    Calculate CAPM betas for all S&P 500 stocks.
    
    Parameters
    ----------
    returns_path : str
        Path to monthly returns CSV
    rf_path : str
        Path to Fama-French data (contains RF)
    market_ticker : str
        Market index ticker (default: S&P 500)
    output_path : str
        Where to save results
    
    Returns
    -------
    betas_df : pd.DataFrame
        DataFrame with beta estimates for each stock
    """
    
    # 1) Load stock returns
    print("\n1. Loading stock returns...")
    returns_df = pd.read_csv(returns_path, parse_dates=['Date'], index_col='Date')
    print(f"   Loaded {len(returns_df)} months for {len(returns_df.columns)} stocks")
    
    # 2) Load risk-free rate
    print("\n2. Loading risk-free rate...")
    ff_data = pd.read_csv(rf_path, parse_dates=['Date'], index_col='Date')
    rf_series = ff_data['RF']
    print(f"   Risk-free rate: {rf_series.mean()*100:.2f}% average monthly")
    
    # 3) Calculate excess returns for all stocks
    print("\n3. Calculating excess returns...")
    stock_excess_returns = calculate_excess_returns(returns_df, rf_series)
    print(f"   Stock excess returns calculated")
    
    # 4) Calculate market excess return
    # Option A: Use S&P 500 equal-weighted return from our stocks
    print("\n4. Calculating market excess return...")
    if market_ticker in returns_df.columns:
        market_return = returns_df[market_ticker]
    else:
        # Use equal-weighted average as market proxy
        market_return = returns_df.mean(axis=1)
        print("   Using equal-weighted portfolio as market proxy")
    
    market_excess = calculate_market_excess_return(market_return, rf_series)
    print(f"   Market risk premium: {market_excess.mean()*100:.2f}% average monthly")
    
    # 5) Calculate beta for each stock
    print("\n5. Calculating CAPM betas for all stocks...")
    results = []
    
    for ticker in stock_excess_returns.columns:
        beta_results = calculate_capm_beta(
            stock_excess_returns[ticker],
            market_excess
        )
        
        results.append({
            'Ticker': ticker,
            'Beta': beta_results['beta'],
            'Alpha': beta_results['alpha'],
            'R²': beta_results['r_squared'],
            'Adj_R²': beta_results['adj_r_squared'],
            'Std_Error': beta_results['std_error'],
            't_Statistic': beta_results['t_stat'],
            'p_value': beta_results['p_value'],
            'F_Statistic': beta_results['f_stat'],
            'F_pvalue': beta_results['f_pvalue'],
            'N_Observations': beta_results['n_obs']
        })
    
    betas_df = pd.DataFrame(results)
    
    # 6) Summary statistics
    print("\n" + "="*80)
    print("BETA SUMMARY STATISTICS")
    print("="*80)
    print(f"Total stocks analyzed: {len(betas_df)}")
    print(f"Stocks with valid betas: {betas_df['Beta'].notna().sum()}")
    print(f"\nBeta distribution:")
    print(betas_df['Beta'].describe())
    print(f"\nModel Fit:")
    print(f"Average R²: {betas_df['R²'].mean():.4f}")
    print(f"Average Adj R²: {betas_df['Adj_R²'].mean():.4f}")
    print(f"Median R²: {betas_df['R²'].median():.4f}")
    print(f"Median Adj R²: {betas_df['Adj_R²'].median():.4f}")
    
    # Identify high/low beta stocks
    print(f"\n" + "-"*80)
    print("TOP 5 HIGH-BETA STOCKS (Most Volatile)")
    print("-"*80)
    high_beta = betas_df.nlargest(5, 'Beta')[['Ticker', 'Beta', 'R²', 'Adj_R²']]
    print(high_beta.to_string(index=False))
    
    print(f"\n" + "-"*80)
    print("TOP 5 LOW-BETA STOCKS (Defensive)")
    print("-"*80)
    low_beta = betas_df.nsmallest(5, 'Beta')[['Ticker', 'Beta', 'R²', 'Adj_R²']]
    print(low_beta.to_string(index=False))
    
    # 7) Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    betas_df.to_csv(output_path, index=False)
    print(f"\n✅ Betas saved to: {output_path}")
    
    return betas_df


def calculate_rolling_beta(stock_excess_returns, market_excess_returns, window=60):
    """
    Calculate rolling beta over time.
    
    Parameters
    ----------
    stock_excess_returns : pd.Series
        Stock excess returns
    market_excess_returns : pd.Series
        Market excess returns
    window : int
        Rolling window in months (default: 60 = 5 years)
    
    Returns
    -------
    rolling_beta : pd.Series
        Time series of rolling betas
    """
    # Align data
    data = pd.DataFrame({
        'stock': stock_excess_returns,
        'market': market_excess_returns
    }).dropna()
    
    # Calculate rolling covariance and variance
    rolling_cov = data['stock'].rolling(window).cov(data['market'])
    rolling_var = data['market'].rolling(window).var()
    
    # Beta = Cov / Var
    rolling_beta = rolling_cov / rolling_var
    
    return rolling_beta


def plot_beta_distribution(betas_df, save_path="results/beta_distribution.png"):
    """
    Plot distribution of betas.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    betas_df['Beta'].hist(bins=50, ax=ax1, edgecolor='black')
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Market Beta = 1')
    ax1.axvline(betas_df['Beta'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median = {betas_df["Beta"].median():.2f}')
    ax1.set_xlabel('Beta', fontsize=12)
    ax1.set_ylabel('Number of Stocks', fontsize=12)
    ax1.set_title('Distribution of CAPM Betas', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Beta vs R²
    ax2 = axes[1]
    scatter = ax2.scatter(betas_df['Beta'], betas_df['R²'], 
                         alpha=0.5, s=30, c=betas_df['Beta'], cmap='coolwarm')
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Beta', fontsize=12)
    ax2.set_ylabel('R² (Model Fit)', fontsize=12)
    ax2.set_title('Beta vs Model Quality', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Beta')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Beta distribution plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Calculate betas
    betas_df = calculate_all_betas()
    
    # Plot distribution
    plot_beta_distribution(betas_df)