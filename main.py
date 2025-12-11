import pandas as pd
from src.data_loader import (
    load_sp500_companies,
    load_rf,
    load_sp500_monthly_returns,
    classify_sp500_factors,
)
from src.models import build_factor_ml_dataset

def main():
    sp500_companies = load_sp500_companies()
    rf = load_rf()
    sp500_monthly_returns = load_sp500_monthly_returns(
        start="2024-11-01",
        end="2025-11-01",
    )
    
    print("\n" + "="*80)
    print("S&P 500 COMPANIES (First 5 Rows)")
    print("="*80)
    print(sp500_companies.head())

    print("\n" + "="*80)
    print("FAMA–FRENCH RISK-FREE & MARKET DATA (First 5 Rows)")
    print("="*80)
    print(rf.head())

    print("\n" + "="*80)
    print("S&P 500 MONTHLY RETURNS (First 5 Rows)")
    print("="*80)
    print(sp500_monthly_returns.head())
    print("="*80 + "\n")

    tickers = pd.read_csv("data/raw/sp500_tickers.csv", header=None)[0].tolist()
    ff5_class = classify_sp500_factors(tickers)
    ff5_class.to_csv("data/processed/sp500_ff5_classifications.csv", index=False)

    print("\n" + "="*80)
    print("FF5 APPROXIMATE CLASSIFICATIONS (First 5 Rows)")
    print("="*80)
    print(ff5_class.head())
    print("="*80 + "\n")

    factor_ml_dataset = build_factor_ml_dataset()

    print("\n" + "="*80)
    print("FACTOR ML DATASET (First 5 Rows)")
    print("="*80)
    print(factor_ml_dataset.head())
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
