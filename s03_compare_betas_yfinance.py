# compare_betas_yfinance.py
"""
Compare our computed CAPM betas vs Yahoo Finance betas.
Performs only ONE bulk Yahoo Finance request for all tickers.
Runtime: ~2-3 minutes
"""

import os
import pandas as pd
import yfinance as yf

# --- Config ---
INPUT_FILE   = "Output/sp500_betas_alphas_2020_2024_monthly_excess.csv"
YAHOO_CACHE  = "Data/yahoo_betas.csv"
OUTPUT_FILE  = "Output/beta_comparison.csv"

# --- Load our computed betas ---
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Could not find {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
if "Symbol" not in df.columns or "Beta" not in df.columns:
    raise ValueError("❌ Input file missing 'Symbol' or 'Beta' column")

symbols = sorted(df["Symbol"].dropna().unique().tolist())
print(f"✅ Loaded {len(symbols)} symbols from {INPUT_FILE}")

# --- Try using cached Yahoo betas ---
if os.path.exists(YAHOO_CACHE):
    yahoo_cache = pd.read_csv(YAHOO_CACHE)
    print(f"✅ Using cached Yahoo betas ({len(yahoo_cache)} entries)")
else:
    yahoo_cache = pd.DataFrame(columns=["Symbol", "Yahoo_Beta"])

# --- Determine which tickers need updating ---
cached_symbols = set(yahoo_cache["Symbol"])
missing = [s for s in symbols if s not in cached_symbols]
print(f"⬇️ Need to fetch {len(missing)} new Yahoo betas...")

# --- One bulk Yahoo Finance request ---
if missing:
    print("📡 Fetching all missing tickers in one API request...")
    tickers_str = " ".join(missing)
    tickers_data = yf.Tickers(tickers_str)  # Single bulk call

    new_rows = []
    for sym in missing:
        try:
            info = tickers_data.tickers[sym].fast_info  # much faster and lighter
            beta_yahoo = getattr(info, "beta", None)
            if beta_yahoo is None:
                # fallback: try .info if fast_info doesn't expose beta
                beta_yahoo = tickers_data.tickers[sym].info.get("beta", None)
            new_rows.append({"Symbol": sym, "Yahoo_Beta": beta_yahoo})
        except Exception as e:
            print(f"⚠️ Failed for {sym}: {e}")
            new_rows.append({"Symbol": sym, "Yahoo_Beta": None})

    # --- Update cache ---
    if new_rows:
        yahoo_cache = pd.concat([yahoo_cache, pd.DataFrame(new_rows)], ignore_index=True)
        yahoo_cache = yahoo_cache.drop_duplicates(subset="Symbol", keep="last")
        os.makedirs("Data", exist_ok=True)
        yahoo_cache.to_csv(YAHOO_CACHE, index=False)
        print(f"💾 Updated Yahoo beta cache: {YAHOO_CACHE}")
else:
    print("✅ All tickers already cached; no new downloads needed.")

# --- Merge results ---
merged = df[["Symbol", "Beta"]].merge(yahoo_cache, on="Symbol", how="left")
merged = merged.rename(columns={"Beta": "Our_Beta"})
merged = merged[["Symbol", "Our_Beta", "Yahoo_Beta"]]

# --- Save comparison ---
os.makedirs("Output", exist_ok=True)
merged.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Comparison saved to: {OUTPUT_FILE}")
print(merged.head(10))
