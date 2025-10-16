import pandas as pd

# --- Load semicolon-separated CSV ---
df = pd.read_csv("SPY_Data.csv", sep=';', skipinitialspace=True)

# --- Clean and normalize columns ---
df.columns = [col.strip() for col in df.columns]

# --- Clean Close column: remove '$' and spaces, fix decimal commas ---
df["Close"] = (
    df["Close"]
    .astype(str)
    .str.replace(r"[^0-9,.-]", "", regex=True)  # remove non-numeric symbols ($, spaces)
    .str.replace(",", ".", regex=False)         # change decimal comma to dot
)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# --- Clean Log_Weekly_Returns column (if present) ---
if "Log_Weekly_Returns" in df.columns:
    df["Log_Weekly_Returns"] = (
        df["Log_Weekly_Returns"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["Log_Weekly_Returns"] = pd.to_numeric(df["Log_Weekly_Returns"], errors="coerce")

# --- Save cleaned CSV (comma-separated, YYYY-MM-DD dates) ---
output_file = "SPY_Data_Formatted.csv"
df.to_csv(output_file, index=False, sep=",")

print(f"✅ File saved as '{output_file}' with ',' separator.")
