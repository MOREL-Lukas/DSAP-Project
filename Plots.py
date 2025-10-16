    """
    # --- Step 4: Plot adjusted closing price over time ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['adjusted close'], label='Adjusted Closing Price')
    plt.title(f'{symbol} Adjusted Closing Prices (Up to 2024)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    # --- Step 5: Save chart ---
    chart_filename = os.path.join(DATA_DIR, f"{safe_symbol}_adjusted_closing_prices_up_to_2024.png")
    plt.savefig(chart_filename)
    plt.close()
    print(f"📊 Chart saved as '{chart_filename}'\n")
    """