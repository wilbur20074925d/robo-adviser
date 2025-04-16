import yfinance as yf
import pandas as pd
import os

# 10 representative bond ETFs from Yahoo Finance
bond_tickers = ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'MUB', 'EMB']

# Dictionary to store each bond's data
bond_data = {}

# Loop through each bond and download 10 years of monthly adjusted close prices
for ticker in bond_tickers:
    print(f"Fetching data for {ticker}...")
    try:
        data = yf.Ticker(ticker).history(period="10y", interval="1d")
        if not data.empty:
            bond_data[ticker] = data['Close']
        else:
            print(f"No data returned for {ticker}. Skipping.")
    except Exception as e:
        print(f"Failed to get data for {ticker}. Reason: {e}")

# Combine into a single DataFrame
bond_prices_df = pd.DataFrame(bond_data)

# Drop rows with any missing values (optional)
bond_prices_df.dropna(inplace=True)

# Save to CSV
output_path = "bond_etf_10yr_prices.csv"
bond_prices_df.to_csv(output_path)

print(f"\n✅ Saved 10-year monthly data to: {output_path}")
print(bond_prices_df.head())