import yfinance as yf
import pandas as pd
import os

# 10 representative US market ETFs from Yahoo Finance
etf_tickers = [
    'VOO',  # Stock: S&P 500
    'QQQ',  # Stock: NASDAQ
    'IWF',  # Stock: Russell 1000
    'IWM',  # Stock: Russell 2000
    'VIG',  # Stock: Stocks with High Dividend
    'BIL',  # T-Bill: 1-3 Month T-Bills
    'SHY',  # Treasury Bond: 1-3 Year Treasury Bond
    'IEF',  # Treasury Bond: 7-10 Year Treasury Bond
    'TLT',  # Treasury Bond: 20+ Year Treasury Bond
    'GLD',  # Commodity: GOLD
    ]

# Dictionary to store each etf's data
etf_data = {}

# Loop through each etf and download 10 years of monthly adjusted close prices
def fetch_tickers(etf_tickers, csv_name): 
    for ticker in etf_tickers:
        print(f"Fetching data for {ticker}...")
        try:
            data = yf.Ticker(ticker).history(period="10y", interval="1d")
            if not data.empty:
                etf_price = data['Close']
                nav = etf_price / etf_price.iloc[0]
                etf_data[ticker] = nav
            else:
                print(f"No data returned for {ticker}. Skipping.")
        except Exception as e:
            print(f"Failed to get data for {ticker}. Reason: {e}")

    # Combine into a single DataFrame
    etf_prices_df = pd.DataFrame(etf_data)

    # Drop rows with any missing values (optional)
    etf_prices_df.dropna(inplace=True)

    # Save to CSV
    output_path = csv_name
    etf_prices_df.to_csv(output_path)

    print(f"\nSaved 10-year monthly data to: {output_path}")
    print(etf_prices_df.head())

fetch_tickers(etf_tickers, "us_etf_portfolio_10yr_nav.csv")