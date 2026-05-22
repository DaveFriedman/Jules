import pandas as pd
import numpy as np
import yfinance as yf

def fetch_real_stock_prices(ticker="GOOGL", start="2026-04-01", end="2026-05-22"):
    """Fetches real historical stock prices using yfinance."""
    print(f"Fetching data for {ticker} from {start} to {end}...")
    stock_data = yf.download(ticker, start=start, end=end)

    if stock_data.empty:
        return pd.DataFrame()

    # Reset index so 'Date' becomes a column and extract just the Date and Close price
    df = stock_data.reset_index()

    # yfinance output might have multi-level columns if multiple tickers are passed,
    # or just simple columns for one ticker. Let's handle standard single ticker format.
    # Usually it's Date, Open, High, Low, Close, Adj Close, Volume
    # We will just use 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-index if necessary (happens in newer yfinance versions sometimes)
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

    # We only need Date and Close. Ensure they exist.
    if 'Date' in df.columns and 'Close' in df.columns:
        df = df[['Date', 'Close']].rename(columns={'Close': 'Price'})
    else:
        raise ValueError("Could not find 'Date' and 'Close' columns in yfinance data.")

    return df

def apply_moving_averages(df):
    if df.empty:
        return df
    df['7MA'] = df['Price'].rolling(window=7).mean()
    df['30MA'] = df['Price'].rolling(window=30).mean()
    return df

def trading_algorithm(df):
    cash = 10000.0
    shares = 0
    initial_value = cash

    if df.empty:
        print("No data available to run the trading algorithm.")
        return initial_value, initial_value, cash, shares

    print(f"{'Date':<12} | {'Price':<8} | {'7MA':<8} | {'30MA':<8} | {'Action':<15} | {'Cash':<10} | {'Shares':<6} | {'Total Value'}")
    print("-" * 95)

    df['Signal'] = 0.0

    # Need at least 30 periods for moving average
    if len(df) > 30:
        # Signal is 1 if 7MA > 30MA, else 0
        df.loc[30:, 'Signal'] = np.where(df['7MA'][30:] > df['30MA'][30:], 1.0, 0.0)

    # Position: shift signal by 1 to represent executing at the next open?
    # Let's just execute at the close of the day we get the signal.
    # Crossover: diff of signal. +1 means golden cross (buy), -1 means death cross (sell).
    df['Position'] = df['Signal'].diff()

    for idx, row in df.iterrows():
        action = "Hold"
        price = row['Price']

        # Ensure price is a scalar float just in case
        if isinstance(price, pd.Series):
             price = price.iloc[0]
        price = float(price)

        # We can only trade if we have 30MA
        if not pd.isna(row['30MA']):
            # Ensure row['Position'] is scalar
            pos = row['Position']
            if isinstance(pos, pd.Series):
                pos = pos.iloc[0]

            if pos == 1.0 and cash >= price:
                # Buy
                shares_to_buy = int(cash // price)
                if shares_to_buy > 0:
                    cash -= shares_to_buy * price
                    shares += shares_to_buy
                    action = f"BUY {shares_to_buy}"
            elif pos == -1.0 and shares > 0:
                # Sell
                cash += shares * price
                action = f"SELL {shares}"
                shares = 0

        total_val = cash + shares * price

        # Formatting for ledger
        price_str = f"{price:.2f}"

        ma7 = row['7MA']
        if isinstance(ma7, pd.Series): ma7 = ma7.iloc[0]
        ma7_str = f"{float(ma7):.2f}" if not pd.isna(ma7) else "N/A"

        ma30 = row['30MA']
        if isinstance(ma30, pd.Series): ma30 = ma30.iloc[0]
        ma30_str = f"{float(ma30):.2f}" if not pd.isna(ma30) else "N/A"

        date_str = row['Date'].strftime('%Y-%m-%d')

        print(f"{date_str:<12} | {price_str:<8} | {ma7_str:<8} | {ma30_str:<8} | {action:<15} | {cash:<10.2f} | {shares:<6} | {total_val:.2f}")

    final_price = float(df.iloc[-1]['Price'])
    final_value = cash + shares * final_price
    return initial_value, final_value, cash, shares

def main():
    df = fetch_real_stock_prices(ticker="GOOGL", start="2026-04-01", end="2026-05-22")
    df = apply_moving_averages(df)

    print("Fetched GOOGL prices from 2026-04-01 to 2026-05-21 and running Golden Cross trading algorithm")
    print("=" * 95)
    init_val, final_val, cash, shares = trading_algorithm(df)

    print("=" * 95)
    print("Final Portfolio Performance:")
    print(f"Initial Value:  ${init_val:.2f}")
    print(f"Final Value:    ${final_val:.2f}")
    print(f"Final Cash:     ${cash:.2f}")
    print(f"Final Shares:   {shares}")
    print(f"Return:         {((final_val - init_val) / init_val * 100):.2f}%")

if __name__ == '__main__':
    main()
