import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simulate_stock_prices(S0=150, mu=0.2, sigma=0.4, days=60):
    np.random.seed(42) # For reproducibility
    dt = 1/252
    prices = [S0]
    for _ in range(1, days):
        # Geometric Brownian Motion
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.sqrt(dt) * np.random.normal()
        price = prices[-1] * np.exp(drift + shock)
        prices.append(price)

    dates = [datetime.today() - timedelta(days=days - i) for i in range(days)]
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    return df

def apply_moving_averages(df):
    df['7MA'] = df['Price'].rolling(window=7).mean()
    df['30MA'] = df['Price'].rolling(window=30).mean()
    return df

def trading_algorithm(df):
    cash = 10000.0
    shares = 0
    initial_value = cash

    print(f"{'Date':<12} | {'Price':<8} | {'7MA':<8} | {'30MA':<8} | {'Action':<15} | {'Cash':<10} | {'Shares':<6} | {'Total Value'}")
    print("-" * 95)

    df['Signal'] = 0.0
    # Signal is 1 if 7MA > 30MA, else 0
    df.loc[30:, 'Signal'] = np.where(df['7MA'][30:] > df['30MA'][30:], 1.0, 0.0)

    # Position: shift signal by 1 to represent executing at the next open?
    # Let's just execute at the close of the day we get the signal.
    # Crossover: diff of signal. +1 means golden cross (buy), -1 means death cross (sell).
    df['Position'] = df['Signal'].diff()

    for idx, row in df.iterrows():
        action = "Hold"
        price = row['Price']

        # We can only trade if we have 30MA
        if not pd.isna(row['30MA']):
            if row['Position'] == 1.0 and cash >= price:
                # Buy
                shares_to_buy = int(cash // price)
                if shares_to_buy > 0:
                    cash -= shares_to_buy * price
                    shares += shares_to_buy
                    action = f"BUY {shares_to_buy}"
            elif row['Position'] == -1.0 and shares > 0:
                # Sell
                cash += shares * price
                action = f"SELL {shares}"
                shares = 0

        total_val = cash + shares * price

        # Formatting for ledger
        price_str = f"{row['Price']:.2f}"
        ma7_str = f"{row['7MA']:.2f}" if not pd.isna(row['7MA']) else "N/A"
        ma30_str = f"{row['30MA']:.2f}" if not pd.isna(row['30MA']) else "N/A"
        date_str = row['Date'].strftime('%Y-%m-%d')

        print(f"{date_str:<12} | {price_str:<8} | {ma7_str:<8} | {ma30_str:<8} | {action:<15} | {cash:<10.2f} | {shares:<6} | {total_val:.2f}")

    final_value = cash + shares * df.iloc[-1]['Price']
    return initial_value, final_value, cash, shares

def main():
    df = simulate_stock_prices(S0=150, days=60)
    df = apply_moving_averages(df)

    print("Simulated 60 days of GOOGL prices and Golden Cross trading algorithm")
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
