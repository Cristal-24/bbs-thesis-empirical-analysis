import pandas as pd
import os
import matplotlib.pyplot as plt

# === 1. Load data ===
file_path = r"C:\Users\crist\Desktop\SORBONNE\PSME\THESIS Micro\6_Portfolios_2x3.CSV"

# Skip the first 15 rows to get to the actual data
df_raw = pd.read_csv(file_path, skiprows=15, encoding='latin1', sep=",")

# Keep only the first 1182 rows (table goes from row 17 to 1198)
df_raw = df_raw.iloc[:1182].copy()

# Rename columns
df_raw.columns = ['Date', 'Small_Low', 'Small_Med', 'Small_High',
                  'Big_Low', 'Big_Med', 'Big_High']

# === 2. Filter valid date rows and convert ===
df_raw = df_raw[df_raw['Date'].astype(str).str.match(r'^\d{6}$')].copy()
df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%Y%m')

# === 3. Convert returns to decimals ===
df_returns = df_raw.copy()
for col in df_returns.columns[1:]:
    df_returns[col] = pd.to_numeric(df_returns[col], errors='coerce') / 100

# === 4. Compute cumulative prices ===
initial_price = 100
df_cum_prices = df_returns.copy()
df_cum_prices.iloc[:, 1:] = initial_price * (1 + df_returns.iloc[:, 1:]).cumprod()




# === 7. Compute Relative Prices ===
df_relative_prices = df_cum_prices.copy()

# For each row, divide each value by the mean across portfolios (columns 1 onward)
df_relative_prices.iloc[:, 1:] = df_cum_prices.iloc[:, 1:].div(df_cum_prices.iloc[:, 1:].mean(axis=1), axis=0)

# Preview the relative prices dataset
print("\n📊 Relative Prices Preview (first 5 rows):")
print(df_relative_prices.head())




# === ✅ 5. Preview the first 5 rows ===
print("\n📘 First 5 rows of returns (in decimal form):")
print(df_returns.head())

print("\n📈 First 5 rows of cumulative prices:")
print(df_cum_prices.head())

# === 6. Plot cumulative prices ===
plt.figure(figsize=(12, 6))
for col in df_cum_prices.columns[1:]:
    plt.plot(df_cum_prices['Date'], df_cum_prices[col], label=col)

plt.title("Cumulative Price Evolution of 6 Portfolios (Starting from 100)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# === 6. Plot relative prices ===
plt.figure(figsize=(12, 6))

for col in df_relative_prices.columns[1:]:  # Skip the 'Date' column
    plt.plot(df_relative_prices['Date'], df_relative_prices[col], label=col)

plt.title("Relative Prices of Portfolios Over Time (Normalized to Cross-Sectional Mean)")
plt.xlabel("Date")
plt.ylabel("Relative Price")
plt.axhline(1, color='gray', linestyle='--', linewidth=1)  # Optional: reference line at 1
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
