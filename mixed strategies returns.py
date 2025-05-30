import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Step 1: Load the necessary data
merged_data = pd.read_excel('C:/Users/crist/Desktop/SORBONNE/PSME/THESIS Micro/merged_actual_predicted_returns.xlsx')
sp500_data = pd.read_excel('C:/Users/crist/Desktop/SORBONNE/PSME/THESIS Micro/sp500_returns_rolling_variance.xlsx')

# Prepare S&P500 returns
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
sp500_data = sp500_data[['Date', 'Return']].rename(columns={'Return': 'Index_Return'})

# Merge S&P500 Rolling Variance into merged_data
sp500_variance = sp500_data[['Date']]  # Just keep the Date for now
full_data = merged_data.merge(sp500_data[['Date', 'Index_Return']], on='Date', how='left')

# Step 2: Optimization - Rebuild Price-Contingent Strategy Returns
optimized_returns = []
optimized_weights = []

# Find first valid index where Rolling Variance is available
rolling_variance_data = pd.read_excel('C:/Users/crist/Desktop/SORBONNE/PSME/THESIS Micro/sp500_returns_rolling_variance.xlsx')
rolling_variance_data['Date'] = pd.to_datetime(rolling_variance_data['Date'])
full_data = full_data.merge(rolling_variance_data[['Date', 'Rolling_Variance_60M']], on='Date', how='left')

start_index = full_data['Rolling_Variance_60M'].first_valid_index()

for i in range(start_index, len(full_data)):
    predicted = full_data.loc[i, [
        'Predicted_Small_Low', 'Predicted_Small_Med', 'Predicted_Small_High',
        'Predicted_Big_Low', 'Predicted_Big_Med', 'Predicted_Big_High'
    ]].values
    
    past_actual_returns = full_data.loc[i-60:i-1, [
        'Small_Low', 'Small_Med', 'Small_High', 
        'Big_Low', 'Big_Med', 'Big_High'
    ]]
    cov_matrix = past_actual_returns.cov().values
    
    target_variance = full_data.loc[i, 'Rolling_Variance_60M']
    
    n_assets = len(predicted)
    initial_guess = np.ones(n_assets) / n_assets

    def objective(q):
        return -np.dot(q, predicted)

    def constraint_variance(q):
        return np.dot(q.T, np.dot(cov_matrix, q)) - target_variance

    constraints = {'type': 'eq', 'fun': constraint_variance}

    result = minimize(objective, initial_guess, constraints=constraints)

    if result.success:
        weights = result.x
        realized_return = np.dot(weights, full_data.loc[i, [
            'Small_Low', 'Small_Med', 'Small_High', 
            'Big_Low', 'Big_Med', 'Big_High'
        ]].values)
    else:
        weights = np.full(n_assets, np.nan)
        realized_return = np.nan

    optimized_returns.append(realized_return)
    optimized_weights.append(weights)

# Step 3: Build the new strategy results DataFrame
strategy_results = pd.DataFrame({
    'Date': full_data.loc[start_index:, 'Date'].values,
    'Strategy_Return': optimized_returns
})

# Step 4: Merge with S&P500 returns
strategy_results = strategy_results.merge(sp500_data, on='Date', how='inner')  # Match dates

# Step 5: Build 50/50 Mixed Strategy Return
strategy_results['Mixed_Return'] = 0.5 * strategy_results['Strategy_Return'] + 0.5 * strategy_results['Index_Return']

# Step 6: Build Cumulative Returns
strategy_results['Cumulative_Strategy'] = 100 * (1 + strategy_results['Strategy_Return']).cumprod()
strategy_results['Cumulative_Index'] = 100 * (1 + strategy_results['Index_Return']).cumprod()
strategy_results['Cumulative_Mixed'] = 100 * (1 + strategy_results['Mixed_Return']).cumprod()

# Step 7: Plot cumulative returns
plt.figure(figsize=(12, 8))
plt.plot(strategy_results['Date'], strategy_results['Cumulative_Strategy'], label='Price-Contingent Strategy', linewidth=2)
plt.plot(strategy_results['Date'], strategy_results['Cumulative_Index'], label='S&P500 Index', linewidth=2)
plt.plot(strategy_results['Date'], strategy_results['Cumulative_Mixed'], label='50% Strategy + 50% Index Mix', linewidth=2)
plt.title('Comparison of Cumulative Returns', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Return (Starting from 100)', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Print first few rows
print("\n📈 Combined Cumulative Returns Table (first 10 months):")
print(strategy_results[['Date', 'Cumulative_Strategy', 'Cumulative_Index', 'Cumulative_Mixed']].head(10))

print("\n✅ Mixed Strategy cumulative returns built, plotted, and printed successfully!")
