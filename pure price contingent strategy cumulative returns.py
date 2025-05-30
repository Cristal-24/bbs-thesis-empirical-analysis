import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Step 1: Load your merged and full data
merged_data = pd.read_excel('C:/Users/crist/Desktop/SORBONNE/PSME/THESIS Micro/merged_actual_predicted_returns.xlsx')
sp500_data = pd.read_excel('C:/Users/crist/Desktop/SORBONNE/PSME/THESIS Micro/sp500_returns_rolling_variance.xlsx')

sp500_data = sp500_data[['Date', 'Rolling_Variance_60M']]
full_data = merged_data.merge(sp500_data, on='Date', how='left')

# Step 2: Optimization again
optimized_returns = []
optimized_weights = []

# Find first valid index where Rolling Variance is available
start_index = full_data['Rolling_Variance_60M'].first_valid_index()

for i in range(start_index, len(full_data)):
    current_date = full_data.loc[i, 'Date']
    
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

# Create strategy results DataFrame
strategy_results = pd.DataFrame({
    'Date': full_data.loc[start_index:, 'Date'].values,
    'Strategy_Return': optimized_returns
})

# Step 3: Build cumulative returns starting from 100
strategy_results['Cumulative_Return'] = 100 * (1 + strategy_results['Strategy_Return']).cumprod()

# Step 4: Plot cumulative returns
plt.figure(figsize=(12, 7))
plt.plot(strategy_results['Date'], strategy_results['Cumulative_Return'], color='blue', linewidth=2)
plt.title('Cumulative Return of Price-Contingent Strategy (Starting from 100)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Print cumulative returns
print("\n📈 Cumulative Returns Table (first 20 months):")
print(strategy_results[['Date', 'Cumulative_Return']].head(20))

print("\n✅ Full optimization, cumulative returns built, plotted, and printed successfully!")
