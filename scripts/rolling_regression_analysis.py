import pandas as pd
import statsmodels.api as sm

# === 1. Load the returns data ===
file_path = r"C:\Users\crist\Desktop\SORBONNE\PSME\THESIS Micro\returns_corrected_in_decimals.xlsx"
df_returns = pd.read_excel(file_path)

# Make sure Date is datetime and set as index
df_returns['Date'] = pd.to_datetime(df_returns['Date'])
df_returns.set_index('Date', inplace=True)

# === 2. Rolling regression setup ===
window_size = 60
target = 'Big_High'  # Portfolio we want to analyze
predictors = [col for col in df_returns.columns if col != target]

# === 3. Perform rolling regression ===
results_list = []

for i in range(window_size, len(df_returns)):
    window_df = df_returns.iloc[i-window_size:i]
    y = window_df[target]
    X = window_df[predictors]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    # Prepare next observation with same structure
    next_row = df_returns.iloc[i]
    X_next = pd.DataFrame([next_row[predictors]])  # Only predictors first
    X_next.insert(0, 'const', 1.0)  # Insert constant manually as first column

    predicted_return = model.predict(X_next)[0]

    results_list.append({
        'Date': df_returns.index[i],
        'R_squared': model.rsquared,
        'Predicted': predicted_return,
        'Actual': next_row[target],
        'Intercept': model.params['const'],
        **{f'beta_{col}': model.params[col] for col in predictors}
    })

# === 4. Save results to Excel ===
df_rolling_results = pd.DataFrame(results_list)

output_path = r"C:\Users\crist\Desktop\SORBONNE\PSME\THESIS Micro\Small_Low\roll_REG_results_Big_High.xlsx"
df_rolling_results.to_excel(output_path, index=False)

print("✅ Rolling regression for Big_High completed and saved successfully!")
print(f"📄 Saved at: {output_path}")
print(df_rolling_results.head())
