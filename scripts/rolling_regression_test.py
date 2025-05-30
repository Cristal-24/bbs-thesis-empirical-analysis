import pandas as pd
import statsmodels.api as sm

# Load your csv file
df = pd.read_csv(
    r"C:\Users\crist\Desktop\SORBONNE\PSME\THESIS Micro\6_Portfolios_2x3.CSV",
    skiprows=15,
    sep=";"
)

# Rename first column to "Date"
df.rename(columns={df.columns[0]: "Date"}, inplace=True)

# Keep only rows where the Date is 6 digits like YYYYMM
df = df[df["Date"].astype(str).str.match(r"^\d{6}$")]

# Convert to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")

# Filter data for the 60-month window ending in January 2005
end_date = pd.to_datetime("2005-01-01")
start_date = end_date - pd.DateOffset(months=60)

window_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

# 1. Set Y (the dependent variable)
y = pd.to_numeric(window_df["SMALL LoBM"], errors='coerce')

# 2. Set X (independent variables — the other 5 portfolios)
X = window_df[["ME1 BM2", "SMALL HiBM", "BIG LoBM", "ME2 BM2", "BIG HiBM"]].apply(pd.to_numeric, errors='coerce')

# 3. Drop any rows with NaNs caused by conversion
valid = X.notnull().all(axis=1) & y.notnull()
X = X[valid]
y = y[valid]

# 4. Add a constant (intercept) to X
X = sm.add_constant(X)

# 5. Run the regression
model = sm.OLS(y, X)
results = model.fit()

# 6. Show the regression summary
print(results.summary())


import matplotlib.pyplot as plt

# 1. Get predicted values
predicted = results.predict(X)

# 2. Create the plot
plt.figure(figsize=(10, 5))
plt.plot(window_df["Date"].loc[predicted.index], y, label="Actual Returns", marker='o')
plt.plot(window_df["Date"].loc[predicted.index], predicted, label="Predicted Returns", marker='x')
plt.title("Actual vs Predicted Returns for SMALL LoBM (2000–2004)")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 3. Show the plot
plt.show()

