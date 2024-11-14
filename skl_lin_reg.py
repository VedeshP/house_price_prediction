import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('housing.csv')

print(data.info())
print(data.describe())
print(" ")
print(data.isnull().sum())

X = data.drop('median_house_value', axis=1)
# Use one-hot encoding on the categorical column
X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)
# Fill missing values in X with the mean of each column
# X = X.fillna(X.mean())

# X = X.fillna(X.median())

# This does best of the three as rmse is least 
X = X.dropna()

"""
"""
y = data['median_house_value']
y = y[X.index]  # Ensure y aligns with X after dropping rows

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
rmse = mse ** 0.5
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)

# Ideal Line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values")
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Line Plot for a subset of data
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:50], label="Actual Values", marker='o')
plt.plot(y_pred[:50], label="Predicted Values", marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Median House Value")
plt.title("Comparison of Predicted and Actual Values for a Subset")
plt.legend()
plt.show()