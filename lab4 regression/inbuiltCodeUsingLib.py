import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data from CSV using pandas
data = pd.read_csv('single.csv')  # Replace with your actual dataset path

# Print the column names to check what's available in the CSV
print(data.columns)

# Replace 'X' and 'y' with your actual column names
X = data[['X']].values  # scikit-learn expects X to be a 2D array
y = data['y'].values

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Compute evaluation metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R2 Score): {r2:.3f}")

# Interpretation of R-squared as an "accuracy-like" metric
print(f"R-squared (R2 Score) as accuracy-like measure: {r2:.3f}")

# Plotting the data and regression line using matplotlib and seaborn
plt.figure(figsize=(10, 6))
sns.set(style="darkgrid")

# Scatter plot for actual data
plt.scatter(X, y, color='blue', label='Actual data', marker='o')

# Plot for predicted regression line
plt.plot(X, predictions, color='red', label='Regression line', linewidth=2)

# Plot for residuals
plt.vlines(X.flatten(), ymin=y, ymax=predictions, colors='green', linestyle='dotted', label='Residuals')

# Adding labels and title
plt.title('Linear Regression Fit with scikit-learn')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.show()
