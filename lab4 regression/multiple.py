import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data from your house prediction CSV
data = pd.read_csv('multiple.csv')  # Replace with your actual dataset path

# Print the first few rows to understand the dataset structure
print(data.head())

# One-hot encode the 'City' categorical column
data_encoded = pd.get_dummies(data, columns=['City'], drop_first=True)

# Assuming your dataset has features like 'Area', 'Bedrooms', and one-hot encoded 'City'
X = data_encoded[['Area', 'Bedrooms'] + [col for col in data_encoded.columns if 'City_' in col]].values  # Independent variables (multiple features)
y = data['Price'].values  # Dependent variable (target - house prices)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Compute evaluation metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Print the coefficients (slopes) and intercept
print("Coefficients (slopes):", model.coef_)
print("Intercept:", model.intercept_)

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R2 Score): {r2:.3f}")

# Display accuracy in percentage
accuracy = r2 * 100
print(f"Accuracy (R2 Score as percentage): {accuracy:.2f}%")

# Visualize the relationship between actual and predicted values
plt.figure(figsize=(8, 6))
sns.set(style="darkgrid")

# Scatter plot for actual vs predicted
plt.scatter(y, predictions, color='blue', label='Predicted vs Actual')

# Line representing perfect prediction
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', label='Perfect fit')

# Adding labels and title
plt.title('House Price Prediction: Actual vs Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()

# Show the plot
plt.show()
