import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression class
class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None

    # Method to fit the model
    def fit(self, X, y):
        # Add a column of ones to X to account for the intercept (b0)
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        
        # Calculate coefficients using the normal equation: (X^T X)^-1 X^T y
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # Method to make predictions
    def predict(self, X):
        # Add a column of ones to X for the intercept
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        
        # Predict: y = X_b * coefficients
        return X_b.dot(self.coefficients)

    # Get coefficients (slopes and intercept)
    def get_coefficients(self):
        return self.coefficients

# Load data from CSV using pandas
data = pd.read_csv('multiple.csv')  # Replace with your actual dataset path

# Assuming your CSV has columns 'Area', 'Bedrooms', 'Price'
X = data[['Area', 'Bedrooms']].values  # Independent variables (features)
y = data['Price'].values  # Dependent variable (target)

# Create MultipleLinearRegression object and fit the data
model = MultipleLinearRegression()
model.fit(X, y)

# Get predictions
predictions = model.predict(X)

# Print the coefficients and predictions
print("Coefficients (intercept and slopes):", model.get_coefficients())
print("Predictions:", predictions)

# Compute evaluation metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R2 Score): {r2:.3f}")

# Display accuracy in percentage
accuracy_percentage = r2 * 100
print(f"Accuracy (R2 Score as percentage): {accuracy_percentage:.2f}%")

# Plotting the data and regression predictions using matplotlib and seaborn
plt.figure(figsize=(8, 6))
sns.set(style="darkgrid")

# Scatter plot for actual data
plt.scatter(y, predictions, color='blue', label='Predicted vs Actual')

# Line representing perfect prediction
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', label='Perfect fit')

# Adding labels and title
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()

# Show the plot
plt.show()
