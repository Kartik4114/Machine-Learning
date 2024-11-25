import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression class
class LinearRegression:
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    # Method to fit the model
    def fit(self, X, y):
        n = len(X)
        
        # Calculate the means of X and y
        mean_x = np.mean(X)
        mean_y = np.mean(y)
        
        # Calculate slope (b1) and intercept (b0)
        numerator = np.sum((X - mean_x) * (y - mean_y))
        denominator = np.sum((X - mean_x)**2)
        self.slope = numerator / denominator
        self.intercept = mean_y - self.slope * mean_x

    # Method to make predictions
    def predict(self, X):
        return self.slope * X + self.intercept

    # Get slope and intercept
    def get_slope(self):
        return self.slope
    
    def get_intercept(self):
        return self.intercept

# Load data from CSV using pandas
data = pd.read_csv('data2.csv')  # Replace with your actual dataset path

# Assuming your CSV has columns named 'X' and 'y'
X = data['X'].values
y = data['y'].values

# Create LinearRegression object and fit the data
model = LinearRegression()
model.fit(X, y)

# Get predictions
predictions = model.predict(X)

# Print the slope, intercept, and predictions
print("Slope:", model.get_slope())
print("Intercept:", model.get_intercept())
print("Predictions:", predictions)

# Compute evaluation metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R2 Score): {r2:.3f}")

# Interpretation of R-squared as an "accuracy-like" metric
print(f"R-squared (R2 Score) as accuracy-like measure: {r2:.3f}")

# Plotting the data and regression line using matplotlib and seaborn
plt.figure(figsize=(8, 6))
sns.set(style="darkgrid")

# Scatter plot for actual data
plt.scatter(X, y, color='blue', label='Actual data')

# Plot for predicted regression line
plt.plot(X, predictions, color='red', label='Regression line')

# Adding labels and title
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.show()
