import pandas as pd
import matplotlib.pyplot as plt

# Load the normalized files
file1 = pd.read_csv('file1_normalized.csv')
file2 = pd.read_csv('file2_normalized.csv')

# Plot histograms for numerical columns in file1
file1.select_dtypes(include=['float64', 'int64']).hist(figsize=(12, 10), bins=30)
plt.suptitle('Histograms for file1')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()

# Plot histograms for numerical columns in file2
file2.select_dtypes(include=['float64', 'int64']).hist(figsize=(12, 10), bins=30)
plt.suptitle('Histograms for file2')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()
