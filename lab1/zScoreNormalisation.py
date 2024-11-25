import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the files
file1 = pd.read_csv('file1.csv')
file2 = pd.read_csv('file2.csv')

# Initialize the StandardScaler
scaler = StandardScaler()

# Normalize the data in both files
file1_normalized = file1.copy()
file2_normalized = file2.copy()

# Apply Z-Score Normalization to numerical columns
for column in file1_normalized.select_dtypes(include=['float64', 'int64']).columns:
    file1_normalized[[column]] = scaler.fit_transform(file1_normalized[[column]])

for column in file2_normalized.select_dtypes(include=['float64', 'int64']).columns:
    file2_normalized[[column]] = scaler.fit_transform(file2_normalized[[column]])

# Save the normalized files
file1_normalized.to_csv('file1_standardized.csv', index=False)
file2_normalized.to_csv('file2_standardized.csv', index=False)
