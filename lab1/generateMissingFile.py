import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('tested.csv')

# Identify missing values in critical and non-critical columns
missing_critical = df[df[['Age', 'Fare']].isnull().any(axis=1)]
missing_non_critical = df[df[['Cabin', 'Ticket']].isnull().any(axis=1)]

# Save to file1.csv (critical columns)
missing_critical.to_csv('file1.csv', index=False)

# Save to file2.csv (non-critical columns)
missing_non_critical.to_csv('file2.csv', index=False)
