import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the normalized data
file1_normalized = pd.read_csv('file1_normalized.csv')
file2_normalized = pd.read_csv('file2_normalized.csv')

# Select 2-3 features to visualize
features = ['PassengerId', 'Survived', 'Pclass']  # Replace with actual feature names

# Set up the plotting environment
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))

# Plot histograms
for i, feature in enumerate(features):
    sns.histplot(file1_normalized[feature], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Histogram of {feature} (File 1)')

    sns.histplot(file2_normalized[feature], kde=True, ax=axes[i, 1], color='orange')
    axes[i, 1].set_title(f'Histogram of {feature} (File 2)')

# Create a new figure for scatter plots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Scatter plots between features
for i, feature in enumerate(features):
    sns.scatterplot(x=file1_normalized[feature], y=file2_normalized[feature], ax=axes[i])
    axes[i].set_title(f'Scatter Plot of {feature} (File 1 vs File 2)')
    axes[i].set_xlabel(f'{feature} (File 1)')
    axes[i].set_ylabel(f'{feature} (File 2)')

# Create a new figure for mean error plots
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate the mean and standard deviation for each feature in both files
means_file1 = file1_normalized[features].mean()
stds_file1 = file1_normalized[features].std()
means_file2 = file2_normalized[features].mean()
stds_file2 = file2_normalized[features].std()

# Plot mean and error bars
ax.errorbar(features, means_file1, yerr=stds_file1, fmt='o', capsize=5, label='File 1', color='skyblue')
ax.errorbar(features, means_file2, yerr=stds_file2, fmt='o', capsize=5, label='File 2', color='orange')
ax.set_title('Mean and Error Bars for Selected Features')
ax.set_xlabel('Features')
ax.set_ylabel('Mean ± Standard Deviation')
ax.legend()

# Adjust the layout
plt.tight_layout()

# Show all plots
plt.show()
