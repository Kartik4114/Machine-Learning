import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the normalized files
file1_normalized = pd.read_csv('file1_normalized.csv')
file2_normalized = pd.read_csv('file2_normalized.csv')

# Replace 'Feature1' and 'Feature2' with actual column names
features = ['Age', 'Fare']  # Example feature names

# Plot histograms for file1
plt.figure(figsize=(12, 6))
for i, feature in enumerate(features):
    plt.subplot(2, len(features), i + 1)
    sns.histplot(file1_normalized[feature], kde=True)
    plt.title(f'File1 - {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

# Plot histograms for file2
for i, feature in enumerate(features):
    plt.subplot(2, len(features), len(features) + i + 1)
    sns.histplot(file2_normalized[feature], kde=True)
    plt.title(f'File2 - {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Plot scatter plots for pairs of features
plt.figure(figsize=(12, 6))
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        plt.subplot(len(features) - 1, len(features) - 1, (i * (len(features) - 1)) + j)
        plt.scatter(file1_normalized[features[i]], file1_normalized[features[j]], alpha=0.5)
        plt.title(f'File1: {features[i]} vs {features[j]}')
        plt.xlabel(features[i])
        plt.ylabel(features[j])

        plt.subplot(len(features) - 1, len(features) - 1, (i * (len(features) - 1)) + j + len(features) - 1)
        plt.scatter(file2_normalized[features[i]], file2_normalized[features[j]], alpha=0.5)
        plt.title(f'File2: {features[i]} vs {features[j]}')
        plt.xlabel(features[i])
        plt.ylabel(features[j])

plt.tight_layout()
plt.show()
