import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('Mall_Customers.csv')

# Delete rows with missing values
df.dropna(inplace=True)

# Selecting features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Step 1: Initial scatter plot of the data points
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], color='blue', s=100)
plt.title('Initial Scatter Plot of Data Points')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Step 2: Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Elbow curve to find the optimal number of clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(K, inertia, marker='o', linestyle='--')
plt.title('Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.show()

# Step 4: Applying K-Means with the optimal number of clusters (e.g., 5 from elbow curve)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)

# Step 5: Visualizing the clusters with centroids
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', 
            marker='*',s=200, label='Centroids')
plt.title('K-Means Clustering with Centroids (Scikit-Learn)')
plt.xlabel('Annual Income (Standardized)')
plt.ylabel('Spending Score (Standardized)')
plt.legend()
plt.show()

# Step 6: Confusion matrix heatmap (assuming cluster predictions vs true labels)
# In this case, we don't have true labels, but we can use the predicted labels for simulation
y_true = kmeans.labels_  # In practice, this should be true labels if available
y_pred = kmeans.predict(X_scaled)

# Create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plotting the confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Output cluster centers
print("Cluster centers (standardized):")
print(kmeans.cluster_centers_)
