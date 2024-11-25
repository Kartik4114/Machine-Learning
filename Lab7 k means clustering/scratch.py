import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to calculate the Euclidean distance between points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# K-Means Clustering from scratch
class KMeansScratch:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Randomly initialize cluster centers
        np.random.seed(42)
        random_idx = np.random.permutation(X.shape[0])
        self.centroids = X[random_idx[:self.n_clusters]]
        
        for i in range(self.max_iter):
            # Assign clusters
            self.labels = self.assign_clusters(X)
            
            # Compute new centroids
            new_centroids = self.calculate_centroids(X)
            
            # If centroids do not change, break
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def assign_clusters(self, X):
        # Assign each point to the nearest centroid
        labels = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            # np.argmin(distances) returns the index of the minimum value in the array
            labels.append(np.argmin(distances))  
        return np.array(labels)

    def calculate_centroids(self, X):
        # Compute the centroids as the mean of the points assigned to each cluster
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for idx in range(self.n_clusters):
            # Get all points that belong to cluster idx
            points = X[self.labels == idx]
            centroids[idx] = np.mean(points, axis=0) if len(points) > 0 else self.centroids[idx]
        return centroids

    def predict(self, X):
        return self.assign_clusters(X)

# Step 2: Plot the elbow curve to find optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeansScratch(n_clusters=k)
    kmeans.fit(X_scaled)
    # Inertia is the sum of squared distances to the nearest centroid
    inertia_val = np.sum([euclidean_distance(X_scaled[i],
        kmeans.centroids[kmeans.labels[i]])**2 for i in range(X_scaled.shape[0])])
    inertia.append(inertia_val)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.show()

# Step 3: Applying K-Means with the optimal number of clusters (e.g., 5 from elbow curve)
kmeans_scratch = KMeansScratch(n_clusters=5)
kmeans_scratch.fit(X_scaled)

# Step 4: Visualizing the clusters with centroids
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_scratch.labels, cmap='viridis', s=100)
plt.scatter(kmeans_scratch.centroids[:, 0], kmeans_scratch.centroids[:, 1],
            color='red', marker='*', s=200, label='Centroids')
plt.title('K-Means Clustering with Centroids (Scratch)')
plt.xlabel('Annual Income (Standardized)')
plt.ylabel('Spending Score (Standardized)')
plt.legend()
plt.show()

# Step 5: Confusion matrix heatmap (assuming cluster predictions vs true labels)
# Here, since we don't have true labels, we simulate a confusion matrix using the predicted labels
y_true = kmeans_scratch.labels  # Predicted labels (in real scenario, compare with actual labels)
y_pred = kmeans_scratch.predict(X_scaled)

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
print(kmeans_scratch.centroids)
