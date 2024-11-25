import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Helper functions
def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X_column, y, threshold):
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])
    
    n = len(y)
    weighted_avg_entropy = (len(y[left_indices]) / n) * left_entropy + (len(y[right_indices]) / n) * right_entropy
    
    return entropy(y) - weighted_avg_entropy

def best_split(X, y):
    best_gain = -1
    split_idx, split_threshold = None, None
    
    n_features = X.shape[1]
    
    for feature_idx in range(n_features):
        X_column = X[:, feature_idx]
        thresholds = np.unique(X_column)
        
        for threshold in thresholds:
            gain = information_gain(X_column, y, threshold)
            
            if gain > best_gain:
                best_gain = gain
                split_idx = feature_idx
                split_threshold = threshold
                
    return split_idx, split_threshold

# Node class for decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_labels = np.unique(y)
        
        if len(unique_labels) == 1 or n_samples < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_feature, best_threshold = best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _most_common_label(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        most_common = unique_labels[np.argmax(counts)]
        return most_common
    
    def fit(self, X, y):
        self.root = self._grow_tree(np.array(X), np.array(y))
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

# Load your custom Iris dataset
df = pd.read_csv('Iris.csv')

# Drop the Id column
df = df.drop(columns=['Id'])

# Encode Species column
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Split the data into features and labels
X = df.drop(columns=['Species']).values
y = df['Species'].values

# Split dataset manually
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree
clf = DecisionTree(max_depth=3)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate accuracy manually
accuracy = (np.sum(y_pred == y_test) / len(y_test))*100
print(f"Accuracy: {accuracy:.2f} %")

# Manually calculate the confusion matrix
def confusion_matrix_manual(y_true, y_pred):
    K = len(np.unique(y_true))  # Number of classes 
    result = np.zeros((K, K), dtype=int)
    
    for i in range(len(y_true)):
        result[y_true[i]][y_pred[i]] += 1
    
    return result

cm = confusion_matrix_manual(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize the confusion matrix using seaborn
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Setosa', 'Versicolor', 'Virginica'],
            yticklabels=['Setosa', 'Versicolor', 'Virginica'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Print the dataset as a table using Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))  # Set the size of the figure
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
plt.title("Iris Dataset")
plt.show()

# Visualize data points with a scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=df['Species'], cmap='viridis', edgecolor='k', s=100)
plt.title("Iris Dataset Scatter Plot")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(handles=scatter.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.grid()
plt.show()

# Create a small plot showing accuracy
plt.figure(figsize=(4, 2))
plt.axis('off')  # Turn off the axis
plt.text(0.5, 0.5, f'Accuracy: {accuracy * 100:.2f}%', fontsize=14, ha='center', va='center')
plt.title('Model Accuracy', fontsize=12)
plt.show()

# Visualize the decision tree structure in textual format
def print_tree(node, feature_names, spacing=""):
    """World's most elegant tree printing function."""
    # Base case: we've reached a leaf
    if node.value is not None:
        print(spacing + "Predict", node.value)
        return

    # Print the question at this node
    print(spacing + f"[{feature_names[node.feature]} <= {node.threshold}]")

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.left, feature_names, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.right, feature_names, spacing + "  ")

# Print the tree structure
print("Decision Tree Structure:")
print_tree(clf.root, df.columns[:-1])