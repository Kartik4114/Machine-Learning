import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# Load the Iris dataset
df = pd.read_csv('iris.csv')
# X = df.drop(columns=['Species', 'Id'])  # Remove target column
X = df.drop(columns=['species'])  # Remove target column
y = df['species']
feature_names = X.columns

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=feature_names)  # Keep column names for readability

# Initialize lists to store results
selected_features = []  # This will store the features as we add them
accuracies = []         # To store accuracy for each step
combinations_list = []   # To store feature combinations at each step
best_accuracy = 0

# Start measuring time
start_time = time.time()

# Iterate to add features one by one to maximize accuracy
while len(selected_features) < len(feature_names):
    best_feature = None
    best_combo_accuracy = 0
    
    # Test adding each remaining feature one at a time
    for feature in feature_names:
        if feature in selected_features:
            continue  # Skip already selected features
        
        # Create a new feature set with the current selection + this feature
        current_features = selected_features + [feature]
        X_selected = X[current_features]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        
        # Train and evaluate the KNN model
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Trying feature set: {current_features}, Accuracy: {accuracy:.4f}")
        
        # Check if this is the best accuracy so far in this iteration
        if accuracy > best_combo_accuracy:
            best_combo_accuracy = accuracy
            best_feature = feature
    
    # If adding the best feature improves accuracy, update selected features and store results
    if best_combo_accuracy > best_accuracy:
        best_accuracy = best_combo_accuracy
        selected_features.append(best_feature)
        accuracies.append(best_accuracy)
        combinations_list.append(list(selected_features))
        
        print(f"Selected feature '{best_feature}', Updated feature set: {selected_features}, "
        f"Accuracy: {best_accuracy:.4f}")
        
        # Stop if 100% accuracy is reached
        if best_accuracy == 1.0:
            print("Reached 100% accuracy. Stopping.")
            break
    else:
        # If no improvement, stop the process
        print("No further accuracy improvement. Stopping.")
        break

# End measuring time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal computation time: {elapsed_time:.2f} seconds")

# Plotting accuracy for each combination of features
plt.figure(figsize=(14, 8))
plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', color='y')
plt.xticks(range(1, len(accuracies) + 1), [str(combo) for combo in combinations_list], rotation=45, ha='right')
plt.xlabel("Feature Combination", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Accuracy vs. Feature Combinations", fontsize=16)
plt.legend(['Accuracy'], fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
