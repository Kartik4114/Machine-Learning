import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('./Lab6 Decision Tree/Iris.csv')

# Dropping the 'Id' column as it is not a feature
data = data.drop('Id', axis=1)

# Splitting the data into features (X) and target (y)
X = data.drop('Species', axis=1)
y = data['Species']

# Splitting into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=y.unique())

print(f'Accuracy: {accuracy:.2f}')
print(f'Classification Report:\n{class_report}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=y.unique(), yticklabels=y.unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plotting the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dt, feature_names=X.columns, class_names=y.unique(), filled=True, rounded=True)
plt.title('Decision Tree')
plt.show()
