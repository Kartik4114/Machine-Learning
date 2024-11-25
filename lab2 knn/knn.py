import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
data = pd.read_csv('titanic.csv')
imputer = SimpleImputer(strategy='mean')
data['Age'] = imputer.fit_transform(data[['Age']])

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

data = data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, errors='ignore')

# Splitting the data
x = data.drop('2urvived', axis=1)
y = data['2urvived']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xTrain, yTrain)

yPred = knn.predict(xTest)

accuracy = accuracy_score(yTest, yPred)
classReport = classification_report(yTest, yPred)
confMatrix = confusion_matrix(yTest, yPred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Classification Report:\n{classReport}')
print(f'Confusion Matrix:\n{confMatrix}')

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confMatrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plotting the accuracy
plt.figure(figsize=(4, 4))
plt.bar(['Accuracy'], [accuracy], color='green')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.show()
