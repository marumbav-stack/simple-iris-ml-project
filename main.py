
"""
Iris Flower Classification
A simple machine learning project.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions using the test data
predictions = model.predict(X_test)

# 5. Calculate the model's accuracy
accuracy = accuracy_score(y_test, predictions)

print("IRIS FLOWER CLASSIFICATION")
print("--------------------------")
print(f"Model accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(classification_report(
    y_test, predictions, target_names=iris.target_names
))

# 6. Predict the type of a sample flower
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)[0]

print("Sample measurements:", sample[0])
print("Predicted flower type:", iris.target_names[prediction])
