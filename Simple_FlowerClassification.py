# ========================================================================================================================
# DECISION TREE - COMPLETE PYTHON GUIDE with Simple Classification and Iris Flower Classification
# ========================================================================================================================

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# ========================================
# EXAMPLE 1: Simple Classification
# Predicting if someone will play tennis
# ========================================

print("=" * 50)
print("EXAMPLE 1: Tennis Playing Decision")
print("=" * 50)

# Create simple dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 
                'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
                   'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("\nDataset:")
print(df.head(10))

# Convert categorical variables to numerical
df_encoded = pd.get_dummies(df.drop('PlayTennis', axis=1))
X = df_encoded
y = df['PlayTennis'].map({'Yes': 1, 'No': 0})

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Create and train decision tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)
accuracy = accuracy_score(y, predictions)
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.head())

# ========================================
# EXAMPLE 2: Real-World Dataset
# Iris Flower Classification
# ========================================

print("\n" + "=" * 50)
print("EXAMPLE 2: Iris Flower Classification")
print("=" * 50)

from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target

print("\nDataset Info:")
print(f"Samples: {X_iris.shape[0]}")
print(f"Features: {X_iris.shape[1]}")
print(f"Classes: {len(np.unique(y_iris))} - {iris.target_names}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train decision tree
dt_classifier = DecisionTreeClassifier(
    max_depth=3,           # Maximum depth of tree
    min_samples_split=2,   # Minimum samples to split a node
    min_samples_leaf=1,    # Minimum samples in leaf node
    random_state=42
)

dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate model
train_acc = dt_classifier.score(X_train, y_train)
test_acc = dt_classifier.score(X_test, y_test)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy: {test_acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
