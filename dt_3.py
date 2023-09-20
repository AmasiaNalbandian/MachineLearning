import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('./datasets/games.csv')


# Define the target column and set random_state
target_column = 'winner'
random_state = 45
test_size = 0.2
# Extract features (X) and target variable (y)

# List of categorical columns to keep
columns_to_keep = ["turns", "white_rating", "black_rating", "opening_name"]

# Drop all other categorical columns
X = data.drop(columns=[target_column])
y = data[target_column]
data = data[columns_to_keep]

num_entries = data.shape[0]
print("Number of entries:", num_entries)
num_features = data.shape[1]
print("Number of features:", num_features)

# Encode categorical features using one-hot encoding
categorical_cols = ["opening_name"]
X_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)

# Create a GridSearchCV object for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 10, None],  # You can include 'None' for unlimited depth
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best model and best hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Use the best model for predictions
y_pred = best_model.predict(X_test)

# Evaluate the model's accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Now, let's plot the learning curve for the best model
clf = best_model  # Use the best model found by GridSearchCV

train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), return_times=False
)


# Calculate mean and standard deviation of training and testing scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curves
plt.figure(figsize=(10, 6))
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")

plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Accuracy")
plt.plot(train_sizes, test_mean, 'o-', color="orange", label="Cross-validation Accuracy")

plt.legend(loc="best")
plt.show()
