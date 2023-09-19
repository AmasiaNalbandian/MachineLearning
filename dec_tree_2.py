import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('./datasets/BankChurners.csv')

# Define the target column and set random_state
target_column = 'Attrition_Flag'
random_state = 40
test_size = 0.4

# Extract features (X) and target variable (y)
X = data.drop(columns=[target_column])
y = data[target_column]

# Encode categorical features using one-hot encoding
categorical_cols = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]

# Perform one-hot encoding on categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)

# Create a GridSearchCV object for hyperparameter tuning
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
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
