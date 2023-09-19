import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('./datasets/BankChurners.csv')

# Define the target column
target_column = 'Attrition_Flag'

# Extract features (X) and target variable (y)
X = data.drop(columns=[target_column])
y = data[target_column]

# Encode categorical features using one-hot encoding
categorical_cols = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]

# Perform one-hot encoding on categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Encode the target variable 'Attrition_Flag' into numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Standardize features (important for k-NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the k-NN classifier
k = 7  # Adjust the number of neighbors as needed
knn_classifier = KNeighborsClassifier(n_neighbors=100) # TODO: Changing neighbors impacts performance! 
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
