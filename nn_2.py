import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# Encode the target variable 'Attrition_Flag' into one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)


# Build a multi-class neural network model using Keras
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Use 'softmax' for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(np.asarray(X_train).astype(np.float32), np.asarray(y_train).astype(np.float32), epochs=10, batch_size=32, validation_split=0.2)
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
y_pred_proba = model.predict(np.asarray(X_test).astype(np.float32))
y_pred = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class predictions

# Generate a classification report
report = classification_report(y_test, y_pred)

print("Classification Report:\n", report)
