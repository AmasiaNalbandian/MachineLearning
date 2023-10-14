import pandas as pd

# Load the dataset
data = pd.read_csv('./datasets/BankChurners.csv')

# Define the target column and set random_state
target_column = 'Attrition_Flag'

# List of categorical columns to keep
columns_to_keep = ["Customer_Age", "Gender", "Education_Level", "Marital_Status", "Income_Category"]

# Drop all other categorical columns
X = data[columns_to_keep]
y = data[target_column]

num_entries = data.shape[0]
print("Number of entries:", num_entries)
num_features = X.shape[1]
print("Number of features:", num_features)

# Find unique values in the "Marital_Status" column
unique_marital_statuses = X['Marital_Status'].unique()

# Find unique values in the "Education_Level" column
unique_education_levels = X['Education_Level'].unique()
income_levels = X['Income_Category'].unique()

print("Unique Marital Statuses:", unique_marital_statuses)
print("Unique Education Levels:", unique_education_levels)
print("Income Category Levels:", income_levels)
