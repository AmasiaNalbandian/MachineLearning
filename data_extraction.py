import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA = '/kaggle/input/credit-card-apprivals/cc_approvals.data'
# DATA = 'StudentsPerformance.csv'

def get_dataset():
    nRowsRead = 1000  # specify 'None' if you want to read the whole file
    df = pd.read_csv('./datasets/StudentsPerformance.csv', delimiter=',', nrows=nRowsRead, header=0)
    
    # Assuming 'score_total' is the name of your target variable column
    X = df.drop('score_total', axis=1)  # Features
    y = df['score_total']  # Target variable

    # Apply label encoding to categorical columns
    categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Optionally, apply scaling to numeric features
    numeric_columns = ['math_score', 'reading_score', 'writing_score']
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    nRow, nCol = df.shape
    # print(f'There are {nRow} rows and {nCol} columns')
    
    # Print a subset of the data to inspect
    # print("Sample data:")
    # print(X.head())
    
    return X, y
    