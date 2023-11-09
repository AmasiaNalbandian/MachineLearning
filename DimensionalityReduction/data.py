import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import constants
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import constants
import numpy as np
from sklearn.preprocessing import LabelEncoder
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

np.random.seed(26)

def get_chess_data():
    target_column = 'winner'
    d = pd.read_csv('./datasets/games.csv')

    filtered_data = d[(d['turns'] >= 15) & (d['winner'] != 'draw')].copy()

    # parse out the first move
    filtered_data['moves'] = filtered_data['moves'].str.split()
    filtered_data['first_move'] = filtered_data['moves'].str[0]

    # List the columns to keep - and drop the rest 
    columns_to_keep = ["turns", "white_rating", "black_rating", "rated", "winner", "first_move"]
    data = filtered_data[columns_to_keep]

    # Create one-hot encoding for the "first_move" column, and drop the first_move column
    one_hot_encoded = pd.get_dummies(data['first_move'], prefix='first_move', drop_first=True)
    data_encoded = pd.concat([data, one_hot_encoded], axis=1)
    data_encoded = data_encoded.drop(columns=['first_move'])

    # Set the X and y for the data 
    X = data_encoded.drop(columns=[target_column])
    y = data_encoded[target_column]
    encoder = LabelEncoder()
    y = encoder.fit_transform(data_encoded[target_column])

    # Print out data and stats to ensure correct
    num_entries = data.shape[0]
    print("Number of entries:", num_entries)
    num_features = data.shape[1]
    print("Number of features:", num_features)
    print(data_encoded.head(7))
    counts = data_encoded['winner'].value_counts()
    print("counts", counts)
    sns.countplot(data=data_encoded, x='winner')
    plt.xlabel('Winner')
    plt.ylabel('Count')
    plt.title('Games Won by Black vs. White')
    # plt.show()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)

    return X_train, X_test, y_train, y_test

def get_bank_dataset():
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

    # Encode categorical features using one-hot encoding
    categorical_cols = ["Gender", "Education_Level", "Marital_Status", "Income_Category"]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    num_features_encoded = X_encoded.shape[1]
    print("Number of features after one-hot encoding:", num_features_encoded)

    # Perform undersampling
    # undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=constants.CV)
    # X_resampled, y_resampled = undersampler.fit_resample(X_encoded, y)
    # num_features_resampled = X_resampled.shape[1]
    # print("Number of features after undersampling:", num_features_resampled)

    #perform oversampling instead
    oversampler = RandomOverSampler(sampling_strategy='minority', random_state=constants.CV)
    X_resampled, y_resampled = oversampler.fit_resample(X_encoded, y)
    num_features_resampled = X_resampled.shape[1]
    print("Number of features after oversampling:", num_features_resampled)

    sns.countplot(data=pd.DataFrame({'Attrition_Flag': y_resampled}), x='Attrition_Flag')
    plt.xlabel('Attrition_Flag')
    plt.ylabel('Count')
    plt.title('Attrition Flag Counts (After Undersampling)')
    plt.show()

    counts = pd.Series(y_resampled).value_counts()

    # Now 'counts' contains the counts of each category ('Existing Customer' and 'Attrited Customer')
    print(counts)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)
    return X_train, X_test, y_train, y_test
