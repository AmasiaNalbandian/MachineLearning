import mlrose_hiive
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import RANDOM_STATE
from NNOptimizerOptuna import nn
from sklearn.preprocessing import LabelEncoder


# These are the hyperparameters you found using Optuna:
best_hidden_layer_size = 64
best_activation = 'relu'
best_solver = 'sgd'
best_alpha = 7.342732836786994e-05

# Define neural network with different optimization algorithms
def train_nn_with_optimizer(X_train_scaled, X_test_scaled, y_train, y_test, algorithm, name):
    # Define optimization parameters
    global_max_iters = 1000
    global_max_attempts = 100

    print("max iterations", global_max_iters)
    nn_model = mlrose_hiive.NeuralNetwork(
        hidden_nodes=[best_hidden_layer_size], # Use the best hidden layer size
        activation=best_activation,            # Use the best activation
        algorithm=algorithm,
        max_iters=global_max_iters,
        max_attempts=global_max_attempts,
        learning_rate=0.1,                     # This learning rate might need adjustment
        curve=True,
        random_state=RANDOM_STATE
    )
    
    start_time = time.time()
    nn_model.fit(X_train_scaled, y_train)
    end_time = time.time() - start_time

    train_accuracy = nn_model.score(X_train_scaled, y_train)
    test_accuracy = nn_model.score(X_test_scaled, y_test)
    
    print(f"{name} Training Accuracy: {train_accuracy:.4f}")
    print(f"{name} Testing Accuracy: {test_accuracy:.4f}")
    print(f"{name} took {end_time} seconds to execute.")
    
    return nn_model

#functiont o run optuna and get best hyper params
def run_optuna(X_train_scaled, X_test_scaled, y_train, y_test):
    best_nn_model, best_nn_params = nn(X_train_scaled, X_test_scaled, y_train, y_test)
    # plot_learning_curve(best_nn_model, X_train_scaled, y_train, scoring='accuracy', title=f"Chess Neural Network Learning Curve\nOptimized Paramters by Optuna")    

#plot curves
def plot_curve(curve, title):
    # Plotting the learning curve for RHC optimizer
    plt.figure(figsize=(10, 6))
    plt.plot(curve[:,0], label='Learning Curve', color='blue')
    plt.title(f'Learning Curve for {title}')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load your dataset
def get_data():
    d = pd.read_csv('./datasets/games.csv')

    filtered_data = d[(d['turns'] >= 15) & (d['winner'] != 'draw')].copy()

    # Define the target column and set random_state
    target_column = 'winner'

    # parse out the first move
    filtered_data['moves'] = filtered_data['moves'].str.split()
    filtered_data['first_move'] = filtered_data['moves'].str[0]
    # filtered_data['second_move'] = filtered_data['moves'].str[1] #TODO: If I want to include the second move

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
    return X, y


