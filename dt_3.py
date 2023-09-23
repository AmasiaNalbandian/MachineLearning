import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import constants

def dt(X_train, X_test, y_train, y_test, cv):
    best_model, best_params = useGridSearch(X_train, y_train)
    # best_model, best_params = useRandomizedSearch(X_train, y_train)

    # Use the best model for predictions
    y_pred = best_model.predict(X_test)

    # Evaluate the model's accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Best Hyperparameters:", best_params)
    print("cross validation:", constants.CV)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    return best_model, best_params


def useGridSearch(X_train, y_train):
    # Create a GridSearchCV object for hyperparameter tuning
    param_grid = {
        'max_depth': [8,9,7,6,5],  # You can include 'None' for unlimited depth
        'min_samples_split': [2, 5, 3, 13],
        'min_samples_leaf': [1, 5, 9]
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=constants.CV, scoring='accuracy')


    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and best hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params


def useRandomizedSearch(X_train, y_train):
    # Define the hyperparameter space with distributions
    param_dist = {
        'max_depth': randint(1, 10),  # Randomly sample integers between 1 and 10
        'min_samples_split': randint(2, 20),  # Randomly sample integers between 2 and 20
        'min_samples_leaf': randint(1, 10),  # Randomly sample integers between 1 and 10
    }

    # Create a RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        DecisionTreeClassifier(),
        param_distributions=param_dist,
        n_iter=10,  # Number of parameter settings that are sampled
        cv=constants.CV,
        scoring='accuracy',
        random_state=constants.RANDOM_STATE
    )

    # Fit the randomized search to the training data
    random_search.fit(X_train, y_train)

    # Get the best model and best hyperparameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params
