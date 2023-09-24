from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint
from sklearn.ensemble import AdaBoostClassifier
import constants

def dt(X_train, X_test, y_train, y_test, param_grid, cv=constants.CV):
    best_model, best_params = useGridSearch(X_train, y_train, param_grid)
    # best_model, best_params = useRandomizedSearch(X_train, y_train)

    # Use the best model for predictions
    y_pred = best_model.predict(X_test)

    # Evaluate the model's accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Best Hyperparameters:", best_params)
    print("cross validation:", cv)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    return best_model, best_params


def useGridSearch(X_train, y_train, param_grid):
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

def ada_boosted_dt(X_train, X_test, y_train, y_test, param_grid, cv=constants.CV):
    # Create a base Decision Tree classifier
    base_dt = DecisionTreeClassifier()

    # Create an AdaBoost classifier with Decision Tree as the base estimator
    ada_dt_classifier = AdaBoostClassifier(base_estimator=base_dt)

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=ada_dt_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_ada_dt_model = grid_search.best_estimator_
    best_ada_dt_params = grid_search.best_params_

    # Use the best model for predictions
    y_pred = best_ada_dt_model.predict(X_test)

    # Generate a classification report
    report = classification_report(y_test, y_pred)

    print("Best AdaBoost Decision Tree Hyperparameters:", best_ada_dt_params)
    print("Classification Report:\n", report)

    return best_ada_dt_model, best_ada_dt_params
