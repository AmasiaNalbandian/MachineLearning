from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from stats import get_classification_report
import constants
import time

def tune_neural_network(X_train, y_train, param_grid, cv, scoring='accuracy'):
    mlp_classifier = MLPClassifier(max_iter=1000, random_state=constants.RANDOM_STATE)

    grid_search = GridSearchCV(estimator=mlp_classifier, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

def nn(X_train, X_test, y_train, y_test, param_grid, cv=constants.CV):
    best_nn_model, best_nn_params = tune_neural_network(X_train, y_train, param_grid, cv)
    print("Best Hyperparameters:", best_nn_params)

    start_time = time.time()
    y_test_pred = best_nn_model.predict(X_test)
    elapsed_time = time.time() - start_time

    y_train_pred = best_nn_model.predict(X_train)

    get_classification_report(y_train, y_train_pred, "NN train report")
    get_classification_report(y_test, y_test_pred, "NN test report")

    print(f"NN took {elapsed_time} seconds to execute.")
    
    return best_nn_model, best_nn_params
