from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import constants

def tune_neural_network(X_train, y_train, param_grid, cv, scoring='accuracy'):
    mlp_classifier = MLPClassifier(max_iter=1000, random_state=constants.RANDOM_STATE)

    grid_search = GridSearchCV(estimator=mlp_classifier, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

def nn(X_train, X_test, y_train, y_test, cv):
    param_grid = {
        'hidden_layer_sizes': [(32,), (64,), (128,), (256,), (512,)],
        'learning_rate_init': [0.001, 0.01, 0.1, 0.0001]
    }

    best_nn_model, best_nn_params = tune_neural_network(X_train, y_train, param_grid, cv)
    print("Best Hyperparameters:", best_nn_params)

    # You can use best_nn_model for predictions and further evaluation
    y_pred = best_nn_model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    return best_nn_model, best_nn_params
