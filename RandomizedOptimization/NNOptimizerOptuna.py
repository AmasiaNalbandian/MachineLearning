import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from stats import get_classification_report
from constants import CV, RANDOM_STATE
import time

def objective(trial, X_train, y_train, cv):

    # Suggest hyperparameters
    hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 50, 300)
    activation = trial.suggest_categorical("activation", ["relu", "logistic", "tanh"])
    solver = trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"])
    alpha = trial.suggest_float("alpha", 1e-5, 1e-3, log=True)
    
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    
    # Use cross-validation to compute accuracy
    from sklearn.model_selection import cross_val_score
    return cross_val_score(mlp_classifier, X_train, y_train, cv=cv, scoring='accuracy').mean()

def nn(X_train, X_test, y_train, y_test, cv=CV):

    study = optuna.create_study(direction='maximize')  # we want to maximize accuracy
    study.optimize(lambda trial: objective(trial, X_train, y_train, cv), n_trials=50)  # adjust n_trials as needed

    best_params = study.best_params
    best_accuracy = study.best_value
    print(f"Best hyperparameters: {best_params}")
    print(f"Best cross-validated accuracy: {best_accuracy}")
    
    best_nn_model = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation=best_params['activation'],
        solver=best_params['solver'],
        alpha=best_params['alpha'],
        max_iter=1000,
        random_state=RANDOM_STATE
    ).fit(X_train, y_train)
    
    print("Best Hyperparameters:", best_params)

    start_time = time.time()
    y_test_pred = best_nn_model.predict(X_test)
    elapsed_time = time.time() - start_time

    y_train_pred = best_nn_model.predict(X_train)

    get_classification_report(y_train, y_train_pred, "NN train report")
    get_classification_report(y_test, y_test_pred, "NN test report")

    print(f"NN took {elapsed_time} seconds to execute.")
    
    return best_nn_model, best_params

