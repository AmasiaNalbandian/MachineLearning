from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import constants
from stats import get_classification_report
import time

def tune_knn(X_train, y_train, param_grid, cv, scoring='accuracy'):
    knn_classifier = KNeighborsClassifier()

    grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

def knn(X_train, X_test, y_train, y_test, param_grid, cv=constants.CV):
    best_knn_model, best_knn_params = tune_knn(X_train, y_train, param_grid, cv)
    print("Best Hyperparameters:", best_knn_params)

    start_time = time.time()
    y_test_pred = best_knn_model.predict(X_test)
    elapsed_time = time.time() - start_time

    y_train_pred = best_knn_model.predict(X_train)

    # Print the classification reports
    get_classification_report(y_train, y_train_pred, "knn train report")
    get_classification_report(y_test, y_test_pred, "knn test report")

    print(f"K-NN took {elapsed_time} seconds to execute.")

    return best_knn_model, best_knn_params
