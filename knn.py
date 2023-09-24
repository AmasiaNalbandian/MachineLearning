from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def tune_knn(X_train, y_train, param_grid, cv, scoring='accuracy'):
    knn_classifier = KNeighborsClassifier()

    grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

def knn(X_train, X_test, y_train, y_test, cv):
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['distance'],
        'p': [1, 2]
    }

    best_knn_model, best_knn_params = tune_knn(X_train, y_train, param_grid, cv)
    print("Best Hyperparameters:", best_knn_params)

    # You can use best_knn_model for predictions and further evaluation
    y_pred = best_knn_model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    return best_knn_model, best_knn_params
