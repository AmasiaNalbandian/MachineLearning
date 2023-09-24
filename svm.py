from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import constants

def tune_svm(X_train, y_train, param_grid, cv, scoring='accuracy'):
    svm_classifier = SVC(random_state=constants.RANDOM_STATE)

    grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

def svm(X_train, X_test, y_train, y_test, param_grid, cv=constants.CV):
    X_train, y_train = get_subset(X_train, y_train)
    print("running v2")

    best_svm_model, best_svm_params = tune_svm(X_train, y_train, param_grid, cv)
    print("Best Hyperparameters:", best_svm_params)

    # You can use best_svm_model for predictions and further evaluation
    y_pred = best_svm_model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    return best_svm_model, best_svm_params

# This function is used to get a subset of data to experiment faster with the SVMs
def get_subset(X_train, y_train, subset_size=100):
    random_indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_subset = X_train[random_indices]
    y_subset = y_train.iloc[random_indices]

    return X_subset, y_subset