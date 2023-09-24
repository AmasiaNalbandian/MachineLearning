from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, explained_variance_score, median_absolute_error
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report

def get_classification_report(y_set, y_pred, title=None, digits=6):
    report = classification_report(y_set, y_pred, digits=digits)
    print(f"\n{title}\n", report)


#Print all the stats
def print_stats(y_test, y_pred):
    # Evaluate the model using regression metrics
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    #Evaluate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    # Root mean squared Error (RMSE):
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    # r-squared
    r2 = r2_score(y_test, y_pred)
    print("r2: ", r2)

    #Explained Mean Variance
    evs = explained_variance_score(y_test, y_pred)
    print("EVS:", evs)

    #MAPE:
    print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

    #Median Absolute Error(MedAE):
    medae = median_absolute_error(y_test, y_pred)
    print("MedAE: ", medae)


def plot_val_curve(estimator, X, y):
    param_name = "max_depth"
    param_range = np.arange(1, 21)
    plot_validation_curve(estimator, X, y, param_name, param_range, title="Validation Curve for Max Depth")
    plt.show()

def plot_learn_curve(estimator, X, y):
    plot_learning_curve(estimator, X, y, title="Learning Curve")
    plt.show()

#Mean absolute percentage error 
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# """
# Generate a validation curve plot for a given hyperparameter.

# Parameters:
#     - estimator: The model.
#     - X: The feature matrix.
#     - y: The target variable.
#     - param_name: Name of the hyperparameter.
#     - param_range: Range of hyperparameter values to explore.
#     - title: Title for the chart.
#     - cv: Cross-validation strategy (e.g., StratifiedKFold).
#     - scoring: Scoring metric (default is 'neg_mean_squared_error').
# """
def plot_validation_curve(estimator, X, y, param_name, param_range, title, cv=None, scoring='neg_mean_squared_error'):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("MSE Score")
    plt.grid()

    plt.semilogx(param_range, train_scores_mean, 'o-', label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.semilogx(param_range, test_scores_mean, 'o-', label="Cross-validation score", color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.legend(loc="best")
    return plt



# """
# Generate a simple plot of the test and training learning curve.

# Parameters:
#     - estimator: The model to use for learning curve generation.
#     - X: The feature matrix.
#     - y: The target variable.
#     - title: Title for the chart.
#     - cv: Cross-validation strategy (e.g., StratifiedKFold).
#     - n_jobs: Number of CPU cores to use for cross-validation (-1 to use all available cores).
#     - train_sizes: Relative or absolute numbers of training examples to use for the curve.
# """
def plot_learning_curve(estimator, X, y, title, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error'
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("MSE Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt