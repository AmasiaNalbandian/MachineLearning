from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(26)

# def get_dt(X_train_pca, y_train, X_test_pca, y_test):
#     dt_classifier = DecisionTreeClassifier(random_state=42)
#     dt_classifier.fit(X_train_pca, y_train)
#     y_pred = dt_classifier.predict(X_test_pca)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy of the decision tree classifier on the test set: {accuracy:.2%}")


# def run_pca_with_decision_tree(X, y, n_components=2):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # PCA transformation
#     pca = PCA(n_components=n_components)
#     X_train_pca = pca.fit_transform(X_train)
#     X_test_pca = pca.transform(X_test)
    
#     # Decision Tree classifier
#     dt_classifier = DecisionTreeClassifier(random_state=42)
#     dt_classifier.fit(X_train_pca, y_train)
    
#     # Predictions
#     y_pred = dt_classifier.predict(X_test_pca)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy of the decision tree classifier on the test set: {accuracy:.2%}")
    
#     return dt_classifier, pca

import matplotlib.pyplot as plt

def run_decision_tree(X_train, X_test, y_train, y_test):
    # Decision Tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


def compare_dimensionality_reduction(X_train_scaled, X_test_scaled, y_train, y_test, X_train_dim_red, X_test_dim_red, dr_technique_name="PCA"):
    # Run without dimensionality reduction
    acc_without_dr = run_decision_tree(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Run with dimensionality reduction
    acc_with_dr = run_decision_tree(X_train_dim_red, X_test_dim_red, y_train, y_test)

    # Plotting
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(2)
    plt.bar(index, [acc_without_dr, acc_with_dr], bar_width, label=('Without ' + dr_technique_name, 'With ' + dr_technique_name))
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title(f'Comparison of Decision Tree Classifier With and Without {dr_technique_name}')
    plt.xticks(index + bar_width / 2, ('Without ' + dr_technique_name, 'With ' + dr_technique_name))
    plt.tight_layout()
    plt.show()

    return acc_without_dr, acc_with_dr



from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, color="r", alpha=0.1)
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, color="g", alpha=0.1)
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt

def compare_dimensionality_reduction_lc(X_train_scaled, X_test_scaled, y_train, y_test, X_train_dim_red, X_test_dim_red, dr_technique_name="PCA"):
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Plot learning curve for decision tree without dimensionality reduction
    plt_without_dr = plot_learning_curve(dt_classifier, f'Decision Tree learning curve without DR',
                                         X_train_scaled, y_train, cv=5)
    plt_without_dr.show()

    # Plot learning curve for decision tree with dimensionality reduction
    plt_with_dr = plot_learning_curve(dt_classifier, f'Decision Tree learning curve for {dr_technique_name}',
                                      X_train_dim_red, y_train, cv=5)
    plt_with_dr.show()

# Call your compare function
