from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from stats import print_stats, plot_val_curve, plot_learn_curve

def run_dt(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a DecisionTreeRegressor model
    regressor = DecisionTreeRegressor(random_state=42)

    # Train the model
    regressor.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = regressor.predict(X_test)

    print_stats(y_test, y_pred)


    plot_val_curve(regressor, X, y)
    plot_learn_curve(regressor, X, y)

def run_dt_class(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=42)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
