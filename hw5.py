# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use

    train_error =
    test_error =

    return train_error, test_error


def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_trees =

    # Split data
    X_train =
    y_train =
    X_test =
    y_test =

    train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, num_trees)


if __name__ == "__main__":
    main_hw5()
