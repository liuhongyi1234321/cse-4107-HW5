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

    train_error = []
    test_error = []
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    weights = np.ones(n_train) / n_train
    F_train = np.zeros(n_train)
    F_test = np.zeros(n_test)
    for i in range(n_trees):
        stump, stump_pred_train = train_weighted_stump(X_train, y_train, weights)
        err = np.sum(weights * (stump_pred_train != y_train))
        err = np.clip(err, 1e-10, 1 - 1e-10)
        if err > 0.5:
            break
        alpha = 0.5 * np.log((1 - err) / err)
        weights = weights * np.exp(-alpha * y_train * stump_pred_train)
        weights = weights / np.sum(weights)
        F_train += alpha * stump_pred_train
        stump_pred_test = stump.predict(X_test)
        F_test += alpha * stump_pred_test
        train_err = np.mean(np.sign(F_train) != y_train)
        test_err = np.mean(np.sign(F_test) != y_test)
        train_error.append(train_err)
        test_error.append(test_err)

    return train_error, test_error

def train_weighted_stump(X, y, sample_weights):
    """
    Train a decision stump (depth-1 decision tree) on weighted data.
    """
    stump = DecisionTreeClassifier(max_depth=1, criterion="entropy")
    stump.fit(X, y, sample_weight=sample_weights)
    predictions = stump.predict(X)
    return stump, predictions


def build_binary_task(data, pos_class, neg_class):
    mask = np.isin(data[:, 0], [pos_class, neg_class])
    subset = data[mask]
    X = subset[:, 1:]
    y = np.where(subset[:, 0] == pos_class, 1, -1)
    return X, y
    
def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_trees = 200

    # Split data
    X_train1, y_train1 = build_binary_task(og_train_data, 1, 3)
    X_test1, y_test1 = build_binary_task(og_test_data, 1, 3)

    X_train2, y_train2 = build_binary_task(og_train_data, 3, 5)
    X_test2, y_test2 = build_binary_task(og_test_data, 3, 5)


    train_error1, test_error1 = adaboost_trees(X_train1, y_train1, X_test1, y_test1, num_trees)
    train_error2, test_error2 = adaboost_trees(X_train2, y_train2, X_test2, y_test2, num_trees)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    rounds1 = np.arange(1, len(train_error1) + 1)
    axes[0].plot(rounds1, train_error1, label="Train Error")
    axes[0].plot(rounds1, test_error1, label="Test Error")
    axes[0].set_title("AdaBoost (digits 1 vs 3)")
    axes[0].set_xlabel("Number of Weak Learners")
    axes[0].set_ylabel("Error Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rounds2 = np.arange(1, len(train_error2) + 1)
    axes[1].plot(rounds2, train_error2, label="Train Error")
    axes[1].plot(rounds2, test_error2, label="Test Error")
    axes[1].set_title("AdaBoost (digits 3 vs 5)")
    axes[1].set_xlabel("Number of Weak Learners")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if train_error1:
        print(f"(1 vs 3) final train error: {train_error1[-1]:.4f}")
    if test_error1:
        print(f"(1 vs 3) final test error: {test_error1[-1]:.4f}")
    if train_error2:
        print(f"(3 vs 5) final train error: {train_error2[-1]:.4f}")
    if test_error2:
        print(f"(3 vs 5) final test error: {test_error2[-1]:.4f}")

if __name__ == "__main__":
    main_hw5()
