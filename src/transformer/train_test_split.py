import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def stratified_shuffle_split(X, y):
    split = StratifiedShuffleSplit(n_splits=1, random_state=10, test_size=0.2)

    for train, test in split.split(X, y):
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

    return X_train, y_train, X_test, y_test


def standard_split(X, y, test_size, random_state):
    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=["tracker"]
    )
    return X_train, y_train, X_test, y_test


def plot_tracker_distribution(train_set, test_set):
    plt.figure(figsize=(10, 15))

    plt.subplot(121)
    plt.pie(
        train_set.value_counts(), labels=["non_tracker", "tracker"],
        autopct="%1.2f%%"
    )
    plt.title("Training Dataset")

    plt.subplot(122)
    plt.pie(
        test_set.value_counts(), labels=["non_tracker", "tracker"],
        autopct="%1.2f%%"
    )
    plt.title("Test Dataset")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = pd.read_parquet("../../data/interim/processed_test.parquet.gzip")
    data = data.iloc[:, 6:]

    X_train, y_train, X_test, y_test = stratified_shuffle_split(
        data.iloc[:, :-1], data[["tracker"]]
    )

    plot_tracker_distribution(y_test, y_test)
