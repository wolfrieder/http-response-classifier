import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, \
    StratifiedGroupKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys


def split_dataset(splitter, X, y):
    for train, test in splitter.split(X, y):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

    return X_train, y_train, X_test, y_test


def stratified_shuffle_split(X, y):
    split = StratifiedShuffleSplit(n_splits=1, random_state=10, test_size=0.2)
    return split_dataset(split, X, y)


def stratified_kfold_split(X, y):
    split = StratifiedKFold(n_splits=2, random_state=10, shuffle=True)
    return split_dataset(split, X, y)


def stratified_group_kfold_split(X, y):
    split = StratifiedGroupKFold(n_splits=2, random_state=10, shuffle=True)
    return split_dataset(split, X, y)


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
    browser = sys.argv[1]
    directory = sys.argv[2]
    dir_path = f"{browser}/{directory}"

    data = pd.read_parquet(f"data/interim/{dir_path}/{sys.argv[3]}_processed.parquet.gzip")

    # remove non-headers
    data = data.iloc[:, 6:]

    X_train, y_train, X_test, y_test = stratified_shuffle_split(
        data.iloc[:, :-1], data[["tracker"]]
    )

    # plot_tracker_distribution(y_test, y_test)

    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    train_set.to_parquet(
        f"data/processed/{dir_path}/train_set_{sys.argv[4]}.parquet.gzip", compression="gzip"
    )

    test_set.to_parquet(
        f"data/processed/{dir_path}/test_set_{sys.argv[4]}.parquet.gzip", compression="gzip"
    )
