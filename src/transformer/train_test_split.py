import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.model_selection import train_test_split
from alive_progress import alive_bar


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
        X, y, test_size=test_size, random_state=random_state, stratify=["tracker"]
    )
    return X_train, y_train, X_test, y_test


def plot_tracker_distribution(train_set, test_set):
    plt.figure(figsize=(10, 15))

    plt.subplot(121)
    plt.pie(
        train_set.value_counts(), labels=["non_tracker", "tracker"], autopct="%1.2f%%"
    )
    plt.title("Training Dataset")

    plt.subplot(122)
    plt.pie(
        test_set.value_counts(), labels=["non_tracker", "tracker"], autopct="%1.2f%%"
    )
    plt.title("Test Dataset")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with alive_bar(100, force_tty=True, manual=True, title="Train-Test-Split") as bar:
        bar.text('Read-in parameters')
        browser = sys.argv[1]
        directory = sys.argv[2]
        dir_path = f"{browser}/{directory}"
        bar(0.05)

        bar.text("Read-in data")
        data = pd.read_parquet(f"../../../data/interim/{dir_path}/{sys.argv[3]}.parquet.gzip")
        bar(0.1)

        bar.text("Create directory if it doesn't exist")
        try:
            os.makedirs(f"../../data/processed/{dir_path}", exist_ok=True)
            print(f"Directory {dir_path} created successfully.")
        except OSError as error:
            print(f"Directory {dir_path} can not be created.")

        bar(0.15)

        bar.text("Read-in second dataset")
        if len(sys.argv) > 4:
            data_2 = pd.read_parquet(
                f"../../../data/interim/{dir_path}/{sys.argv[5]}.parquet.gzip"
            )
            bar(0.2)
            bar.text("Concat data")
            concat_data = [data, data_2]
            data = pd.concat(concat_data, ignore_index=True)
            del data_2
            bar(0.3)

            bar.text("Reindex columns")
            temp_cols = data.columns.tolist()
            index_col = data.columns.get_loc("tracker")
            new_col_order = (
                temp_cols[0:index_col]
                + temp_cols[index_col + 1:]
                + temp_cols[index_col: index_col + 1]
            )
            data = data[new_col_order]

        bar(0.5)

        bar.text("Split dataset")
        X_train, y_train, X_test, y_test = stratified_shuffle_split(
            data.iloc[:, :-1], data[["tracker"]]
        )

        # plot_tracker_distribution(y_train, y_test)
        del data
        bar(0.6)

        bar.text("Concat Training Data")
        train_set = pd.concat([X_train, y_train], axis=1)
        bar(0.7)

        bar.text("Concat Test Data")
        test_set = pd.concat([X_test, y_test], axis=1)
        bar(0.9)

        bar.text("Write datasets to parquet files")

        train_set.to_parquet(
            f"../../../data/processed/{dir_path}/train_set_{sys.argv[4]}{sys.argv[6]}.parquet.gzip",
            compression="gzip",
        )

        test_set.to_parquet(
            f"../../../data/processed/{dir_path}/test_set_{sys.argv[4]}{sys.argv[6]}.parquet.gzip",
            compression="gzip",
        )
        bar(1)
