import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import kstest, shapiro, probplot
import xgboost


def variance_per_column_2(column):
    tracker_ratio = train_data[train_data[column].notnull()].tracker.value_counts()
    try:
        trackers = tracker_ratio[1]
    except KeyError:
        trackers = 0
    try:
        non_trackers = tracker_ratio[0]
    except KeyError:
        non_trackers = 0
    return [column, trackers, non_trackers]


# https://sparkbyexamples.com/pandas/pandas-change-position-of-a-column/
def label_as_last_column(dataset):
    temp_cols = dataset.columns.tolist()
    index_col = dataset.columns.get_loc("tracker")
    new_col_order = (
        temp_cols[0:index_col]
        + temp_cols[index_col + 1:]
        + temp_cols[index_col: index_col + 1]
    )
    return new_col_order


if __name__ == "__main__":
    train_data = pd.read_parquet(
        "../data/processed/chrome/08_12_2022/train_set_01_processed.parquet.gzip"
    )
    test_data = pd.read_parquet(
        "../data/processed/chrome/08_12_2022/test_set_01_processed.parquet.gzip"
    )

    # exclude metadata columns
    train_data = train_data.iloc[:, 4:]
    test_data = test_data.iloc[:, 4:]

    number_of_elements_reduced = np.array(
        [variance_per_column_2(column) for column in train_data.iloc[:, :-4].columns]
    )
    summary_table = pd.DataFrame(
        number_of_elements_reduced, columns=["header_name", "trackers", "non_trackers"]
    )
    summary_table["trackers"] = summary_table["trackers"].astype("Int32")
    summary_table["non_trackers"] = summary_table["non_trackers"].astype("float32")

    na_ratio_greater_than_85 = summary_table[
        summary_table["tracker_na_ratio"] >= 0.85
    ].header_name.values.tolist()

    for elem in na_ratio_greater_than_85:
        train_data[f'{elem}_binary'] = np.where(train_data[elem].isnull(), 0, 1)
        test_data[f'{elem}_binary'] = np.where(test_data[elem].isnull(), 0, 1)

    train_data.drop(na_ratio_greater_than_85, axis=1, inplace=True)
    test_data.drop(na_ratio_greater_than_85, axis=1, inplace=True)

    train_data.drop(['last-modified', 'date'], axis=1, inplace=True)
    test_data.drop(['last-modified', 'date'], axis=1, inplace=True)

    # put tracker column at the end
    reordered_cols = label_as_last_column(train_data)
    train_data = train_data[reordered_cols]
    test_data = test_data[reordered_cols]

    X_train, y_train = train_data.iloc[:, :-1], train_data[["tracker"]]
    X_test, y_test = test_data.iloc[:, :-1], test_data[["tracker"]]

    list_of_categorical_cols = list(
        X_train.iloc[:, :-3].select_dtypes("category").columns.values.tolist()
    )
    list_of_integer_cols = list(
        X_train.iloc[:, :-3].select_dtypes("Int64").columns.values.tolist()
    )
