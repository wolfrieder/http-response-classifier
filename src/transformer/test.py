import os
import sys

# import pandas as pd
import ray

# import modin.pandas as pd
import pandas as pd

# import yaml
# from pandarallel import pandarallel
import swifter
from rapidfuzz import process
import numpy as np
from alive_progress import alive_bar
from collections import Counter


def new_fuzzy_string_matching_for_column(col_name, col_values):
    fuzzy_result = pd.DataFrame(
        process.extract(
            col_name, col_values, processor=None, score_cutoff=80, limit=100
        ),
        columns=["fuzzy_match", "w_ratio", "index"],
    )
    fuzzy_result["col_name"] = col_name
    return fuzzy_result


def find_cols_with_similar_values(fuzzy_match, column):
    value_fuzzy = set(data[fuzzy_match].values)
    value_column = set(data[column].values)

    try:
        value_fuzzy.remove(None)
        value_column.remove(None)
    except KeyError:
        pass

    if (len([True for i in value_fuzzy if i in value_column]) / len(value_fuzzy)) > 0.5:
        return fuzzy_match, column
    else:
        return None


def select_similar_columns(fuzzy_match, column):
    row = match2.loc[
        (match2["fuzzy_match"] == fuzzy_match) & (match2["col_name"] == column)
    ]
    index = row.index[0]
    match2.drop(index, inplace=True)
    return row


def merge_similar_columns2(fuzzy_match, col_name):
    boolean_mask = data[fuzzy_match].notnull()
    new_values = data[boolean_mask][fuzzy_match].to_numpy()
    indices_fuzzy_matches = data.index[boolean_mask].tolist()

    current_values = data[col_name].to_numpy()
    np.put(current_values, indices_fuzzy_matches, new_values)


def merge_similar_columns2_test(fuzzy_match, col_name):
    boolean_mask = data_test[fuzzy_match].notnull()
    new_values = data_test[boolean_mask][fuzzy_match].to_numpy()
    indices_fuzzy_matches = data_test.index[boolean_mask].tolist()

    current_values = data_test[col_name].to_numpy()
    np.put(current_values, indices_fuzzy_matches, new_values)


def reduced_variance_per_column(column):
    unique_values = int(len(set(data[column]))) - 1
    na_values = data[column].isna().sum()
    return [column, unique_values, round(na_values / len(data), 3)]


def variance_per_column_2(column):
    tracker_ratio = data[data[column].notnull()].tracker.value_counts()
    try:
        trackers = tracker_ratio[1]
    except KeyError:
        trackers = 0
    try:
        non_trackers = tracker_ratio[0]
    except KeyError:
        non_trackers = 0
    return [column, trackers, non_trackers]


def concise_information(col_list, classification, dataset):
    indices = list()

    for col in col_list:
        indices.append(
            dataset[
                (dataset[col].notnull()) & (dataset["tracker"] == classification)
            ].index.tolist()
        )

    return indices


def find_cols_to_combine(information_table):
    only_non_trackers = information_table[
        information_table["ratio"] <= 10
    ].header_name.values.tolist()
    only_trackers = information_table[
        information_table["ratio2"] <= 10
    ].header_name.values.tolist()
    return only_non_trackers, only_trackers


def concise_information_wrapper(dataset, table):
    only_non_tracker_cols, only_tracker_cols = find_cols_to_combine(table)

    dataset["comb_col_non_tracker"] = 0
    dataset["comb_col_tracker"] = 0

    col_tracker = dict(
        Counter(
            list(
                np.concatenate(concise_information(only_tracker_cols, 1, dataset)).flat
            )
        )
    )
    col_non_tracker = dict(
        Counter(
            list(
                np.concatenate(
                    concise_information(only_non_tracker_cols, 0, dataset)
                ).flat
            )
        )
    )

    for key, value in col_tracker.items():
        dataset.at[key, "comb_col_tracker"] = value

    for key, value in col_non_tracker.items():
        dataset.at[key, "comb_col_non_tracker"] = value


def test_new_categories_update(element, dataset):
    categories = dataset[element].astype("category").cat.categories.values.tolist()
    try:
        np.array(categories, dtype="int64")
        return {element: "Int64"}
    except (ValueError, OverflowError):
        return None


def create_categories_list(dataset):
    dtype_list = {i: "category" for i in dataset.columns.values[:-2]}
    current_columns = dataset.columns.values[6:-2].tolist()
    int64_columns = [
        test_new_categories_update(element, dataset) for element in current_columns
    ]

    int64_columns = list(filter(lambda x: type(x) is dict, int64_columns))
    int64_columns = {k: v for d in int64_columns for k, v in d.items()}

    dtype_list.update(int64_columns)
    del dtype_list["query"]
    del dtype_list["protocol"]
    return dtype_list


def create_summary_table(dataset):
    number_of_elements_reduced = np.array(
        [variance_per_column_2(column) for column in dataset.iloc[:, 4:-2].columns]
    )
    summary_table_2 = pd.DataFrame(
        number_of_elements_reduced, columns=["header_name", "trackers", "non_trackers"]
    )
    summary_table_2["trackers"] = summary_table_2["trackers"].astype("Int32")
    summary_table_2["non_trackers"] = summary_table_2["non_trackers"].astype("float32")
    summary_table_2["ratio"] = (
        summary_table_2["trackers"] / summary_table_2["non_trackers"]
    ) * 100
    summary_table_2["ratio2"] = (
        summary_table_2["non_trackers"] / summary_table_2["trackers"]
    ) * 100
    return summary_table_2


if __name__ == "__main__":
    with alive_bar(100, force_tty=True, manual=True, title="Data Processing") as bar:
        bar.text('Read-in data')
        data = pd.read_parquet(
            "../../data/processed/chrome/08_12_2022/train_set_01.parquet.gzip"
        )

        data_test = pd.read_parquet(
            "../../data/processed/chrome/08_12_2022/test_set_01.parquet.gzip"
        )
        bar(0.05)

        bar.text('Count number of headers per message')
        data["header_count"] = data.iloc[:, 6:-1].notnull().sum(axis=1)
        bar(0.1)
        data_test["header_count"] = data_test.iloc[:, 6:-1].notnull().sum(axis=1)
        bar(0.15)

        bar.text('Remove empty columns')
        empty_columns = [col for col in data if data[col].isnull().all() == True]
        data.drop(empty_columns, axis=1, inplace=True)
        data_test.drop(empty_columns, axis=1, inplace=True)
        bar(0.2)

        # Is that even necessary? Check papers; also in context of HTTP traffic.
        # bar.text("Remove duplicated observations")
        # data = data[~data.iloc[:, 6:-2].duplicated(keep="first")].reset_index(drop=True)

        bar.text('Fuzzy match')
        data_column_values = data.columns.values[6:-2].tolist()
        match = [
            new_fuzzy_string_matching_for_column(j, data_column_values[i + 1:])
            for i, j in enumerate(data_column_values)
            if i != len(data_column_values) - 1
        ]

        match2 = pd.concat(match, ignore_index=True)
        del match
        bar(0.3)

        bar.text('Find fuzzy matches with similar columns')

        result = [
            find_cols_with_similar_values(col, col2)
            for col, col2 in zip(match2["fuzzy_match"], match2["col_name"])
        ]
        bar(0.4)

        bar.text('Reset index')
        data.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        similar_values = [
            select_similar_columns(col[0], col[1]) for col in result if col is not None
        ]

        similar_values_test = pd.concat(similar_values, ignore_index=True)
        similar_values = pd.concat(similar_values, ignore_index=True)
        similar_values.apply(
            lambda x: merge_similar_columns2(x["fuzzy_match"], x["col_name"]), axis=1
        )

        similar_values_test.apply(
            lambda x: merge_similar_columns2_test(x["fuzzy_match"], x["col_name"]),
            axis=1,
        )

        del match2

        columns_to_remove = list(set(similar_values.fuzzy_match.values.tolist()))
        data.drop(columns_to_remove, axis=1, inplace=True)
        data_test.drop(columns_to_remove, axis=1, inplace=True)

        del result
        del similar_values
        del similar_values_test
        del columns_to_remove

        bar(0.6)

        bar.text('Data types')
        list_of_dtypes = create_categories_list(data)
        list_of_dtypes_test = create_categories_list(data_test)
        data.drop(["protocol", "query"], axis=1, inplace=True)
        data_test.drop(["protocol", "query"], axis=1, inplace=True)
        data = data.astype(list_of_dtypes)
        data_test = data_test.astype(list_of_dtypes_test)

        bar(0.7)

        bar.text('Find columns with na_ratio of 1')
        number_of_elements = np.array(
            [
                reduced_variance_per_column(column)
                for column in data.iloc[:, 4:-2].columns
            ]
        )
        summary_table = pd.DataFrame(
            number_of_elements, columns=["header_name", "unique_values", "na_ratio"]
        )
        summary_table["unique_values"] = summary_table["unique_values"].astype("Int32")
        summary_table["na_ratio"] = summary_table["na_ratio"].astype("float32")
        remove_headers_with_one_na_ratio = summary_table[
            summary_table["na_ratio"] == 1
        ].header_name.values.tolist()
        remove_headers_with_one_value = summary_table[
            (summary_table["unique_values"] <= 1) & (summary_table["na_ratio"] != 1)
        ].header_name.values.tolist()

        bar(0.8)

        bar.text('Combine columns')
        summary_table2 = create_summary_table(data)

        concise_information_wrapper(data, summary_table2)
        concise_information_wrapper(data_test, summary_table2)

        bar(0.9)

        bar.text('Drop columns')

        data.drop(remove_headers_with_one_na_ratio, axis=1, inplace=True)
        data.drop(remove_headers_with_one_value, axis=1, inplace=True)

        data_test.drop(remove_headers_with_one_na_ratio, axis=1, inplace=True)
        data_test.drop(remove_headers_with_one_value, axis=1, inplace=True)

        del remove_headers_with_one_na_ratio
        del number_of_elements
        del summary_table

        bar(0.95)

        bar.text('Remove remaining combined columns')
        summary_table2 = create_summary_table(data)
        only_non_tracker_col, only_tracker_col = find_cols_to_combine(summary_table2)

        data.drop(only_non_tracker_col, axis=1, inplace=True)
        data.drop(only_tracker_col, axis=1, inplace=True)

        data_test.drop(only_non_tracker_col, axis=1, inplace=True)
        data_test.drop(only_tracker_col, axis=1, inplace=True)

        bar(0.99)

        bar.text('Write data to parquet.gzip')

        data.to_parquet(
            "../../data/processed/chrome/08_12_2022/train_set_01_processed.parquet.gzip",
            compression="gzip",
        )

        data_test.to_parquet(
            "../../data/processed/chrome/08_12_2022/test_set_01_processed.parquet.gzip",
            compression="gzip",
        )
        bar(1)
