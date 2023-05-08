import sys
import os
from alive_progress import alive_bar

sys.path.append("../../")
from src.pipeline_functions.data_preprocessing_functions import *


def run(
    browser_one: str,
    date_one: str,
    train_data_file_name: str,
    browser_two: str,
    date_two: str,
    test_data_file_name: str,
    other_test_data: str,
) -> None:
    dir_path_one = f"{browser_one}/{date_one}"
    dir_path_two = f"{browser_two}/{date_two}"

    try:
        os.makedirs(f"../../../data/processed/{dir_path_one}", exist_ok=True)
        print(f"Directory {dir_path_one} created successfully.")
    except OSError as error:
        print(f"Directory {dir_path_one} can not be created.")

    try:
        os.makedirs(f"../../../data/processed/{dir_path_two}", exist_ok=True)
        print(f"Directory {dir_path_two} created successfully.")
    except OSError as error:
        print(f"Directory {dir_path_two} can not be created.")

    dir_path_one = f"../../../data/processed/{dir_path_one}/{train_data_file_name}"
    dir_path_two = f"../../../data/processed/{dir_path_two}/{test_data_file_name}"

    other_test_data = True if other_test_data == "other" else False

    preprocessing_data(dir_path_one, dir_path_two, other_test_data)


def preprocessing_data(
    train_data_file_path: str, test_data_file_path: str, other_test_data: bool
) -> None:
    with alive_bar(100, force_tty=True, manual=True, title="Data Processing") as bar:

        bar.text("Read-in data")
        data_train = pd.read_parquet(
            f"{train_data_file_path}.parquet.gzip",
            engine="pyarrow",
            dtype_backend="pyarrow",
        )
        data_test = pd.read_parquet(
            f"{test_data_file_path}.parquet.gzip",
            engine="pyarrow",
            dtype_backend="pyarrow",
        )
        bar(0.05)

        bar.text("Remove empty columns")
        empty_columns = [
            col for col in data_train if data_train[col].isnull().all() == True
        ]
        data_train.drop(empty_columns, axis=1, inplace=True)
        if other_test_data:
            columns = empty_columns + data_test.columns.values.tolist()
            empty_columns = [x for x in columns if columns.count(x) > 1]
        data_test.drop(empty_columns, axis=1, inplace=True)

        if other_test_data:
            test_columns = data_test.columns.values.tolist()
            train_columns = data_train.columns.values.tolist()
            cols_not_in_train = list(set(test_columns).difference(train_columns))
            data_test.drop(cols_not_in_train, axis=1, inplace=True)

        data_train.reset_index(drop=True, inplace=True)
        bar(0.15)

        # bar.text("Remove duplicated observations")
        # data_train = data[~data_train.iloc[:, 6:-1].duplicated(keep="first")].reset_index(drop=True)
        # bar(0.2)

        bar.text("Fuzzy match")
        data_column_values = data_train.columns.values[6:-1].tolist()
        match = [
            new_fuzzy_string_matching_for_column(j, data_column_values[i + 1 :])
            for i, j in enumerate(data_column_values)
            if i != len(data_column_values) - 1
        ]

        match2 = pd.concat(match, ignore_index=True)
        del match
        bar(0.3)

        bar.text("Find fuzzy matches with similar columns")

        result = [
            find_cols_with_similar_values(col, col2, data_train)
            for col, col2 in zip(match2["fuzzy_match"], match2["col_name"])
        ]
        bar(0.4)

        bar.text("Reset index")
        data_train.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        similar_values = [
            select_similar_columns(col[0], col[1], match2)
            for col in result
            if col is not None
        ]

        similar_values_test = pd.concat(similar_values, ignore_index=True)
        similar_values = pd.concat(similar_values, ignore_index=True)
        if other_test_data:
            other_columns = data_test.columns.values.tolist()
            values = list(
                set(
                    similar_values.col_name.values.tolist()
                    + similar_values.fuzzy_match.values.tolist()
                )
            )
            cols_not_in_other = list(set(values).difference(other_columns))
            similar_values_test = similar_values_test[
                ~similar_values_test["fuzzy_match"].isin(cols_not_in_other)
            ]
            similar_values_test = similar_values_test[
                ~similar_values_test["col_name"].isin(cols_not_in_other)
            ]

        similar_values.apply(
            lambda x: merge_similar_columns(
                x["fuzzy_match"], x["col_name"], data_train
            ),
            axis=1,
        )

        similar_values_test.apply(
            lambda x: merge_similar_columns(x["fuzzy_match"], x["col_name"], data_test),
            axis=1,
        )

        del match2

        columns_to_remove = list(set(similar_values.fuzzy_match.values.tolist()))
        data_train.drop(columns_to_remove, axis=1, inplace=True)
        if other_test_data:
            columns_to_remove = list(
                set(similar_values_test.fuzzy_match.values.tolist())
            )
        data_test.drop(columns_to_remove, axis=1, inplace=True)

        del result
        del similar_values
        del similar_values_test
        del columns_to_remove

        bar(0.6)

        bar.text("Removing URL Parameters and parsing data types")
        # remove non http headers
        data_train = data_train.iloc[:, 6:]
        data_test = data_test.iloc[:, 6:]
        list_of_dtypes = create_categories_list(data_train)
        list_of_dtypes_test = create_categories_list(data_test)
        data_train = data_train.astype(list_of_dtypes)
        data_test = data_test.astype(list_of_dtypes_test)

        bar(0.7)

        bar.text("Find columns with na_ratio of 1")

        summary_table = create_summary_table(data_train.iloc[:, :-1])
        remove_headers_with_one_na_ratio = summary_table[
            summary_table["na_ratio"] == 1
        ].header_name.values.tolist()
        remove_headers_with_one_value = summary_table[
            (summary_table["unique_values"] <= 1) & (summary_table["na_ratio"] != 1)
        ].header_name.values.tolist()

        bar(0.8)

        bar.text("Drop columns")

        data_train.drop(remove_headers_with_one_na_ratio, axis=1, inplace=True)
        data_train.drop(remove_headers_with_one_value, axis=1, inplace=True)

        if other_test_data:
            other_columns = data_test.columns.values.tolist()
            col_intersection = list(
                set(remove_headers_with_one_na_ratio).intersection(other_columns)
            )
            data_test.drop(col_intersection, axis=1, inplace=True)

            other_columns = data_test.columns.values.tolist()
            col_intersection = list(
                set(remove_headers_with_one_value).intersection(other_columns)
            )
            data_test.drop(col_intersection, axis=1, inplace=True)

            test_columns = data_test.columns.values.tolist()
            train_columns = data_train.columns.values.tolist()
            cols_not_in_train = list(set(test_columns).difference(train_columns))
            data_test.drop(cols_not_in_train, axis=1, inplace=True)
        else:
            data_test.drop(remove_headers_with_one_na_ratio, axis=1, inplace=True)
            data_test.drop(remove_headers_with_one_value, axis=1, inplace=True)

        del remove_headers_with_one_na_ratio
        del summary_table

        bar(0.85)

        bar.text("Combine columns")
        summary_table2 = create_summary_table_2(data_train)

        concise_information_wrapper(data_train, summary_table2)
        if other_test_data:
            other_summary_table = summary_table2[
                ~summary_table2.header_name.isin(
                    list(
                        set(data_train.columns.values.tolist()).difference(
                            data_test.columns.values.tolist()
                        )
                    )
                )
            ]
            concise_information_wrapper(data_test, other_summary_table)
        else:
            concise_information_wrapper(data_test, summary_table2)

        bar(0.95)

        bar.text("Removing remaining combined columns")
        only_non_tracker_col, only_tracker_col = find_cols_to_combine(summary_table2)

        data_train.drop(only_non_tracker_col, axis=1, inplace=True)
        data_train.drop(only_tracker_col, axis=1, inplace=True)

        if other_test_data:
            other_columns = data_test.columns.values.tolist()
            chrome_columns = data_train.columns.values.tolist()
            cols_not_in_chrome = list(set(other_columns).difference(chrome_columns))
            data_test.drop(cols_not_in_chrome, axis=1, inplace=True)
        else:
            data_test.drop(only_non_tracker_col, axis=1, inplace=True)
            data_test.drop(only_tracker_col, axis=1, inplace=True)

        data_train["tracker"] = data_train["tracker"].astype("Int32")
        data_test["tracker"] = data_test["tracker"].astype("Int32")
        reordered_cols = label_as_last_column(data_train)
        if len(data_test.columns) != len(reordered_cols):
            missing_cols = list(
                set(reordered_cols).difference(data_test.columns.tolist())
            )
            data_test = data_test.reindex(
                columns=data_test.columns.tolist() + missing_cols
            )
        data_test = data_test[reordered_cols]
        bar(0.99)

        bar.text("Write data to parquet.gzip")

        data_train.to_parquet(
            f"{train_data_file_path}_processed.parquet.gzip",
            compression="gzip",
        )

        data_test.to_parquet(
            f"{test_data_file_path}_processed.parquet.gzip",
            compression="gzip",
        )
        bar(1)
