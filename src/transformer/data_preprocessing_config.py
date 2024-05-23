import json
from alive_progress import alive_bar

from src.pipeline_functions.data_preprocessing_functions import *


def run(
    browser: str, date: str, file_name: str, config_path: str, other_test_data: bool
) -> None:
    dir_path = f"{browser}/{date}"

    dir_path = f"data/processed/{dir_path}/{file_name}"

    if "train_set" in file_name:
        preprocessing_train_data(dir_path, config_path)
    else:
        preprocessing_test_data(dir_path, config_path, other_test_data)


def preprocessing_train_data(file_path: str, config_path: str) -> None:
    with alive_bar(100, force_tty=True, manual=True, title="Data Processing") as bar:
        bar.text("Read-in data")
        httpMessage = "response" if "response" in config_path else "request"
        print(f"{file_path}_{httpMessage}.parquet.gzip")
        data = pd.read_parquet(
            f"{file_path}_{httpMessage}.parquet.gzip",
            engine="pyarrow",
            dtype_backend="pyarrow",
        )
        train_columns = data.columns.values.tolist()
        bar(0.05)

        bar.text("Remove empty columns")
        empty_columns = [col for col in data if data[col].isnull().all() == True]
        data.drop(empty_columns, axis=1, inplace=True)
        data.reset_index(drop=True, inplace=True)
        bar(0.15)

        bar.text("Fuzzy match")
        data_column_values = data.columns.values[6:-2].tolist()
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
            find_cols_with_similar_values(col, col2, data)
            for col, col2 in zip(match2["fuzzy_match"], match2["col_name"])
        ]
        bar(0.4)

        bar.text("Reset index")
        data.reset_index(drop=True, inplace=True)

        similar_values = [
            select_similar_columns(col[0], col[1], match2)
            for col in result
            if col is not None
        ]
        bar(0.5)

        bar.text("Merge fuzzy matches")
        similar_values_train = pd.concat(similar_values, ignore_index=True)
        similar_values_config_file = pd.concat(similar_values, ignore_index=True)

        similar_values_train.apply(
            lambda x: merge_similar_columns(x["fuzzy_match"], x["col_name"], data),
            axis=1,
        )

        del match2

        columns_to_remove = list(set(similar_values_train.fuzzy_match.values.tolist()))
        data.drop(columns_to_remove, axis=1, inplace=True)

        del result

        bar(0.6)

        bar.text("Removing URL Parameters and parsing data types")
        # remove non http headers
        data = data.iloc[:, 6:]
        list_of_dtypes = create_categories_list(data)
        data = data.astype(list_of_dtypes)

        bar(0.7)

        bar.text("Find columns with na_ratio of 1")

        summary_table = create_summary_table(data.iloc[:, :-2])
        remove_headers_with_one_na_ratio = summary_table[
            summary_table["na_ratio"] == 1
        ].header_name.values.tolist()
        remove_headers_with_one_value = summary_table[
            (summary_table["unique_values"] <= 1) & (summary_table["na_ratio"] != 1)
        ].header_name.values.tolist()

        bar(0.8)

        bar.text("Drop columns")

        data.drop(remove_headers_with_one_na_ratio, axis=1, inplace=True)
        data.drop(remove_headers_with_one_value, axis=1, inplace=True)
        train_columns_2 = data.columns.values.tolist()

        del summary_table

        bar(0.85)

        bar.text("Combine columns")
        summary_table2 = create_summary_table_2(data)

        concise_information_wrapper(data, summary_table2)

        bar(0.95)

        bar.text("Removing remaining combined columns")
        only_non_tracker_col, only_tracker_col = find_cols_to_combine(summary_table2)

        data.drop(only_non_tracker_col, axis=1, inplace=True)
        data.drop(only_tracker_col, axis=1, inplace=True)
        reordered_cols = label_as_last_column(data)

        data["tracker"] = data["tracker"].astype("Int32")
        data["httpMessageId"] = data["httpMessageId"].astype("Int32")
        bar(0.99)

        if config_path:
            config = {
                "train_columns": train_columns,
                "empty_columns": empty_columns,
                "similar_values": similar_values_config_file.to_dict("records"),
                "columns_to_remove": columns_to_remove,
                "train_columns_2": train_columns_2,
                "remove_headers_with_one_na_ratio": remove_headers_with_one_na_ratio,
                "remove_headers_with_one_value": remove_headers_with_one_value,
                "other_summary_table": summary_table2.to_dict("records"),
                "only_non_tracker_col": only_non_tracker_col,
                "only_tracker_col": only_tracker_col,
                "reordered_cols": reordered_cols,
            }

            # Save the configuration dictionary as a JSON file
            with open(config_path, "w") as outfile:
                json.dump(config, outfile)

        bar.text("Write data to parquet.gzip")
        print(f"{file_path}_processed_{httpMessage}.parquet.gzip")
        data.to_parquet(
            f"{file_path}_processed_{httpMessage}.parquet.gzip",
            compression="gzip",
        )

        bar(1)


def preprocessing_test_data(file_path, config_path, other_test_data: bool = False):
    with alive_bar(100, force_tty=True, manual=True, title="Data Processing") as bar:

        bar.text("Read-in data")
        httpMessage = "response" if "response" in config_path else "request"
        data = pd.read_parquet(
            f"{file_path}_{httpMessage}.parquet.gzip",
            engine="pyarrow",
            dtype_backend="pyarrow",
        )

        with open(config_path, "r") as infile:
            config = json.load(infile)
        bar(0.05)

        bar.text("Remove empty columns")
        empty_columns = config["empty_columns"]
        if other_test_data:
            columns = config["empty_columns"] + data.columns.values.tolist()
            empty_columns = [x for x in columns if columns.count(x) > 1]
        data.drop(empty_columns, axis=1, inplace=True)

        if other_test_data:
            test_columns = data.columns.values.tolist()
            train_columns = config["train_columns"]
            cols_not_in_train = list(set(test_columns).difference(train_columns))
            data.drop(cols_not_in_train, axis=1, inplace=True)

        data.reset_index(drop=True, inplace=True)
        bar(0.15)

        bar.text("Merge fuzzy matches")
        similar_values_test = pd.DataFrame.from_dict(config["similar_values"])

        if other_test_data:
            other_columns = data.columns.values.tolist()
            values = list(
                set(
                    similar_values_test.col_name.values.tolist()
                    + similar_values_test.fuzzy_match.values.tolist()
                )
            )
            cols_not_in_other = list(set(values).difference(other_columns))
            similar_values_test = similar_values_test[
                ~similar_values_test.isin(cols_not_in_other)
            ]
            similar_values_test = similar_values_test[
                ~similar_values_test.isin(cols_not_in_other)
            ]
            similar_values_test.dropna(axis=0, inplace=True)

        bar(0.2)

        similar_values_test.apply(
            lambda x: merge_similar_columns(x["fuzzy_match"], x["col_name"], data),
            axis=1,
        )
        columns_to_remove = list(
            set(
                pd.DataFrame.from_dict(
                    config["similar_values"]
                ).fuzzy_match.values.tolist()
            )
        )
        if other_test_data:
            columns_to_remove = list(
                set(similar_values_test.fuzzy_match.values.tolist())
            )
        data.drop(columns_to_remove, axis=1, inplace=True)
        bar(0.3)

        bar.text("Removing URL Parameters and parsing data types")
        data = data.iloc[:, 6:]
        list_of_dtypes_test = create_categories_list(data)
        empty_columns = data.columns[data.dtypes == "null[pyarrow]"].tolist()
        if len(empty_columns) > 0:
            dtypes_of_empty_columns = dict.fromkeys(empty_columns, "object")
            data = data.astype(dtypes_of_empty_columns)
        bar(0.4)
        data = data.astype(list_of_dtypes_test)
        bar(0.6)

        bar.text("Drop columns with na_ratio of 1")
        if other_test_data:
            other_columns = data.columns.values.tolist()
            col_intersection = list(
                set(config["remove_headers_with_one_na_ratio"]).intersection(
                    other_columns
                )
            )
            data.drop(col_intersection, axis=1, inplace=True)

            other_columns = data.columns.values.tolist()
            col_intersection = list(
                set(config["remove_headers_with_one_value"]).intersection(other_columns)
            )
            data.drop(col_intersection, axis=1, inplace=True)

            test_columns = data.columns.values.tolist()
            train_columns = config["train_columns_2"]
            cols_not_in_train = list(set(test_columns).difference(train_columns))
            data.drop(cols_not_in_train, axis=1, inplace=True)
        else:
            data.drop(config["remove_headers_with_one_na_ratio"], axis=1, inplace=True)
            data.drop(config["remove_headers_with_one_value"], axis=1, inplace=True)

        bar(0.7)

        bar.text("Combine columns")
        if other_test_data:
            other_summary_table = pd.DataFrame.from_dict(config["other_summary_table"])[
                ~pd.DataFrame.from_dict(config["other_summary_table"]).header_name.isin(
                    list(
                        set(config["train_columns_2"]).difference(
                            data.columns.values.tolist()
                        )
                    )
                )
            ]
            concise_information_wrapper(data, other_summary_table)
        else:
            concise_information_wrapper(
                data, pd.DataFrame.from_dict(config["other_summary_table"])
            )

        bar(0.8)

        bar.text("Removing remaining combined columns")
        if other_test_data:
            other_columns = data.columns.values.tolist()
            cols_not_in_chrome = list(
                set(other_columns).difference(
                    config["train_columns_2"]
                    + ["comb_col_non_tracker", "comb_col_tracker"]
                )
            )
            data.drop(cols_not_in_chrome, axis=1, inplace=True)
        else:
            data.drop(config["only_non_tracker_col"], axis=1, inplace=True)
            data.drop(config["only_tracker_col"], axis=1, inplace=True)

        data["tracker"] = data["tracker"].astype("Int32")
        data["httpMessageId"] = data["httpMessageId"].astype("Int32")

        # Identify columns in data that are not in reordered_cols
        columns_to_remove = set(data.columns) - set(config["reordered_cols"])

        # Remove these columns from data
        data.drop(columns=columns_to_remove, inplace=True)

        if len(data.columns) != len(config["reordered_cols"]):
            missing_cols = list(
                set(config["reordered_cols"]).difference(data.columns.tolist())
            )
            data = data.reindex(columns=data.columns.tolist() + missing_cols)
        data = data[config["reordered_cols"]]
        bar(0.9)

        bar.text("Write data to parquet.gzip")
        data.to_parquet(
            f"{file_path}_processed_{httpMessage}.parquet.gzip",
            compression="gzip",
        )
        bar(1)
