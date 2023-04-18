
from alive_progress import alive_bar
from src.pipeline_functions.data_preprocessing_functions import *


if __name__ == "__main__":
    with alive_bar(100, force_tty=True, manual=True, title="Data Processing") as bar:
        bar.text('Read-in data')
        data = pd.read_parquet(
            "../../data/processed/chrome/08_12_2022/train_set_01.parquet.gzip"
        )

        data_test = pd.read_parquet(
            "../../../data/processed/chrome/08_12_2022/test_set_01.parquet.gzip"
        )

        # data_test = pd.read_parquet('../../data/interim/firefox/08_12_2022/http.0.parquet.gzip')
        # data_test = pd.read_parquet('../../data/processed/brave/08_12_2022/test_set_0123.parquet.gzip')
        bar(0.05)

        bar.text('Remove empty columns')
        empty_columns = [col for col in data if data[col].isnull().all() == True]
        data.drop(empty_columns, axis=1, inplace=True)
        data_test.drop(empty_columns, axis=1, inplace=True)
        bar(0.15)

        # Is that even necessary? Check papers; also in context of HTTP traffic.
        # bar.text("Remove duplicated observations")
        # data = data[~data.iloc[:, 6:-1].duplicated(keep="first")].reset_index(drop=True)
        # bar(0.2)

        bar.text('Fuzzy match')
        data_column_values = data.columns.values[6:-1].tolist()
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
            find_cols_with_similar_values(col, col2, data)
            for col, col2 in zip(match2["fuzzy_match"], match2["col_name"])
        ]
        bar(0.4)

        bar.text('Reset index')
        data.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        similar_values = [
            select_similar_columns(col[0], col[1], match2) for col in result if col is not None
        ]

        similar_values_test = pd.concat(similar_values, ignore_index=True)
        similar_values = pd.concat(similar_values, ignore_index=True)
        similar_values.apply(
            lambda x: merge_similar_columns(x["fuzzy_match"], x["col_name"], data), axis=1
        )

        similar_values_test.apply(
            lambda x: merge_similar_columns(x["fuzzy_match"], x["col_name"], data_test),
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
        # remove non http headers
        data = data.iloc[:, 6:]
        data_test = data_test.iloc[:, 6:]
        list_of_dtypes = create_categories_list(data)
        list_of_dtypes_test = create_categories_list(data_test)
        data = data.astype(list_of_dtypes)
        data_test = data_test.astype(list_of_dtypes_test)

        bar(0.7)

        bar.text('Find columns with na_ratio of 1')

        summary_table = create_summary_table(data.iloc[:, :-1])
        remove_headers_with_one_na_ratio = summary_table[
            summary_table["na_ratio"] == 1
        ].header_name.values.tolist()
        remove_headers_with_one_value = summary_table[
            (summary_table["unique_values"] <= 1) & (summary_table["na_ratio"] != 1)
        ].header_name.values.tolist()

        bar(0.8)

        bar.text('Drop columns')

        data.drop(remove_headers_with_one_na_ratio, axis=1, inplace=True)
        data.drop(remove_headers_with_one_value, axis=1, inplace=True)

        data_test.drop(remove_headers_with_one_na_ratio, axis=1, inplace=True)
        data_test.drop(remove_headers_with_one_value, axis=1, inplace=True)

        del remove_headers_with_one_na_ratio
        del summary_table

        bar(0.85)

        bar.text('Combine columns')
        summary_table2 = create_summary_table_2(data)

        concise_information_wrapper(data, summary_table2)
        concise_information_wrapper(data_test, summary_table2)

        bar(0.95)

        bar.text('Remove remaining combined columns')
        only_non_tracker_col, only_tracker_col = find_cols_to_combine(summary_table2)

        data.drop(only_non_tracker_col, axis=1, inplace=True)
        data.drop(only_tracker_col, axis=1, inplace=True)

        data_test.drop(only_non_tracker_col, axis=1, inplace=True)
        data_test.drop(only_tracker_col, axis=1, inplace=True)

        bar(0.99)

        bar.text('Write data to parquet.gzip')

        data.to_parquet(
            "../../../data/processed/chrome/08_12_2022/train_set_01_processed.parquet.gzip",
            compression="gzip",
        )

        data_test.to_parquet(
            "../../../data/processed/chrome/08_12_2022/test_set_01_processed.parquet.gzip",
            compression="gzip",
        )
        bar(1)
