import pandas as pd
import numpy as np


def median_and_mode_imputation(element, classification):
    if element in list_of_integer_cols:
        data.loc[data['tracker'] == classification, element] = data.loc[data['tracker'] == classification, element]\
            .fillna(data[data['tracker']==classification][element].median())

    if element in list_of_categorical_cols:
        data.loc[data['tracker'] == classification, element] = data.loc[data['tracker'] == classification, element]\
            .fillna(data[data['tracker']==classification][element].mode().iloc[0])


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


if __name__ == '__main__':
    data = pd.read_parquet("../../data/processed/chrome/08_12_2022/train_set_01_processed.parquet.gzip")
    print("Imputation")

    number_of_elements_reduced = np.array(
        [variance_per_column_2(column) for column in data.iloc[:, 4:-4].columns]
    )
    summary_table = pd.DataFrame(
        number_of_elements_reduced, columns=["header_name", "trackers", "non_trackers"]
    )
    summary_table["trackers"] = summary_table["trackers"].astype("Int32")
    summary_table["non_trackers"] = summary_table["non_trackers"].astype("float32")

    list_of_categorical_cols = list(data.iloc[:, 4:-4].select_dtypes('category').columns.values.tolist())
    list_of_integer_cols = list(data.iloc[:, 4:-4].select_dtypes('Int64').columns.values.tolist())

    number_of_trackers = len(data[data['tracker'] == 1])
    number_of_non_trackers = len(data[data['tracker'] == 0])
    summary_table['ratio_tracker'] = summary_table['trackers'] / number_of_trackers
    summary_table['ratio_non_tracker'] = summary_table['non_trackers'] / number_of_non_trackers

    impute_col_list_t = summary_table[summary_table['ratio_tracker'] > 0.4].header_name.values.tolist()
    impute_col_list_nt = summary_table[summary_table['ratio_non_tracker'] > 0.4].header_name.values.tolist()

    for header in impute_col_list_t:
        median_and_mode_imputation(header, 0)

    for header in impute_col_list_t:
        median_and_mode_imputation(header, 1)

    for header in impute_col_list_nt:
        median_and_mode_imputation(header, 1)

    for header in impute_col_list_nt:
        median_and_mode_imputation(header, 0)