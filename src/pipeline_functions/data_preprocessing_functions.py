import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
import numpy as np
from rapidfuzz import process
from collections import Counter


def test_new_categories_update(
    element: str, dataset: pd.DataFrame
) -> Optional[Dict[str, str]]:
    """
    Test if the given column of the dataset can be converted to the Int64 data
    type. If so, return a dictionary with the column name as the key and 'Int64'
    as the value. Otherwise, return None.

    Parameters
    ----------
    element : str
        The column name to be tested.
    dataset : pd.DataFrame
        The DataFrame containing the column.

    Returns
    -------
    Optional[Dict[str, str]]
        A dictionary with the column name and 'Int64' if the column can be
        converted to the Int64 data type, None otherwise.
    """
    try:
        categories = dataset[element].astype("category").cat.categories.values.tolist()
    except ValueError:
        return None

    try:
        np.array(categories, dtype="int64")
        return {element: "Int64"}
    except (ValueError, OverflowError):
        return None


def create_categories_list(dataset: pd.DataFrame) -> Dict[str, Union[str, None]]:
    """
    Create a dictionary of column names and their corresponding data types for
    the given dataset. The data types are determined based on the column values.
    If a column can be converted to Int64, its data type is set to 'Int64',
    otherwise it is set to 'category'.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame for which to create the dictionary of column names
        and data types.

    Returns
    -------
    Dict[str, Union[str, None]]
        A dictionary of column names and their corresponding data types.
    """
    dtype_list = {i: "category" for i in dataset.columns.values[:-1]}
    current_columns = dataset.columns.values[:-1].tolist()
    int64_columns = [
        test_new_categories_update(element, dataset) for element in current_columns
    ]

    int64_columns = list(filter(lambda x: type(x) is dict, int64_columns))
    int64_columns = {k: v for d in int64_columns for k, v in d.items()}

    dtype_list.update(int64_columns)
    return dtype_list


def new_fuzzy_string_matching_for_column(
    col_name: str, col_values: List[str]
) -> pd.DataFrame:
    """
    Find fuzzy matches for a given column name with a list of column values.

    Parameters
    ----------
    col_name : str
        The column name for which fuzzy matches should be found.
    col_values : List[str]
        A list of column values to compare with the given column name.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the fuzzy matches, their respective scores,
        and the original column name.
    """
    fuzzy_result = pd.DataFrame(
        process.extract(
            col_name, col_values, processor=None, score_cutoff=80, limit=100
        ),
        columns=["fuzzy_match", "w_ratio", "index"],
    )
    fuzzy_result["col_name"] = col_name
    return fuzzy_result


def find_cols_with_similar_values(
    fuzzy_match: str, column: str, dataset: pd.DataFrame
) -> Optional[Tuple[str, str]]:
    """
    Compare two columns and return their names if more than 50% of their values
    are similar.

    Parameters
    ----------
    fuzzy_match : str
        The name of the first column.
    column : str
        The name of the second column.
    dataset : pd.DataFrame
        The DataFrame that contains the columns to compare.

    Returns
    -------
    Optional[Tuple[str, str]]
        A tuple containing the column names if more than 50% of their values are
        similar, otherwise None.
    """
    value_fuzzy = set(dataset[fuzzy_match].dropna())
    value_column = set(dataset[column].dropna())

    common_values = len(value_fuzzy.intersection(value_column))

    if common_values / len(value_fuzzy) > 0.5:
        return fuzzy_match, column
    else:
        return None


def select_similar_columns(
    fuzzy_match: str, column: str, match_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Select a row from the `match_df` DataFrame based on the provided column
    names and remove it from the DataFrame.

    Parameters
    ----------
    fuzzy_match : str
        The name of the first column.
    column : str
        The name of the second column.
    match_df: pd.DataFrame
        A dataframe containing the fuzzy matches and their w_ratio similarity.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the selected row.
    """
    row = match_df.loc[
        (match_df["fuzzy_match"] == fuzzy_match) & (match_df["col_name"] == column)
    ]
    match_df.drop(row.index[0], inplace=True)
    return row


def merge_similar_columns(fuzzy_match: str, col_name: str, df: pd.DataFrame) -> None:
    """
    Merge the values of two columns in the given DataFrame by replacing null
    values in the second column with the corresponding values from the first
    column.

    Parameters
    ----------
    fuzzy_match : str
        The name of the first column.
    col_name : str
        The name of the second column.
    df : pd.DataFrame
        The DataFrame to process.

    Returns
    -------
    None
    """
    boolean_mask = df[fuzzy_match].notnull()
    new_values = df.loc[boolean_mask, fuzzy_match].to_numpy()
    indices_fuzzy_matches = boolean_mask[boolean_mask].index.tolist()

    current_values = df[col_name].to_numpy()
    np.put(current_values, indices_fuzzy_matches, new_values)
    df[col_name] = current_values


def create_summary_table(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table containing the number of unique values and the NA
    ratio for each column in a DataFrame.

    Parameters
    ----------
    dataset : pd.DataFrame
        The DataFrame for which to compute the summary table.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the number of unique values and the NA ratio for
        each column in the input DataFrame.
    """
    table_result = dataset.apply(
        lambda x: pd.Series(
            {
                "header_name": x.name,
                "unique_values": x.nunique(dropna=True),
                "na_ratio": round(x.isna().mean(), 3),
            }
        )
    ).T

    table_result["unique_values"] = table_result["unique_values"].astype("Int32")
    table_result["na_ratio"] = table_result["na_ratio"].astype("float32")
    table_result.reset_index(drop=True, inplace=True)

    return table_result


def count_trackers_and_non_trackers(
    column: pd.Series, tracker: pd.Series
) -> List[Union[str, int]]:
    """
    Count the number of trackers and non-trackers in a given column of a
    DataFrame.

    Parameters
    ----------
    column : pd.Series
        The column to count trackers and non-trackers.
    tracker : pd.Series
        The 'tracker' column from the DataFrame.

    Returns
    -------
    List[Union[str, int]]
        A list containing the column name, the number of trackers, and the
        number of non-trackers.
    """
    column_name = column.name
    notnull_mask = column.notnull()
    tracker_ratio = tracker[notnull_mask].value_counts()
    try:
        trackers = tracker_ratio[1]
    except KeyError:
        trackers = 0
    try:
        non_trackers = tracker_ratio[0]
    except KeyError:
        non_trackers = 0
    return [column_name, trackers, non_trackers]


def create_summary_table_2(dataset: pd.DataFrame) -> pd.DataFrame:
    number_of_elements_reduced = np.array(
        [
            count_trackers_and_non_trackers(dataset[column], dataset["tracker"])
            for column in dataset.iloc[:, 4:-1].columns
        ]
    )
    result_table = pd.DataFrame(
        number_of_elements_reduced, columns=["header_name", "trackers", "non_trackers"]
    )
    result_table["trackers"] = result_table["trackers"].astype("Int32")
    result_table["non_trackers"] = result_table["non_trackers"].astype("float32")
    result_table["ratio"] = (
        result_table["trackers"] / result_table["non_trackers"]
    ) * 100
    result_table["ratio2"] = (
        result_table["non_trackers"] / result_table["trackers"]
    ) * 100
    return result_table


def update_combined_columns(
    dataset: pd.DataFrame, col_list: List[str], classification: int, column_name: str
) -> None:
    """
    Update the combined columns in the dataset based on given column list and
    classification.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to update.
    col_list : List[str]
        The list of columns to process.
    classification : int
        The classification value (0 or 1) to filter rows in the dataset.
    column_name : str
        The name of the column to update in the dataset.
    """
    indices = [
        dataset[
            (dataset[col].notnull()) & (dataset["tracker"] == classification)
        ].index.tolist()
        for col in col_list
    ]
    indices_concat = list(np.concatenate(indices).flat)
    count_indices = dict(Counter(indices_concat))

    for key, value in count_indices.items():
        dataset.at[key, column_name] = value


def find_cols_to_combine(
    information_table: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Find columns to combine based on the given information table.

    Parameters
    ----------
    information_table : pd.DataFrame
        A summary table with column information.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists of column names: one for non-trackers and
        one for trackers.
    """
    only_non_trackers = information_table[
        information_table["ratio"] <= 10
    ].header_name.values.tolist()
    only_trackers = information_table[
        information_table["ratio2"] <= 10
    ].header_name.values.tolist()
    return only_non_trackers, only_trackers


def concise_information_wrapper(dataset: pd.DataFrame, table: pd.DataFrame) -> None:
    """
    Process dataset with concise information and update the dataset with
    combined columns.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to process and update.
    table : pd.DataFrame
        A summary table with column information.
    """

    only_non_tracker_cols, only_tracker_cols = find_cols_to_combine(table)

    dataset["comb_col_non_tracker"] = 0
    dataset["comb_col_tracker"] = 0

    update_combined_columns(dataset, only_tracker_cols, 1, "comb_col_tracker")
    update_combined_columns(dataset, only_non_tracker_cols, 0, "comb_col_non_tracker")


def label_as_last_column(dataset: pd.DataFrame) -> List[str]:
    """
    Rearrange the columns of a DataFrame to place the "tracker" column at the end.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame containing the "tracker" column.

    Returns
    -------
    List[str]
        A list of column names with the "tracker" column moved to the end.
    """
    temp_cols = dataset.columns.tolist()
    index_col = dataset.columns.get_loc("tracker")
    new_col_order = (
        temp_cols[0:index_col]
        + temp_cols[index_col + 1 :]
        + temp_cols[index_col : index_col + 1]
    )
    return new_col_order
