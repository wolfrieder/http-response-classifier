import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
import numpy as np
from rapidfuzz import process


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
    categories = dataset[element].astype("category").cat.categories.values.tolist()
    try:
        np.array(categories, dtype="int64")
        return {element: "Int64"}
    except (ValueError, OverflowError):
        return None


def create_categories_list(dataset: pd.DataFrame) -> Dict[str, Union[str, None]]:
    """
    Create a dictionary of column names and their corresponding data types for the
    given dataset. The data types are determined based on the column values.
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
    current_columns = dataset.columns.values[4:-1].tolist()
    int64_columns = [
        test_new_categories_update(element, dataset) for element in current_columns
    ]

    int64_columns = list(filter(lambda x: type(x) is dict, int64_columns))
    int64_columns = {k: v for d in int64_columns for k, v in d.items()}

    dtype_list.update(int64_columns)
    del dtype_list["query"]
    del dtype_list["protocol"]
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

    Returns
    -------
    Optional[Tuple[str, str]]
        A tuple containing the column names if more than 50% of their values are
        similar, otherwise None.
    """
    value_fuzzy = set(dataset[fuzzy_match].dropna().values)
    value_column = set(dataset[column].dropna().values)

    common_values = len(value_fuzzy.intersection(value_column))
    len_value_fuzzy = len(value_fuzzy)

    if common_values / len_value_fuzzy > 0.5:
        return fuzzy_match, column
    else:
        return None


def select_similar_columns(
    fuzzy_match: str, column: str, match_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Select a row from the `match2` DataFrame based on the provided column names
    and remove it from the DataFrame.

    Parameters
    ----------
    fuzzy_match : str
        The name of the first column.
    column : str
        The name of the second column.

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
