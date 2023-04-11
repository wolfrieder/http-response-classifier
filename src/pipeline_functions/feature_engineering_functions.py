from collections import Counter
from typing import Tuple, Union, List, Any

import pandas as pd


# from https://towardsdatascience.com/dealing-with-features-that-have-high-cardinality-1c9212d7ff1b
def cumulatively_categorise(
    column: pd.Series, threshold: float = 0.75, return_categories_list: bool = True
) -> Union[pd.Series, Tuple[pd.Series, List[str]]]:
    """
    Categorize the values in a pandas Series based on their frequency. The function
    groups the most commonncategories until the cumulative frequency reaches a
    specified threshold. The remaining categories are grouped into a single
    "Other" category.

    Parameters
    ----------
    column : pd.Series
        The input pandas Series to be categorized.
    threshold : float, optional, default=0.75
        The cumulative frequency threshold, between 0 and 1, at which the most
        common categories should be grouped.
    return_categories_list : bool, optional, default=True
        If True, the function returns a tuple containing the transformed Series
        and a list of unique categories
        (including "Other"). If False, only the transformed Series is returned.

    Returns
    -------
    Union[pd.Series, Tuple[pd.Series, List[str]]]
        If return_categories_list is True, returns a tuple (new_column,
        categories_list), where new_column is the
        transformed pandas Series and categories_list is a list of unique categories
        (including "Other"). Ifnreturn_categories_list is False, only the transformed
        Series (new_column) is returned.

    Examples
    --------
    >>> import pandas as pd
    >>> column = pd.Series(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'E', 'F'])
    >>> new_column, categories_list = cumulatively_categorise(column, threshold=0.75)
    >>> new_column
    0       A
    1       A
    2       B
    3       B
    4       C
    5       C
    6    Other
    7    Other
    8    Other
    dtype: object
    >>> categories_list
    ['A', 'B', 'C', 'Other']
    """
    threshold_value = int(threshold * len(column))
    categories_list = []
    s = 0
    counts = Counter(column)

    for i, j in counts.most_common():
        s += dict(counts)[i]
        categories_list.append(i)
        if s >= threshold_value:
            break

    categories_list.append("Other")
    new_column = column.apply(lambda x: x if x in categories_list else "Other")

    if return_categories_list:
        return new_column, categories_list
    else:
        return new_column


def variance_per_column(column: str, train_data: pd.DataFrame) -> List[Any]:
    """
    Calculate the number of tracker and non-tracker instances in a specified
    column of a DataFrame. The function handles missing values by filtering out
    rows with null values in the specified column.

    Parameters
    ----------
    column : str
        The name of the column in the DataFrame for which the tracker and
        non-tracker instances are to be calculated.
    train_data : pd.DataFrame
        The input DataFrame containing the data.

    Returns
    -------
    List[Any]
        A list containing the column name, number of tracker instances, and
        number of non-tracker instances.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'column1': [1, 0, 1, None, 0], 'tracker': [1, 1, 0, 0, 1]}
    >>> train_data = pd.DataFrame(data)
    >>> variance_per_column('column1', train_data)
    ['column1', 2, 1]
    """
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


def label_as_last_column(dataset: pd.DataFrame) -> List[str]:
    """
    Reorder the columns of a DataFrame, moving the "tracker" column to the end.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame with a "tracker" column.

    Returns
    -------
    List[str]
        A list of column names in the new order, with the "tracker" column last.
    """
    temp_cols = dataset.columns.tolist()
    index_col = dataset.columns.get_loc("tracker")
    new_col_order = (
        temp_cols[0:index_col]
        + temp_cols[index_col + 1:]
        + temp_cols[index_col: index_col + 1]
    )
    return new_col_order


def compute_imputed_value(
    element: str,
    classification: int,
    check: float,
    train_data: pd.DataFrame,
    list_of_categorical_cols: List[str],
) -> Union[int, str]:
    """
    Compute the imputed value for a given element and classification.

    Parameters
    ----------
    element : str
        The column name in the DataFrame.
    classification : int
        The classification value, either 0 or 1.
    check : float
        The ratio threshold for imputation.
    train_data: pd.DataFrame
        The input DataFrame containing the data.
    list_of_categorical_cols: str
        A list of column names in the DataFrame that are categorical.

    Returns
    -------
    Union[int, str]
        The computed imputed value, either an integer or a string.
    """
    if element in ["content-length", "age"]:
        value = int(
            train_data[train_data["tracker"] == classification][element].median()
        )
        if check < 0.4:
            value = -1
    elif element in list_of_categorical_cols:
        value = (
            train_data[train_data["tracker"] == classification][element].mode().iloc[0]
        )
        if check < 0.4:
            value = "Missing"
            train_data[element].cat.add_categories("Missing", inplace=True)
    else:
        value = None

    return value


def impute_value(
    element: str,
    classification: int,
    train_data: pd.DataFrame,
    summary_table: pd.DataFrame,
    imputed_values_dict: dict,
) -> None:
    """
    Impute missing values in the given element (column) based on the
    classification value.

    This function computes the imputed value depending on the column type
    (numerical or categorical) and the given classification. It then imputes the
    missing values in the specified column
    for rows that match the classification.

    Parameters
    ----------
    element : str
        The column name in the DataFrame.
    classification : int
        The classification value, either 0 or 1.
    train_data: pd.DataFrame
        The input DataFrame containing the data.
    summary_table: pd.DataFrame
        A DataFrame containing the summary statistics.
    imputed_values_dict: dict
        A dictionary containing the imputed values.
    """
    check = (
        summary_table.loc[summary_table.header_name == element, "ratio_tracker"].values[
            0
        ]
        if classification == 1
        else summary_table.loc[
            summary_table.header_name == element, "ratio_non_tracker"
        ].values[0]
    )

    imputed_value = compute_imputed_value(element, classification, check)

    if imputed_value is not None:
        imputed_values_dict[classification].append({element: imputed_value})
        train_data.loc[
            train_data["tracker"] == classification, element
        ] = train_data.loc[train_data["tracker"] == classification, element].fillna(
            imputed_value
        )
