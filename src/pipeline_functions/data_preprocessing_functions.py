import pandas as pd
from typing import Dict, Optional, Union
import numpy as np


def test_new_categories_update(element: str, dataset: pd.DataFrame) -> Optional[Dict[str, str]]:
    """
        Test if the given column of the dataset can be converted to the Int64 data type. If so, return a dictionary
        with the column name as the key and 'Int64' as the value. Otherwise, return None.

        Parameters
        ----------
        element : str
            The column name to be tested.
        dataset : pd.DataFrame
            The DataFrame containing the column.

        Returns
        -------
        Optional[Dict[str, str]]
            A dictionary with the column name and 'Int64' if the column can be converted to the Int64 data type,
            None otherwise.
        """
    categories = dataset[element].astype("category").cat.categories.values.tolist()
    try:
        np.array(categories, dtype="int64")
        return {element: "Int64"}
    except (ValueError, OverflowError):
        return None


def create_categories_list(dataset: pd.DataFrame) -> Dict[str, Union[str, None]]:
    """
        Create a dictionary of column names and their corresponding data types for the given dataset. The data types
        are determined based on the column values. If a column can be converted to Int64, its data type is set to
        'Int64', otherwise it is set to 'category'.

        Parameters
        ----------
        dataset : pd.DataFrame
            The input DataFrame for which to create the dictionary of column names and data types.

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