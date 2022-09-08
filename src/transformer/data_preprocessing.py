import ray
import modin.pandas as md
import json
from rapidfuzz import fuzz, process
import pandas as pd
import numpy as np


def fuzzy_string_matching_for_column(col_name, col_values):
    result = pd.DataFrame(
        process.extract(
            col_name, col_values, processor=None, score_cutoff=80, limit=100
        ),
        columns=["fuzzy_match", "w_ratio", "index"],
    )
    result["col_name"] = col_name
    return result


def fuzzy_string_matching_wrapper(data: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    data

    Returns
    -------
    pd.DataFrame

    """
    data_column_values = data.columns.values[6:-1].tolist()

    match = [
        fuzzy_string_matching_for_column(j, data_column_values[i + 1:])
        for i, j in enumerate(data_column_values)
        if i != len(data_column_values) - 1
    ]

    return pd.concat(match, ignore_index=True)


def overwrite_dtypes_of_dataset(
    data: pd.DataFrame, columns: np.ndarray
) -> pd.DataFrame:
    """

    Parameters
    ----------
    data
    columns

    Returns
    -------
    pd.DataFrame

    """
    with open("../../data/processed/dict_categories2.json", "r") as categories:
        dict_categories = json.loads(categories.read())
    # dict_categories = {
    #     key: value for (key, value) in dict_categories.items() if key in columns
    # }
    return data.astype(dict_categories)


def create_target_column(data: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    data

    Returns
    -------
    pd.DataFrame

    """
    data["tracker"] = np.where(
        np.logical_or(data.easylist == 1, data.easyprivacy == 1), 1, 0
    )
    data["tracker"] = data["tracker"].astype(np.int32)
    data.drop(["easylist", "easyprivacy"], axis=1, inplace=True)
    return data


if __name__ == "__main__":
    # ray.init()
    dataset = pd.read_parquet(
        "../../data/interim/tranco_16_05_22_10k_run_06/part_0.parquet.gzip",
    )
    dataset = create_target_column(dataset)
    dataset = overwrite_dtypes_of_dataset(dataset, dataset.columns.values)
    dataset.to_parquet(
        f"../../data/interim/processed_test.parquet.gzip", compression="gzip"
    )
