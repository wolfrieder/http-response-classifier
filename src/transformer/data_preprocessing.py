import sys

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


if __name__ == "__main__":
    # ray.init()
    browser = sys.argv[1]
    directory = sys.argv[2]
    dir_path = f"{browser}/{directory}"

    dataset = pd.read_parquet(
        f"data/interim/{dir_path}/{sys.argv[3]}.parquet.gzip",
    )
    # dataset = overwrite_dtypes_of_dataset(dataset, dataset.columns.values)
    # dataset.to_parquet(
    #     f"data/interim/{dir_path}/{sys.argv[3]}_processed.parquet.gzip", compression="gzip"
    # )
