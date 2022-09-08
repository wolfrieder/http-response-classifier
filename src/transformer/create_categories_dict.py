import json

import modin.pandas as md
import pandas as pd
import ray

from pandarallel import pandarallel


def create_json_of_category_dtypes(filename: str) -> dict:
    """

    Parameters
    ----------
    filename

    Returns
    -------

    """
    dataset = md.read_parquet(
        f"../../data/" f"interim/tranco_16_05_22_10k_run_06/{filename}.parquet.gzip"
    )
    # TODO: remove test limit
    dataset = dataset.iloc[:, 8:100]
    current_memory = dataset.memory_usage(deep=True)

    category_list = ray.get(
        [
            check_col_with_category_dtype.remote(column, current_memory, dataset)
            for column in dataset
        ]
    )
    category_list_final = [i for i in category_list if i is not None]
    dict_categories_result = {i: "category" for i in category_list_final}
    return dict_categories_result


@ray.remote
def check_col_with_category_dtype(col_name: str, current_memory,
                                  data: md.DataFrame):
    """

    Parameters
    ----------
    col_name
    current_memory
    data

    Returns
    -------

    """
    index = data.columns.get_loc(col_name)
    if col_name in ["query"]:
        return None
    print(index, col_name)
    new_memory = data[col_name].astype("category").memory_usage(deep=True)
    if new_memory < current_memory[index + 1]:
        return col_name


def update_category_json(columns: pd.DataFrame) -> None:
    new_dtypes = list(filter(lambda x: type(x) is dict, columns[0].tolist()))
    new_dtypes_dict = {k: v for d in new_dtypes for k, v in d.items()}
    with open("../../data/processed/dict_categories.json", "r") as categories:
        dict_categories_old = json.loads(categories.read())

    dict_categories_old.update(new_dtypes_dict)

    with open("dict_categories.json", "w") as f:
        json.dump(dict_categories_old, f)


def find_int64_columns(element):
    if element in [
        "x-nws-log-uuid",
        "traceid",
        "x-oss-hash-crc64ecma",
        "x-neory-subid",
        "gsid",
        "cdn-request-id",
        "dd.trace_id",
        "x-amz-meta-cld-surrogate-key",
        "dd-trace-id",
        "x-fc-code-checksum",
        "tracecode",
        "x-cos-hash-crc64ecma",
    ]:
        return None
    try:
        # print(element)
        dataset[element] = dataset[element].astype("Int64")
        return {element: "Int64"}
    except ValueError:
        pass


if __name__ == "__main__":
    # ray.init()
    # dict_categories = create_json_of_category_dtypes("part_0")
    # with open("../../data/processed/dict_categories.json", "w") as f:
    #     json.dump(dict_categories, f, indent=10)
    # ray.shutdown()
    dataset = pd.read_parquet(
        "../../data/interim/tranco_16_05_22_10k_run_06/part_0.parquet.gzip",
    )

    pandarallel.initialize(progress_bar=True)
    # TODO error? but works in jupyter
    int64_dtypes = dataset.iloc[:, 8:].parallel_apply(
        lambda x: find_int64_columns(x.name)
    )
    # update_category_json(int64_dtypes)
