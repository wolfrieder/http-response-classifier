import time

import numpy as np
import pandas as pd
import ray

from src.common_functions import read_json_file, read_parquet_file, \
    write_to_parquet_file, check_if_dir_exists


def combine_datasets(names: list[str], target_file: str) -> None:
    """

    Parameters
    ----------
    names
    target_file

    Returns
    -------
    object, type of objs

    """
    result = pd.concat(
        [read_parquet_file(i, target_file) for i in names], ignore_index=True
    )
    write_to_parquet_file(result, "all_parts", target_file)
    return print("Finished")


@ray.remote
def process_header_rows(row: list) -> pd.DataFrame:
    """

    Parameters
    ----------
    row

    Returns
    -------
    object, type of objs

    """
    processed = [pd.DataFrame(element) for element in row]
    df_row = pd.concat(processed, axis=1)
    columns = np.vectorize(str.lower)(df_row.iloc[0].values)
    values = np.vectorize(str.lower)(df_row.iloc[1].values)
    result = pd.DataFrame(values.reshape(1, -1), columns=columns)
    # TODO find better solution for duplicated header fields
    return result.loc[:, ~result.columns.duplicated()]


@ray.remote
def process_label_rows(row: list) -> pd.DataFrame:
    """

    Parameters
    ----------
    row

    Returns
    -------
    object, type of objs

    """
    df_row = pd.DataFrame(row)
    columns = np.vectorize(str.lower)(df_row.blocklist.values)
    return pd.DataFrame(df_row.isLabeled.values.reshape(1, -1), columns=columns)


@ray.remote
def process_url_rows(row: dict) -> pd.DataFrame:
    """

    Parameters
    ----------
    row

    Returns
    -------
    object, type of objs

    """
    return pd.json_normalize(row)


@ray.remote
def concat_splits(row: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    row

    Returns
    -------
    object, type of objs

    """
    return pd.concat(row, ignore_index=True)


# https://datagy.io/python-split-list-into-chunks/
def parse_response_headers(chunklist: list[pd.DataFrame], n_chunks: int) \
        -> pd.DataFrame:
    """

    Parameters
    ----------
    chunklist
    n_chunks

    Returns
    -------
    object, type of objs

    """
    chunked_list = [
        chunklist[i: i + n_chunks] for i in range(0, len(chunklist), n_chunks)
    ]
    chunked_list2 = ray.get([concat_splits.remote(row) for row in chunked_list])
    return pd.concat(chunked_list2, ignore_index=True)


def prepare_initial_dataset(file_name: str, target_file: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    file_name
    target_file

    Returns
    -------
    object, type of objs

    """
    data = read_json_file(file_name, target_file).dropna().reset_index(drop=True)
    return data[data["responseHeaders"].map(len) != 0].reset_index(drop=True)


def parse_dataset(origin_file_name: str, origin_dir_name: str,
                  target_file_name: str, target_dir_name: str,
                  n_chunks: int) -> None:
    """

    Parameters
    ----------
    target_dir_name
    origin_file_name
    origin_dir_name
    target_file_name
    n_chunks
    """
    print(
        f"Prepare initial dataset: "
        f"Path: data/raw/{origin_dir_name}/{origin_file_name}.json, "
        f"Chunk-size: {n_chunks} ",
        f"Target Filename: {target_file_name}",
    )

    check_if_dir_exists(target_dir_name)

    response_data = prepare_initial_dataset(origin_file_name, origin_dir_name)

    print("Parse HTTP Header Fields")
    parsed_headers = ray.get(
        [
            process_header_rows.remote(i)
            for i in response_data[["responseHeaders"]].responseHeaders
        ]
    )
    final_response_headers = parse_response_headers(parsed_headers, n_chunks)

    print("Parse HTTP Labels")
    parsed_labels = ray.get(
        [process_label_rows.remote(row) for row in response_data[["labels"]].labels]
    )
    final_response_labels = pd.concat(parsed_labels, ignore_index=True)

    print("Parse URLs")
    parsed_urls = ray.get(
        [process_url_rows.remote(row) for row in response_data[["url"]].url]
    )
    final_response_urls = pd.concat(parsed_urls, ignore_index=True)

    print("Combine Results and Write to data/interim/test as {target_dir_name}")
    result = pd.concat(
        [final_response_labels, final_response_urls, final_response_headers],
        axis=1
    )
    result = result.loc[:, ~result.columns.duplicated()]
    write_to_parquet_file(result, target_file_name, target_dir_name)
    print("End")


if __name__ == "__main__":
    # ray.shutdown()
    ray.init()
    pd.set_option("display.max_columns", 500)

    start = time.perf_counter()
    parse_dataset('http.1', 'tranco_16_05_22_10k_run_06/http', 'part_1',
                  'data/interim/tranco_16_05_22_10k_run_06', 3000)
    # combine_datasets(['data1', 'data2'], "interim/tranco_16_05_22_10k_run_06")
    stop = time.perf_counter()
    print("end time:", stop - start)

    # TODO: check, maybe alternative solution
    # buggy = pd.DataFrame(filter(lambda x: len(x) != 0,
    # header['responseHeaders']))
