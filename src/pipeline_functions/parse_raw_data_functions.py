from typing import List

import numpy as np
import pandas as pd
import ray
import sys
import os.path

# from . import file_operations
# from src.pipeline_functions.file_operations import read_json_file, \
#     write_to_parquet_file
# sys.path.append('../../../src/pipeline_functions/file_operations.py')


@ray.remote
def process_header_rows(row: list) -> pd.DataFrame:
    """
    Process a list of header rows and combine them into a single DataFrame.

    This function takes a list of header rows as input, concatenates them into
    a single DataFrame, converts column names and values to lowercase, removes
    duplicate columns, and returns the resulting DataFrame.

    Parameters
    ----------
    row : list
        A list of header rows, where each row is a list of key-value pairs.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the processed header rows with lowercase
        column names and values, and duplicate columns removed.

    Examples
    --------
    >>> header_rows = [[
    ...     ['content-type', 'application/javascript'],
    ...     ['date', 'Thu, 11 Aug 2022 21:14:37 GMT']
    ... ], [
    ...     ['content-type', 'text/html'],
    ...     ['date', 'Fri, 12 Aug 2022 22:15:40 GMT']
    ... ]]
    >>> result = ray.get(process_header_rows.remote(header_rows))
    >>> print(result)
          content-type                         date
    0  application/javascript  thu, 11 aug 2022 21:14:37 gmt
    1            text/html  fri, 12 aug 2022 22:15:40 gmt
    """
    df_row = pd.concat([pd.DataFrame(element) for element in row], axis=1)
    columns = np.vectorize(str.lower)(df_row.iloc[0].values)
    values = np.vectorize(str.lower)(df_row.iloc[1].values)
    result = pd.DataFrame(values.reshape(1, -1), columns=columns)
    return result.loc[:, ~result.columns.duplicated()]


@ray.remote
def process_label_rows(row: list) -> pd.DataFrame:
    """
    Process a list of label rows and combine them into a single DataFrame.

    This function takes a list of label rows as input, each as a dictionary
    with 'isLabeled' and 'blocklist' keys, converts the 'blocklist' values
    to lowercase, and returns the resulting DataFrame.

    Parameters
    ----------
    row : list
        A list of label rows, where each row is a dictionary with
        'isLabeled' and 'blocklist' keys.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the processed label rows with lowercase
        column names.

    Examples
    --------
    >>> label_rows = [
    ...     {'isLabeled': False, 'blocklist': 'EasyList'},
    ...     {'isLabeled': False, 'blocklist': 'EasyPrivacy'}
    ... ]
    >>> result = ray.get(process_label_rows.remote(label_rows))
    >>> print(result)
      easylist  easyprivacy
    0    False        False
    """
    columns = np.vectorize(str.lower)([r["blocklist"] for r in row])
    row_data = [r["isLabeled"] for r in row]
    return pd.DataFrame(np.array(row_data).reshape(1, -1), columns=columns)


@ray.remote
def process_url_rows(row: dict) -> pd.DataFrame:
    """
        Normalize semi-structured JSON data into a flat table.

        This function takes a dictionary containing nested JSON data and returns
        a flattened Pandas DataFrame.

        Parameters
        ----------
        row : dict
            A dictionary containing nested JSON data.

        Returns
        -------
        pd.DataFrame
            A flattened DataFrame representation of the input JSON data.

        Examples
        --------
        >>> data = {
        ...     'statusCode': 200,
        ...     'fromCache': False,
        ...     'responseHeaders': {'Pragma': 'no-cache', 'Content-Type': 'application/json'}
        ... }
        >>> result = ray.get(process_url_rows.remote(data))
        >>> print(result)
          statusCode  fromCache responseHeaders.Pragma responseHeaders.Content-Type
    0         200      False               no-cache             application/json
    """
    return pd.json_normalize(row)


@ray.remote
def concat_splits(row: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate a list of DataFrames along rows and reset index.

    This function takes a list of DataFrames as input, concatenates them along
    rows (axis=0), and resets the index.

    Parameters
    ----------
    row : List[pd.DataFrame]
        A list of Pandas DataFrames to be concatenated.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame with index reset.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    >>> result = ray.get(concat_splits.remote([df1, df2]))
    >>> print(result)
       A  B
    0  1  3
    1  2  4
    2  5  7
    3  6  8
    """
    return pd.concat(row, ignore_index=True)


# https://datagy.io/python-split-list-into-chunks/
def parse_response_headers(
    chunklist: List[pd.DataFrame], n_chunks: int
) -> pd.DataFrame:
    """
    Process a list of DataFrames by concatenating them in chunks.

    This function takes a list of DataFrames as input, divides the list into
    chunks of size `n_chunks`, and concatenates the DataFrames within each chunk.
    The resulting DataFrames are then concatenated along rows (axis=0) and the
    index is reset.

    Parameters
    ----------
    chunklist : List[pd.DataFrame]
        A list of Pandas DataFrames to be processed.
    n_chunks : int
        The size of the chunks to divide the input list of DataFrames.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame with index reset.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    >>> df3 = pd.DataFrame({'A': [9, 10], 'B': [11, 12]})
    >>> df4 = pd.DataFrame({'A': [13, 14], 'B': [15, 16]})
    >>> result = parse_response_headers([df1, df2, df3, df4], n_chunks=2)
    >>> print(result)
        A   B
    0   1   3
    1   2   4
    2   5   7
    3   6   8
    4   9  11
    5  10  12
    6  13  15
    7  14  16
    """
    chunked_list = [
        chunklist[i : i + n_chunks] for i in range(0, len(chunklist), n_chunks)
    ]
    chunked_list2 = ray.get([concat_splits.remote(row) for row in chunked_list])
    return pd.concat(chunked_list2, ignore_index=True)


def prepare_initial_dataset(
    file_name: str, target_file: str, target_data_dir, compression_alg
) -> pd.DataFrame:
    """
    Prepare the initial dataset by reading a JSON file, filtering, and resetting the index.

    This function reads a JSON file from the specified target file location, drops
    any rows with missing values, and filters rows with non-empty 'responseHeaders'.
    The index of the resulting DataFrame is reset before returning.

    Parameters
    ----------
    file_name : str
        The name of the JSON file to read.
    target_file : str
        The target file location containing the JSON file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the prepared initial dataset.

    Examples
    --------
    >>> file_name = 'raw_data.json'
    >>> target_file = 'path/to/input/file'
    >>> initial_dataset = prepare_initial_dataset(file_name, target_file)
    """
    data = (
        read_json_file(file_name, target_file, target_data_dir, compression_alg)
        .dropna()
        .reset_index(drop=True)
    )
    return data.loc[data["responseHeaders"].map(len) != 0].reset_index(drop=True)


def create_target_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a target column named "tracker" in the input DataFrame by merging
    two existing columns, "easylist" and "easyprivacy". The new "tracker" column
    will contain a 1 where either "easylist" or "easyprivacy" is equal to 1, and
    0 otherwise.
    The original "easylist" and "easyprivacy" columns will be dropped.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the "easylist" and "easyprivacy" columns.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with the new "tracker" column and the original
        "easylist" and "easyprivacy" columns removed.

    Examples
    --------
    >>> df = pd.DataFrame({"easylist": [1, 0, 1, 0], "easyprivacy": [0, 1, 1, 0]})
    >>> create_target_column(df)
       tracker
    0       1
    1       1
    2       1
    3       0
    """
    data["tracker"] = ((data.easylist == 1) | (data.easyprivacy == 1)).astype(np.int32)
    data.drop(["easylist", "easyprivacy"], axis=1, inplace=True)
    return data


def parse_dataset(
    origin_file_name: str,
    origin_dir_name: str,
    target_file_name: str,
    target_dir_name: str,
    n_chunks: int,
) -> None:
    """
    Parse and process a dataset by performing various tasks, such as processing
    HTTP header fields, labels, and URLs. The dataset is then combined, and the
    target column is created. Finally, the resulting DataFrame is written to a
    Parquet file.

    Parameters
    ----------
    origin_file_name : str
        The name of the original input file (without extension) containing the dataset.
    origin_dir_name : str
        The name of the directory where the original input file is located.
    target_file_name : str
        The name of the target output file (without extension) where the processed
        DataFrame will be saved.
    target_dir_name : str
        The name of the directory where the target output file will be saved.
    n_chunks : int
        The number of chunks to use for parallel processing.

    Examples
    --------
    >>> parse_dataset("input_data", "raw_data", "output_data", "interim_data", 10)
    """

    target_data_dir = "merged" if "merged" in origin_file_name else "raw"
    compression_alg = "gz" if "merged" in origin_file_name else "gzip"

    print(
        f"Prepare initial dataset: "
        f"Path: data/{target_data_dir}/{origin_dir_name}/{origin_file_name}.json.{compression_alg}, "
        f"Chunk-size: {n_chunks} ",
        f"Target Filename: {target_file_name}",
    )

    # check_if_dir_exists(target_dir_name)

    response_data = prepare_initial_dataset(
        origin_file_name, origin_dir_name, target_data_dir, compression_alg
    )

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

    print(f"Combine Results and Write to data/interim as data/{target_dir_name}")
    result = pd.concat(
        [final_response_labels, final_response_urls, final_response_headers], axis=1
    )
    result = result.loc[:, ~result.columns.duplicated()]
    result = create_target_column(result)
    write_to_parquet_file(result, target_file_name, target_dir_name)
    print("End")


def read_json_file(
    name: str, target_file_name: str, target_data_dir, compression_alg
) -> pd.DataFrame:
    """
    Read a JSON file and return a pandas DataFrame.

    Parameters
    ----------
    name: String
        Name of the file to read from.
    target_file_name: String
        Name of the file directory to read from (Path).
        Note: data/raw/ is already defined.

    Returns
    -------
    object, type of objs

    """
    return pd.read_json(
        f"../../../data/{target_data_dir}/{target_file_name}/{name}.json.{compression_alg}"
    )


def read_parquet_file(name: str, target_file_name: str) -> pd.DataFrame:
    """
    Read a parquet file and return a pandas DataFrame.

    Parameters
    ----------
    name: String
        Name of the file to read from.
    target_file_name: String
        Name of the file directory to read from (Path).
        Note: data/ is already defined.

    Returns
    -------
    object, type of objs

    """
    return pd.read_parquet(f"data/{target_file_name}/{name}.parquet.gzip")


def write_to_parquet_file(
    dataframe: pd.DataFrame, file_name: str, target_dir_name: str
) -> None:
    """
    Takes a pandas DataFrame and writes it to a parquet file.

    Parameters
    ----------
    dataframe: DataFrame object
        The pandas DataFrame to write to parquet.
    file_name: String
        The file_name of the file to write.
    target_dir_name: String
        Name of the file directory to write to (Path).

    Returns
    -------
    None

    """
    dataframe.to_parquet(
        f"{target_dir_name}/{file_name}.parquet.gzip", compression="gzip"
    )
    return print("Finished")


def check_if_dir_exists(path: str) -> None:
    """
    This function checks if the given path for a directory exists.

    Parameters
    ----------
    path: String
        The path of the directory.
    """
    if os.path.isdir(path):
        pass
    else:
        raise FileNotFoundError(f"Directory {path} does not exist.")


def combine_datasets(names: list[str], target_file: str) -> None:
    """
    Combine multiple datasets into a single dataset and save it to a target file.

    This function reads multiple datasets from parquet files, concatenates them into
    a single DataFrame, and writes the result to a new parquet file named "all_parts"
    in the specified target file location.

    Parameters
    ----------
    names : list[str]
        A list of dataset names to be combined.
    target_file : str
        The target file location to save the combined dataset.

    Returns
    -------
    None

    Examples
    --------
    >>> names = ['dataset1', 'dataset2', 'dataset3']
    >>> target_file = 'path/to/output/file'
    >>> combine_datasets(names, target_file)
    Finished
    """
    result = pd.concat(
        (read_parquet_file(i, target_file) for i in names), ignore_index=True
    )
    output_path = os.path.join(target_file, "all_parts")
    write_to_parquet_file(result, output_path, "TEST")
    return print("Finished")
