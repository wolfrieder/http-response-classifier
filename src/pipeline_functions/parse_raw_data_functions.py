import os.path
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import copy
import math


def process_label_rows(
    data: List[List[Dict[str, Union[bool, str, List]]]]
) -> np.ndarray:
    """
    Process a list of label row data and return a NumPy array.

    Args:
        data (List[List[Dict[str, Union[bool, str, List]]]]): A list containing
        label row data.

    Returns:
        np.ndarray: A NumPy array with processed label data.
    """
    columns = [r["blocklist"].lower() for row in data for r in row]
    unique_columns = list(set(columns))

    row_data = np.zeros((len(data), len(unique_columns)), dtype=bool)

    col_to_idx = {col: idx for idx, col in enumerate(unique_columns)}

    for i, row in enumerate(data):
        for r in row:
            col = r["blocklist"].lower()
            row_data[i, col_to_idx[col]] = r["isLabeled"]

    return row_data


def process_url_rows(row: List[Dict]) -> np.ndarray:
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
    np.ndarray
        A NumPy array representation of the input JSON data.
    """
    return pd.DataFrame.from_records(row).to_numpy()


def prepare_initial_dataset(
    file_name: str, target_file: str, target_data_dir, compression_alg
) -> pd.DataFrame:
    """
    Prepare the initial dataset by reading a JSON file, filtering, and resetting
    the index.

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


def create_target_column(data: np.ndarray) -> np.ndarray:
    """
    Create a target column named "tracker" in the input numpy array by merging
    two existing columns, "easylist" and "easyprivacy". The new "tracker" column
    will contain a 1 where either "easylist" or "easyprivacy" is equal to 1, and
    0 otherwise.
    The original "easylist" and "easyprivacy" columns should be removed before
    using this function.

    Parameters
    ----------
    data : np.ndarray
        The input 2D numpy array containing the boolean values representing the
        presence of "easylist" and "easyprivacy" labels.

    Returns
    -------
    np.ndarray
        A 1D numpy array with the new "tracker" values, where each value is 1 if
        either "easylist" or "easyprivacy" was equal to 1 in the corresponding
        row of the input array, and 0 otherwise.
    """
    merged = np.any(data, axis=1).astype(int)
    return merged


def remove_value_at_index(
    arr: np.ndarray, header_name: str, column_names: List[str]
) -> np.ndarray:
    """
    Remove the value at the specified index from each element in a 2D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        A 2D numpy array where each element represents a feature vector.
    header_name : str
        The header name of the column to be removed.
    column_names : List[str]
        A list of header column names.

    Returns
    -------
    np.ndarray
        A 2D numpy array with the specified value removed from each element.
    """
    index = column_names.index(header_name)
    result_array = np.hstack((arr[:, :index], arr[:, index + 1 :]))
    return result_array


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
    path = f"../../../data/{target_data_dir}/{target_file_name}/{name}.json.{compression_alg}"
    print(f"\nDEBUG: File exists? {os.path.isfile(path)}\n")

    return pd.read_json(path, orient="records", compression="gzip")


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


def concatenate_dicts(
    data: List[Dict[str, str]], column_names: List[str]
) -> np.ndarray:
    """
    Concatenate a list of dictionaries containing header data into a single NumPy array.

    Parameters
    ----------
    data : List[Dict[str, str]]
        A list of dictionaries, where each dictionary contains header keys and values.
    column_names : List[str]
        A list of header column names.

    Returns
    -------
    np.ndarray
        A NumPy array containing the concatenated header data.
    """
    data_array = np.empty((len(data), len(column_names)), dtype=object)
    col_to_idx = {col: idx for idx, col in enumerate(column_names)}

    for i, d in enumerate(data):
        for col, value in d.items():
            data_array[i, col_to_idx[col]] = value

    data_array[data_array is None] = np.nan
    return data_array


def process_header_rows(row: list) -> dict:
    """
    Process a list of header rows and return a dictionary with lowercase keys
    and values.

    Parameters
    ----------
    row : list
        A list of header rows, where each row is a list containing a key and a
        value as strings.

    Returns
    -------
    dict
        A dictionary containing the header keys and values, where keys and
        values are lowercase.
    """
    http_message = np.array(row)
    header_keys = np.char.lower(http_message[:, 0])
    header_values = np.char.lower(http_message[:, 1])
    parsed_row_as_dict = dict(zip(header_keys, header_values))
    return parsed_row_as_dict


def concat_arrays(
    header_array: np.ndarray,
    url_array: np.ndarray,
    label_array: np.ndarray,
) -> np.ndarray:
    """
    Concatenate header_array, url_array, and label_array column-wise.

    Parameters
    ----------
    header_array : np.ndarray
        2D NumPy array containing header information.
    url_array : np.ndarray
        2D NumPy array containing URL information.
    label_array : np.ndarray
        1D NumPy array containing label information.

    Returns
    -------
    np.ndarray
        A 2D NumPy array that is the concatenation of the input arrays.
    """
    # print(f"header_array shape: {header_array.shape}")
    # print(f"url_array shape: {url_array.shape}")
    # print(f"label_array shape: {label_array.shape}")

    final_array = np.hstack((url_array, header_array))
    final_array = np.column_stack((final_array, label_array))

    return final_array


def rename_duplicates(original_arr: List[str], duplicates_arr: List[str]) -> List[str]:
    """
    Rename the duplicated elements in the original array of strings based on the
    given duplicates array. The duplicated strings in the original array will be
    renamed by appending '_dlc' to their original names.

    Parameters
    ----------
    original_arr : List[str]
        The original array of strings containing the elements to be checked for
        duplicates and potentially renamed.
    duplicates_arr : List[str]
        The array of strings containing the duplicated elements that should be
        renamed in the original array.

    Returns
    -------
    List[str]
        The updated array of strings with the duplicated elements renamed.

    Examples
    --------
    >>> original_arr = ['a', 'b', 'c', 'a', 'b']
    >>> duplicates_arr = ['a', 'b']
    >>> rename_duplicates(original_arr, duplicates_arr)
    ['a', 'b', 'c', 'a_dlc', 'b_dlc']
    """
    if not duplicates_arr:
        return original_arr

    renamed_arr = copy.deepcopy(original_arr)
    duplicates_set = copy.deepcopy(duplicates_arr)

    for i, header_name in enumerate(renamed_arr):
        if header_name in duplicates_set:
            print(
                f"Duplicate header name found at index {i} and renamed to "
                f"{header_name}_dlc"
            )
            renamed_arr[i] = f"{header_name}_dlc"
            duplicates_set.remove(header_name)

    return renamed_arr


def find_duplicates(strings: List[str]) -> List[str]:
    """
    Find duplicate strings in a list and return a list of the duplicated strings.

    Parameters
    ----------
    strings : List[str]
        A list of strings to find duplicates in.

    Returns
    -------
    List[str]
        A list containing the duplicated strings found in the input list.

    Examples
    --------
    >>> strings = ["apple", "orange", "banana", "apple", "orange"]
    >>> find_duplicates(strings)
    ['apple', 'orange']
    """
    duplicates = []
    unique_strings = set()

    for string in strings:
        if string in unique_strings:
            duplicates.append(string)
        else:
            unique_strings.add(string)

    return duplicates


def rename_duplicate_keys(
    data: List[Dict[str, str]], key_mapping: Dict[str, str]
) -> List[Dict[str, str]]:
    """
    Rename the specified keys in a list of dictionaries.

    Parameters
    ----------
    data : List[Dict[str, str]]
        A list of dictionaries containing the keys to be renamed.
    key_mapping : Dict[str, str]
        A dictionary mapping the original keys to their new names.

    Returns
    -------
    List[Dict[str, str]]
        A new list of dictionaries with the specified keys renamed according to the key_mapping.

    Examples
    --------
    >>> data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    >>> key_mapping = {"a": "c"}
    >>> rename_duplicate_keys(data, key_mapping)
    [{"c": 1, "b": 2}, {"c": 3, "b": 4}]
    """
    return [{key_mapping.get(k, k): v for k, v in d.items()} for d in data]


def create_key_mapping(duplicate_keys: List[str]) -> Dict[str, str]:
    """
    Create a key mapping dictionary for renaming duplicate keys.

    Parameters
    ----------
    duplicate_keys : List[str]
        A list of duplicate keys to be renamed.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping the original keys to their new names, which include '_dlc' suffix.

    Examples
    --------
    >>> duplicate_keys = ["a", "b"]
    >>> create_key_mapping(duplicate_keys)
    {"a": "a_dlc", "b": "b_dlc"}
    """
    return {key: f"{key}_dlc" for key in duplicate_keys}


def parse_chunks(
        header_array: np.ndarray,
        url_array: np.ndarray,
        label_array: np.ndarray,
        chunk_size: int
) -> List[np.ndarray]:
    """
    Split and concatenate arrays into chunks of the specified size.

    The function divides the input arrays into chunks of the specified size,
    concatenates the corresponding parts of each array, and returns a list of
    the resulting chunks.

    Parameters
    ----------
    header_array : np.ndarray
        2D NumPy array containing header information.
    url_array : np.ndarray
        2D NumPy array containing URL information.
    label_array : np.ndarray
        1D NumPy array containing label information.
    chunk_size : int
        The size of each chunk to be parsed.

    Returns
    -------
    List[np.ndarray]
        A list of concatenated 2D NumPy arrays, one for each chunk.
    """
    num_rows = header_array.shape[0]
    num_chunks = math.ceil(num_rows / chunk_size)
    result_chunks = []

    print(f"num_rows: {num_rows}, num_chunks: {num_chunks}")

    for i in range(num_chunks):
        # print(f'iteration: {i}')
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_rows)

        # print(f"start_idx: {start_idx}, end_idx: {end_idx}")

        part_header = header_array[start_idx:end_idx, :]
        part_url = url_array[start_idx:end_idx, :]
        part_label = label_array[start_idx:end_idx]

        combined = concat_arrays(part_header, part_url, part_label)
        result_chunks.append(combined.T)

    return result_chunks
