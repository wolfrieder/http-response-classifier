import pandas as pd
import os.path


def read_json_file(name: str, target_file_name: str) -> pd.DataFrame:
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
    return pd.read_json(f"../..data/raw/{target_file_name}/{name}.json")


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
    write_to_parquet_file(result, output_path)
    return print("Finished")
