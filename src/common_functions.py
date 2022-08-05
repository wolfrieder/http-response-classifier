import pandas as pd


def read_json_file(name: str, target_file_name: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    target_file_name
    name

    Returns
    -------
    pd.DataFrame

    """
    return pd.read_json(f'data/raw/{target_file_name}/{name}.json')


def read_parquet_file(name: str, target_file_name: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    name
    target_file_name

    Returns
    -------
    pd.DataFrame

    """
    return pd.read_parquet(f'data/{target_file_name}/{name}.parquet.gzip')


def write_to_parquet_file(dataframe: pd.DataFrame, name: str, target_file_name: str) -> None:
    """

    Parameters
    ----------
    dataframe
    name
    target_file_name

    Returns
    -------
    None

    """
    dataframe.to_parquet(f'data/{target_file_name}/{name}.parquet.gzip', compression='gzip')
    return print('Finished')
