import pandas as pd


def read_json_file(name: str, target_file: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    target_file
    name

    Returns
    -------
    pd.DataFrame

    """
    return pd.read_json(f'data/raw/{target_file}/{name}.json')


def read_parquet_file(name: str, target_file: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    name
    target_file

    Returns
    -------
    pd.DataFrame

    """
    return pd.read_parquet(f'data/{target_file}/{name}.parquet.gzip')


def write_to_parquet_file(dataframe: object, name: str, target_file: str) -> None:
    """

    Parameters
    ----------
    dataframe
    name
    target_file

    Returns
    -------
    None

    """
    dataframe.to_parquet(f'data/{target_file}/{name}.parquet.gzip', compression='gzip')
    return print('Finished')
