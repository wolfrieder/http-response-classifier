from typing import Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.model_selection import train_test_split


def split_dataset(
    splitter: Any, X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the input data into training and testing sets using a specified splitter.

    Parameters
    ----------
    X : pd.DataFrame
        The input feature data to split.

    y : pd.Series
        The target variable data to split.

    splitter : scikit-learn splitter object
        The splitter object to use for splitting the data into train and test sets.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        A tuple containing the feature data and target variable data for the
        training and testing sets, respectively.
    """
    for train, test in splitter.split(X, y):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

    return X_train, y_train, X_test, y_test


def stratified_shuffle_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified shuffle split of the input data into training and testing sets.

    Parameters
    ----------
    X : pandas.DataFrame
        The input feature data to split.

    y : pandas.Series
        The target variable data to split.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
        A tuple containing the feature data and target variable data for the
        training and testing sets, respectively.

    """
    split = StratifiedShuffleSplit(n_splits=1, random_state=10, test_size=0.2)
    return split_dataset(split, X, y)


def stratified_kfold_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified k-fold split of the input data into training and testing sets.

    Parameters
    ----------
    X : pandas.DataFrame
        The input feature data to split.

    y : pandas.Series
        The target variable data to split.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
        A tuple containing the feature data and target variable data for the
        training and testing sets, respectively.

    """
    split = StratifiedKFold(n_splits=2, random_state=10, shuffle=True)
    return split_dataset(split, X, y)


def stratified_group_kfold_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified group k-fold split of the input data into training and testing sets.

    Parameters
    ----------
    X : pandas.DataFrame
        The input feature data to split.

    y : pandas.Series
        The target variable data to split.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
        A tuple containing the feature data and target variable data for the
        training and testing sets, respectively.

    """
    split = StratifiedGroupKFold(n_splits=2, random_state=10, shuffle=True)
    return split_dataset(split, X, y)


def standard_split(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the input data into training and testing sets with a random split and
    stratified sampling.

    Parameters
    ----------
    X : pandas.DataFrame
        The input feature data to split.

    y : pandas.Series
        The target variable data to split.

    test_size : float
        The proportion of the data to use for testing.

    random_state : int
        The random seed used to generate the split.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
        A tuple containing the feature data and target variable data for the
        training and testing sets, respectively.

    """
    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=["tracker"]
    )
    return X_train, y_train, X_test, y_test


def plot_tracker_distribution(train_set: pd.Series, test_set: pd.Series) -> None:
    """
    Plot the distribution of the 'tracker' variable in the training and test datasets.

    Parameters
    ----------
    train_set : pandas.Series
        The training dataset containing the 'tracker' variable.
    test_set : pandas.Series
        The test dataset containing the 'tracker' variable.

    Returns
    -------
    None
        This function only produces a plot.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    train_set.value_counts().plot(
        kind="pie", ax=axes[0], labels=["non_tracker", "tracker"], autopct="%1.2f%%"
    )
    axes[0].set_title("Training Dataset")

    test_set.value_counts().plot(
        kind="pie", ax=axes[1], labels=["non_tracker", "tracker"], autopct="%1.2f%%"
    )
    axes[1].set_title("Test Dataset")

    plt.show()