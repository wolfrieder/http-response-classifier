import warnings

import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# from sklearn.impute import KNNImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer, FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, shapiro, probplot
import category_encoders as ce
import mlflow
import os
import logging
import pickle


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate classification metrics for a given set of true labels and predictions.

    Parameters
    ----------
    y_true : pd.Series
        The true labels for the test set.
    y_pred : np.ndarray
        The predicted labels for the test set.
    y_pred_proba : np.ndarray
        The predicted probabilities for the positive class.

    Returns
    -------
    dict
        A dictionary containing various classification metrics.
    """
    score = metrics.log_loss(y_true, y_pred_proba)
    auc_score = metrics.roc_auc_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "log_loss": score,
        "auc": auc_score,
        "balanced_accuracy": bal_acc,
        "f1": f1_score,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
    }


def perform_cross_validation(
    X: pd.DataFrame, y: pd.Series, clf: Pipeline, cv: StratifiedKFold
) -> List[Dict[str, float]]:
    """
    Perform cross-validation for a given classifier and dataset.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target vector.
    clf : Pipeline
        The classifier pipeline.
    cv : StratifiedKFold
        The cross-validator providing train/test indices for each fold.

    Returns
    -------
    list
        A list of dictionaries containing classification metrics for each fold.
    """
    all_metrics = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        fold_metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        all_metrics.append(fold_metrics)

    return all_metrics


def mean_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute the mean values of classification metrics across folds.

    Parameters
    ----------
    all_metrics : list
        A list of dictionaries containing classification metrics for each fold.

    Returns
    -------
    dict
        A dictionary containing the mean values of classification metrics.
    """
    return {
        metric: np.mean([fold_metrics[metric] for fold_metrics in all_metrics])
        for metric in all_metrics[0].keys()
    }
