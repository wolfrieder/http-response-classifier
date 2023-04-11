import warnings

import yaml
import pandas as pd
import numpy as np

from sklearn.preprocessing import Normalizer, FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
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

    # TODO: Add more metrics here.
    # print(metrics.classification_report(y_test, y_pred))
    #
    # disp_1 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    # disp_2 = metrics.PrecisionRecallDisplay.from_estimator(
    #     clf, X_test, y_test, name="Random Forest"
    # )
    # disp_3 = metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
    # mlflow.log_figure(disp_1.figure_, "cm.png")
    # mlflow.log_figure(disp_2.figure_, "prec_recall.png")
    # mlflow.log_figure(disp_3.figure_, "roc.png")

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

def train_and_evaluate_models(models: Dict[str, BaseEstimator], X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> pd.DataFrame:
    """
    Train and evaluate different classification models.

    Parameters
    ----------
    models : Dict[str, BaseEstimator]
        A dictionary containing model names as keys and classifier instances as values.
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target vector.
    cv : StratifiedKFold
        The cross-validator providing train/test indices for each fold.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the mean metrics for each model.
    """
    all_mean_metrics = {}

    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")

        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        all_metrics = perform_cross_validation(X, y, clf, cv)
        mean_metrics = mean_metrics(all_metrics)

        all_mean_metrics[model_name] = mean_metrics

    return pd.DataFrame(all_mean_metrics).T
