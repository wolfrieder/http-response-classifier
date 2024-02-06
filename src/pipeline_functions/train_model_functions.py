from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    make_scorer,
    matthews_corrcoef,
    jaccard_score,
    average_precision_score,
    fbeta_score,
)
from sklearn.model_selection import BaseCrossValidator, RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from joblib import Parallel, delayed
import pickle
import gzip


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate classification metrics for a given set of true labels and
    predictions.

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
    aupcr = metrics.average_precision_score(y_true, y_pred_proba)
    f1_score = metrics.f1_score(y_true, y_pred)
    # fbeta_score = metrics.fbeta_score(y_true, y_pred, beta=0.5)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    # jaccard = metrics.jaccard_score(y_true, y_pred)

    return {
        "accuracy": np.round(metrics.accuracy_score(y_true, y_pred), 3),
        "log_loss": np.round(score, 3),
        "auc": np.round(auc_score, 3),
        "aupcr": np.round(aupcr, 3),
        "balanced_accuracy": np.round(bal_acc, 3),
        "f1": np.round(f1_score, 3),
        # "fbeta": np.round(fbeta_score, 3),
        "precision": np.round(precision, 3),
        "recall": np.round(recall, 3),
        "mcc": np.round(mcc, 3),
        # "jaccard": np.round(jaccard, 3),
    }


def calculate_confusion_matrix_elements(
    y_true: pd.Series, y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Calculate the confusion matrix elements (FP, TN, FN, TP) for the given true
    and predicted values.

    Parameters
    ----------
    y_true : pd.Series
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    Dict[str, int]
        A dictionary containing the confusion matrix elements (FP, TN, FN, TP).
    """
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return {"FP": fp, "TN": tn, "FN": fn, "TP": tp}


def calculate_single_bootstrap_sample(y_true, y_pred, y_pred_proba, random_state):
    """
    Calculate classification metrics for a single bootstrap sample.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    y_pred_proba : np.ndarray
        Estimated probabilities as returned by a classifier.
    random_state : np.random.RandomState
        Random state to use for generating the bootstrap sample.

    Returns
    -------
    dict
        Dictionary of calculated metrics for the single bootstrap sample.
    """
    sample_idx = random_state.choice(len(y_true), len(y_true), replace=True)
    y_true_sample = y_true.iloc[sample_idx]
    y_pred_sample = y_pred[sample_idx]
    y_pred_proba_sample = y_pred_proba[sample_idx]

    return calculate_metrics(y_true_sample, y_pred_sample, y_pred_proba_sample)


def calculate_confidence_intervals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstrap_samples: int = 100,
    random_state: int = 10,
    n_jobs: int = -1,
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for various classification metrics using
    bootstrap resampling.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    y_pred_proba : np.ndarray
        Estimated probabilities as returned by a classifier.
    n_bootstrap_samples : int, optional, default=1000
        Number of bootstrap samples to create.
    random_state : int, optional, default=10
        Seed used by the random number generator.
    n_jobs : int, optional, default=-1
        The number of jobs to run in parallel. -1 means using all processors.

    Returns
    -------
    confidence_intervals : dict
        Dictionary with keys as metric names and values as tuples of the lower
        and upper confidence interval bounds.
    """
    rng = np.random.RandomState(random_state)
    random_states = [
        np.random.RandomState(rng.randint(np.iinfo(np.int32).max))
        for _ in range(n_bootstrap_samples)
    ]

    bootstrap_metrics = Parallel(n_jobs=n_jobs, timeout=400)(
        delayed(calculate_single_bootstrap_sample)(y_true, y_pred, y_pred_proba, rs)
        for rs in random_states
    )

    confidence_intervals = {}
    for metric in bootstrap_metrics[0].keys():
        sorted_bootstrap_values = sorted(
            [sample_metrics[metric] for sample_metrics in bootstrap_metrics]
        )
        ci_lower = np.round(np.percentile(sorted_bootstrap_values, 2.5), 3)
        ci_upper = np.round(np.percentile(sorted_bootstrap_values, 97.5), 3)
        confidence_intervals[metric + "_CI"] = (ci_lower, ci_upper)

    return confidence_intervals


def train_models(
    models: Dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: RepeatedStratifiedKFold,
    clf_preprocessor: ColumnTransformer = None,
) -> pd.DataFrame:
    """
    Train models using cross-validation on the training data.

    Parameters
    ----------
    models : Dict[str, BaseEstimator]
        A dictionary containing model names as keys and estimator instances as
        values.
    X_train : pd.DataFrame
        Training feature data.
    y_train : pd.Series
        Training target data.
    cv : RepeatedStratifiedKFold
        Cross-validation object.
    clf_preprocessor : Pipeline
        A scikit-learn Pipeline object with the preprocessing step.

    Returns
    -------
    pd.DataFrame
        A DataFrame with each row representing a model and each column
        representing a metric.
    """
    all_mean_metrics = {}
    scoring_metrics = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "neg_log_loss": "neg_log_loss",
        "mcc": make_scorer(matthews_corrcoef),
        # "jaccard": make_scorer(jaccard_score),
        "aupcr": make_scorer(average_precision_score),
        # "fbeta": make_scorer(fbeta_score, beta=0.5),
    }

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        if type(clf_preprocessor) is not None:
            clf_pipeline = Pipeline(
                steps=[("preprocessor", clf_preprocessor), ("classifier", model)]
            )
        else:
            clf_pipeline = Pipeline(steps=[("classifier", model)])

        cv_results = cross_validate(
            clf_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring_metrics,
            return_estimator=True,
            n_jobs=-1,
        )
        mean_cv_metrics = {
            metric: np.mean(cv_results["test_" + metric])
            for metric in scoring_metrics.keys()
        }
        all_mean_metrics[model_name] = mean_cv_metrics

        best_estimator_idx = np.argmax(cv_results["test_roc_auc"])
        best_estimator = cv_results["estimator"][best_estimator_idx]

        filename = f"{model_name}_BE.sav.gz"
        gzip_path = f"models/chrome/08_12_2022/{filename}"

        with gzip.GzipFile(gzip_path, "wb") as model_gzip:
            pickle.dump(best_estimator["classifier"], model_gzip)

    return pd.DataFrame(all_mean_metrics).T


def test_models(
    models: List[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> List[pd.DataFrame]:
    """
    Evaluate the performance of multiple classification models on the test
    dataset and calculate the confidence intervals for each model.

    Parameters
    ----------
    models : List[str]
        A list of model names to be evaluated. Each model should have a
        corresponding saved binary file in the specified directory with the
        format "{model_name}_binary.sav".

    X_test : pd.DataFrame
        The test dataset containing the feature values.

    y_test : pd.Series
        The true labels for the test dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the evaluation metrics for each model, with model
        names as index and metrics as columns.

    Notes
    -----
    The function also prints a DataFrame showing the confidence interval lower
    and upper bounds for each model and metric. The confidence intervals are
    calculated using the `calculate_confidence_intervals` function.
    """
    all_mean_metrics = {}
    ci_results = {}

    for index, model_name in enumerate(models):
        print(f"Evaluating {model_name}...")

        filename = f"{model_name}_BE.sav.gz"
        gzip_path = f"models/chrome/08_12_2022/{filename}"

        with gzip.GzipFile(gzip_path, "rb") as f:
            best_estimator = pickle.load(f)

        y_pred_test = best_estimator.predict(X_test)
        y_pred_proba_test = best_estimator.predict_proba(X_test)[:, 1]

        confidence_intervals = calculate_confidence_intervals(
            y_test, y_pred_test, y_pred_proba_test
        )

        ci_results[model_name] = confidence_intervals

        test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
        test_metrics.update(calculate_confusion_matrix_elements(y_test, y_pred_test))

        final_metrics = {}
        final_metrics.update({"test_" + k: v for k, v in test_metrics.items()})
        all_mean_metrics[model_name] = final_metrics

    data = {model_name: {} for model_name in models}
    for model_name, ci in ci_results.items():
        for metric, bounds in ci.items():
            lower_key = f"{metric}_lower"
            upper_key = f"{metric}_upper"
            data[model_name][lower_key] = bounds[0]
            data[model_name][upper_key] = bounds[1]

    ci_df = pd.DataFrame.from_dict(data, orient="index")

    return [pd.DataFrame(all_mean_metrics).T, ci_df]
