from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from joblib import Parallel, delayed


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
    fbeta_score = metrics.fbeta_score(y_true, y_pred, beta=0.5)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    jaccard = metrics.jaccard_score(y_true, y_pred)

    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "log_loss": score,
        "auc": auc_score,
        "aupcr": aupcr,
        "balanced_accuracy": bal_acc,
        "f1": f1_score,
        "fbeta": fbeta_score,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "jaccard": jaccard,
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
    sample_idx = random_state.choice(len(y_true), len(y_true), replace=True)
    y_true_sample = y_true.iloc[sample_idx]
    y_pred_sample = y_pred[sample_idx]
    y_pred_proba_sample = y_pred_proba[sample_idx]

    return calculate_metrics(y_true_sample, y_pred_sample, y_pred_proba_sample)


def calculate_confidence_intervals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstrap_samples: int = 1000,
    random_state: int = 10,
    n_jobs: int = -1,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.RandomState(random_state)
    random_states = [
        np.random.RandomState(rng.randint(np.iinfo(np.int32).max))
        for _ in range(n_bootstrap_samples)
    ]

    bootstrap_metrics = Parallel(n_jobs=n_jobs)(
        delayed(calculate_single_bootstrap_sample)(y_true, y_pred, y_pred_proba, rs)
        for rs in random_states
    )

    confidence_intervals = {}
    for metric in bootstrap_metrics[0].keys():
        sorted_bootstrap_values = sorted(
            [sample_metrics[metric] for sample_metrics in bootstrap_metrics]
        )
        ci_lower = np.percentile(sorted_bootstrap_values, 2.5)
        ci_upper = np.percentile(sorted_bootstrap_values, 97.5)
        confidence_intervals[metric + "_CI"] = (ci_lower, ci_upper)

    return confidence_intervals


def train_and_evaluate_models(
    models: Dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: BaseCrossValidator,
    clf_preprocessor: ColumnTransformer = None,
) -> pd.DataFrame:
    """
    Train and evaluate models using cross-validation on the training data and
    evaluate them on the test data.

    Parameters
    ----------
    models : Dict[str, BaseEstimator]
        A dictionary containing model names as keys and estimator instances as
        values.
    X_train : pd.DataFrame
        Training feature data.
    y_train : pd.Series
        Training target data.
    X_test : pd.DataFrame
        Testing feature data.
    y_test : pd.Series
        Testing target data.
    cv : BaseCrossValidator
        Cross-validation object (e.g., StratifiedKFold, KFold).
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
    }

    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")

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

        best_estimator_idx = np.argmax(cv_results["test_roc_auc"])
        best_estimator = cv_results["estimator"][best_estimator_idx]
        y_pred_test = best_estimator.predict(X_test)
        y_pred_proba_test = best_estimator.predict_proba(X_test)[:, 1]

        # confidence_intervals = calculate_confidence_intervals(
        #     y_test, y_pred_test, y_pred_proba_test
        # )
        #
        # print(pd.DataFrame(confidence_intervals))

        test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
        test_metrics.update(calculate_confusion_matrix_elements(y_test, y_pred_test))

        mean_cv_metrics.update({"test_" + k: v for k, v in test_metrics.items()})
        all_mean_metrics[model_name] = mean_cv_metrics

    return pd.DataFrame(all_mean_metrics).T
