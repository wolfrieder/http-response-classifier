import warnings

import yaml
from sklearn.dummy import DummyClassifier
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, shapiro, probplot
import category_encoders as ce
import mlflow
import os
import logging
import pickle
from sklearn.naive_bayes import GaussianNB

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def calculate_metrics(data, y_true):
    y_pred = clf.predict(data)
    pred_probs = clf.predict_proba(data)
    score = metrics.log_loss(y_true, pred_probs)
    auc_score = metrics.roc_auc_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    print(
        "Model accuracy score : {0:0.4f}".format(metrics.accuracy_score(y_test, y_pred))
    )
    print("Model log-loss score : {0:0.4f}".format(score))
    print("Model auc score : {0:0.4f}".format(auc_score))
    print("Balanced accuracy score : {0:0.4f}".format(bal_acc))
    print("F1 score : {0:0.4f}".format(f1_score))
    print("Precision score : {0:0.4f}".format(precision))
    print("Recall score : {0:0.4f}".format(recall))
    print("Matthews correlation coefficient score : {0:0.4f}".format(mcc))
    print(metrics.classification_report(y_test, y_pred))

    disp_1 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp_2 = metrics.PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test, name="Random Forest"
    )
    disp_3 = metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
    mlflow.log_figure(disp_1.figure_, "cm.png")
    mlflow.log_figure(disp_2.figure_, "prec_recall.png")
    mlflow.log_figure(disp_3.figure_, "roc.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with open("../../../params.yaml", "rb") as f:
        params = yaml.load(f, yaml.FullLoader)

    mlflow.set_tracking_uri(params["ml_flow"]["MLFLOW_TRACKING_URI"])
    os.environ["MLFLOW_TRACKING_USERNAME"] = params["ml_flow"][
        "MLFLOW_TRACKING_USERNAME"
    ]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = params["ml_flow"][
        "MLFLOW_TRACKING_PASSWORD"
    ]

    mlflow.set_experiment('philips_experiments')
    # mlflow.set_experiment("imputation_by_label_experiments")
    # mlflow.set_experiment("simple_imputation_experiments")
    # mlflow.set_experiment("all_binary_experiments")

    mlflow.sklearn.autolog()
    # mlflow.xgboost.autolog()
    # mlflow.lightgbm.autolog()

    train_data = pd.read_parquet(
        "../../../data/processed/chrome/08_12_2022/train_set_01_featurized.parquet.gzip"
    )
    test_data = pd.read_parquet(
        "../../../data/processed/chrome/08_12_2022/test_set_01_featurized.parquet.gzip"
    )

    X_train, y_train = train_data.iloc[:, :-1], train_data[["tracker"]]
    X_test, y_test = test_data.iloc[:, :-1], test_data[["tracker"]]

    # models
    # model = RandomForestClassifier(
    #     n_estimators=100,
    #     n_jobs=-1,
    #     random_state=10,
    #     criterion="log_loss",
    #     max_features=None,
    # )

    model = KNeighborsClassifier(n_jobs=-1)
    # model = DecisionTreeClassifier()
    # model = GradientBoostingClassifier()
    # model = HistGradientBoostingClassifier()
    # model = xgb.XGBClassifier()
    # model = lgb.LGBMClassifier(class_weight="balanced")
    # model = CatBoostClassifier(thread_count=-1)
    # model = MLPClassifier()
    # model = LogisticRegression(n_jobs=-1, random_state=10)
    # model = GaussianNB()

    # pipeline
    with mlflow.start_run():
        # clf = lgb.LGBMClassifier(class_weight="balanced")

        numeric_transformer = Pipeline(
            steps=[("scaler", FunctionTransformer(np.log1p))]
        )
        norm_transformer = Pipeline(steps=[("norm_scaler", Normalizer())])
        # minmax_transformer = Pipeline(steps=[("mmscaler", MinMaxScaler(feature_range=[-1, 1]))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", ce.WOEEncoder(), selector(dtype_include="category")),
                ("num", numeric_transformer, ["content-length"]),
                ("age", norm_transformer, ["age"]),
            ]
        )

        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        clf.fit(X_train, y_train["tracker"])
        calculate_metrics(X_test, y_test)
    mlflow.end_run()
