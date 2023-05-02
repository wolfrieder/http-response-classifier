import warnings

import yaml
import pandas as pd
import numpy as np
# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, shapiro, probplot
import category_encoders as ce
import mlflow
import os
import logging
import pickle

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with open("../../../params.yaml", "rb") as f:
        params = yaml.safe_load(f)

    mlflow.set_tracking_uri(params["ml_flow"]["MLFLOW_TRACKING_URI"])
    os.environ["MLFLOW_TRACKING_USERNAME"] = params["ml_flow"][
        "MLFLOW_TRACKING_USERNAME"
    ]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = params["ml_flow"][
        "MLFLOW_TRACKING_PASSWORD"
    ]

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

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=10, criterion="gini",
                                                max_features=None),
        # "KNN": KNeighborsClassifier(n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=10),
        "Gradient Boosting": GradientBoostingClassifier(random_state=10),
        # "XGBoost": XGBClassifier(random_state=10, use_label_encoder=False, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(random_state=10, class_weight="balanced"),
        "CatBoost": CatBoostClassifier(random_state=10, verbose=0),
        "MLP": MLPClassifier(random_state=10),
        "Logistic Regression": LogisticRegression(random_state=10),
        "Gaussian NB": GaussianNB()
    }

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
