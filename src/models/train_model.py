import os
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

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


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
    mlflow.set_experiment("test_experiment")

    mlflow.sklearn.autolog()
    # mlflow.xgboost.autolog()

    train_data = pd.read_parquet(
        "../../../data/processed/chrome/08_12_2022/train_set_01_featurized.parquet.gzip"
    )
    test_data = pd.read_parquet(
        "../../../data/processed/chrome/08_12_2022/test_set_01_featurized.parquet.gzip"
    )

    with mlflow.start_run():

        X_train, y_train = train_data.iloc[:, :-1], train_data[["tracker"]]
        X_test, y_test = test_data.iloc[:, :-1], test_data[["tracker"]]

        # models
        rf_model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=10,
            criterion="log_loss",
            max_features=None,
        )

        # knn_model = KNeighborsClassifier(n_jobs=-1)
        # dt_model = DecisionTreeClassifier()
        # gbm_model = GradientBoostingClassifier()
        # hgb_model = HistGradientBoostingClassifier()
        # xgb_model = xgb.XGBClassifier()
        # lgb_model = lgb.LGBMClassifier(class_weight="balanced")
        # cat_model = CatBoostClassifier(thread_count=-1)
        # mlp_model = MLPClassifier()

        # pipeline
        numeric_transformer = Pipeline(
            steps=[("scaler", FunctionTransformer(np.log1p))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, ["content-length"]),
                ("cat", ce.WOEEncoder(), selector(dtype_include="category")),
            ]
        )

        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier",
                                                               rf_model)])

        clf.fit(X_train, y_train["tracker"].to_numpy())
        y_pred = clf.predict(X_test)
        print(
            "Model accuracy score : {0:0.4f}".format(
                metrics.accuracy_score(y_test, y_pred)
            )
        )

        clf_probs = clf.predict_proba(X_test)
        score = metrics.log_loss(y_test, clf_probs)
        print("Model log-loss score : {0:0.4f}".format(score))
        test_roc = metrics.RocCurveDisplay.from_predictions(y_test, y_pred)
        test_cm = metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, cmap="spring"
        )
        mlflow.log_figure(test_cm.figure_, "test_confusion_matrix.png")
        mlflow.log_figure(test_roc.figure_, "test_roc.png")
        plt.show()
        f1_score = metrics.f1_score(y_test, y_pred)
        bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)

        print("Balanced accuracy score : {0:0.4f}".format(bal_acc))
        print("F1 score : {0:0.4f}".format(f1_score))
        precision = metrics.precision_score(y_test, y_pred)
        print("Precision score : {0:0.4f}".format(precision))
        mlflow.end_run()

    filename = "random_forest_v2_binary.sav"
    pickle.dump(clf['classifier'], open(f'../../../models/chrome/08_12_2022/{filename}', 'wb'))
