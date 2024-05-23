import logging
import sys

import category_encoders as ce
from alive_progress import alive_bar
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer, FunctionTransformer, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.append("../../")
from src.pipeline_functions.train_model_functions import *

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def run(
    browser: str,
    date: str,
    train_data_file_name: str,
    strategy: str,
    experiment_name: str,
) -> None:
    dir_path_one = f"{browser}/{date}"

    dir_path_one = f"data/processed/{dir_path_one}/{train_data_file_name}"
    result_csv_filename = f"models/result_metrics/{experiment_name}.csv"

    train_models_run(dir_path_one, strategy, result_csv_filename)


def train_models_run(
    train_data_file_path: str,
    strategy: str,
    result_csv_filename: str,
) -> None:
    with alive_bar(100, force_tty=True, manual=True, title="Training Models") as bar:
        httpMessage = "response" if "response" in train_data_file_path else "request"
        bar.text("Read-in data")
        train_data = pd.read_parquet(f"{train_data_file_path}.parquet.gzip")
        bar(0.1)

        bar.text("Split data into features and targets")
        X_train, y_train = train_data.iloc[:, :-2], train_data[["tracker"]]
        bar(0.2)

        bar.text("Define models")
        models = {
            "Logistic_Regression": LogisticRegression(
                random_state=10, n_jobs=-1, max_iter=1000, solver="newton-cholesky"
            ),
            "Gaussian_NB": GaussianNB(),
            "Bernoulli_NB": BernoulliNB(),
            "Decision_Tree": DecisionTreeClassifier(random_state=10),
            "Random_Forest": RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=10,
                criterion="gini",
                max_features=None,
            ),
            "Extra_Trees_Classifier": ExtraTreesClassifier(random_state=10, n_jobs=-1),
            "Ada_Boost": AdaBoostClassifier(random_state=10),
            "Gradient_Boosting": GradientBoostingClassifier(random_state=10),
            "LightGBM": LGBMClassifier(random_state=10, n_jobs=-1),
            "Hist_GB": HistGradientBoostingClassifier(random_state=10),
            "XGBoost": XGBClassifier(
                random_state=10,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            # "MLP": MLPClassifier(
            #     solver="adam",
            #     random_state=10,
            #     activation="relu",
            #     alpha=1e-05,
            #     learning_rate_init=0.001,
            #     hidden_layer_sizes=(128, 64),
            #     learning_rate="adaptive",
            # ),
        }
        bar(0.3)

        bar.text("Define CV Method")
        cv = RepeatedStratifiedKFold(random_state=10, n_splits=5, n_repeats=5)
        bar(0.4)

        bar.text("Train and evaluate models")
        if strategy == "binary":
            result_df = train_models(
                models, X_train, y_train["tracker"], httpMessage, cv
            )
        else:
            numeric_transformer = Pipeline(
                steps=[("scaler", FunctionTransformer(np.log1p))]
            )
            norm_transformer = Pipeline(steps=[("norm_scaler", Normalizer())])
            minmax_transformer = Pipeline(
                steps=[("mmscaler", MinMaxScaler(feature_range=[-1, 1]))]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", ce.WOEEncoder(), selector(dtype_include="category")),
                    ("num", numeric_transformer, ["content-length"]),
                    ("age", norm_transformer, ["age"]),
                ]
            )

            result_df = train_models(
                models,
                X_train,
                y_train["tracker"],
                cv,
                preprocessor,
            )
        bar(0.9)

        bar.text("Export results to CSV")
        result_df.to_csv(f"{result_csv_filename}", index=True)
        bar(1)
