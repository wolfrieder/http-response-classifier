import sys

from alive_progress import alive_bar

sys.path.append("../../")
from src.pipeline_functions.train_model_functions import *


def run(
    browser_two: str,
    date_two: str,
    test_data_file_name: str,
    experiment_name: str,
) -> None:
    dir_path_two = f"{browser_two}/{date_two}"

    dir_path_two = f"../../../data/processed/{dir_path_two}/{test_data_file_name}"
    result_csv_filename = f"../../../models/result_metrics/{experiment_name}.csv"

    test_models(dir_path_two, result_csv_filename)


def test_models(
    test_data_file_path: str,
    result_csv_filename: str,
) -> None:
    with alive_bar(
        100, force_tty=True, manual=True, title="Training and Validating Models"
    ) as bar:
        print(result_csv_filename)
        bar.text("Read-in data")
        test_data = pd.read_parquet(f"{test_data_file_path}.parquet.gzip")
        bar(0.1)

        bar.text("Split data into features and targets")
        X_test, y_test = test_data.iloc[:, :-1], test_data[["tracker"]]
        bar(0.2)

        bar.text("Define models")
        models = [
            "Logistic Regression"
            "Gaussian NB"
            "Bernoulli NB"
            "Decision Tree"
            "Random Forest"
            "Extra Trees Classifier"
            "Ada Boost"
            "Gradient Boosting"
            "LightGBM"
            "Hist Gradient Boosting"
            "XGBoost"
            "MLP"
        ]
        bar(0.3)

        bar.text("Evaluate models")
        result_df = train_and_evaluate_models(models, X_test, y_test["tracker"])
        bar(0.9)

        bar.text("Export results to CSV")
        result_df.to_csv(f"{result_csv_filename}", index=True)
        bar(1)
