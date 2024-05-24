import sys

from alive_progress import alive_bar

sys.path.append("../../")
from src.pipeline_functions.train_model_functions import *


def run(
    browser: str,
    date: str,
    test_data_file_name: str,
    http_message: str,
) -> None:
    dir_path = f"{browser}/{date}"
    experiment_name = f"{browser}_{date}_{test_data_file_name}_{http_message}"

    dir_path = (
        f"data/processed/{dir_path}/{test_data_file_name}_featurized_{http_message}_BE"
    )
    result_csv_filename = f"models/result_metrics/{experiment_name}"

    test_models_run(dir_path, result_csv_filename, http_message)


def test_models_run(
    test_data_file_path: str,
    result_csv_filename: str,
    http_message: str,
) -> None:
    with alive_bar(100, force_tty=True, manual=True, title="Validating Models") as bar:

        bar.text("Read-in data")
        test_data = pd.read_parquet(f"{test_data_file_path}.parquet.gzip")
        bar(0.1)

        bar.text("Split data into features and targets")
        X_test, y_test = test_data.iloc[:, :-2], test_data[["tracker"]]
        bar(0.2)

        bar.text("Define models")
        models = [
            "Logistic_Regression",
            "Gaussian_NB",
            "Bernoulli_NB",
            "Decision_Tree",
            "Random_Forest",
            "Extra_Trees_Classifier",
            "Ada_Boost",
            "Gradient_Boosting",
            "LightGBM",
            "Hist_GB",
            "XGBoost",
        ]
        bar(0.3)

        bar.text("Evaluate models")
        result_df_list = test_models(models, X_test, y_test["tracker"], http_message)
        bar(0.9)

        bar.text("Export results to CSV")
        result_df_list[0].to_csv(f"{result_csv_filename}.csv", index=True)
        result_df_list[1].to_csv(f"{result_csv_filename}_CI.csv", index=True)
        bar(1)
