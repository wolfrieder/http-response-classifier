import sys

sys.path.append("src")
from src.transformer import parse_raw_data, data_preprocessing, train_test_split
from src.transformer import data_preprocessing_config
from src.features import feature_engineering
from src.models import train_model, test_model


def run_parse_raw_data(browser: str, date: str, filename: str, data_dir: str) -> None:
    parse_raw_data.run(browser, date, filename, data_dir)


def run_train_test_split(browser: str, date: str, filename_one: str) -> None:
    train_test_split.run(browser, date, filename_one)


def run_preprocessing_data(
    browser_one: str,
    date_one: str,
    train_data_file_name: str,
    config_path: str,
    other_test_data: bool,
) -> None:
    data_preprocessing_config.run(
        browser_one, date_one, train_data_file_name, config_path, other_test_data
    )


def run_feature_engineering(
    browser: str, date: str, filename: str, strategy: str
) -> None:
    feature_engineering.run(browser, date, filename, strategy)


def run_train_model(
    browser: str, date: str, filename: str, strategy: str, experiment_name: str
) -> None:
    train_model.run(browser, date, filename, strategy, experiment_name)


def run_test_model(
    browser: str, date: str, filename: str
) -> None:
    test_model.run(browser, date, filename)


if __name__ == "__main__":
    script_to_run = sys.argv[1]
    args = sys.argv[2:]

    print(f"Running script: {script_to_run}")

    if script_to_run == "parse_raw_data.py":
        run_parse_raw_data(*args)
    elif script_to_run == "train_test_split.py":
        run_train_test_split(*args)
    elif script_to_run == "data_preprocessing_config.py":
        run_preprocessing_data(*args)
    elif script_to_run == "feature_engineering.py":
        run_feature_engineering(*args)
    elif script_to_run == "train_model.py":
        run_train_model(*args)
    elif script_to_run == "test_model.py":
        run_test_model(*args)
    else:
        print(f"Unknown script: {script_to_run}")
