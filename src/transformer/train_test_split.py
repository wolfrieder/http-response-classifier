import os

import pandas as pd
from src.pipeline_functions.train_test_split_functions import split_wrapper
from alive_progress import alive_bar


def run(browser: str, date: str, filename_one: str) -> None:
    with alive_bar(100, force_tty=True, manual=True, title="Train-Test-Split") as bar:
        bar.text("Read-in parameters")
        dir_path = f"{browser}/{date}"
        bar(0.05)

        bar.text("Read-in data")
        data = pd.read_parquet(
            f"data/processed/{dir_path}/{filename_one}.parquet.gzip",
            engine="pyarrow",
            dtype_backend="pyarrow",
        )
        bar(0.15)

        bar.text("Split dataset into train and test set")
        train_set, test_set = split_wrapper(
            data.iloc[:, :-1], data["tracker"], 0.2, "stratified_shuffle_split"
        )
        bar(0.4)

        # plot_tracker_distribution(y_train, y_test)

        bar.text("Delete data")
        del data
        bar(0.5)

        bar.text("Split data into train and validation set")
        # train_set, validation_set = split_wrapper(
        #     train_set.iloc[:, :-1],
        #     train_set["tracker"],
        #     0.1,
        #     "stratified_shuffle_split",
        # )
        bar(0.7)

        bar.text("Split data into train and calibration set")
        train_set, calibration_set = split_wrapper(
            train_set.iloc[:, :-1],
            train_set["tracker"],
            0.1,
            "stratified_shuffle_split",
        )
        bar(0.9)

        bar.text("Write datasets to parquet files")

        train_set.to_parquet(
            f"data/processed/{dir_path}/train_set.parquet.gzip",
            compression="gzip",
        )

        test_set.to_parquet(
            f"data/processed/{dir_path}/test_set.parquet.gzip",
            compression="gzip",
        )

        # validation_set.to_parquet(
        #     f"data/processed/{dir_path}/validation_set.parquet.gzip",
        #     compression="gzip",
        # )

        calibration_set.to_parquet(
            f"data/processed/{dir_path}/calibration_set.parquet.gzip",
            compression="gzip",
        )
        bar(1)
