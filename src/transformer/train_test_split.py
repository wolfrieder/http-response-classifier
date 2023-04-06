import os
import sys

import pandas as pd
from src.pipeline_functions.train_test_split_functions import stratified_shuffle_split
from alive_progress import alive_bar


if __name__ == "__main__":
    with alive_bar(100, force_tty=True, manual=True, title="Train-Test-Split") as bar:
        bar.text('Read-in parameters')
        browser = sys.argv[1]
        directory = sys.argv[2]
        dir_path = f"{browser}/{directory}"
        bar(0.05)

        bar.text("Read-in data")
        data = pd.read_parquet(f"../../../data/interim/{dir_path}/{sys.argv[3]}.parquet.gzip")
        bar(0.1)

        bar.text("Create directory if it doesn't exist")
        try:
            os.makedirs(f"../../../data/processed/{dir_path}", exist_ok=True)
            print(f"Directory {dir_path} created successfully.")
        except OSError as error:
            print(f"Directory {dir_path} can not be created.")

        bar(0.15)

        bar.text("Read-in second dataset")
        if len(sys.argv) > 4:
            data_2 = pd.read_parquet(
                f"../../../data/interim/{dir_path}/{sys.argv[5]}.parquet.gzip"
            )
            bar(0.2)
            bar.text("Concat data")
            concat_data = [data, data_2]
            data = pd.concat(concat_data, ignore_index=True)
            del data_2
            bar(0.3)

            bar.text("Reindex columns")
            temp_cols = data.columns.tolist()
            index_col = data.columns.get_loc("tracker")
            new_col_order = (
                temp_cols[0:index_col]
                + temp_cols[index_col + 1:]
                + temp_cols[index_col: index_col + 1]
            )
            data = data[new_col_order]

        bar(0.5)

        bar.text("Split dataset")
        X_train, y_train, X_test, y_test = stratified_shuffle_split(
            data.iloc[:, :-1], data[["tracker"]]
        )

        # plot_tracker_distribution(y_train, y_test)
        del data
        bar(0.6)

        bar.text("Concat Training Data")
        train_set = pd.concat([X_train, y_train], axis=1)
        bar(0.7)

        bar.text("Concat Test Data")
        test_set = pd.concat([X_test, y_test], axis=1)
        bar(0.9)

        bar.text("Write datasets to parquet files")

        train_set.to_parquet(
            f"../../../data/processed/{dir_path}/train_set_{sys.argv[4]}{sys.argv[6]}.parquet.gzip",
            compression="gzip",
        )

        test_set.to_parquet(
            f"../../../data/processed/{dir_path}/test_set_{sys.argv[4]}{sys.argv[6]}.parquet.gzip",
            compression="gzip",
        )
        bar(1)
