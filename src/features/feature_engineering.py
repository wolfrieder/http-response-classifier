import warnings
import sys
import numpy as np
from alive_progress import alive_bar

sys.path.append("../../")
from src.pipeline_functions.feature_engineering_functions import *

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def run(
    browser: str, date: str, file_name: str, strategy: str, http_message: str
) -> None:
    dir_path = f"{browser}/{date}"
    dir_path = f"data/processed/{dir_path}/{file_name}"

    print(dir_path)

    if strategy == "binary_encoding":
        featurize_data_binary_encoding(dir_path, http_message)


def featurize_data_binary_encoding(file_path: str, http_message: str) -> None:
    with alive_bar(
        100, force_tty=True, manual=True, title="Feature Engineering"
    ) as bar:
        bar.text("Read-in data")
        data = pd.read_parquet(f"{file_path}_processed_{http_message}.parquet.gzip")
        bar(0.1)

        bar.text("Prepare data")
        number_of_features = len(data.columns) - 4
        bar(0.2)

        bar.text("Binary Encoding of Features")
        for elem in data.iloc[:, :-4].columns.values.tolist():
            data[f"{elem}_binary"] = np.where(data[elem].isnull(), 0, 1)
        bar(0.6)

        bar.text("Removing old columns")
        data = data.iloc[:, number_of_features:]
        bar(0.7)

        bar.text("Parsing to uint8 data type")
        for elem in data.columns.values.tolist():
            data[elem] = data[elem].astype("uint8")
        bar(0.8)

        bar.text("Reordering features")
        reordered_cols = feature_as_last_column(data, "httpMessageId")
        data = data[reordered_cols]
        reordered_cols = feature_as_last_column(data, "tracker")
        data = data[reordered_cols]
        bar(0.9)

        bar.text("Export data to local parquet files")
        data.to_parquet(
            f"{file_path}_featurized_{http_message}_BE.parquet.gzip",
            compression="gzip",
        )
        bar(1)
