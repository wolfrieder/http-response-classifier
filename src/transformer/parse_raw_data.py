import os
import sys
import time

sys.path.append('../../')
from src.pipeline_functions.parse_raw_data_functions import *


def run(browser, directory, file_name):
    # ray.shutdown()
    # ray.init()
    # pd.set_option("display.max_columns", 500)

    start = time.perf_counter()
    # browser = sys.argv[1]
    # directory = sys.argv[2]
    dir_path = f"{browser}/{directory}"

    try:
        os.makedirs(f"../../../data/interim/{dir_path}", exist_ok=True)
        print(f"Directory {dir_path} created successfully.")
    except OSError as error:
        print(f"Directory {dir_path} can not be created.")

    # parse_dataset(
    #     sys.argv[3], dir_path, sys.argv[3], f"interim/{browser}/{directory}", 3000
    # )

    parse_dataset(
        file_name, dir_path, file_name, f"interim/{browser}/{directory}", 50000
    )

    stop = time.perf_counter()
    print("end time:", stop - start)


# if __name__ == "__main__":
#     ray.shutdown()
#     ray.init()
#     # pd.set_option("display.max_columns", 500)
#
#     start = time.perf_counter()
#     browser = sys.argv[1]
#     directory = sys.argv[2]
#     dir_path = f"{browser}/{directory}"
#
#     try:
#         os.makedirs(f"../../../data/interim/{dir_path}", exist_ok=True)
#         print(f"Directory {dir_path} created successfully.")
#     except OSError as error:
#         print(f"Directory {dir_path} can not be created.")
#
#     parse_dataset(
#         sys.argv[3], dir_path, sys.argv[3], f"interim/{browser}/{directory}", 3000
#     )
#
#     stop = time.perf_counter()
#     print("end time:", stop - start)
