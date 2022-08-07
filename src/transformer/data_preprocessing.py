import ray
import pandas as pd


@ray.remote
def test2(index, col_name, old_memory, data2):
    if col_name in ["query"]:
        return
    print(index, col_name)
    new_memory = data2[col_name].astype("category").memory_usage(deep=True)
    if new_memory < old_memory[index + 1]:
        return col_name


if __name__ == "__main__":
    # ray.init()
    dataset = ray.data.read_parquet(
        "/Users/wolfrieder/Desktop/master_thesis/thesis_project_v2/data/interim"
        "/tranco_16_05_22_10k_run_06/part_1.parquet.gzip"
    )
    a = dataset.schema()
    data22 = pd.read_parquet(
        "../data/interim/tranco_16_05_22_10k_run_06/part_0.parquet.gzip"
    )
    current_memory = data22.memory_usage()

    result = ray.get(
        [test2.remote(idx, column, current_memory) for idx, column in enumerate(data22)]
    )
