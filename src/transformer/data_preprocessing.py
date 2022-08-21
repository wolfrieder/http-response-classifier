import ray
import modin.pandas as md
import json
from rapidfuzz import fuzz, utils, process
import pandas as pd


@ray.remote
def check_col_with_category_dtype(col_name: str, current_memory, data: md.DataFrame):
    index = data.columns.get_loc(col_name)
    if col_name in ["query"]:
        return None
    print(index, col_name)
    new_memory = data[col_name].astype("category").memory_usage(deep=True)
    if new_memory < current_memory[index + 1]:
        return col_name


def create_json_of_category_dtypes(filename: str) -> None:
    dataset = md.read_parquet(
        f"/Users/wolfrieder/Desktop/master_thesis/thesis_project_v2/data/interim/tranco_16_05_22_10k_run_06/{filename}.parquet.gzip"
    )
    dataset = dataset.iloc[:, 8:20]
    current_memory = dataset.memory_usage(deep=True)

    category_list = ray.get(
        [
            check_col_with_category_dtype.remote(column, current_memory, dataset)
            for column in enumerate(dataset)
        ]
    )
    category_list_final = [i for i in category_list if i is not None]
    dict_categories = {i: "category" for i in category_list_final}
    return dict_categories
    # with open("dict_categories.json", "w") as f:
    #     json.dump(dict_categories, f)


# @ray.remote
def fuzzy_string_matching_for_column(col_name, col_values):
    list_of_ratios = [
        calculate_ratios(col_name, col_name_2) for col_name_2 in col_values
    ]
    final_list_of_ratios = pd.concat(list_of_ratios, ignore_index=True)
    final_list_of_ratios = final_list_of_ratios[
        (final_list_of_ratios["ratio"] >= 80)
        | (final_list_of_ratios["token_sort_ratio"] >= 80)
        | (final_list_of_ratios["partial_ratio"] >= 90)
        | (final_list_of_ratios["token_set_ratio"] >= 90)
    ]
    if final_list_of_ratios.empty:
        return None
    return final_list_of_ratios


# @ray.remote
def calculate_ratios(col_name, col_name_2):
    ratio = fuzz.ratio(col_name, col_name_2)
    partial_ratio = fuzz.partial_ratio(col_name, col_name_2)
    token_sort_ratio = fuzz.token_sort_ratio(col_name, col_name_2)
    token_set_ratio = fuzz.token_set_ratio(col_name, col_name_2)
    w_ratio = fuzz.WRatio(col_name, col_name_2)

    ratio_results_dict = pd.DataFrame(
        data={
            "ratio": ratio,
            "partial_ratio": partial_ratio,
            "token_sort_ratio": token_sort_ratio,
            "token_set_ratio": token_set_ratio,
            "w_ratio": w_ratio,
            "header_fields": f"{col_name}=={col_name_2}",
        },
        index=[0],
    )
    return ratio_results_dict


def new_fuzzy_string_matching_for_column(col_name, col_values):
    result = pd.DataFrame(
        process.extract(
            col_name, col_values, processor=None, score_cutoff=80, limit=100
        ),
        columns=["fuzzy_match", "w_ratio", "index"],
    )
    result["col_name"] = col_name
    return result


if __name__ == "__main__":
    ray.init()
    print(create_json_of_category_dtypes("part_0"))
