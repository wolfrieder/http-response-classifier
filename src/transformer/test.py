import os
import sys
# import pandas as pd
import ray

# import modin.pandas as pd
import pandas as pd
# import yaml
# from pandarallel import pandarallel
import swifter
from rapidfuzz import process
import numpy as np

# pandarallel.initialize(progress_bar=True)


# params = yaml.safe_load(open('params.yaml'))

# print(params['browsers'])
#
# print(params['browsers']['chrome'])

# print(pd.read_json(f'../../data/raw/{params["browsers"]["chrome"]}/'
#                    f'{params["file_name"]["date"]}/http.0.json'))

# print(sys.argv)

# dict_categories = {"test": sys.argv[3]}
# number = sys.argv[3].split('.')[1]

# with open(f"data/raw/{sys.argv[2]}/{sys.argv[3]}.json", "w") as f:
#     orjson.dump(dict_categories, f)

# test = pd.read_json('../../data/raw/chrome/08_12_2022/http.0.json.gzip',
#                     compression='gzip')


def new_fuzzy_string_matching_for_column(col_name, col_values):
    fuzzy_result = pd.DataFrame(
        process.extract(
            col_name, col_values, processor=None, score_cutoff=80, limit=100
        ),
        columns=["fuzzy_match", "w_ratio", "index"],
    )
    fuzzy_result["col_name"] = col_name
    return fuzzy_result


def find_cols_with_similar_values(fuzzy_match, column):
    value_fuzzy = set(data[fuzzy_match].values)
    value_column = set(data[column].values)

    try:
        value_fuzzy.remove(None)
        value_column.remove(None)
    except KeyError:
        pass

    if (len([True for i in value_fuzzy if i in value_column]) / len(value_fuzzy)) > 0.5:
        return fuzzy_match, column
    else:
        return None


def select_similar_columns(fuzzy_match, column):
    row = match2.loc[
        (match2["fuzzy_match"] == fuzzy_match) & (match2["col_name"] == column)
    ]
    index = row.index[0]
    match2.drop(index, inplace=True)
    return row


def merge_similar_columns2(fuzzy_match, col_name):
    boolean_mask = data[fuzzy_match].notnull()
    new_values = data[boolean_mask][fuzzy_match].to_numpy()
    indices_fuzzy_matches = data.index[boolean_mask].tolist()

    current_values = data[col_name].to_numpy()
    np.put(current_values, indices_fuzzy_matches, new_values)


def merge_similar_columns2_test(fuzzy_match, col_name):
    boolean_mask = data_test[fuzzy_match].notnull()
    new_values = data_test[boolean_mask][fuzzy_match].to_numpy()
    indices_fuzzy_matches = data_test.index[boolean_mask].tolist()

    current_values = data_test[col_name].to_numpy()
    np.put(current_values, indices_fuzzy_matches, new_values)


def reduced_variance_per_column(column):
    unique_values = int(len(set(data[column]))) - 1
    na_values = data[column].isna().sum()
    return [column, unique_values, round(na_values / len(data), 3)]


def variance_per_column_2(column):
    tracker_ratio = data[data[column].notnull()].tracker.value_counts()
    try:
        trackers = tracker_ratio[1]
    except KeyError:
        trackers = 0
    try:
        non_trackers = tracker_ratio[0]
    except KeyError:
        non_trackers = 0
    return [column, trackers, non_trackers]


def concise_information(col_list, classification):
    indices = set()

    for col in col_list:
        indices.update(
            data[
                (data[col].notnull()) & (data["tracker"] == classification)
            ].index.tolist()
        )

    return indices


def test_new_categories_update(element):
    categories = data[element].astype("category").cat.categories.values.tolist()
    try:
        np.array(categories, dtype="int64")
        return {element: "Int64"}
    except (ValueError, OverflowError):
        return None


def impute_value(element, classification):
    if element in list_of_integer_cols:
        data.loc[data['tracker'] == classification, element] = data.loc[data['tracker'] == classification, element]\
            .fillna(data[data['tracker']==classification][element].median())

    if element in list_of_categorical_cols:
        data.loc[data['tracker'] == classification, element] = data.loc[data['tracker'] == classification, element]\
            .fillna(data[data['tracker']==classification][element].mode().iloc[0])


print("Read data")
data = pd.read_parquet(
    "../../data/processed/chrome/08_12_2022/train_set_01.parquet.gzip"
)

data_test = pd.read_parquet(
    "../../data/processed/chrome/08_12_2022/test_set_01.parquet.gzip"
)

print("Count number of headers per message")
data["header_count"] = data.iloc[:, 6:-1].notnull().sum(axis=1)
data_test["header_count"] = data.iloc[:, 6:-1].notnull().sum(axis=1)

print("Remove empty columns")
empty_columns = [col for col in data if data[col].isnull().all() == True]
data.drop(empty_columns, axis=1, inplace=True)
data_test.drop(empty_columns, axis=1, inplace=True)

print("Remove duplicated observations")
data = data[~data.iloc[:, 6:-2].duplicated(keep="first")].reset_index(drop=True)

print("Fuzzy match")
data_column_values = data.columns.values[6:-2].tolist()
match = [
    new_fuzzy_string_matching_for_column(j, data_column_values[i + 1:])
    for i, j in enumerate(data_column_values)
    if i != len(data_column_values) - 1
]

match2 = pd.concat(match, ignore_index=True)
del match

print("Find fuzzy matches with similar columns")

result = [
    find_cols_with_similar_values(col, col2)
    for col, col2 in zip(match2["fuzzy_match"], match2["col_name"])
]

print("Reset index")
data.reset_index(drop=True, inplace=True)
data_test.reset_index(drop=True, inplace=True)

similar_values = [
    select_similar_columns(col[0], col[1]) for col in result if col is not None
]
similar_values = pd.concat(similar_values, ignore_index=True)
similar_values.apply(
    lambda x: merge_similar_columns2(x["fuzzy_match"], x["col_name"]), axis=1
)

similar_values_test = pd.concat(similar_values, ignore_index=True)
similar_values_test.apply(
    lambda x: merge_similar_columns2_test(x["fuzzy_match"], x["col_name"]), axis=1
)

del match2

columns_to_remove = list(set(similar_values.fuzzy_match.values.tolist()))
data.drop(columns_to_remove, axis=1, inplace=True)
data_test.drop(columns_to_remove, axis=1, inplace=True)

del result
del similar_values
del similar_values_test
del columns_to_remove

print("Data types")
columns_as_category = {i: "category" for i in data.columns.values[:-2]}
column_test = data.columns.values[6:-2].tolist()
braze2 = [test_new_categories_update(element) for element in column_test]

braze2 = list(filter(lambda x: type(x) is dict, braze2))
braze2 = {k: v for d in braze2 for k, v in d.items()}

columns_as_category.update(braze2)
del columns_as_category["query"]
del columns_as_category["protocol"]
data.drop(["protocol", "query"], axis=1, inplace=True)
data_test.drop(["protocol", "query"], axis=1, inplace=True)
data = data.astype(columns_as_category)
data_test = data_test.astype(columns_as_category)

print("Remove columns with 1.0 na_ratio")
number_of_elements = np.array(
    [reduced_variance_per_column(column) for column in data.iloc[:, 4:-2].columns]
)
summary_table = pd.DataFrame(
    number_of_elements, columns=["header_name", "unique_values", "na_ratio"]
)
summary_table["unique_values"] = summary_table["unique_values"].astype("Int32")
summary_table["na_ratio"] = summary_table["na_ratio"].astype("float32")
remove_headers_with_one_na_ratio = summary_table[
    summary_table["na_ratio"] == 1
].header_name.values.tolist()
remove_headers_with_one_value = summary_table[
    (summary_table["unique_values"] <= 1) & (summary_table["na_ratio"] != 1)
].header_name.values.tolist()

print("Drop columns")

data.drop(remove_headers_with_one_na_ratio, axis=1, inplace=True)
data.drop(remove_headers_with_one_value, axis=1, inplace=True)

data_test.drop(remove_headers_with_one_na_ratio, axis=1, inplace=True)
data_test.drop(remove_headers_with_one_value, axis=1, inplace=True)

del remove_headers_with_one_na_ratio
del number_of_elements
del summary_table

print("Combine columns")
number_of_elements_reduced = np.array(
    [variance_per_column_2(column) for column in data.iloc[:, 4:-2].columns]
)
summary_table = pd.DataFrame(
    number_of_elements_reduced, columns=["header_name", "trackers", "non_trackers"]
)
summary_table["trackers"] = summary_table["trackers"].astype("Int32")
summary_table["non_trackers"] = summary_table["non_trackers"].astype("float32")
summary_table["ratio"] = (
    summary_table["trackers"] / summary_table["non_trackers"]
) * 100
summary_table["ratio2"] = (
    summary_table["non_trackers"] / summary_table["trackers"]
) * 100

only_non_tracker_col = summary_table[
    summary_table["ratio"] <= 10
].header_name.values.tolist()
only_tracker_col = summary_table[
    summary_table["ratio2"] <= 10
].header_name.values.tolist()

data["comb_col_non_tracker"] = 0
data["comb_col_tracker"] = 0

for idx in concise_information(only_tracker_col, 1):
    data.at[idx, "comb_col_tracker"] = 1

for idx in concise_information(only_non_tracker_col, 0):
    data.at[idx, "comb_col_non_tracker"] = 1

data.drop(only_non_tracker_col, axis=1, inplace=True)
data.drop(only_tracker_col, axis=1, inplace=True)

print("Imputation")

list_of_categorical_cols = list(data.iloc[:, 4:-4].select_dtypes('category').columns.values.tolist())
list_of_integer_cols = list(data.iloc[:, 4:-4].select_dtypes('Int64').columns.values.tolist())

number_of_trackers = len(data[data['tracker'] == 1])
number_of_non_trackers = len(data[data['tracker'] == 0])
summary_table['ratio_tracker'] = summary_table['trackers'] / number_of_trackers
summary_table['ratio_non_tracker'] = summary_table['non_trackers'] / number_of_non_trackers

impute_col_list_t = summary_table[summary_table['ratio_tracker'] > 0.4].header_name.values.tolist()
impute_col_list_nt = summary_table[summary_table['ratio_non_tracker'] > 0.4].header_name.values.tolist()

for header in impute_col_list_t:
    impute_value(header, 0)

for header in impute_col_list_t:
    impute_value(header, 1)

for header in impute_col_list_nt:
    impute_value(header, 1)

for header in impute_col_list_nt:
    impute_value(header, 0)

data.to_parquet(
    "../../data/processed/chrome/08_12_2022/train_set_01_processed.parquet.gzip", compression="gzip"
)

print("end")
