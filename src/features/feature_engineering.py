import numpy as np
# from scipy.stats import kstest, shapiro, probplot

from src.pipeline_functions.feature_engineering_functions import *

import warnings


def impute_value(element, classification):
    check = (
        summary_table.loc[summary_table.header_name == element, "ratio_tracker"].values[
            0
        ]
        if classification == 1
        else summary_table.loc[
            summary_table.header_name == element, "ratio_non_tracker"
        ].values[0]
    )
    if element in ["content-length", "age"]:
        value = int(
            train_data[train_data["tracker"] == classification][element].median()
        )
        if check < 0.4:
            value = -1
        imputed_values_dict[classification].append({element: value})
        train_data.loc[
            train_data["tracker"] == classification, element
        ] = train_data.loc[train_data["tracker"] == classification, element].fillna(
            value
        )

    if element in list_of_categorical_cols:
        value = (
            train_data[train_data["tracker"] == classification][element].mode().iloc[0]
        )
        if check < 0.4:
            value = "Missing"
            train_data[element].cat.add_categories("Missing", inplace=True)
        imputed_values_dict[classification].append({element: value})
        train_data.loc[
            train_data["tracker"] == classification, element
        ] = train_data.loc[train_data["tracker"] == classification, element].fillna(
            value
        )


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

if __name__ == "__main__":
    train_data = pd.read_parquet(
        "../../data/processed/chrome/08_12_2022/train_set_01_processed.parquet.gzip"
    )
    test_data = pd.read_parquet(
        "../../data/processed/chrome/08_12_2022/test_set_01_processed.parquet.gzip"
    )

    # test_data = pd.read_parquet(
    #     "../../data/processed/brave/08_12_2022/test_set_all_processed.parquet.gzip"
    # )

    # exclude metadata columns
    train_data = train_data.iloc[:, 4:]
    test_data = test_data.iloc[:, 4:]

    for df in [train_data, test_data]:
        df["comb_col_non_tracker"] = df["comb_col_non_tracker"].astype("uint8")
        df["comb_col_tracker"] = df["comb_col_tracker"].astype("uint8")
        # df["header_count"] = df["header_count"].astype("uint8")
        df["tracker"] = df["tracker"].astype("uint8")

    number_of_elements_reduced = np.array(
        [variance_per_column(column, train_data) for column in train_data.iloc[:, :-3].columns]
    )
    summary_table = pd.DataFrame(
        number_of_elements_reduced, columns=["header_name", "trackers", "non_trackers"]
    )
    summary_table["trackers"] = summary_table["trackers"].astype("Int32")
    summary_table["non_trackers"] = summary_table["non_trackers"].astype("float32")

    number_of_trackers = len(train_data[train_data["tracker"] == 1])
    number_of_non_trackers = len(train_data[train_data["tracker"] == 0])
    summary_table["ratio_tracker"] = summary_table["trackers"] / number_of_trackers
    summary_table["ratio_non_tracker"] = (
        summary_table["non_trackers"] / number_of_non_trackers
    )
    summary_table["tracker_na_ratio"] = (
        train_data[train_data["tracker"] == 1].iloc[:, :-3].isnull().mean().values
    )
    summary_table["non_tracker_na_ratio"] = (
        train_data[train_data["tracker"] == 0].iloc[:, :-3].isnull().mean().values
    )

    na_ratio_greater_than_85 = summary_table[
        summary_table["tracker_na_ratio"] >= 0.85
    ].header_name.values.tolist()

    for elem in na_ratio_greater_than_85:
        train_data[f"{elem}_binary"] = np.where(train_data[elem].isnull(), 0, 1)
        test_data[f"{elem}_binary"] = np.where(test_data[elem].isnull(), 0, 1)

    for df in [train_data, test_data]:
        df.drop(na_ratio_greater_than_85, axis=1, inplace=True)
        df.drop(["last-modified", "date"], axis=1, inplace=True)

    list_of_integer_cols = list(
        train_data.select_dtypes("Int64").columns.values.tolist()
    )
    # list_of_float_cols = list(
    #     train_data.select_dtypes("Float64").columns.values.tolist()
    # )

    binary_cols = list(filter(lambda x: "_binary" in x, list_of_integer_cols))
    for elem in binary_cols:
        train_data[elem] = train_data[elem].astype("uint8")
        test_data[elem] = train_data[elem].astype("uint8")

    # Data preprocessing, following lines should be moved
    train_data.replace(" ", np.nan, inplace=True)
    test_data.replace(" ", np.nan, inplace=True)

    train_data.replace("", np.nan, inplace=True)
    test_data.replace("", np.nan, inplace=True)

    # etag
    for df in [train_data, test_data]:
        df.etag = df.etag.astype("object")
        df.etag.replace(to_replace=r"^w\/", value="", regex=True, inplace=True)
        df.etag.replace(to_replace=r"\"", value="", regex=True, inplace=True)
        df.etag = df.etag.astype("category")
        df["etag_length"] = df.etag.apply(len)
        df["etag_length"].fillna(-1, inplace=True)
        df.etag_length = df.etag_length.astype("int16")

    # access-control-allow-origin
    for df in [train_data, test_data]:
        transformed_column, new_category_list = cumulatively_categorise(
            df["access-control-allow-origin"], return_categories_list=True
        )
        df["access-control-allow-origin_cumulative"] = transformed_column
        df["access-control-allow-origin_cumulative"] = df[
            "access-control-allow-origin_cumulative"
        ].astype("category")
        df.drop("access-control-allow-origin", axis=1, inplace=True)

    # age
    for df in [train_data, test_data]:
        df.age = df.age.astype("object")
        df.age.replace(to_replace=r"^0.*\d", value=0, regex=True, inplace=True)
        df.age.replace(to_replace=r"^60;", value=60, regex=True, inplace=True)
        df.age.replace("null", np.nan, inplace=True)
        df.age = df.age.astype("float64")
        df.drop(df[df.age < -2].index, inplace=True)

    # server
    server_values = [
        "nginx",
        "apache",
        "ecacc",
        "ecs",
        "oracle",
        "mt3",
        "microsoft",
        "jetty",
        "ats",
        "openresty",
    ]

    for df in [train_data, test_data]:
        df.server = df.server.astype("object")
        for elem in server_values:
            df.server.replace(
                to_replace=rf"^{elem}.*", value=f"{elem}", regex=True, inplace=True
            )
        transformed_column_server, new_category_list_server = cumulatively_categorise(
            df.server, threshold=0.9, return_categories_list=True
        )
        df.server = transformed_column_server
        df.server = df.server.astype("category")

    # Imputation
    impute_col_list_t = summary_table[
        summary_table["ratio_tracker"] > 0.4
    ].header_name.values.tolist()
    impute_col_list_nt = summary_table[
        summary_table["ratio_non_tracker"] > 0.4
    ].header_name.values.tolist()

    list_of_categorical_cols = list(
        train_data.select_dtypes("category").columns.values.tolist()
    )

    current_header = train_data.columns.tolist()
    impute_col_list_t = list(filter(lambda x: x in current_header, impute_col_list_t))
    impute_col_list_nt = list(filter(lambda x: x in current_header, impute_col_list_nt))

    imputed_values_dict = {0: [], 1: []}

    for label in [0, 1]:
        for header in impute_col_list_t:
            impute_value(header, label)

    for label in [0, 1]:
        for header in impute_col_list_nt:
            impute_value(header, label)

    for label in [0, 1]:
        for elem in imputed_values_dict[label]:
            ((key, value),) = elem.items()
            if key in list_of_categorical_cols:
                if value not in test_data[key].cat.categories:
                    test_data[key].cat.add_categories("Missing", inplace=True)
            test_data.loc[test_data["tracker"] == label, key] = test_data.loc[
                test_data["tracker"] == label, key
            ].fillna(value)

    for df in [train_data, test_data]:
        df["access-control-allow-origin_cumulative"].cat.add_categories(
            "Missing", inplace=True
        )
        df["access-control-allow-origin_cumulative"].fillna("Missing", inplace=True)

    for df in [train_data, test_data]:
        for elem in [
            "pragma",
            "p3p",
            "x-xss-protection",
            "x-content-type-options",
            "strict-transport-security",
            "access-control-allow-credentials",
            "timing-allow-origin",
        ]:
            df[f"{elem}_binary"] = np.where(df[elem].isnull(), 0, 1)
            df[f"{elem}_binary"] = df[f"{elem}_binary"].astype("uint8")
            df.drop(elem, axis=1, inplace=True)

    # put tracker column at the end
    reordered_cols = label_as_last_column(train_data)
    train_data = train_data[reordered_cols]
    test_data = test_data[reordered_cols]

    train_data.to_parquet(
        f"../../data/processed/chrome/08_12_2022/train_set_01_featurized_no_imp.parquet.gzip",
        compression="gzip",
    )

    test_data.to_parquet(
        f"../../data/processed/chrome/08_12_2022/test_set_01_featurized_no_imp.parquet.gzip",
        compression="gzip",
    )
