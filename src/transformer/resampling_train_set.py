import pandas as pd

import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from src.pipeline_functions.resampling_functions import *


if __name__ == "__main__":
    print("Start")
    train_data = pd.read_parquet(
        "../../data/processed/chrome/08_12_2022/train_set_01.parquet.gzip"
    )
    train_data = train_data.sample(30000, random_state=10)
    X_train, y_train = train_data.iloc[:, :-1], train_data["tracker"].to_numpy()
    del train_data

    columns_as_category = {i: 'category' for i in X_train.columns.values}
    # column_test = X_train.columns.values[6:].tolist()
    #
    # braze3 = [test_new_categories_update(element) for element in column_test]
    # braze3 = list(filter(lambda x: type(x) is dict, braze3))
    # braze3 = {k: v for d in braze3 for k, v in d.items()}

    print("halftime")
    # columns_as_category.update(braze3)
    del columns_as_category["query"]
    del columns_as_category["protocol"]
    X_train.drop(['protocol', 'query'], axis=1, inplace=True)
    X_train = X_train.astype(columns_as_category)

    # [impute_value(value, X_train) for value in X_train.columns.values]

    # test = X_train.columns.get_indexer(list(X_train.iloc[:, 4:].select_dtypes('category').columns.values.tolist())).tolist()
    print('resampling')
    X_resampled, y_resampled = smoten_oversampling(X_train.iloc[:, 4:], y_train)
    print("end")
