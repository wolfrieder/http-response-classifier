import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTENC, SMOTEN


def smotenc_oversampling(X, y, categorical_features):
    sm = SMOTENC(random_state=10, categorical_features=categorical_features,
                 n_jobs=-1)
    X_resampled_values, y_resampled_values = sm.fit_resample(X, y)
    return X_resampled_values, y_resampled_values


def smoten_oversampling(X, y):
    sm = SMOTEN(random_state=10, n_jobs=-1, sampling_strategy='minority')
    X_resampled_values, y_resampled_values = sm.fit_resample(X, y)
    return X_resampled_values, y_resampled_values


def random_oversampling(X, y):
    ros = RandomOverSampler(random_state=10)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled


def test_new_categories_update(element):
    categories = X_train[element].astype("category").cat.categories.values.tolist()
    try:
        np.array(categories, dtype='int64')
        return {element: "Int64"}
    except (ValueError, OverflowError):
        return None


def impute_value(element):
    if X_train[element].dtype == 'category':
        X_train[element] = X_train[element].cat.add_categories('missing')
        X_train[element].fillna('missing', inplace=True)
    else:
        X_train[element].fillna(91219942022, inplace=True)


if __name__ == "__main__":
    print("Start")
    train_data = pd.read_parquet(
        "../../data/processed/chrome/08_12_2022/train_set_01.parquet.gzip"
    )
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

    [impute_value(value) for value in X_train.columns.values]

    # test = X_train.columns.get_indexer(list(X_train.iloc[:, 4:].select_dtypes('category').columns.values.tolist())).tolist()
    print('resampling')
    X_resampled, y_resampled = smoten_oversampling(X_train.iloc[:, 4:], y_train)
    print("end")
