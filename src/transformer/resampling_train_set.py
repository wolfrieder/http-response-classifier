import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTENC


def smotenc_oversampling(X, y, categorical_features):
    sm = SMOTENC(random_state=10, categorical_features=categorical_features)
    X_resampled_values, y_resampled_values = sm.fit_resample(X, y)
    return X_resampled_values, y_resampled_values


def random_oversampling(X, y):
    ros = RandomOverSampler(random_state=10)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled


if __name__ == "__main__":
    print("resampling")
    train_data = pd.read_parquet("../../data/processed/train_set_0.parquet.gzip")
    test_data = pd.read_parquet("../../data/processed/test_set_0.parquet.gzip")
    X_train, y_train = train_data.iloc[:, :-1], train_data[["tracker"]]
    X_test, y_test = test_data.iloc[:, :-1], test_data[["tracker"]]

    test = train_data.columns.get_indexer(list(train_data.select_dtypes('category').columns.values))

    # X_resampled, y_resampled = smotenc_oversampling(X_train.iloc[:, 0:4], y_train.iloc[:, 0:4], [0,1,2,3])
