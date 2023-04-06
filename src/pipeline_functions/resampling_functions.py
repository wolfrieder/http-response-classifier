import numpy as np
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


def test_new_categories_update(element, X_train):
    categories = X_train[element].astype("category").cat.categories.values.tolist()
    try:
        np.array(categories, dtype='int64')
        return {element: "Int64"}
    except (ValueError, OverflowError):
        return None


def impute_value(element, X_train):
    if X_train[element].dtype == 'category':
        X_train[element] = X_train[element].cat.add_categories('missing')
        X_train[element].fillna('missing', inplace=True)
    else:
        X_train[element].fillna(91219942022, inplace=True)
