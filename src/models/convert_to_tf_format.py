from typing import List, Union
from sklearn.base import BaseEstimator
from onnx import ModelProto
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle
import os
# import tensorflow as tf
# from onnx_tf.backend import prepare


def get_sklearn_model(path: str) -> BaseEstimator:
    """
    Load a scikit-learn model from a given path.

    Parameters
    ----------
    path : str
        The path to the saved model.

    Returns
    -------
    BaseEstimator
        The loaded scikit-learn model.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def create_path(browser: str, date: str, model_name: str) -> str:
    """
    Create a path for the model.

    Parameters
    ----------
    browser : str
        The browser name.
    date : str
        The date in the format "dd_mm_yyyy".
    model_name : str
        The name of the model.

    Returns
    -------
    str
        The path to the model.
    """
    starting_path = "../../models"
    path = os.path.join(starting_path, browser, date, model_name)
    return path


def convert_to_onnx(
    browser: str, date: str, model_name: str, num_features: int
) -> List[Union[BaseEstimator, ModelProto]]:
    """
    Convert a scikit-learn model to ONNX format.

    Parameters
    ----------
    browser : str
        The browser name.
    date : str
        The date in the format "dd_mm_yyyy".
    model_name : str
        The name of the model.
    num_features : int
        The number of features in the model's input.

    Returns
    -------
    List[Union[BaseEstimator, ModelProto]]
        A list containing the scikit-learn model and its ONNX representation.
    """
    path = create_path(browser, date, model_name)
    sklearn_model = get_sklearn_model(path)
    onnx_model = convert_sklearn(
        sklearn_model,
        initial_types=[("int_input", FloatTensorType([None, num_features]))],
        verbose=True,
    )
    return [sklearn_model, onnx_model]


if __name__ == "__main__":
    converted_model = convert_to_onnx(
        "chrome", "08_12_2022", "decision_tree_binary.sav", 150
    )

    onnx_model = converted_model[1]
    # tf_rep = prepare(onnx_model)
    # tf_rep.export_graph("decision_tree_tf")
