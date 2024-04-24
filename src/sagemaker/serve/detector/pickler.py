"""Save the object using cloudpickle"""

from __future__ import absolute_import
from typing import Any
from pathlib import Path
import cloudpickle

PKL_FILE_NAME = "serve.pkl"


def save_pkl(save_path: Path, obj: Any):
    """Save obj with cloudpickle under save_path"""
    if not save_path.exists():
        save_path.mkdir(parents=True)
    with open(save_path.joinpath(PKL_FILE_NAME), mode="wb") as file:
        cloudpickle.dump(obj, file)


def save_xgboost(save_path: Path, xgb_model: Any):
    """Save xgboost model to json format using save_model"""
    if not save_path.exists():
        save_path.mkdir(parents=True)
    xgb_model.save_model(str(save_path.joinpath("model.json")))


def load_xgboost_from_json(model_save_path: str, class_name: str):
    """Load xgboost model from json format"""
    try:
        kls = _get_class_from_name(class_name=class_name)
        xgb_model = kls()
        xgb_model.load_model(model_save_path)
        return xgb_model
    except Exception as e:
        raise ValueError(
            (
                "Unable to instantiate %s due to %s, please provide"
                "your custom code for loading the model with InferenceSpec"
            )
            % (class_name, e)
        )


def _get_class_from_name(class_name: str):
    """Given a full class name like xgboost.sklearn.XGBClassifier, return the class"""
    parts = class_name.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)

    return m
