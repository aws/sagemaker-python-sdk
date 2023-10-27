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
