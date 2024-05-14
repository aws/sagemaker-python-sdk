"""Module for artifacts preparation for tensorflow_serving"""

from __future__ import absolute_import
from pathlib import Path
import shutil
from typing import List, Dict, Any

from sagemaker.serve.model_format.mlflow.utils import (
    _get_saved_model_path_for_tensorflow_and_keras_flavor,
    _move_contents,
)
from sagemaker.serve.detector.dependency_manager import capture_dependencies
from sagemaker.serve.validations.check_integrity import (
    generate_secret_key,
    compute_hash,
)
from sagemaker.remote_function.core.serialization import _MetaData


def prepare_for_tf_serving(
    model_path: str,
    shared_libs: List[str],
    dependencies: Dict[str, Any],
) -> str:
    """Prepares the model for serving.

    Args:
        model_path (str): Path to the model directory.
        shared_libs (List[str]): List of shared libraries.
        dependencies (Dict[str, Any]): Dictionary of dependencies.

    Returns:
        str: Secret key.
    """

    _model_path = Path(model_path)
    if not _model_path.exists():
        _model_path.mkdir()
    elif not _model_path.is_dir():
        raise Exception("model_dir is not a valid directory")

    code_dir = _model_path.joinpath("code")
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).parent.joinpath("inference.py"), code_dir)

    shared_libs_dir = _model_path.joinpath("shared_libs")
    shared_libs_dir.mkdir(exist_ok=True)
    for shared_lib in shared_libs:
        shutil.copy2(Path(shared_lib), shared_libs_dir)

    capture_dependencies(dependencies=dependencies, work_dir=code_dir)

    saved_model_bundle_dir = _model_path.joinpath("1")
    saved_model_bundle_dir.mkdir(exist_ok=True)
    mlflow_saved_model_dir = _get_saved_model_path_for_tensorflow_and_keras_flavor(model_path)
    if not mlflow_saved_model_dir:
        raise ValueError("SavedModel is not found for Tensorflow or Keras flavor.")
    _move_contents(src_dir=mlflow_saved_model_dir, dest_dir=saved_model_bundle_dir)

    secret_key = generate_secret_key()
    with open(str(code_dir.joinpath("serve.pkl")), "rb") as f:
        buffer = f.read()
    hash_value = compute_hash(buffer=buffer, secret_key=secret_key)
    with open(str(code_dir.joinpath("metadata.json")), "wb") as metadata:
        metadata.write(_MetaData(hash_value).to_json())

    return secret_key
