"""Summary of MyModule.

Extended discussion of my module.
"""

from __future__ import absolute_import
import os
from pathlib import Path
import shutil
from typing import List

from sagemaker.session import Session
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.detector.dependency_manager import capture_dependencies
from sagemaker.serve.validations.check_integrity import (
    generate_secret_key,
    compute_hash,
)
from sagemaker.serve.validations.check_image_uri import is_1p_image_uri
from sagemaker.remote_function.core.serialization import _MetaData


def prepare_for_torchserve(
    model_path: str,
    shared_libs: List[str],
    dependencies: dict,
    session: Session,
    image_uri: str,
    inference_spec: InferenceSpec = None,
) -> str:
    """This is a one-line summary of the function.

    Args:to
        model_path (str) : Argument
        shared_libs (List[]) : Argument
        dependencies (dict) : Argument
        session (Session) : Argument
        inference_spec (InferenceSpec, optional) : Argument
            (default is None)

    Returns:
        ( str ) :

    """
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.mkdir()
    elif not model_path.is_dir():
        raise Exception("model_dir is not a valid directory")

    if inference_spec:
        inference_spec.prepare(str(model_path))

    code_dir = model_path.joinpath("code")
    code_dir.mkdir(exist_ok=True)
    # https://github.com/aws/sagemaker-python-sdk/issues/4288
    if is_1p_image_uri(image_uri=image_uri) and "xgboost" in image_uri:
        shutil.copy2(Path(__file__).parent.joinpath("xgboost_inference.py"), code_dir)
        os.rename(
            str(code_dir.joinpath("xgboost_inference.py")), str(code_dir.joinpath("inference.py"))
        )
    else:
        shutil.copy2(Path(__file__).parent.joinpath("inference.py"), code_dir)

    shared_libs_dir = model_path.joinpath("shared_libs")
    shared_libs_dir.mkdir(exist_ok=True)
    for shared_lib in shared_libs:
        shutil.copy2(Path(shared_lib), shared_libs_dir)

    capture_dependencies(dependencies=dependencies, work_dir=code_dir)

    secret_key = generate_secret_key()
    with open(str(code_dir.joinpath("serve.pkl")), "rb") as f:
        buffer = f.read()
    hash_value = compute_hash(buffer=buffer, secret_key=secret_key)
    with open(str(code_dir.joinpath("metadata.json")), "wb") as metadata:
        metadata.write(_MetaData(hash_value).to_json())

    return secret_key
