"""Summary of MyModule.

Extended discussion of my module.
"""

from __future__ import absolute_import
import os
from pathlib import Path
import shutil
from typing import List

from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.detector.dependency_manager import capture_dependencies
from sagemaker.serve.validations.check_integrity import (
    generate_secret_key,
    compute_hash,
)
from sagemaker.core.remote_function.core.serialization import _MetaData
from sagemaker.serve.spec.inference_base import CustomOrchestrator, AsyncCustomOrchestrator


def prepare_for_smd(
    model_path: str,
    shared_libs: List[str],
    dependencies: dict,
    inference_spec: InferenceSpec = None,
) -> str:
    """Prepares artifacts for SageMaker model deployment.

    Args:to
        model_path (str) : Argument
        shared_libs (List[]) : Argument
        dependencies (dict) : Argument
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

    if inference_spec and isinstance(inference_spec, InferenceSpec):
        inference_spec.prepare(str(model_path))

    code_dir = model_path.joinpath("code")
    code_dir.mkdir(exist_ok=True)

    if inference_spec and isinstance(inference_spec, (CustomOrchestrator, AsyncCustomOrchestrator)):
        shutil.copy2(Path(__file__).parent.joinpath("custom_execution_inference.py"), code_dir)
        os.rename(
            str(code_dir.joinpath("custom_execution_inference.py")),
            str(code_dir.joinpath("inference.py")),
        )

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
