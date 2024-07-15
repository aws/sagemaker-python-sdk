# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Shared resources for prepare step of model deployment"""

from __future__ import absolute_import
import logging

from sagemaker.serve.model_server.tgi.prepare import _copy_jumpstart_artifacts
from sagemaker.serve.utils.local_hardware import _check_disk_space, _check_docker_disk_usage

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
from sagemaker.remote_function.core.serialization import _MetaData

logger = logging.getLogger(__name__)


def _create_dir_structure(model_path: str) -> tuple:
    """Create the expected model directory structure for the Multi Model server"""
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.mkdir(parents=True)
    elif not model_path.is_dir():
        raise ValueError("model_dir is not a valid directory")

    code_dir = model_path.joinpath("code")
    code_dir.mkdir(exist_ok=True, parents=True)

    _check_disk_space(model_path)
    _check_docker_disk_usage()

    return model_path, code_dir


def prepare_mms_js_resources(
    model_path: str,
    js_id: str,
    shared_libs: List[str] = None,
    dependencies: str = None,
    model_data: str = None,
) -> tuple:
    """Prepare serving when a JumpStart model id is given

    Args:
        model_path (str) : Argument
        js_id (str): Argument
        shared_libs (List[]) : Argument
        dependencies (str) : Argument
        model_data (str) : Argument

    Returns:
        ( str ) :

    """
    model_path, code_dir = _create_dir_structure(model_path)

    return _copy_jumpstart_artifacts(model_data, js_id, code_dir)


def prepare_for_mms(
    model_path: str,
    shared_libs: List[str],
    dependencies: dict,
    session: Session,
    image_uri: str,
    inference_spec: InferenceSpec = None,
) -> str:
    """Prepares for InferenceSpec using model_path, writes inference.py, and captures dependencies to generate secret_key.

    Args:to
        model_path (str) : Argument
        shared_libs (List[]) : Argument
        dependencies (dict) : Argument
        session (Session) : Argument
        inference_spec (InferenceSpec, optional) : Argument
            (default is None)
    Returns:
        ( str ) : secret_key
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

    shutil.copy2(Path(__file__).parent.joinpath("inference.py"), code_dir)

    logger.info("Finished writing inference.py to code directory")

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
