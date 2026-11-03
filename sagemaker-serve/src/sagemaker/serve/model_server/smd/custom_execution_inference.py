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
"""This module is for SageMaker inference.py.

This file is copied into the model artifacts as ``code/inference.py`` and runs inside
the SageMaker Distribution serving container, whose installed ``sagemaker`` version is
set by the image rather than by the SDK that built the artifacts. It therefore must not
import from ``sagemaker``: the integrity check is inlined below instead of imported from
``sagemaker.serve.validations.check_integrity`` so that it cannot drift against the
container's copy of that module.
"""

from __future__ import absolute_import
import asyncio
import hashlib
import hmac
import json
import logging
import os
import platform
from pathlib import Path
import cloudpickle

logger = LOGGER = logging.getLogger("sagemaker")

DEFAULT_CODE_DIRECTORY = "/opt/ml/model/code"


def _get_code_directory() -> Path:
    """Returns the directory holding serve.pkl and metadata.json."""
    return Path(os.getenv("SAGEMAKER_INFERENCE_CODE_DIRECTORY") or DEFAULT_CODE_DIRECTORY)


def initialize_custom_orchestrator():
    """Initializes the custom orchestrator."""
    serve_path = _get_code_directory().joinpath("serve.pkl")
    with open(str(serve_path), mode="rb") as pkl_file:
        return cloudpickle.load(pkl_file)


def _run_preflight_diagnostics():
    _py_vs_parity_check()
    _pickle_file_integrity_check()


def _py_vs_parity_check():
    container_py_vs = platform.python_version()
    local_py_vs = os.getenv("LOCAL_PYTHON")

    if not local_py_vs or container_py_vs.split(".")[1] != local_py_vs.split(".")[1]:
        logger.warning(
            f"The local python version {local_py_vs} differs from the python version "
            f"{container_py_vs} on the container. Please align the two to avoid unexpected behavior"
        )


def _perform_integrity_check(buffer: bytes, metadata_path: Path):
    """Validates the integrity of bytes by comparing the SHA-256 hash value."""
    if not Path.exists(metadata_path):
        raise ValueError("Path to metadata.json does not exist")

    with open(str(metadata_path), "rb") as md:
        try:
            expected_hash_value = json.loads(md.read()).get("sha256_hash")
        except json.JSONDecodeError:
            raise ValueError("Corrupt metadata file. It is not a valid json file.")

    if not expected_hash_value:
        raise ValueError("metadata.json does not contain a sha256_hash value")

    actual_hash_value = hashlib.sha256(buffer).hexdigest()

    if not hmac.compare_digest(expected_hash_value, actual_hash_value):
        raise ValueError("Integrity check for the serialized function or data failed.")


def _pickle_file_integrity_check():
    code_directory = _get_code_directory()
    with open(str(code_directory.joinpath("serve.pkl")), "rb") as f:
        buffer = f.read()

    _perform_integrity_check(buffer=buffer, metadata_path=code_directory.joinpath("metadata.json"))


_run_preflight_diagnostics()
custom_orchestrator, _ = initialize_custom_orchestrator()


async def handler(request):
    """Custom service entry point function.

    :param request: raw input from request
    :return: outputs to be send back to client
    """
    if asyncio.iscoroutinefunction(custom_orchestrator.handle):
        return await custom_orchestrator.handle(request.body)
    else:
        return custom_orchestrator.handle(request.body)
