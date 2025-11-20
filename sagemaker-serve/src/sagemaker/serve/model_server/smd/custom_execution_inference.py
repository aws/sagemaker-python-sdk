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
"""This module is for SageMaker inference.py."""

from __future__ import absolute_import
import asyncio
import os
import platform
import cloudpickle
import logging
from pathlib import Path
from sagemaker.serve.validations.check_integrity import perform_integrity_check


logger = LOGGER = logging.getLogger("sagemaker")


def initialize_custom_orchestrator():
    """Initializes the custom orchestrator."""
    code_dir = os.getenv("SAGEMAKER_INFERENCE_CODE_DIRECTORY", None)
    serve_path = Path(code_dir).joinpath("serve.pkl")
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


def _pickle_file_integrity_check():
    with open("/opt/ml/model/code/serve.pkl", "rb") as f:
        buffer = f.read()

    metadata_path = Path("/opt/ml/model/code/metadata.json")
    perform_integrity_check(buffer=buffer, metadata_path=metadata_path)


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
