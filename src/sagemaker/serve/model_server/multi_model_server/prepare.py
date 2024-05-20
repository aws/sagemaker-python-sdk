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
from pathlib import Path
from typing import List

from sagemaker.serve.model_server.tgi.prepare import _copy_jumpstart_artifacts
from sagemaker.serve.utils.local_hardware import _check_disk_space, _check_docker_disk_usage

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
