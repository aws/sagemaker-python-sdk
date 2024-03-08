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
