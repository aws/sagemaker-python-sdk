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
"""SageMaker model builder dependency managing module.

This must be kept independent of SageMaker PySDK
"""

from __future__ import absolute_import

from pathlib import Path
import logging
import shutil
import subprocess
import sys


_SUPPORTED_SUFFIXES = [".txt"]
# TODO : Move PKL_FILE_NAME to common location
PKL_FILE_NAME = "serve.pkl"

logger = logging.getLogger(__name__)


def capture_dependencies(dependencies: str, work_dir: Path, capture_all: bool = False):
    """Placeholder docstring"""
    path = work_dir.joinpath("requirements.txt")
    if "auto" in dependencies and dependencies["auto"]:
        command = [
            sys.executable,
            Path(__file__).parent.joinpath("pickle_dependencies.py"),
            "--pkl_path",
            work_dir.joinpath(PKL_FILE_NAME),
            "--dest",
            path,
        ]

        if capture_all:
            command.append("--capture_all")

        subprocess.run(
            command,
            env={"SETUPTOOLS_USE_DISTUTILS": "stdlib"},
            check=True,
        )

    if "requirements" in dependencies:
        _capture_from_customer_provided_requirements(dependencies["requirements"], path)

    if "custom" in dependencies:

        with open(path, "a+") as f:
            for package in dependencies["custom"]:
                f.write(f"{package}\n")


def _capture_from_customer_provided_requirements(requirements_file: str, output_path: Path):
    """Placeholder docstring"""
    input_path = Path(requirements_file)
    if not input_path.is_file() or not _is_valid_requirement_file(input_path):
        raise Exception(f"Path: {requirements_file} to requirements.txt doesn't exist")
    logger.debug("Packaging provided requirements.txt from %s", requirements_file)
    with open(output_path, "a+") as f:
        shutil.copyfileobj(open(input_path, "r"), f)


def _is_valid_requirement_file(path):
    """Placeholder docstring"""
    # In the future, we can also check the if the content of customer provided file has valid format
    for suffix in _SUPPORTED_SUFFIXES:
        if path.name.endswith(suffix):
            return True
    return False


# only required for dev testing
def prepare_wheel(code_artifact_client, whl_dir: str):
    """Placeholder docstring"""
    # pull from code artifact
    input_dict = {
        "domain": "galactus-preview-builds",
        "domainOwner": "661407751302",
        "repository": "dev",
        "format": "pypi",
        "package": "sagemaker",
        "packageVersion": "2.193.1.dev0",
        "asset": "sagemaker-2.193.1.dev0-py2.py3-none-any.whl",
    }

    response = code_artifact_client.get_package_version_asset(**input_dict)
    whl_binary = response.get("asset").read()

    with open(f"{whl_dir}/sagemaker-2.185.1.dev0-py2.py3-none-any.whl", "wb") as binary_file:
        binary_file.write(whl_binary)
