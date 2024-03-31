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
import subprocess
import sys
import re

_SUPPORTED_SUFFIXES = [".txt"]
# TODO : Move PKL_FILE_NAME to common location
PKL_FILE_NAME = "serve.pkl"

logger = logging.getLogger(__name__)


def capture_dependencies(dependencies: dict, work_dir: Path, capture_all: bool = False):
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

        with open(path, "r") as f:
            autodetect_depedencies = f.read().splitlines()
        autodetect_depedencies.append("sagemaker[huggingface]>=2.199")
    else:
        autodetect_depedencies = ["sagemaker[huggingface]>=2.199"]

    module_version_dict = _parse_dependency_list(autodetect_depedencies)

    if "requirements" in dependencies:
        module_version_dict = _process_customer_provided_requirements(
            requirements_file=dependencies["requirements"], module_version_dict=module_version_dict
        )
    if "custom" in dependencies:
        module_version_dict = _process_custom_dependencies(
            custom_dependencies=dependencies.get("custom"), module_version_dict=module_version_dict
        )
    with open(path, "w") as f:
        for module, version in module_version_dict.items():
            f.write(f"{module}{version}\n")


def _process_custom_dependencies(custom_dependencies: list, module_version_dict: dict):
    """Placeholder docstring"""
    custom_module_version_dict = _parse_dependency_list(custom_dependencies)
    module_version_dict.update(custom_module_version_dict)
    return module_version_dict


def _process_customer_provided_requirements(requirements_file: str, module_version_dict: dict):
    """Placeholder docstring"""
    requirements_file = Path(requirements_file)
    if not requirements_file.is_file() or not _is_valid_requirement_file(requirements_file):
        raise Exception(f"Path: {requirements_file} to requirements.txt doesn't exist")
    logger.debug("Packaging provided requirements.txt from %s", requirements_file)
    with open(requirements_file, "r") as f:
        custom_dependencies = f.read().splitlines()

    module_version_dict.update(_parse_dependency_list(custom_dependencies))
    return module_version_dict


def _is_valid_requirement_file(path):
    """Placeholder docstring"""
    # In the future, we can also check the if the content of customer provided file has valid format
    for suffix in _SUPPORTED_SUFFIXES:
        if path.name.endswith(suffix):
            return True
    return False


def _parse_dependency_list(depedency_list: list) -> dict:
    """Placeholder docstring"""

    # Divide a string into 2 part, first part is the module name
    # and second part is its version constraint or the url
    # checkout tests/unit/sagemaker/serve/detector/test_dependency_manager.py
    # for examples
    pattern = r"^([\w.-]+)(@[^,\n]+|((?:[<>=!~]=?[\w.*-]+,?)+)?)$"

    module_version_dict = {}

    for dependency in depedency_list:
        if dependency.startswith("#"):
            continue
        match = re.match(pattern, dependency)
        if match:
            package = match.group(1)
            # Group 2 is either a URL or version constraint, if present
            url_or_version = match.group(2) if match.group(2) else ""
            module_version_dict.update({package: url_or_version})
        else:
            module_version_dict.update({dependency: ""})

    return module_version_dict
