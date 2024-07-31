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
"""Placeholder docstring"""
from __future__ import absolute_import

import os
import re
import sys
from ast import literal_eval
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

sys.stderr.write(
    """
===============================
Unsupported installation method
===============================

This version of sagemaker no longer supports installation with `python setup.py install`.

Please use `python -m pip install .` instead.
"""
)

HERE = Path(__file__).parent.absolute()
PYPROJECT = HERE.joinpath("pyproject.toml").read_text(encoding="utf-8")
BUILD_SCRIPT = HERE.joinpath("hatch_build.py").read_text(encoding="utf-8")


def get_dependencies():
    pattern = r"^dependencies = (\[.*?\])$"
    array = re.search(pattern, PYPROJECT, flags=re.MULTILINE | re.DOTALL).group(1)
    return literal_eval(array)


def get_optional_dependencies():
    pattern = r"^def get_optional_dependencies.+"
    function = re.search(pattern, BUILD_SCRIPT, flags=re.MULTILINE | re.DOTALL).group(0)
    identifiers = {}
    exec(function, None, identifiers)
    return identifiers["get_optional_dependencies"](str(HERE))


setup(
    name="sagemaker",
    version=HERE.joinpath("VERSION").read_text().strip(),
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.whl"]},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    install_requires=get_dependencies(),
    extras_require=get_optional_dependencies(),
    entry_points={
        "console_scripts": [
            "sagemaker-upgrade-v2=sagemaker.cli.compatibility.v2.sagemaker_upgrade_v2:main",
        ]
    },
    scripts=[
        "src/sagemaker/serve/model_server/triton/pack_conda_env.sh",
    ],
)
