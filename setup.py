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
from glob import glob
import sys

from setuptools import find_packages, setup


def read(fname):
    """
    Args:
        fname:
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    return read("VERSION").strip()


def read_requirements(filename):
    """Reads requirements file which lists package dependencies.

    Args:
        filename: type(str) Relative file path of requirements.txt file

    Returns:
        list of dependencies extracted from file
    """
    with open(os.path.abspath(filename)) as fp:
        deps = [line.strip() for line in fp.readlines()]
    return deps


# Declare minimal set for installation
required_packages = [
    "attrs>=23.1.0,<24",
    "boto3>=1.33.3,<2.0",
    "cloudpickle==2.2.1",
    "google-pasta",
    "numpy>=1.9.0,<2.0",
    "protobuf>=3.12,<5.0",
    "smdebug_rulesconfig==1.0.1",
    "importlib-metadata>=1.4.0,<7.0",
    "packaging>=20.0",
    "pandas",
    "pathos",
    "schema",
    "PyYAML~=6.0",
    "jsonschema",
    "platformdirs",
    "tblib>=1.7.0,<4",
    "urllib3>=1.26.8,<3.0.0",
    "requests",
    "docker",
    "tqdm",
    "psutil",
]

# Specific use case dependencies
# Keep format of *_requirements.txt to be tracked by dependabot
extras = {
    "local": read_requirements("requirements/extras/local_requirements.txt"),
    "scipy": read_requirements("requirements/extras/scipy_requirements.txt"),
    "feature-processor": read_requirements(
        "requirements/extras/feature-processor_requirements.txt"
    ),
    "huggingface": read_requirements("requirements/extras/huggingface_requirements.txt"),
}
# Meta dependency groups
extras["all"] = [item for group in extras.values() for item in group]
# Tests specific dependencies (do not need to be included in 'all')
test_dependencies = read_requirements("requirements/extras/test_requirements.txt")
# test dependencies are a superset of testing and extra dependencies
test_dependencies.extend(extras["all"])
# remove torch and torchvision if python version is not 3.10/3.11
if sys.version_info.minor != 10 or sys.version_info.minor != 11:
    test_dependencies = [
        module
        for module in test_dependencies
        if not (
            module.startswith("transformers")
            or module.startswith("sentencepiece")
            or module.startswith("torch")
            or module.startswith("torchvision")
        )
    ]

extras["test"] = (test_dependencies,)

setup(
    name="sagemaker",
    version=read_version(),
    description="Open source library for training and deploying models on Amazon SageMaker.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.whl"]},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    long_description=read("README.rst"),
    author="Amazon Web Services",
    url="https://github.com/aws/sagemaker-python-sdk/",
    license="Apache License 2.0",
    keywords="ML Amazon AWS AI Tensorflow MXNet",
    python_requires=">= 3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=required_packages,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "sagemaker-upgrade-v2=sagemaker.cli.compatibility.v2.sagemaker_upgrade_v2:main",
        ]
    },
    scripts=[
        "src/sagemaker/serve/model_server/triton/pack_conda_env.sh",
    ],
)
