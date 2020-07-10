# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Functions for generating ECR image URIs for pre-built SageMaker Docker images."""
from __future__ import absolute_import

import json
import os

from sagemaker import utils

ECR_URI_TEMPLATE = "{registry}.dkr.{hostname}/{repository}:{tag}"


def retrieve(framework, region, version=None, py_version=None, instance_type=None):
    """Retrieves the ECR URI for the Docker image matching the given arguments.

    Args:
        framework (str): The name of the framework.
        region (str): The AWS region.
        version (str): The framework version. This is required if there is
            more than one supported version for the given framework.
        py_version (str): The Python version. This is required if there is
            more than one supported Python version for the given framework version.
        instance_type (str): The SageMaker instance type. For supported types, see
            https://aws.amazon.com/sagemaker/pricing/instance-types. This is required if
            there are different images for different processor types.

    Returns:
        str: the ECR URI for the corresponding SageMaker Docker image.

    Raises:
        ValueError: If the framework version, Python version, processor type, or region is
            not supported given the other arguments.
    """
    config = config_for_framework(framework)
    version_config = config["versions"][_version_for_config(version, config, framework)]

    registry = _registry_from_region(region, version_config["registries"])
    hostname = utils._botocore_resolver().construct_endpoint("ecr", region)["hostname"]

    repo = version_config["repository"]

    _validate_py_version(py_version, version_config["py_versions"], framework, version)
    tag = "{}-{}-{}".format(version, _processor(instance_type, config["processors"]), py_version)

    return ECR_URI_TEMPLATE.format(registry=registry, hostname=hostname, repository=repo, tag=tag)


def config_for_framework(framework):
    """Loads the JSON config for the given framework."""
    fname = os.path.join(os.path.dirname(__file__), "image_uri_config", "{}.json".format(framework))
    with open(fname) as f:
        return json.load(f)


def _version_for_config(version, config, framework):
    """Returns the version string for retrieving a framework version's specific config."""
    if "version_aliases" in config:
        if version in config["version_aliases"].keys():
            return config["version_aliases"][version]

    available_versions = config["versions"].keys()
    if version in available_versions:
        return version

    raise ValueError(
        "Unsupported {} version: {}. "
        "You may need to upgrade your SDK version (pip install -U sagemaker) for newer versions. "
        "Supported version(s): {}.".format(framework, version, ", ".join(available_versions))
    )


def _registry_from_region(region, registry_dict):
    """Returns the ECR registry (AWS account number) for the given region."""
    available_regions = registry_dict.keys()
    if region not in available_regions:
        raise ValueError(
            "Unsupported region: {}. You may need to upgrade "
            "your SDK version (pip install -U sagemaker) for newer regions. "
            "Supported region(s): {}.".format(region, ", ".join(available_regions))
        )

    return registry_dict[region]


def _processor(instance_type, available_processors):
    """Returns the processor type for the given instance type."""
    if instance_type.startswith("local"):
        processor = "cpu" if instance_type == "local" else "gpu"
    elif not instance_type.startswith("ml."):
        raise ValueError(
            "Invalid SageMaker instance type: {}. See: "
            "https://aws.amazon.com/sagemaker/pricing/instance-types".format(instance_type)
        )
    else:
        family = instance_type.split(".")[1]
        processor = "gpu" if family[0] in ("g", "p") else "cpu"

    if processor in available_processors:
        return processor

    raise ValueError(
        "Unsupported processor type: {} (for {}). "
        "Supported type(s): {}.".format(processor, instance_type, ", ".join(available_processors))
    )


def _validate_py_version(py_version, available_versions, framework, fw_version):
    """Checks if the Python version is one of the supported versions."""
    if py_version not in available_versions:
        raise ValueError(
            "Unsupported Python version for {} {}: {}. You may need to upgrade "
            "your SDK version (pip install -U sagemaker) for newer versions. "
            "Supported Python version(s): {}.".format(
                framework, fw_version, py_version, ", ".join(available_versions)
            )
        )
