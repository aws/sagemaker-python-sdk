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
import logging
import os

from sagemaker import utils

logger = logging.getLogger(__name__)

ECR_URI_TEMPLATE = "{registry}.dkr.{hostname}/{repository}:{tag}"


def retrieve(
    framework,
    region,
    version=None,
    py_version=None,
    instance_type=None,
    accelerator_type=None,
    image_scope=None,
):
    """Retrieves the ECR URI for the Docker image matching the given arguments.

    Args:
        framework (str): The name of the framework or algorithm.
        region (str): The AWS region.
        version (str): The framework or algorithm version. This is required if there is
            more than one supported version for the given framework or algorithm.
        py_version (str): The Python version. This is required if there is
            more than one supported Python version for the given framework version.
        instance_type (str): The SageMaker instance type. For supported types, see
            https://aws.amazon.com/sagemaker/pricing/instance-types. This is required if
            there are different images for different processor types.
        accelerator_type (str): Elastic Inference accelerator type. For more, see
            https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
        image_scope (str): The image type, i.e. what it is used for.
            Valid values: "training", "inference", "eia". If ``accelerator_type`` is set,
            ``image_scope`` is ignored.

    Returns:
        str: the ECR URI for the corresponding SageMaker Docker image.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    config = _config_for_framework_and_scope(framework, image_scope, accelerator_type)

    version = _validate_version_and_set_if_needed(version, config, framework)
    version_config = config["versions"][_version_for_config(version, config)]

    py_version = _validate_py_version_and_set_if_needed(py_version, version_config)
    version_config = version_config.get(py_version) or version_config

    registry = _registry_from_region(region, version_config["registries"])
    hostname = utils._botocore_resolver().construct_endpoint("ecr", region)["hostname"]

    repo = version_config["repository"]
    tag = _format_tag(version, _processor(instance_type, config.get("processors")), py_version)

    return ECR_URI_TEMPLATE.format(registry=registry, hostname=hostname, repository=repo, tag=tag)


def _config_for_framework_and_scope(framework, image_scope, accelerator_type=None):
    """Loads the JSON config for the given framework and image scope."""
    config = config_for_framework(framework)

    if accelerator_type:
        if image_scope not in ("eia", "inference"):
            logger.warning(
                "Elastic inference is for inference only. Ignoring image scope: %s.", image_scope
            )
        image_scope = "eia"

    _validate_arg("image scope", image_scope, config.get("scope", config.keys()))
    return config if "scope" in config else config[image_scope]


def config_for_framework(framework):
    """Loads the JSON config for the given framework."""
    fname = os.path.join(os.path.dirname(__file__), "image_uri_config", "{}.json".format(framework))
    with open(fname) as f:
        return json.load(f)


def _validate_version_and_set_if_needed(version, config, framework):
    """Checks if the framework/algorithm version is one of the supported versions."""
    available_versions = list(config["versions"].keys())

    if len(available_versions) == 1:
        logger.info(
            "Defaulting to only available framework/algorithm version: %s", available_versions[0]
        )
        return available_versions[0]

    available_versions += list(config.get("version_aliases", {}).keys())
    _validate_arg("{} version".format(framework), version, available_versions)

    return version


def _version_for_config(version, config):
    """Returns the version string for retrieving a framework version's specific config."""
    if "version_aliases" in config:
        if version in config["version_aliases"].keys():
            return config["version_aliases"][version]

    return version


def _registry_from_region(region, registry_dict):
    """Returns the ECR registry (AWS account number) for the given region."""
    _validate_arg("region", region, registry_dict.keys())
    return registry_dict[region]


def _processor(instance_type, available_processors):
    """Returns the processor type for the given instance type."""
    if not available_processors:
        logger.info("Ignoring unnecessary instance type: %s.", instance_type)
        return None

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

    _validate_arg("processor", processor, available_processors)
    return processor


def _validate_py_version_and_set_if_needed(py_version, version_config):
    """Checks if the Python version is one of the supported versions."""
    if "repository" in version_config:
        available_versions = version_config.get("py_versions")
    else:
        available_versions = list(version_config.keys())

    if not available_versions:
        if py_version:
            logger.info("Ignoring unnecessary Python version: %s.", py_version)
        return None

    if py_version is None and len(available_versions) == 1:
        logger.info("Defaulting to only available Python version: %s", available_versions[0])
        return available_versions[0]

    _validate_arg("Python version", py_version, available_versions)
    return py_version


def _validate_arg(arg_name, arg, available_options):
    """Checks if the arg is in the available options, and raises a ``ValueError`` if not."""
    if arg not in available_options:
        raise ValueError(
            "Unsupported {arg_name}: {arg}. You may need to upgrade your SDK version "
            "(pip install -U sagemaker) for newer {arg_name}s. Supported {arg_name}(s): "
            "{options}.".format(arg_name=arg_name, arg=arg, options=", ".join(available_options))
        )


def _format_tag(version, processor, py_version):
    """Creates a tag for the image URI."""
    return "-".join([x for x in (version, processor, py_version) if x])
