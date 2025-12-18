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
"""Functions for generating ECR image URIs for pre-built SageMaker Docker images."""
from __future__ import absolute_import

import json
import logging
import os
from typing import Optional
from packaging.version import Version
import requests


from sagemaker.core.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.core.training_compiler_config import TrainingCompilerConfig
from sagemaker.core.fw_utils import (
    GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY,
    GRAVITON_ALLOWED_FRAMEWORKS,
)
from sagemaker.core.common_utils import _botocore_resolver, get_instance_type_family

logger = logging.getLogger(__name__)

ECR_URI_TEMPLATE = "{registry}.dkr.{hostname}/{repository}"
HUGGING_FACE_FRAMEWORK = "huggingface"
HUGGING_FACE_LLM_FRAMEWORK = "huggingface-llm"
HUGGING_FACE_TEI_GPU_FRAMEWORK = "huggingface-tei"
HUGGING_FACE_TEI_CPU_FRAMEWORK = "huggingface-tei-cpu"
HUGGING_FACE_LLM_NEURONX_FRAMEWORK = "huggingface-llm-neuronx"
XGBOOST_FRAMEWORK = "xgboost"
SKLEARN_FRAMEWORK = "sklearn"
TRAINIUM_ALLOWED_FRAMEWORKS = "pytorch"
INFERENCE_GRAVITON = "inference_graviton"
DATA_WRANGLER_FRAMEWORK = "data-wrangler"
STABILITYAI_FRAMEWORK = "stabilityai"
SAGEMAKER_TRITONSERVER_FRAMEWORK = "sagemaker-tritonserver"


def _get_image_tag(
    container_version,
    distributed,
    final_image_scope,
    framework,
    inference_tool,
    instance_type,
    processor,
    py_version,
    tag_prefix,
    version,
):
    """Return image tag based on framework, container, and compute configuration(s)."""
    instance_type_family = get_instance_type_family(instance_type)
    if framework in (XGBOOST_FRAMEWORK, SKLEARN_FRAMEWORK):
        if instance_type_family and final_image_scope == INFERENCE_GRAVITON:
            _validate_arg(
                instance_type_family,
                GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY,
                "instance type",
            )
        if (
            instance_type_family in GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY
            or final_image_scope == INFERENCE_GRAVITON
        ):
            version_to_arm64_tag_mapping = {
                "xgboost": {
                    "1.5-1": "1.5-1-arm64",
                    "1.3-1": "1.3-1-arm64",
                },
                "sklearn": {
                    "1.0-1": "1.0-1-arm64-cpu-py3",
                },
            }
            tag = version_to_arm64_tag_mapping[framework][version]
        else:
            tag = _format_tag(tag_prefix, processor, py_version, container_version, inference_tool)
    else:
        tag = _format_tag(tag_prefix, processor, py_version, container_version, inference_tool)

        if instance_type is not None and _should_auto_select_container_version(
            instance_type, distributed
        ):
            container_versions = {
                "tensorflow-2.3-gpu-py37": "cu110-ubuntu18.04-v3",
                "tensorflow-2.3.1-gpu-py37": "cu110-ubuntu18.04",
                "tensorflow-2.3.2-gpu-py37": "cu110-ubuntu18.04",
                "tensorflow-1.15-gpu-py37": "cu110-ubuntu18.04-v8",
                "tensorflow-1.15.4-gpu-py37": "cu110-ubuntu18.04",
                "tensorflow-1.15.5-gpu-py37": "cu110-ubuntu18.04",
                "mxnet-1.8-gpu-py37": "cu110-ubuntu16.04-v1",
                "mxnet-1.8.0-gpu-py37": "cu110-ubuntu16.04",
                "pytorch-1.6-gpu-py36": "cu110-ubuntu18.04-v3",
                "pytorch-1.6.0-gpu-py36": "cu110-ubuntu18.04",
                "pytorch-1.6-gpu-py3": "cu110-ubuntu18.04-v3",
                "pytorch-1.6.0-gpu-py3": "cu110-ubuntu18.04",
            }
            key = "-".join([framework, tag])
            if key in container_versions:
                tag = "-".join([tag, container_versions[key]])

    # Triton images don't have a trailing -gpu tag. Only -cpu images do.
    if framework == SAGEMAKER_TRITONSERVER_FRAMEWORK:
        if processor == "gpu":
            tag = tag.rstrip("-gpu")

    return tag


def _config_for_framework_and_scope(framework, image_scope, accelerator_type=None):
    """Loads the JSON config for the given framework and image scope."""
    config = config_for_framework(framework)

    if accelerator_type:
        _validate_accelerator_type(accelerator_type)

        if image_scope not in ("eia", "inference"):
            logger.warning(
                "Elastic inference is for inference only. Ignoring image scope: %s.",
                image_scope,
            )
        image_scope = "eia"

    available_scopes = config.get("scope", list(config.keys()))

    if len(available_scopes) == 1:
        if image_scope and image_scope != available_scopes[0]:
            logger.warning(
                "Defaulting to only supported image scope: %s. Ignoring image scope: %s.",
                available_scopes[0],
                image_scope,
            )
        image_scope = available_scopes[0]

    if not image_scope and "scope" in config and set(available_scopes) == {"training", "inference"}:
        logger.info(
            "Same images used for training and inference. Defaulting to image scope: %s.",
            available_scopes[0],
        )
        image_scope = available_scopes[0]

    _validate_arg(image_scope, available_scopes, "image scope")
    return config if "scope" in config else config[image_scope]


def _validate_instance_deprecation(framework, instance_type, version):
    """Check if instance type is deprecated for a certain framework with a certain version"""
    if get_instance_type_family(instance_type) == "p2":
        if (framework == "pytorch" and Version(version) >= Version("1.13")) or (
            framework == "tensorflow" and Version(version) >= Version("2.12")
        ):
            raise ValueError(
                "P2 instances have been deprecated for sagemaker jobs starting PyTorch 1.13 and TensorFlow 2.12"
                "For information about supported instance types please refer to "
                "https://aws.amazon.com/sagemaker/pricing/"
            )


def _validate_for_suppported_frameworks_and_instance_type(framework, instance_type):
    """Validate if framework is supported for the instance_type"""
    # Validate for Trainium allowed frameworks
    if (
        instance_type is not None
        and "trn" in instance_type
        and framework not in TRAINIUM_ALLOWED_FRAMEWORKS
    ):
        _validate_framework(framework, TRAINIUM_ALLOWED_FRAMEWORKS, "framework", "Trainium")

    # Validate for Graviton allowed frameowrks
    if (
        instance_type is not None
        and get_instance_type_family(instance_type) in GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY
        and framework not in GRAVITON_ALLOWED_FRAMEWORKS
    ):
        _validate_framework(framework, GRAVITON_ALLOWED_FRAMEWORKS, "framework", "Graviton")


def config_for_framework(framework):
    """Loads the JSON config for the given framework."""
    fname = os.path.join(os.path.dirname(__file__), "..", "image_uri_config", "{}.json".format(framework))
    with open(fname) as f:
        return json.load(f)


def _get_final_image_scope(framework, instance_type, image_scope):
    """Return final image scope based on provided framework and instance type."""
    if (
        framework in GRAVITON_ALLOWED_FRAMEWORKS
        and get_instance_type_family(instance_type) in GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY
    ):
        return INFERENCE_GRAVITON
    if image_scope is None and framework in (XGBOOST_FRAMEWORK, SKLEARN_FRAMEWORK):
        # Preserves backwards compatibility with XGB/SKLearn configs which no
        # longer define top-level "scope" keys after introducing support for
        # Graviton inference. Training and inference configs for XGB/SKLearn are
        # identical, so default to training.
        return "training"
    return image_scope


def _get_inference_tool(inference_tool, instance_type):
    """Extract the inference tool name from instance type."""
    if not inference_tool:
        instance_type_family = get_instance_type_family(instance_type)
        if instance_type_family.startswith("inf") or instance_type_family.startswith("trn"):
            return "neuron"
    return inference_tool


def _get_latest_versions(list_of_versions):
    """Extract the latest version from the input list of available versions."""
    return sorted(list_of_versions, reverse=True)[0]


def _get_latest_version(framework, version, image_scope):
    """Get the latest version from the input framework"""
    if version:
        return version
    try:
        framework_config = config_for_framework(framework)
    except FileNotFoundError:
        raise ValueError("Invalid framework {}".format(framework))

    if not framework_config:
        raise ValueError("Invalid framework {}".format(framework))

    if not version:
        version = _fetch_latest_version_from_config(framework_config, image_scope)
    return version


def _validate_accelerator_type(accelerator_type):
    """Raises a ``ValueError`` if ``accelerator_type`` is invalid."""
    if not accelerator_type.startswith("ml.eia") and accelerator_type != "local_sagemaker_notebook":
        raise ValueError(
            "Invalid SageMaker Elastic Inference accelerator type: {}. "
            "See https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html".format(accelerator_type)
        )


def _validate_version_and_set_if_needed(version, config, framework, image_scope):
    """Checks if the framework/algorithm version is one of the supported versions."""
    if not config:
        config = config_for_framework(framework)
    available_versions = list(config["versions"].keys())
    aliased_versions = list(config.get("version_aliases", {}).keys())
    if len(available_versions) == 1 and version not in aliased_versions:
        return available_versions[0]
    if not version:
        version = _get_latest_version(framework, version, image_scope)
    _validate_arg(version, available_versions + aliased_versions, "{} version".format(framework))
    return version


def _version_for_config(version, config):
    """Returns the version string for retrieving a framework version's specific config."""
    if "version_aliases" in config:
        if version in config["version_aliases"].keys():
            return config["version_aliases"][version]

    return version


def _registry_from_region(region, registry_dict):
    """Returns the ECR registry (AWS account number) for the given region."""
    _validate_arg(region, registry_dict.keys(), "region")
    return registry_dict[region]


def _processor(instance_type, available_processors, serverless_inference_config=None):
    """Returns the processor type for the given instance type."""
    if not available_processors:
        logger.info("Ignoring unnecessary instance type: %s.", instance_type)
        return None

    if len(available_processors) == 1 and not instance_type:
        logger.info("Defaulting to only supported image scope: %s.", available_processors[0])
        return available_processors[0]

    if serverless_inference_config is not None:
        logger.info("Defaulting to CPU type when using serverless inference")
        return "cpu"

    if not instance_type:
        raise ValueError(
            "Empty SageMaker instance type. For options, see: "
            "https://aws.amazon.com/sagemaker/pricing/instance-types"
        )

    if instance_type.startswith("local"):
        processor = "cpu" if instance_type == "local" else "gpu"
    elif instance_type.startswith("neuron"):
        processor = "neuron"
    else:
        # looks for either "ml.<family>.<size>" or "ml_<family>"
        family = get_instance_type_family(instance_type)
        if family:
            # For some frameworks, we have optimized images for specific families, e.g c5 or p3.
            # In those cases, we use the family name in the image tag. In other cases, we use
            # 'cpu' or 'gpu'.
            if family in available_processors:
                processor = family
            elif family.startswith("inf"):
                processor = "inf"
            elif family.startswith("trn"):
                processor = "trn"
            elif family[0] in ("g", "p"):
                processor = "gpu"
            else:
                processor = "cpu"
        else:
            raise ValueError(
                "Invalid SageMaker instance type: {}. For options, see: "
                "https://aws.amazon.com/sagemaker/pricing/instance-types".format(instance_type)
            )

    _validate_arg(processor, available_processors, "processor")
    return processor


def _should_auto_select_container_version(instance_type, distributed):
    """Returns a boolean that indicates whether to use an auto-selected container version."""
    p4d = False
    if instance_type:
        # looks for either "ml.<family>.<size>" or "ml_<family>"
        family = get_instance_type_family(instance_type)
        if family:
            p4d = family == "p4d"

    return p4d or distributed


def _validate_py_version_and_set_if_needed(py_version, version_config, framework):
    """Checks if the Python version is one of the supported versions."""
    if "repository" in version_config:
        available_versions = version_config.get("py_versions")
    else:
        available_versions = list(version_config.keys())

    if not available_versions:
        if py_version:
            logger.info("Ignoring unnecessary Python version: %s.", py_version)
        return None

    if py_version is None and defaults.SPARK_NAME == framework:
        return None

    if py_version is None and len(available_versions) == 1:
        logger.info("Defaulting to only available Python version: %s", available_versions[0])
        return available_versions[0]

    _validate_arg(py_version, available_versions, "Python version")
    return py_version


def _validate_arg(arg, available_options, arg_name):
    """Checks if the arg is in the available options, and raises a ``ValueError`` if not."""
    if arg not in available_options:
        raise ValueError(
            "Unsupported {arg_name}: {arg}. You may need to upgrade your SDK version "
            "(pip install -U sagemaker) for newer {arg_name}s. Supported {arg_name}(s): "
            "{options}.".format(arg_name=arg_name, arg=arg, options=", ".join(available_options))
        )


def _validate_framework(framework, allowed_frameworks, arg_name, hardware_name):
    """Checks if the framework is in the allowed frameworks, and raises a ``ValueError`` if not."""
    if framework not in allowed_frameworks:
        raise ValueError(
            f"Unsupported {arg_name}: {framework}. "
            f"Supported {arg_name}(s) for {hardware_name} instances: {allowed_frameworks}."
        )


def _format_tag(tag_prefix, processor, py_version, container_version, inference_tool=None):
    """Creates a tag for the image URI."""
    if inference_tool:
        return "-".join(x for x in (tag_prefix, inference_tool, py_version, container_version) if x)
    return "-".join(x for x in (tag_prefix, processor, py_version, container_version) if x)


def _fetch_latest_version_from_config(  # pylint: disable=R0911
    framework_config: dict, image_scope: Optional[str] = None
) -> Optional[str]:
    """Helper function to fetch the latest version as a string from a framework's config

    Args:
        framework_config (dict): A framework config dict.
        image_scope (str): Scope of the image, eg: training, inference
    Returns:
        Version string if latest version found else None
    """
    if image_scope in framework_config:
        if image_scope_config := framework_config[image_scope]:
            if "version_aliases" in image_scope_config:
                if "latest" in image_scope_config["version_aliases"]:
                    return image_scope_config["version_aliases"]["latest"]
    top_version = None
    bottom_version = None

    if "versions" in framework_config:
        versions = list(framework_config["versions"].keys())
        if len(versions) == 1:
            return versions[0]
        top_version = versions[0]
        bottom_version = versions[-1]
        if top_version == "latest" or bottom_version == "latest":
            return None
    elif (
        image_scope is not None
        and image_scope in framework_config
        and "versions" in framework_config[image_scope]
    ):
        versions = list(framework_config[image_scope]["versions"].keys())
        top_version = versions[0]
        bottom_version = versions[-1]
    elif "processing" in framework_config and "versions" in framework_config["processing"]:
        versions = list(framework_config["processing"]["versions"].keys())
        top_version = versions[0]
        bottom_version = versions[-1]
    if top_version and bottom_version:
        if top_version.endswith(".x") or bottom_version.endswith(".x"):
            top_number = int(top_version[:-2])
            bottom_number = int(bottom_version[:-2])
            max_version = max(top_number, bottom_number)
            return f"{max_version}.x"
        if Version(top_version) >= Version(bottom_version):
            return top_version
        return bottom_version

    return None


def _retrieve_pytorch_uri_inputs_are_all_default(
    version: Optional[str] = None,
    py_version: Optional[str] = None,
    instance_type: Optional[str] = None,
    accelerator_type: Optional[str] = None,
    image_scope: Optional[str] = None,
    container_version: str = None,
    distributed: bool = False,
    smp: bool = False,
    training_compiler_config: TrainingCompilerConfig = None,
    sdk_version: Optional[str] = None,
    inference_tool: Optional[str] = None,
    serverless_inference_config: ServerlessInferenceConfig = None,
) -> bool:
    """
    Determine if the inputs for _retrieve_pytorch_uri() are all default values.
    """
    return (
        not version
        and not py_version
        and not instance_type
        and not accelerator_type
        and not image_scope
        and not container_version
        and not distributed
        and not smp
        and not training_compiler_config
        and not sdk_version
        and not inference_tool
        and not serverless_inference_config
    )


def _retrieve_latest_pytorch_training_uri(region: str):
    """
    Retrive the URI for the latest PyTorch training image for CPU
    """
    config = config_for_framework("pytorch")
    image_scope = "training"

    latest_version = _fetch_latest_version_from_config(config, image_scope)
    version_config = config[image_scope]["versions"][latest_version]
    py_version = _validate_py_version_and_set_if_needed(None, version_config, None)

    endpoint_data = _botocore_resolver().construct_endpoint("ecr", region)
    if region == "il-central-1" and not endpoint_data:
        endpoint_data = {"hostname": "ecr.{}.amazonaws.com".format(region)}

    registry = _registry_from_region(region, version_config["registries"])
    hostname = endpoint_data["hostname"]
    repo = version_config["repository"]

    tag = _get_image_tag(
        container_version="ec2",
        distributed=False,
        final_image_scope=image_scope,
        framework="pytorch",
        inference_tool="",
        instance_type="",
        processor="cpu",
        py_version=py_version,
        tag_prefix=latest_version,
        version=latest_version,
    )
    if tag:
        repo += ":{}".format(tag)

    return ECR_URI_TEMPLATE.format(registry=registry, hostname=hostname, repository=repo)
