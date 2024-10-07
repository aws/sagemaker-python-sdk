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
"""ImageSpec class module."""
from __future__ import absolute_import

import re
from enum import Enum
from typing import Optional
from packaging.version import Version

from sagemaker import utils
from sagemaker.image_uris import (
    _validate_version_and_set_if_needed,
    _version_for_config,
    _config_for_framework_and_scope,
    _validate_py_version_and_set_if_needed,
    _registry_from_region,
    ECR_URI_TEMPLATE,
    _get_latest_versions,
    _validate_instance_deprecation,
    _get_image_tag,
    _validate_arg,
)

DEFAULT_TOLERATE_MODEL = False


class Framework(Enum):
    """Framework enum class."""

    HUGGING_FACE = "huggingface"
    HUGGING_FACE_NEURON = "huggingface-neuron"
    HUGGING_FACE_NEURON_X = "huggingface-neuronx"
    HUGGING_FACE_LLM = "huggingface-llm"
    HUGGING_FACE_TEI_GPU = "huggingface-tei"
    HUGGING_FACE_TEI_CPU = "huggingface-tei-cpu"
    HUGGING_FACE_LLM_NEURONX = "huggingface-llm-neuronx"
    HUGGING_FACE_TRAINING_COMPILER = "huggingface-training-compiler"
    XGBOOST = "xgboost"
    XG_BOOST_NEO = "xg-boost-neo"
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    PYTORCH_TRAINING_COMPILER = "pytorch-training-compiler"
    DATA_WRANGLER = "data-wrangler"
    STABILITYAI = "stabilityai"
    SAGEMAKER_TRITONSERVER = "sagemaker-tritonserver"


class ImageScope(Enum):
    """ImageScope enum class."""

    TRAINING = "training"
    INFERENCE = "inference"
    INFERENCE_GRAVITON = "inference-graviton"


class Processor(Enum):
    """Processor enum class."""

    INF = "inf"
    NEURON = "neuron"
    GPU = "gpu"
    CPU = "cpu"
    TRN = "trn"


class ImageSpec:
    """ImageSpec class to get image URI for a specific framework version.

    Attributes:
        framework (Framework): The name of the framework or algorithm.
        processor (Processor): The name of the processor (CPU, GPU, etc.).
        region (str): The AWS region.
        version (str): The framework or algorithm version. This is required if there is
            more than one supported version for the given framework or algorithm.
        py_version (str): The Python version. This is required if there is
            more than one supported Python version for the given framework version.
        instance_type (str): The SageMaker instance type. For supported types, see
            https://aws.amazon.com/sagemaker/pricing. This is required if
            there are different images for different processor types.
        accelerator_type (str): Elastic Inference accelerator type. For more, see
            https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
        image_scope (str): The image type, i.e. what it is used for.
            Valid values: "training", "inference", "inference_graviton", "eia".
            If ``accelerator_type`` is set, ``image_scope`` is ignored.
        container_version (str): the version of docker image.
            Ideally the value of parameter should be created inside the framework.
            For custom use, see the list of supported container versions:
            https://github.com/aws/deep-learning-containers/blob/master/available_images.md
            (default: None).
        distribution (dict): A dictionary with information on how to run distributed training
        sdk_version (str): the version of python-sdk that will be used in the image retrieval.
            (default: None).
        inference_tool (str): the tool that will be used to aid in the inference.
            Valid values: "neuron, neuronx, None"
            (default: None).
    """

    def __init__(
        self,
        framework: Framework,
        processor: Optional[Processor] = Processor.CPU,
        region: Optional[str] = "us-west-2",
        version=None,
        py_version=None,
        instance_type=None,
        accelerator_type=None,
        image_scope: ImageScope = ImageScope.TRAINING,
        container_version=None,
        distribution=None,
        base_framework_version=None,
        sdk_version=None,
        inference_tool=None,
    ):
        self.framework = framework
        self.processor = processor
        self.version = version
        self.image_scope = image_scope
        self.instance_type = instance_type
        self.py_version = py_version
        self.region = region
        self.accelerator_type = accelerator_type
        self.container_version = container_version
        self.distribution = distribution
        self.base_framework_version = base_framework_version
        self.sdk_version = sdk_version
        self.inference_tool = inference_tool

    def update_image_spec(self, **kwargs):
        """Update the ImageSpec object with the given arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def retrieve(self) -> str:
        """Retrieves the ECR URI for the Docker image matching the given arguments.

        Returns:
            str: The ECR URI for the corresponding SageMaker Docker image.

        Raises:
            NotImplementedError: If the scope is not supported.
            ValueError: If the combination of arguments specified is not supported or
                any PipelineVariable object is passed in.
            VulnerableJumpStartModelError: If any of the dependencies required by the script have
                known security vulnerabilities.
            DeprecatedJumpStartModelError: If the version of the model is deprecated.
        """
        config = _config_for_framework_and_scope(
            self.framework.value, self.image_scope.value, self.accelerator_type
        )
        original_version = self.version
        try:
            version = _validate_version_and_set_if_needed(
                self.version, config, self.framework.value
            )
        except ValueError:
            version = None
        if not version:
            version = self._fetch_latest_version_from_config(config)

        version_config = config["versions"][_version_for_config(version, config)]

        if "huggingface" in self.framework.value:
            if version_config.get("version_aliases"):
                full_base_framework_version = version_config["version_aliases"].get(
                    self.base_framework_version, self.base_framework_version
                )
            _validate_arg(
                full_base_framework_version, list(version_config.keys()), "base framework"
            )
            version_config = version_config.get(full_base_framework_version)

        self.py_version = _validate_py_version_and_set_if_needed(
            self.py_version, version_config, self.framework.value
        )
        version_config = version_config.get(self.py_version) or version_config

        registry = _registry_from_region(self.region, version_config["registries"])
        endpoint_data = utils._botocore_resolver().construct_endpoint("ecr", self.region)
        if self.region == "il-central-1" and not endpoint_data:
            endpoint_data = {"hostname": "ecr.{}.amazonaws.com".format(self.region)}
        hostname = endpoint_data["hostname"]

        repo = version_config["repository"]

        # if container version is available in .json file, utilize that
        if version_config.get("container_version"):
            self.container_version = version_config["container_version"][self.processor.value]

        # Append sdk version in case of trainium instances
        if repo in ["pytorch-training-neuron"]:
            if not self.sdk_version:
                sdk_version = _get_latest_versions(version_config["sdk_versions"])
            self.container_version = self.sdk_version + "-" + self.container_version

        if self.framework == Framework.HUGGING_FACE:
            pt_or_tf_version = (
                re.compile("^(pytorch|tensorflow)(.*)$").match(self.base_framework_version).group(2)
            )
            _version = original_version

            if repo in [
                "huggingface-pytorch-trcomp-training",
                "huggingface-tensorflow-trcomp-training",
            ]:
                _version = version
            if repo in [
                "huggingface-pytorch-inference-neuron",
                "huggingface-pytorch-inference-neuronx",
            ]:
                if not sdk_version:
                    self.sdk_version = _get_latest_versions(version_config["sdk_versions"])
                self.container_version = self.sdk_version + "-" + self.container_version
                if config.get("version_aliases").get(original_version):
                    _version = config.get("version_aliases")[original_version]
                if (
                    config.get("versions", {})
                    .get(_version, {})
                    .get("version_aliases", {})
                    .get(self.base_framework_version, {})
                ):
                    _base_framework_version = config.get("versions")[_version]["version_aliases"][
                        self.base_framework_version
                    ]
                    pt_or_tf_version = (
                        re.compile("^(pytorch|tensorflow)(.*)$")
                        .match(_base_framework_version)
                        .group(2)
                    )

            tag_prefix = f"{pt_or_tf_version}-transformers{_version}"
        else:
            tag_prefix = version_config.get("tag_prefix", version)

        if repo == f"{self.framework.value}-inference-graviton":
            self.container_version = f"{self.container_version}-sagemaker"
        _validate_instance_deprecation(self.framework, self.instance_type, version)

        tag = _get_image_tag(
            self.container_version,
            self.distribution,
            self.image_scope.value,
            self.framework.value,
            self.inference_tool,
            self.instance_type,
            self.processor.value,
            self.py_version,
            tag_prefix,
            version,
        )

        if tag:
            repo += ":{}".format(tag)

        return ECR_URI_TEMPLATE.format(registry=registry, hostname=hostname, repository=repo)

    def _fetch_latest_version_from_config(self, framework_config: dict) -> str:
        """Fetches the latest version from the framework config."""
        if self.image_scope.value in framework_config:
            if image_scope_config := framework_config[self.image_scope.value]:
                if version_aliases := image_scope_config["version_aliases"]:
                    if latest_version := version_aliases["latest"]:
                        return latest_version
        versions = list(framework_config["versions"].keys())
        top_version = versions[0]
        bottom_version = versions[-1]

        if Version(top_version) >= Version(bottom_version):
            return top_version
        return bottom_version
