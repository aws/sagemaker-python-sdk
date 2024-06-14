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
"""Holds the util functions used for the optimize function"""
from __future__ import absolute_import

import re
import time
import logging
from typing import List, Dict, Any, Optional

from sagemaker import Session, Model
from sagemaker.enums import Tag
from sagemaker.fw_utils import _is_gpu_instance
from sagemaker.jumpstart.utils import _extract_image_tag_and_version

# TODO: determine how long optimization jobs take
OPTIMIZE_POLLER_MAX_TIMEOUT_SECS = 300
OPTIMIZE_POLLER_INTERVAL_SECS = 30

logger = logging.getLogger(__name__)


def _poll_optimization_job(job_name: str, sagemaker_session: Session) -> bool:
    """Polls optimization job status until success.

    Args:
        job_name (str): The name of the optimization job.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed.

    Returns:
        bool: Whether the optimization job was successful.
    """
    logger.info("Polling status of optimization job %s", job_name)
    start_time = time.time()
    while time.time() - start_time < OPTIMIZE_POLLER_MAX_TIMEOUT_SECS:
        result = sagemaker_session.sagemaker_client.describe_optimization_job(job_name)
        # TODO: use correct condition to determine whether optimization job is complete
        if result is not None:
            return result
        time.sleep(OPTIMIZE_POLLER_INTERVAL_SECS)


def _is_inferentia_or_trainium(instance_type: Optional[str]) -> bool:
    """Checks whether an instance is compatible with Inferentia.

    Args:
        instance_type (str): The instance type used for the compilation job.

    Returns:
        bool: Whether the given instance type is Inferentia or Trainium.
    """
    if isinstance(instance_type, str):
        match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)
        if match:
            if match[1].startswith("inf") or match[1].startswith("trn"):
                return True
    return False


def _is_compatible_with_optimization_job(
    instance_type: Optional[str], image_uri: Optional[str]
) -> bool:
    """Checks whether an instance is compatible with an optimization job.

    Args:
        instance_type (str): The instance type used for the compilation job.
        image_uri (str): The image URI of the optimization job.

    Returns:
        bool: Whether the given instance type is compatible with an optimization job.
    """
    image_tag, image_version = _extract_image_tag_and_version(image_uri)
    if not image_tag or not image_version:
        return False

    return (
        _is_gpu_instance(instance_type) and "djl-inference:" in image_uri and "-lmi" in image_tag
    ) or (
        _is_inferentia_or_trainium(instance_type)
        and "djl-inference:" in image_uri
        and "-neuronx-s" in image_tag
    )


def _generate_optimized_model(pysdk_model: Model, optimization_response: dict) -> Model:
    """Generates a new optimization model.

    Args:
        pysdk_model (Model): A PySDK model.
        optimization_response (dict): The optimization response.

    Returns:
        Model: A deployable optimized model.
    """
    pysdk_model.image_uri = optimization_response["RecommendedInferenceImage"]
    pysdk_model.env = optimization_response["OptimizationEnvironment"]
    pysdk_model.model_data["S3DataSource"]["S3Uri"] = optimization_response["ModelSource"]["S3"]
    pysdk_model.instance_type = optimization_response["DeploymentInstanceType"]
    pysdk_model.add_tags(
        {"key": Tag.OPTIMIZATION_JOB_NAME, "value": optimization_response["OptimizationJobName"]}
    )

    return pysdk_model


def _is_speculation_enabled(deployment_config: Optional[Dict[str, Any]]) -> bool:
    """Checks whether speculation is enabled for this deployment config.

    Args:
        deployment_config (Dict[str, Any]): A deployment config.

    Returns:
        bool: Whether the speculation is enabled for this deployment config.
    """
    if deployment_config is None:
        return False

    acceleration_configs = deployment_config.get("AccelerationConfigs")
    if acceleration_configs:
        for acceleration_config in acceleration_configs:
            if acceleration_config.get("type").lower() == "speculation" and acceleration_config.get(
                "enabled"
            ):
                return True
    return False


def _extract_supported_deployment_config(
    deployment_configs: Optional[List[Dict[str, Any]]],
    speculation_enabled: Optional[bool] = False,
) -> Optional[Dict[str, Any]]:
    """Extracts supported deployment configurations.

    Args:
        deployment_configs (Optional[List[Dict[str, Any]]]): A list of deployment configurations.
        speculation_enabled (Optional[bool]): Whether speculation is enabled.

    Returns:
        Optional[Dict[str, Any]]: Supported deployment configuration.
    """
    if deployment_configs is None:
        return None

    for deployment_config in deployment_configs:
        image_uri: str = deployment_config.get("DeploymentArgs").get("ImageUri")
        instance_type = deployment_config.get("InstanceType")

        if _is_compatible_with_optimization_job(instance_type, image_uri):
            if speculation_enabled:
                if _is_speculation_enabled(deployment_config):
                    return deployment_config
            else:
                return deployment_config
    return None
