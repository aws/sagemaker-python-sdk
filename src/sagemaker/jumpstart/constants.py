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
"""This module stores constants related to SageMaker JumpStart."""
from __future__ import absolute_import
from typing import Set
from enum import Enum
import boto3
from sagemaker.jumpstart.types import JumpStartLaunchedRegionInfo


JUMPSTART_LAUNCHED_REGIONS: Set[JumpStartLaunchedRegionInfo] = set(
    [
        JumpStartLaunchedRegionInfo(
            region_name="us-west-2",
            content_bucket="jumpstart-cache-prod-us-west-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="us-east-1",
            content_bucket="jumpstart-cache-prod-us-east-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="us-east-2",
            content_bucket="jumpstart-cache-prod-us-east-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-west-1",
            content_bucket="jumpstart-cache-prod-eu-west-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-central-1",
            content_bucket="jumpstart-cache-prod-eu-central-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-north-1",
            content_bucket="jumpstart-cache-prod-eu-north-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="me-south-1",
            content_bucket="jumpstart-cache-prod-me-south-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-south-1",
            content_bucket="jumpstart-cache-prod-ap-south-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-west-3",
            content_bucket="jumpstart-cache-prod-eu-west-3",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="af-south-1",
            content_bucket="jumpstart-cache-prod-af-south-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="sa-east-1",
            content_bucket="jumpstart-cache-prod-sa-east-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-east-1",
            content_bucket="jumpstart-cache-prod-ap-east-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-northeast-2",
            content_bucket="jumpstart-cache-prod-ap-northeast-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-west-2",
            content_bucket="jumpstart-cache-prod-eu-west-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-south-1",
            content_bucket="jumpstart-cache-prod-eu-south-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-northeast-1",
            content_bucket="jumpstart-cache-prod-ap-northeast-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="us-west-1",
            content_bucket="jumpstart-cache-prod-us-west-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-southeast-1",
            content_bucket="jumpstart-cache-prod-ap-southeast-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-southeast-2",
            content_bucket="jumpstart-cache-prod-ap-southeast-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ca-central-1",
            content_bucket="jumpstart-cache-prod-ca-central-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="cn-north-1",
            content_bucket="jumpstart-cache-prod-cn-north-1",
        ),
    ]
)

JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT = {
    region.region_name: region for region in JUMPSTART_LAUNCHED_REGIONS
}
JUMPSTART_REGION_NAME_SET = {region.region_name for region in JUMPSTART_LAUNCHED_REGIONS}

JUMPSTART_BUCKET_NAME_SET = {region.content_bucket for region in JUMPSTART_LAUNCHED_REGIONS}

JUMPSTART_DEFAULT_REGION_NAME = boto3.session.Session().region_name or "us-west-2"

JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY = "models_manifest.json"

INFERENCE = "inference"
TRAINING = "training"
SUPPORTED_JUMPSTART_SCOPES = set([INFERENCE, TRAINING])

INFERENCE_ENTRYPOINT_SCRIPT_NAME = "inference.py"
TRAINING_ENTRYPOINT_SCRIPT_NAME = "transfer_learning.py"


class ModelFramework(str, Enum):
    """Enum class for JumpStart model framework.

    The ML framework as referenced in the prefix of the model ID.
    This value does not necessarily correspond to the container name.
    """

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    MXNET = "mxnet"
    HUGGINGFACE = "huggingface"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    SKLEARN = "sklearn"


class VariableScope(str, Enum):
    """Possible value of the ``scope`` attribute for a hyperparameter or environment variable.

    Used for hosting environment variables and training hyperparameters.
    """

    CONTAINER = "container"
    ALGORITHM = "algorithm"


class JumpStartTag(str, Enum):
    """Enum class for tag keys to apply to JumpStart models."""

    INFERENCE_MODEL_URI = "aws-jumpstart-inference-model-uri"
    INFERENCE_SCRIPT_URI = "aws-jumpstart-inference-script-uri"
