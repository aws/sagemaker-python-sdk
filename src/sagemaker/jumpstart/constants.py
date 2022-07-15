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
import boto3
from sagemaker.jumpstart.enums import JumpStartScriptScope
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

INFERENCE_ENTRY_POINT_SCRIPT_NAME = "inference.py"
TRAINING_ENTRY_POINT_SCRIPT_NAME = "transfer_learning.py"

SUPPORTED_JUMPSTART_SCOPES = set(scope.value for scope in JumpStartScriptScope)

ENV_VARIABLE_JUMPSTART_CONTENT_BUCKET_OVERRIDE = "AWS_JUMPSTART_CONTENT_BUCKET_OVERRIDE"
ENV_VARIABLE_JUMPSTART_MODEL_ARTIFACT_BUCKET_OVERRIDE = "AWS_JUMPSTART_MODEL_BUCKET_OVERRIDE"
ENV_VARIABLE_JUMPSTART_SCRIPT_ARTIFACT_BUCKET_OVERRIDE = "AWS_JUMPSTART_SCRIPT_BUCKET_OVERRIDE"
ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE = (
    "AWS_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE"
)
ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE = "AWS_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE"

JUMPSTART_RESOURCE_BASE_NAME = "sagemaker-jumpstart"
