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
from sagemaker.jumpstart.types import JumpStartLaunchedRegionInfo


JUMPSTART_LAUNCHED_REGIONS: Set[JumpStartLaunchedRegionInfo] = set()

JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT = {
    region.region_name: region for region in JUMPSTART_LAUNCHED_REGIONS
}
JUMPSTART_REGION_NAME_SET = {region.region_name for region in JUMPSTART_LAUNCHED_REGIONS}

JUMPSTART_DEFAULT_REGION_NAME = boto3.session.Session().region_name

JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY = "models_manifest.json"
