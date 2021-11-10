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
from typing import Set
from sagemaker.jumpstart.types import JumpStartLaunchedRegionInfo


LAUNCHED_REGIONS: Set[JumpStartLaunchedRegionInfo] = set()

REGION_NAME_TO_LAUNCHED_REGION_DICT = {region.region_name: region for region in LAUNCHED_REGIONS}
REGION_NAME_SET = {region.region_name for region in LAUNCHED_REGIONS}
