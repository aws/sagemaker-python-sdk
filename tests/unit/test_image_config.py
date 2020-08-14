# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

from sagemaker.image_config import ImageConfig

REPOSITORY_ACCESS_MODE_PLATFORM = "Platform"
REPOSITORY_ACCESS_MODE_VPC = "Vpc"


def test_init_with_defaults():
    image_config = ImageConfig()

    assert image_config.repository_access_mode == REPOSITORY_ACCESS_MODE_PLATFORM


def test_init_with_non_defaults():
    image_config = ImageConfig(repository_access_mode=REPOSITORY_ACCESS_MODE_VPC)

    assert image_config.repository_access_mode == REPOSITORY_ACCESS_MODE_VPC
