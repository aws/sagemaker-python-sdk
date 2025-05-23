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
from __future__ import absolute_import

import pytest

from sagemaker.serve.utils import hardware_detector

REGION = "us-west-2"
VALID_INSTANCE_TYPE = "ml.g5.48xlarge"
INVALID_INSTANCE_TYPE = "fl.c5.57xxlarge"
EXPECTED_INSTANCE_GPU_INFO = (8, 183104)


def test_get_gpu_info_success(sagemaker_session):
    gpu_info = hardware_detector._get_gpu_info(VALID_INSTANCE_TYPE, sagemaker_session)

    assert gpu_info == EXPECTED_INSTANCE_GPU_INFO


def test_get_gpu_info_throws(sagemaker_session):
    with pytest.raises(ValueError):
        hardware_detector._get_gpu_info(INVALID_INSTANCE_TYPE, sagemaker_session)


def test_get_gpu_info_fallback_success():
    gpu_info = hardware_detector._get_gpu_info_fallback(VALID_INSTANCE_TYPE, REGION)

    assert gpu_info == EXPECTED_INSTANCE_GPU_INFO


def test_get_gpu_info_fallback_throws():
    with pytest.raises(ValueError):
        hardware_detector._get_gpu_info_fallback(INVALID_INSTANCE_TYPE, REGION)
