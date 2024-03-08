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

from botocore.exceptions import ClientError
import pytest

from sagemaker.serve.utils import hardware_detector

REGION = "us-west-2"
VALID_INSTANCE_TYPE = "ml.g5.48xlarge"
INVALID_INSTANCE_TYPE = "fl.c5.57xxlarge"
EXPECTED_INSTANCE_GPU_INFO = (8, 196608)


def test_get_gpu_info_success(sagemaker_session, boto_session):
    boto_session.client("ec2").describe_instance_types.return_value = {
        "InstanceTypes": [
            {
                "GpuInfo": {
                    "Gpus": [
                        {
                            "Name": "A10G",
                            "Manufacturer": "NVIDIA",
                            "Count": 8,
                            "MemoryInfo": {"SizeInMiB": 24576},
                        }
                    ],
                    "TotalGpuMemoryInMiB": 196608,
                },
            }
        ]
    }

    instance_gpu_info = hardware_detector._get_gpu_info(VALID_INSTANCE_TYPE, sagemaker_session)

    boto_session.client("ec2").describe_instance_types.assert_called_once_with(
        InstanceTypes=["g5.48xlarge"]
    )
    assert instance_gpu_info == EXPECTED_INSTANCE_GPU_INFO


def test_get_gpu_info_throws(sagemaker_session, boto_session):
    boto_session.client("ec2").describe_instance_types.return_value = {"InstanceTypes": [{}]}

    with pytest.raises(ValueError):
        hardware_detector._get_gpu_info(INVALID_INSTANCE_TYPE, sagemaker_session)


def test_get_gpu_info_describe_instance_types_throws(sagemaker_session, boto_session):
    boto_session.client("ec2").describe_instance_types.side_effect = ClientError(
        {
            "Error": {
                "Code": "InvalidInstanceType",
                "Message": f"An error occurred (InvalidInstanceType) when calling the DescribeInstanceTypes "
                f"operation: The following supplied instance types do not exist: [{INVALID_INSTANCE_TYPE}]",
            }
        },
        "DescribeInstanceTypes",
    )

    with pytest.raises(ValueError):
        hardware_detector._get_gpu_info(INVALID_INSTANCE_TYPE, sagemaker_session)


def test_get_gpu_info_fallback_success():
    fallback_instance_gpu_info = hardware_detector._get_gpu_info_fallback(
        VALID_INSTANCE_TYPE, REGION
    )

    assert fallback_instance_gpu_info == EXPECTED_INSTANCE_GPU_INFO


def test_get_gpu_info_fallback_throws():
    with pytest.raises(ValueError):
        hardware_detector._get_gpu_info_fallback(INVALID_INSTANCE_TYPE, REGION)


def test_format_instance_type_success():
    formatted_instance_type = hardware_detector._format_instance_type(VALID_INSTANCE_TYPE)

    assert formatted_instance_type == "g5.48xlarge"


def test_format_instance_type_without_ml_success():
    formatted_instance_type = hardware_detector._format_instance_type("g5.48xlarge")

    assert formatted_instance_type == "g5.48xlarge"
