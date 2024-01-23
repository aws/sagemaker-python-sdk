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
from mock.mock import patch
import pytest

from sagemaker.serve.utils import hardware_detector

REGION = "us-west-2"
VALID_INSTANCE_TYPE = "ml.p5.48xlarge"
INVALID_INSTANCE_TYPE = "fl.c5.57xxlarge"
DESCRIBE_INSTANCE_TYPE_RESULT = {
    "InstanceTypes": [
        {
            "InstanceType": "g5.48xlarge",
            "CurrentGeneration": True,
            "FreeTierEligible": False,
            "SupportedUsageClasses": [
                "on-demand",
                "spot"
            ],
            "SupportedRootDeviceTypes": [
                "ebs"
            ],
            "SupportedVirtualizationTypes": [
                "hvm"
            ],
            "BareMetal": False,
            "Hypervisor": "nitro",
            "ProcessorInfo": {
                "SupportedArchitectures": [
                    "x86_64"
                ],
                "SustainedClockSpeedInGhz": 3.3,
                "Manufacturer": "AMD"
            },
            "VCpuInfo": {
                "DefaultVCpus": 192,
                "DefaultCores": 96,
                "DefaultThreadsPerCore": 2
            },
            "MemoryInfo": {
                "SizeInMiB": 786432
            },
            "InstanceStorageSupported": True,
            "InstanceStorageInfo": {
                "TotalSizeInGB": 7600,
                "Disks": [
                    {
                        "SizeInGB": 3800,
                        "Count": 2,
                        "Type": "ssd"
                    }
                ],
                "NvmeSupport": "required",
                "EncryptionSupport": "required"
            },
            "EbsInfo": {
                "EbsOptimizedSupport": "default",
                "EncryptionSupport": "supported",
                "EbsOptimizedInfo": {
                    "BaselineBandwidthInMbps": 19000,
                    "BaselineThroughputInMBps": 2375.0,
                    "BaselineIops": 80000,
                    "MaximumBandwidthInMbps": 19000,
                    "MaximumThroughputInMBps": 2375.0,
                    "MaximumIops": 80000
                },
                "NvmeSupport": "required"
            },
            "NetworkInfo": {
                "NetworkPerformance": "100 Gigabit",
                "MaximumNetworkInterfaces": 7,
                "MaximumNetworkCards": 1,
                "DefaultNetworkCardIndex": 0,
                "NetworkCards": [
                    {
                        "NetworkCardIndex": 0,
                        "NetworkPerformance": "100 Gigabit",
                        "MaximumNetworkInterfaces": 7,
                        "BaselineBandwidthInGbps": 100.0,
                        "PeakBandwidthInGbps": 100.0
                    }
                ],
                "Ipv4AddressesPerInterface": 50,
                "Ipv6AddressesPerInterface": 50,
                "Ipv6Supported": True,
                "EnaSupport": "required",
                "EfaSupported": True,
                "EfaInfo": {
                    "MaximumEfaInterfaces": 1
                },
                "EncryptionInTransitSupported": True,
                "EnaSrdSupported": True
            },
            "GpuInfo": {
                "Gpus": [
                    {
                        "Name": "A10G",
                        "Manufacturer": "NVIDIA",
                        "Count": 8,
                        "MemoryInfo": {
                            "SizeInMiB": 24576
                        }
                    }
                ],
                "TotalGpuMemoryInMiB": 196608
            },
            "PlacementGroupInfo": {
                "SupportedStrategies": [
                    "cluster",
                    "partition",
                    "spread"
                ]
            },
            "HibernationSupported": False,
            "BurstablePerformanceSupported": False,
            "DedicatedHostsSupported": True,
            "AutoRecoverySupported": False,
            "SupportedBootModes": [
                "legacy-bios",
                "uefi"
            ],
            "NitroEnclavesSupport": "supported",
            "NitroTpmSupport": "supported",
            "NitroTpmInfo": {
                "SupportedVersions": [
                    "2.0"
                ]
            }
        }
    ]
}


@patch("sagemaker.session.Session")
def test_get_gpu_info_success(session):
    session.boto_session.client("ec2").describe_instance_types.return_value = DESCRIBE_INSTANCE_TYPE_RESULT

    gpu_info = hardware_detector._get_gpu_info(VALID_INSTANCE_TYPE, session)

    assert gpu_info == 8


@patch("sagemaker.session.Session")
def test_get_gpu_info_throws(session):
    session.boto_session.client("ec2").describe_instance_types.return_value = {
        "InstanceTypes": [
            {}
        ]
    }

    with pytest.raises(ValueError):
        hardware_detector._get_gpu_info(INVALID_INSTANCE_TYPE, session)


@patch("sagemaker.session.Session")
def test_aggregate_gpu_memory_size_success(session):
    session.boto_session.client("ec2").describe_instance_types.return_value = DESCRIBE_INSTANCE_TYPE_RESULT

    aggregate_memory_size_in_mib = hardware_detector._aggregate_gpu_memory_size(
        VALID_INSTANCE_TYPE,
        session
    )

    assert aggregate_memory_size_in_mib == 196608


@patch("sagemaker.session.Session")
def test_aggregate_gpu_memory_size_throws(session):
    session.boto_session.client("ec2").describe_instance_types.return_value = {
        "InstanceTypes": [
            {}
        ]
    }

    with pytest.raises(ValueError):
        hardware_detector._aggregate_gpu_memory_size(INVALID_INSTANCE_TYPE, session)


def test_get_gpu_info_fallback_success():
    gpu_info = hardware_detector._get_gpu_info_fallback(VALID_INSTANCE_TYPE)

    assert gpu_info == 8


def test_get_gpu_info_fallback_throws():
    with pytest.raises(ValueError):
        hardware_detector._get_gpu_info_fallback(INVALID_INSTANCE_TYPE)


def test_get_gpu_aggregate_memory_size_in_mib_fallback_success():
    aggregate_memory_size_in_mib = hardware_detector._get_aggregate_gpu_memory_size_fallback(VALID_INSTANCE_TYPE)

    assert aggregate_memory_size_in_mib == 655360


def test_get_gpu_aggregate_memory_size_in_mib_fallback_throws():
    with pytest.raises(ValueError):
        hardware_detector._get_aggregate_gpu_memory_size_fallback(INVALID_INSTANCE_TYPE)
