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
"""Utilites for identifying and analyzing instance gpu hardware"""
from __future__ import absolute_import

import logging

from sagemaker import Session

logger = logging.getLogger(__name__)


fallback_gpu_resource_mapping = {
    "ml.p5.48xlarge": {
        "Count": 8,
        "TotalGpuMemoryInMiB": 655360
    },
    "ml.p4d.24xlarge": {
        "Count": 8,
        "TotalGpuMemoryInMiB": 327680
    },
    "ml.p4de.24xlarge": {
        "Count": 8,
        "TotalGpuMemoryInMiB": 610352
    },
    "ml.p3.2xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 16384
    },
    "ml.p3.8xlarge": {
        "Count": 4,
        "TotalGpuMemoryInMiB": 65536
    },
    "ml.p3.16xlarge": {
        "Count": 8,
        "TotalGpuMemoryInMiB": 131072
    },
    "ml.p3dn.24xlarge": {
        "Count": 8,
        "TotalGpuMemoryInMiB": 262144
    },
    "ml.p2.xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 12288
    },
    "ml.p2.8xlarge": {
        "Count": 8,
        "TotalGpuMemoryInMiB": 98304
    },
    "ml.p2.16xlarge": {
        "Count": 16,
        "TotalGpuMemoryInMiB": 196608
    },
    "ml.g4dn.xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 16384
    },
    "ml.g4dn.2xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 16384
    },
    "ml.g4dn.4xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 16384
    },
    "ml.g4dn.8xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 16384
    },
    "ml.g4dn.16xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 16384
    },
    "ml.g4dn.12xlarge": {
        "Count": 4,
        "TotalGpuMemoryInMiB": 65536
    },
    "ml.g5n.xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 22888
    },
    "ml.g5.2xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 24576
    },
    "ml.g5.4xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 24576
    },
    "ml.g5.8xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 24576
    },
    "ml.g5.16xlarge": {
        "Count": 1,
        "TotalGpuMemoryInMiB": 24576
    },
    "ml.g5.12xlarge": {
        "Count": 4,
        "TotalGpuMemoryInMiB": 98304
    },
    "ml.g5.24xlarge": {
        "Count": 4,
        "TotalGpuMemoryInMiB": 98304
    },
    "ml.g5.48xlarge": {
        "Count": 8,
        "TotalGpuMemoryInMiB": 196608
    },
}


instance = None


def _get_gpu_info(instance_type: str, session: Session) -> int:
    """Get GPU info for the provided instance"""
    global instance
    ec2_client = session.boto_session.client("ec2")

    split_instance = instance_type.split(".")
    split_instance.pop(0)

    ec2_instance = ".".join(split_instance)

    if instance is not None and instance.get("InstanceType") == ec2_instance:
        instance_info = instance
    else:
        instance_info = ec2_client.describe_instance_types(InstanceTypes=[ec2_instance]).get("InstanceTypes")[0]
        instance = instance_info

    gpus_info = instance_info.get("GpuInfo")
    if gpus_info:
        return gpus_info.get("Gpus")[0].get("Count")
    raise ValueError("Provided instance_type is not GPU enabled.")


def _aggregate_gpu_memory_size(instance_type: str, session: Session) -> int:
    """Get Aggregate GPU Memory Size in MiB for the provided instance"""
    global instance
    ec2_client = session.boto_session.client("ec2")

    split_instance = instance_type.split(".")
    split_instance.pop(0)

    ec2_instance = ".".join(split_instance)

    if instance is not None and instance.get("InstanceType") == ec2_instance:
        instance_info = instance
    else:
        instance_info = ec2_client.describe_instance_types(InstanceTypes=[ec2_instance]).get("InstanceTypes")[0]
        instance = instance_info

    gpus_info = instance_info.get("GpuInfo")
    if gpus_info:
        return gpus_info.get("TotalGpuMemoryInMiB")
    raise ValueError("Provided instance_type is not GPU enabled.")


def _get_gpu_info_fallback(instance_type: str) -> int:
    """Get GPU info for the provided instance fallback"""
    fallback_instance = fallback_gpu_resource_mapping.get(instance_type)
    if fallback_instance is None or not fallback_instance.get("Count"):
        raise ValueError("Provided instance_type is not GPU enabled.")
    return fallback_instance.get("Count")


def _get_aggregate_gpu_memory_size_fallback(instance_type: str) -> int:
    """Get Aggregate GPU Memory Size in MiB for the provided instance fallback"""
    fallback_instance = fallback_gpu_resource_mapping.get(instance_type)
    if fallback_instance is None or not fallback_instance.get("TotalGpuMemoryInMiB"):
        raise ValueError("Provided instance_type is not GPU enabled.")
    return fallback_instance.get("TotalGpuMemoryInMiB")
