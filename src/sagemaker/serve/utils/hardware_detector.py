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

from botocore.exceptions import ClientError

from sagemaker import Session

logger = logging.getLogger(__name__)

FALLBACK_GPU_RESOURCE_MAPPING = {
    "ml.p5.48xlarge": {"Count": 8, "TotalGpuMemoryInMiB": 655360},
    "ml.p4d.24xlarge": {"Count": 8, "TotalGpuMemoryInMiB": 327680},
    "ml.p4de.24xlarge": {"Count": 8, "TotalGpuMemoryInMiB": 610352},
    "ml.p3.2xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 16384},
    "ml.p3.8xlarge": {"Count": 4, "TotalGpuMemoryInMiB": 65536},
    "ml.p3.16xlarge": {"Count": 8, "TotalGpuMemoryInMiB": 131072},
    "ml.p3dn.24xlarge": {"Count": 8, "TotalGpuMemoryInMiB": 262144},
    "ml.p2.xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 12288},
    "ml.p2.8xlarge": {"Count": 8, "TotalGpuMemoryInMiB": 98304},
    "ml.p2.16xlarge": {"Count": 16, "TotalGpuMemoryInMiB": 196608},
    "ml.g4dn.xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 16384},
    "ml.g4dn.2xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 16384},
    "ml.g4dn.4xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 16384},
    "ml.g4dn.8xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 16384},
    "ml.g4dn.16xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 16384},
    "ml.g4dn.12xlarge": {"Count": 4, "TotalGpuMemoryInMiB": 65536},
    "ml.g5n.xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 22888},
    "ml.g5.2xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 24576},
    "ml.g5.4xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 24576},
    "ml.g5.8xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 24576},
    "ml.g5.16xlarge": {"Count": 1, "TotalGpuMemoryInMiB": 24576},
    "ml.g5.12xlarge": {"Count": 4, "TotalGpuMemoryInMiB": 98304},
    "ml.g5.24xlarge": {"Count": 4, "TotalGpuMemoryInMiB": 98304},
    "ml.g5.48xlarge": {"Count": 8, "TotalGpuMemoryInMiB": 196608},
}


def _get_gpu_info(instance_type: str, session: Session) -> tuple[int, int]:
    """Get GPU info for the provided instance

    @return: Tuple containing: [0]number of GPUs available and [1]aggregate memory size in MiB
    """
    ec2_client = session.boto_session.client("ec2")
    ec2_instance = _format_instance_type(instance_type)

    try:
        instance_info = ec2_client.describe_instance_types(InstanceTypes=[ec2_instance]).get(
            "InstanceTypes"
        )[0]
    except ClientError:
        raise ValueError(f"Provided instance_type is not GPU enabled: [#{ec2_instance}]")

    if instance_info is not None:
        gpus_info = instance_info.get("GpuInfo")
        if gpus_info is not None:
            instance_gpu_info = (
                gpus_info.get("Gpus")[0].get("Count"),
                gpus_info.get("TotalGpuMemoryInMiB"),
            )

            logger.info("GPU Info [%s]: %s", ec2_instance, instance_gpu_info)
            return instance_gpu_info

    raise ValueError(f"Provided instance_type is not GPU enabled: [#{ec2_instance}]")


def _get_gpu_info_fallback(instance_type: str) -> tuple[int, int]:
    """Get GPU info for the provided instance fallback

    @return: Tuple containing: [0]number of GPUs available and [1]aggregate memory size in MiB
    """
    fallback_instance_gpu_info = FALLBACK_GPU_RESOURCE_MAPPING.get(instance_type)
    ec2_instance = _format_instance_type(instance_type)
    if fallback_instance_gpu_info is None:
        raise ValueError(f"Provided instance_type is not GPU enabled: [#{ec2_instance}]")

    fallback_instance_gpu_info = (
        fallback_instance_gpu_info.get("Count"),
        fallback_instance_gpu_info.get("TotalGpuMemoryInMiB"),
    )
    logger.info("GPU Info [%s]: %s", ec2_instance, fallback_instance_gpu_info)
    return fallback_instance_gpu_info


def _format_instance_type(instance_type: str) -> str:
    """Formats provided instance type name"""
    split_instance = instance_type.split(".")

    if len(split_instance) > 2:
        split_instance.pop(0)

    ec2_instance = ".".join(split_instance)
    return ec2_instance
