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
"""Utilities for detecting  available GPUs and Aggregate GPU Memory size of an instance"""
from __future__ import absolute_import

import logging
from typing import Tuple

from botocore.exceptions import ClientError

from sagemaker import Session
from sagemaker import instance_types_gpu_info

logger = logging.getLogger(__name__)


MIB_CONVERSION_FACTOR = 0.00000095367431640625
MEMORY_BUFFER_MULTIPLIER = 1.2  # 20% buffer


def _get_gpu_info(instance_type: str, session: Session) -> Tuple[int, int]:
    """Get GPU info for the provided instance

    Args:
        instance_type (str)
        session: The session to use.

    Returns: tuple[int, int]: A tuple that contains number of GPUs available at index 0,
    and aggregate memory size in MiB at index 1.

    Raises:
        ValueError: If The given instance type does not exist or GPU is not enabled.
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
            gpus = gpus_info.get("Gpus")
            if gpus is not None and len(gpus) > 0:
                count = gpus[0].get("Count")
                total_gpu_memory_in_mib = gpus_info.get("TotalGpuMemoryInMiB")
                if count and total_gpu_memory_in_mib:
                    instance_gpu_info = (
                        count,
                        total_gpu_memory_in_mib,
                    )
                    logger.info("GPU Info [%s]: %s", ec2_instance, instance_gpu_info)
                    return instance_gpu_info

    raise ValueError(f"Provided instance_type is not GPU enabled: [{ec2_instance}]")


def _get_gpu_info_fallback(instance_type: str, region: str) -> Tuple[int, int]:
    """Get GPU info for the provided from the config

    Args:
        instance_type (str):
        region: The AWS region.

    Returns: tuple[int, int]: A tuple that contains number of GPUs available at index 0,
            and aggregate memory size in MiB at index 1.

    Raises:
        ValueError: If The given instance type does not exist.
    """
    instance_types_gpu_info_config = instance_types_gpu_info.retrieve(region)
    fallback_instance_gpu_info = instance_types_gpu_info_config.get(instance_type)

    ec2_instance = _format_instance_type(instance_type)
    if fallback_instance_gpu_info is None:
        raise ValueError(f"Provided instance_type is not GPU enabled: [{ec2_instance}]")

    fallback_instance_gpu_info = (
        fallback_instance_gpu_info.get("Count"),
        fallback_instance_gpu_info.get("TotalGpuMemoryInMiB"),
    )
    logger.info("GPU Info [%s]: %s", ec2_instance, fallback_instance_gpu_info)
    return fallback_instance_gpu_info


def _format_instance_type(instance_type: str) -> str:
    """Formats provided instance type name

    Args:
        instance_type (str):

    Returns: formatted instance type.
    """
    split_instance = instance_type.split(".")

    if len(split_instance) > 2:
        split_instance.pop(0)

    ec2_instance = ".".join(split_instance)
    return ec2_instance


def _total_inference_model_size_mib(model: str, dtype: str) -> int:
    """Calculates the model size from HF accelerate

    This function gets the model size from accelerate. It also adds a
    padding and converts to size MiB. When performing inference, expect
     to add up to an additional 20% to the given model size as found by EleutherAI.
    """
    output = None
    try:
        from accelerate.commands.estimate import estimate_command_parser, gather_data

        args = estimate_command_parser().parse_args([model, "--dtypes", dtype])

        output = gather_data(
            args
        )  # "dtype", "Largest Layer", "Total Size Bytes", "Training using Adam"
    except ImportError:
        logger.error(
            "To enable Model size calculations: Install HuggingFace extras dependencies "
            "using pip install 'sagemaker[huggingface]>=2.212.0'"
        )

    if output is None:
        raise ValueError(f"Could not get Model size for {model}")

    total_memory_size_mib = MEMORY_BUFFER_MULTIPLIER * output[0][2] * MIB_CONVERSION_FACTOR
    logger.info("Total memory size MIB: %s", total_memory_size_mib)
    return total_memory_size_mib
