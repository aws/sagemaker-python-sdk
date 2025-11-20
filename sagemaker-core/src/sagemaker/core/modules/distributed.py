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
"""Distributed module."""
from __future__ import absolute_import

import os

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from sagemaker.core.modules.utils import safe_serialize
from sagemaker.core.training.configs import BaseConfig
from sagemaker.core.training.constants import SM_DRIVERS_LOCAL_PATH


class SMP(BaseConfig):
    """SMP.

    This class is used for configuring the SageMaker Model Parallelism v2 parameters.
    For more information on the model parallelism parameters, see:
    https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-model-parallel-v2-reference.html#distributed-model-parallel-v2-reference-init-config

    Parameters:
        hybrid_shard_degree (Optional[int]):
            Specifies a sharded parallelism degree for the model.
        sm_activation_offloading (Optional[bool]):
            Specifies whether to enable the SMP activation offloading implementation.
        activation_loading_horizon (Optional[int]):
            An integer specifying the activation offloading horizon type for FSDP. This is the
            maximum number of checkpointed or offloaded layers whose inputs can be in the GPU
            memory simultaneously.
        fsdp_cache_flush_warnings (Optional[bool]):
            Detects and warns if cache flushes happen in the PyTorch memory manager, because they
            can degrade computational performance.
        allow_empty_shards (Optional[bool]):
            Whether to allow empty shards when sharding tensors if tensor is not divisible. This is
            an experimental fix for crash during checkpointing in certain scenarios. Disabling this
            falls back to the original PyTorch behavior.
        tensor_parallel_degree (Optional[int]):
            Specifies a tensor parallelism degree. The value must be between 1 and world_size.
        context_parallel_degree (Optional[int]):
            Specifies the context parallelism degree. The value must be between 1 and world_size ,
            and must be <= hybrid_shard_degree.
        expert_parallel_degree (Optional[int]):
            Specifies a expert parallelism degree. The value must be between 1 and world_size.
        random_seed (Optional[int]):
            A seed number for the random operations in distributed modules by SMP tensor
            parallelism or expert parallelism.
    """

    hybrid_shard_degree: Optional[int] = None
    sm_activation_offloading: Optional[bool] = None
    activation_loading_horizon: Optional[int] = None
    fsdp_cache_flush_warnings: Optional[bool] = None
    allow_empty_shards: Optional[bool] = None
    tensor_parallel_degree: Optional[int] = None
    context_parallel_degree: Optional[int] = None
    expert_parallel_degree: Optional[int] = None
    random_seed: Optional[int] = None

    def _to_mp_hyperparameters(self) -> Dict[str, Any]:
        """Converts to the hyperparameters format for the SageMaker Model Parallelism v2."""
        mp_parameters = self.model_dump(exclude_none=True)
        hyperparameters = {
            "mp_parameters": safe_serialize(mp_parameters),
        }
        return hyperparameters


class DistributedConfig(BaseConfig, ABC):
    """Abstract base class for distributed training configurations.

    This class defines the interface that all distributed training configurations
    must implement. It provides a standardized way to specify driver scripts and
    their locations for distributed training jobs.
    """

    @property
    @abstractmethod
    def driver_dir(self) -> str:
        """Directory containing the driver script.

        This property should return the path to the directory containing
        the driver script, relative to the container's working directory.

        Returns:
            str: Path to directory containing the driver script
        """

    @property
    @abstractmethod
    def driver_script(self) -> str:
        """Name of the driver script.

        This property should return the name of the Python script that implements
        the distributed training driver logic.

        Returns:
            str: Name of the driver script file
        """


class Torchrun(DistributedConfig):
    """Torchrun.

    The Torchrun class configures a job that uses ``torchrun`` or
    ``torch.distributed.launch`` in the backend to launch distributed training.

    Parameters:
        process_count_per_node (int):
            The number of processes to run on each node in the training job.
            Will default to the number of GPUs available in the container.
        smp (Optional[SMP]):
            The SageMaker Model Parallelism v2 parameters.
    """

    process_count_per_node: Optional[int] = None
    smp: Optional["SMP"] = None

    @property
    def driver_dir(self) -> str:
        """Directory containing the driver script.

        Returns:
            str: Path to directory containing the driver script
        """
        return os.path.join(SM_DRIVERS_LOCAL_PATH, "distributed_drivers")

    @property
    def driver_script(self) -> str:
        """Name of the driver script.

        Returns:
            str: Name of the driver script file
        """
        return "torchrun_driver.py"


class MPI(DistributedConfig):
    """MPI.

    The MPI class configures a job that uses ``mpirun`` in the backend to launch
    distributed training.

    Parameters:
        process_count_per_node (int):
            The number of processes to run on each node in the training job.
            Will default to the number of GPUs available in the container.
        mpi_additional_options (Optional[str]):
            The custom MPI options to use for the training job.
    """

    process_count_per_node: Optional[int] = None
    mpi_additional_options: Optional[List[str]] = None

    @property
    def driver_dir(self) -> str:
        """Directory containing the driver script.

        Returns:
            str: Path to directory containing the driver script
        """
        return os.path.join(SM_DRIVERS_LOCAL_PATH, "distributed_drivers")

    @property
    def driver_script(self) -> str:
        """Name of the driver script.

        Returns:
            str: Name of the driver script
        """
        return "mpi_driver.py"
