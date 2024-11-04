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
"""This module provides the configuration classes used in `sagemaker.modules`.

Some of these classes are re-exported from `sagemaker-core.shapes`. For convinence,
users can import these classes directly from `sagemaker.modules.configs`.

For more documentation on `sagemaker-core.shapes`, see:
    - https://sagemaker-core.readthedocs.io/en/stable/#sagemaker-core-shapes
"""

from __future__ import absolute_import

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, model_validator

from sagemaker_core.shapes import (
    ResourceConfig,
    StoppingCondition,
    OutputDataConfig,
    Channel,
    DataSource,
    S3DataSource,
    FileSystemDataSource,
    TrainingImageConfig,
    VpcConfig,
)

from sagemaker.modules import logger

__all__ = [
    "SourceCodeConfig",
    "ResourceConfig",
    "StoppingCondition",
    "OutputDataConfig",
    "Channel",
    "DataSource",
    "S3DataSource",
    "FileSystemDataSource",
    "TrainingImageConfig",
    "VpcConfig",
]


class SMDistributedSettings(BaseModel):
    """SMDistributedSettings.

    The SMDistributedSettings is used to configure distributed training when
        using the smdistributed library.

    Attributes:
        enable_dataparallel (Optional[bool]):
            Whether to enable data parallelism.
        enable_modelparallel (Optional[bool]):
            Whether to enable model parallelism.
        modelparallel_parameters (Optional[Dict[str, Any]]):
            The parameters for model parallelism.
    """

    enable_dataparallel: Optional[bool] = False
    enable_modelparallel: Optional[bool] = False
    modelparallel_parameters: Optional[Dict[str, Any]] = None


class DistributionConfig(BaseModel):
    """Base class for distribution configurations."""

    _distribution_type: str


class TorchDistributionConfig(DistributionConfig):
    """TorchDistributionConfig.

    The TorchDistributionConfig uses `torchrun` or `torch.distributed.launch` in the backend to
    launch distributed training.

    SMDistributed Library Information:
        - `TorchDistributionConfig` can be used for SMModelParallel V2.
        - For SMDataParallel or SMModelParallel V1, it is recommended to use the
            `MPIDistributionConfig.`


    Attributes:
        smdistributed_settings (Optional[SMDistributedSettings]):
            The settings for smdistributed library.
        process_count_per_node (int):
            The number of processes to run on each node in the training job.
            Will default to the number of CPUs or GPUs available in the container.
    """

    _distribution_type: str = "torch_distributed"

    smdistributed_settings: Optional[SMDistributedSettings] = None
    process_count_per_node: Optional[int] = None

    @model_validator(mode="after")
    def _validate_model(cls, model):  # pylint: disable=E0213
        """Validate the model."""
        if (
            getattr(model, "smddistributed_settings", None)
            and model.smddistributed_settings.enable_dataparallel
        ):
            logger.warning(
                "For smdistributed data parallelism, it is recommended to use "
                + "MPIDistributionConfig."
            )
        return model


class MPIDistributionConfig(DistributionConfig):
    """MPIDistributionConfig.

    The MPIDistributionConfig uses `mpirun` in the backend to launch distributed training.

    SMDistributed Library Information:
        - `MPIDistributionConfig` can be used for SMDataParallel and SMModelParallel V1.
        - For SMModelParallel V2, it is recommended to use the `TorchDistributionConfig`.

    Attributes:
        smdistributed_settings (Optional[SMDistributedSettings]):
            The settings for smdistributed library.
        process_count_per_node (int):
            The number of processes to run on each node in the training job.
            Will default to the number of CPUs or GPUs available in the container.
        mpi_additional_options (Optional[str]):
            The custom MPI options to use for the training job.
    """

    _distribution_type: str = "mpi"

    smdistributed_settings: Optional[SMDistributedSettings] = None
    process_count_per_node: Optional[int] = None
    mpi_additional_options: Optional[List[str]] = None


class SourceCodeConfig(BaseModel):
    """SourceCodeConfig.

    This config allows the user to specify the source code location, dependencies,
    entry script, or commands to be executed in the training job container.

    Attributes:
        source_dir (Optional[str]):
            The local directory containing the source code to be used in the training job container.
        requirements (Optional[str]):
            The path within `source_dir` to a `requirements.txt` file. If specified, the listed
            requirements will be installed in the training job container.
        entry_script (Optional[str]):
            The path within `source_dir` to the entry script that will be executed in the training
            job container. If not specified, command must be provided.
        command (Optional[str]):
            The command(s) to execute in the training job container. Example: "python my_script.py".
            If not specified, entry_script must be provided.
        distribution (Optional[Union[
            MPIDistributionConfig,
            TorchDistributionConfig,
        ]]):
            The distribution configuration for the training job.
    """

    source_dir: Optional[str] = None
    requirements: Optional[str] = None
    entry_script: Optional[str] = None
    command: Optional[str] = None
