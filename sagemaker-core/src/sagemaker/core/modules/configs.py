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
"""This module provides the configuration classes used in ``sagemaker.modules``.

Some of these classes are re-exported from ``sagemaker_core.shapes``. For convinence,
users can import these classes directly from ``sagemaker.modules.configs``.

For more documentation on ``sagemaker_core.shapes``, see:
    - https://sagemaker-core.readthedocs.io/en/stable/#sagemaker-core-shapes
"""

from __future__ import absolute_import

from typing import Optional, Union
from pydantic import BaseModel, model_validator, ConfigDict

import sagemaker.core.shapes as shapes

# TODO: Can we add custom logic to some of these to set better defaults?
from sagemaker.core.shapes import (
    StoppingCondition,
    RetryStrategy,
    OutputDataConfig,
    Channel,
    ShuffleConfig,
    DataSource,
    S3DataSource,
    FileSystemDataSource,
    TrainingImageConfig,
    TrainingRepositoryAuthConfig,
    Tag,
    InfraCheckConfig,
    RemoteDebugConfig,
    SessionChainingConfig,
    InstanceGroup,
    TensorBoardOutputConfig,
    CheckpointConfig,
    MetricDefinition,
)
from typing import List

__all__ = [
    "SourceCode",
    "StoppingCondition",
    "RetryStrategy",
    "OutputDataConfig",
    "Channel",
    "ShuffleConfig",
    "DataSource",
    "S3DataSource",
    "FileSystemDataSource",
    "TrainingImageConfig",
    "TrainingRepositoryAuthConfig",
    "Tag",
    "InfraCheckConfig",
    "RemoteDebugConfig",
    "SessionChainingConfig",
    "InstanceGroup",
    "TensorBoardOutputConfig",
    "CheckpointConfig",
    "Compute",
    "Networking",
    "InputData",
    "MetricDefinition",
]

from sagemaker.core.modules.utils import convert_unassigned_to_none


class BaseConfig(BaseModel):
    """BaseConfig"""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class SourceCode(BaseConfig):
    """SourceCode.

    The SourceCode class allows the user to specify the source code location, dependencies,
    entry script, or commands to be executed in the training job container.

    Parameters:
        source_dir (Optional[str]):
            The local directory containing the source code to be used in the training job container.
        requirements (Optional[str]):
            The path within ``source_dir`` to a ``requirements.txt`` file. If specified, the listed
            requirements will be installed in the training job container.
        entry_script (Optional[str]):
            The path within ``source_dir`` to the entry script that will be executed in the training
            job container. If not specified, command must be provided.
        command (Optional[str]):
            The command(s) to execute in the training job container. Example: "python my_script.py".
            If not specified, entry_script must be provided.
        ignore_patterns: (Optional[List[str]]) :
            The ignore patterns to ignore specific files/folders when uploading to S3. If not specified,
            default to: ['.env', '.git', '__pycache__', '.DS_Store', '.cache', '.ipynb_checkpoints'].
    """

    source_dir: Optional[str] = None
    requirements: Optional[str] = None
    entry_script: Optional[str] = None
    command: Optional[str] = None
    ignore_patterns: Optional[List[str]] = [
        ".env",
        ".git",
        "__pycache__",
        ".DS_Store",
        ".cache",
        ".ipynb_checkpoints",
    ]


class Compute(shapes.ResourceConfig):
    """Compute.

    The Compute class is a subclass of ``sagemaker_core.shapes.ResourceConfig``
    and allows the user to specify the compute resources for the training job.

    Parameters:
        instance_type (Optional[str]):
            The ML compute instance type. For information about available instance types,
            see https://aws.amazon.com/sagemaker/pricing/.
        instance_count (Optional[int]): The number of ML compute instances to use. For distributed
            training, provide a value greater than 1.
        volume_size_in_gb (Optional[int]):
            The size of the ML storage volume that you want to provision.  ML storage volumes store
            model artifacts and incremental states. Training algorithms might also use the ML
            storage volume for scratch space. Default: 30
        volume_kms_key_id (Optional[str]):
            The Amazon Web Services KMS key that SageMaker uses to encrypt data on the storage
            volume attached to the ML compute instance(s) that run the training job.
        keep_alive_period_in_seconds (Optional[int]):
            The duration of time in seconds to retain configured resources in a warm pool for
            subsequent training jobs.
        instance_groups (Optional[List[InstanceGroup]]):
            A list of instance groups for heterogeneous clusters to be used in the training job.
        enable_managed_spot_training (Optional[bool]):
            To train models using managed spot training, choose True. Managed spot training
            provides a fully managed and scalable infrastructure for training machine learning
            models. this option is useful when training jobs can be interrupted and when there
            is flexibility when the training job is run.
    """

    volume_size_in_gb: Optional[int] = 30
    enable_managed_spot_training: Optional[bool] = None

    @model_validator(mode="after")
    def _model_validator(self) -> "Compute":
        """Convert Unassigned values to None."""
        return convert_unassigned_to_none(self)

    def _to_resource_config(self) -> shapes.ResourceConfig:
        """Convert to a sagemaker_core.shapes.ResourceConfig object."""
        compute_config_dict = self.model_dump()
        resource_config_fields = set(shapes.ResourceConfig.__annotations__.keys())
        filtered_dict = {
            k: v for k, v in compute_config_dict.items() if k in resource_config_fields
        }
        return shapes.ResourceConfig(**filtered_dict)


class Networking(shapes.VpcConfig):
    """Networking.

    The Networking class is a subclass of ``sagemaker_core.shapes.VpcConfig`` and
    allows the user to specify the networking configuration for the training job.

    Parameters:
        security_group_ids (Optional[List[str]]):
            The VPC security group IDs, in the form sg-xxxxxxxx. Specify the
            security groups for the VPC that is specified in the Subnets field.
        subnets (Optional[List[str]]):
            The ID of the subnets in the VPC to which you want to connect your
            training job or model.
        enable_network_isolation (Optional[bool]):
            Isolates the training container. No inbound or outbound network calls can be made,
            except for calls between peers within a training cluster for distributed training.
            If you enable network isolation for training jobs that are configured to use a VPC,
            SageMaker downloads and uploads customer data and model artifacts through the
            specified VPC, but the training container does not have network access.
        enable_inter_container_traffic_encryption (Optional[bool]):
            To encrypt all communications between ML compute instances in distributed training
            choose True. Encryption provides greater security for distributed training, but
            training might take longer. How long it takes depends on the amount of
            communication between compute instances, especially if you use a deep learning
            algorithm in distributed training.
    """

    enable_network_isolation: Optional[bool] = None
    enable_inter_container_traffic_encryption: Optional[bool] = None

    @model_validator(mode="after")
    def _model_validator(self) -> "Networking":
        """Convert Unassigned values to None."""
        return convert_unassigned_to_none(self)

    def _to_vpc_config(self) -> shapes.VpcConfig:
        """Convert to a sagemaker_core.shapes.VpcConfig object."""
        compute_config_dict = self.model_dump()
        resource_config_fields = set(shapes.VpcConfig.__annotations__.keys())
        filtered_dict = {
            k: v for k, v in compute_config_dict.items() if k in resource_config_fields
        }
        return shapes.VpcConfig(**filtered_dict)


class InputData(BaseConfig):
    """InputData.

    This config allows the user to specify an input data source for the training job.

    Will be found at ``/opt/ml/input/data/<channel_name>`` within the training container.
    For convience, can be referenced inside the training container like:

    .. code:: python

        import os
        input_data_dir = os.environ['SM_CHANNEL_<channel_name>']

    Parameters:
        channel_name (str):
            The name of the input data source channel.
        data_source (Union[str, S3DataSource, FileSystemDataSource]):
            The data source for the channel. Can be an S3 URI string, local file path string,
            S3DataSource object, or FileSystemDataSource object.
    """

    channel_name: str = None
    data_source: Union[str, FileSystemDataSource, S3DataSource] = None
