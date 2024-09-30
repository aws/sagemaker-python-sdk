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
"""Configuration classes."""
from __future__ import absolute_import

from typing import Optional
from pydantic import BaseModel

from sagemaker_core.shapes import (
    ResourceConfig,
    StoppingCondition,
    OutputDataConfig,
    AlgorithmSpecification,
    Channel,
    S3DataSource,
    FileSystemDataSource,
    TrainingImageConfig,
    VpcConfig,
)

__all__ = [
    "SourceCodeConfig",
    "ResourceConfig",
    "StoppingCondition",
    "OutputDataConfig",
    "AlgorithmSpecification",
    "Channel",
    "S3DataSource",
    "FileSystemDataSource",
    "TrainingImageConfig",
    "VpcConfig",
]


class SourceCodeConfig(BaseModel):
    """SourceCodeConfig.

    This config allows the user to specify the source code location, dependencies,
    entry script, or commands to be executed in the training job container.

    Attributes:
        command (Optional[str]):
            The command(s) to execute in the training job container. Example: "python my_script.py".
            If not specified, entry_script must be provided
        source_dir (Optional[str]):
            The local directory containing the source code to be used in the training job container.
        requirements (Optional[str]):
            The path within `source_dir` to a `requirements.txt` file. If specified, the listed
            requirements will be installed in the training job container.
        entry_script (Optional[str]):
            The path within `source_dir` to the entry script that will be executed in the training
            job container. If not specified, command must be provided.
    """

    command: Optional[str]
    source_dir: Optional[str]
    requirements: Optional[str]
    entry_script: Optional[str]
