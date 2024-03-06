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
"""This module contains important details related to HubContent data files."""
from __future__ import absolute_import

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from sagemaker.jumpstart.curated_hub.accessors.objectlocation import S3ObjectLocation


class HubContentDependencyType(str, Enum):
    """Enum class for HubContent dependency names"""

    INFERENCE_ARTIFACT = "inference_artifact_s3_reference"
    TRAINING_ARTIFACT = "training_artifact_s3_reference"
    INFERENCE_SCRIPT = "inference_script_s3_reference"
    TRAINING_SCRIPT = "training_script_s3_reference"
    DEFAULT_TRAINING_DATASET = "default_training_dataset_s3_reference"
    DEMO_NOTEBOOK = "demo_notebook_s3_reference"
    MARKDOWN = "markdown_s3_reference"


@dataclass
class FileInfo:
    """Data class for additional S3 file info."""

    location: S3ObjectLocation

    def __init__(
        self,
        bucket: str,
        key: str,
        size: Optional[bytes],
        last_updated: Optional[datetime],
        dependecy_type: Optional[HubContentDependencyType] = None,
    ):
        self.location = S3ObjectLocation(bucket, key)
        self.size = size
        self.last_updated = last_updated
        self.dependecy_type = dependecy_type
