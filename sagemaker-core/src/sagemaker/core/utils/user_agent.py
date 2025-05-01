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

import json
import os

import importlib_metadata

SagemakerCore_PREFIX = "AWS-SageMakerCore"
STUDIO_PREFIX = "AWS-SageMaker-Studio"
NOTEBOOK_PREFIX = "AWS-SageMaker-Notebook-Instance"

NOTEBOOK_METADATA_FILE = "/etc/opt/ml/sagemaker-notebook-instance-version.txt"
STUDIO_METADATA_FILE = "/opt/ml/metadata/resource-metadata.json"

SagemakerCore_VERSION = importlib_metadata.version("sagemaker-core")


def process_notebook_metadata_file() -> str:
    """Check if the platform is SageMaker Notebook, if yes, return the InstanceType

    Returns:
        str: The InstanceType of the SageMaker Notebook if it exists, otherwise None
    """
    if os.path.exists(NOTEBOOK_METADATA_FILE):
        with open(NOTEBOOK_METADATA_FILE, "r") as sagemaker_nbi_file:
            return sagemaker_nbi_file.read().strip()

    return None


def process_studio_metadata_file() -> str:
    """Check if the platform is SageMaker Studio, if yes, return the AppType

    Returns:
        str: The AppType of the SageMaker Studio if it exists, otherwise None
    """
    if os.path.exists(STUDIO_METADATA_FILE):
        with open(STUDIO_METADATA_FILE, "r") as sagemaker_studio_file:
            metadata = json.load(sagemaker_studio_file)
            return metadata.get("AppType")

    return None


def get_user_agent_extra_suffix() -> str:
    """Get the user agent extra suffix string specific to SageMakerCore

    Adhers to new boto recommended User-Agent 2.0 header format

    Returns:
        str: The user agent extra suffix string to be appended
    """
    suffix = "lib/{}#{}".format(SagemakerCore_PREFIX, SagemakerCore_VERSION)

    # Get the notebook instance type and prepend it to the user agent string if exists
    notebook_instance_type = process_notebook_metadata_file()
    if notebook_instance_type:
        suffix = "{} md/{}#{}".format(suffix, NOTEBOOK_PREFIX, notebook_instance_type)

    # Get the studio app type and prepend it to the user agent string if exists
    studio_app_type = process_studio_metadata_file()
    if studio_app_type:
        suffix = "{} md/{}#{}".format(suffix, STUDIO_PREFIX, studio_app_type)

    return suffix
