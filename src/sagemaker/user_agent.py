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
"""Placeholder docstring"""
from __future__ import absolute_import

import platform
import sys
import json
import os

import importlib_metadata

SDK_PREFIX = "AWS-SageMaker-Python-SDK"
STUDIO_PREFIX = "AWS-SageMaker-Studio"
NOTEBOOK_PREFIX = "AWS-SageMaker-Notebook-Instance"

NOTEBOOK_METADATA_FILE = "/etc/opt/ml/sagemaker-notebook-instance-version.txt"
STUDIO_METADATA_FILE = "/opt/ml/metadata/resource-metadata.json"

SDK_VERSION = importlib_metadata.version("sagemaker")
OS_NAME = platform.system() or "UnresolvedOS"
OS_VERSION = platform.release() or "UnresolvedOSVersion"
OS_NAME_VERSION = "{}/{}".format(OS_NAME, OS_VERSION)
PYTHON_VERSION = "Python/{}.{}.{}".format(
    sys.version_info.major, sys.version_info.minor, sys.version_info.micro
)


def process_notebook_metadata_file():
    """Check if the platform is SageMaker Notebook, if yes, return the InstanceType

    Returns:
        str: The InstanceType of the SageMaker Notebook if it exists, otherwise None
    """
    if os.path.exists(NOTEBOOK_METADATA_FILE):
        with open(NOTEBOOK_METADATA_FILE, "r") as sagemaker_nbi_file:
            return sagemaker_nbi_file.read().strip()

    return None


def process_studio_metadata_file():
    """Check if the platform is SageMaker Studio, if yes, return the AppType

    Returns:
        str: The AppType of the SageMaker Studio if it exists, otherwise None
    """
    if os.path.exists(STUDIO_METADATA_FILE):
        with open(STUDIO_METADATA_FILE, "r") as sagemaker_studio_file:
            metadata = json.load(sagemaker_studio_file)
            return metadata.get("AppType")

    return None


def determine_prefix(user_agent=""):
    """Determines the prefix for the user agent string.

    Args:
        user_agent (str): The user agent string to prepend the prefix to.

    Returns:
        str: The user agent string with the prefix prepended.
    """
    prefix = "{}/{}".format(SDK_PREFIX, SDK_VERSION)

    if PYTHON_VERSION not in user_agent:
        prefix = "{} {}".format(prefix, PYTHON_VERSION)

    if OS_NAME_VERSION not in user_agent:
        prefix = "{} {}".format(prefix, OS_NAME_VERSION)

    # Get the notebook instance type and prepend it to the user agent string if exists
    notebook_instance_type = process_notebook_metadata_file()
    if notebook_instance_type:
        prefix = "{} {}/{}".format(prefix, NOTEBOOK_PREFIX, notebook_instance_type)

    # Get the studio app type and prepend it to the user agent string if exists
    studio_app_type = process_studio_metadata_file()
    if studio_app_type:
        prefix = "{} {}/{}".format(prefix, STUDIO_PREFIX, studio_app_type)

    return prefix


def prepend_user_agent(client):
    """Prepends the user agent string with the SageMaker Python SDK version.

    Args:
        client (botocore.client.BaseClient): The client to prepend the user agent string for.
    """
    prefix = determine_prefix(client._client_config.user_agent)

    if client._client_config.user_agent is None:
        client._client_config.user_agent = prefix
    else:
        client._client_config.user_agent = "{} {}".format(prefix, client._client_config.user_agent)
