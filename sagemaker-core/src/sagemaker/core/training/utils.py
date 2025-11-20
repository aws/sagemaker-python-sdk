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
"""Training utilities."""
from __future__ import absolute_import

import os
from typing import Any, Literal
from sagemaker.core.utils.utils import Unassigned


def convert_unassigned_to_none(instance) -> Any:
    """Convert Unassigned values to None for any instance."""
    for name, value in instance.__dict__.items():
        if isinstance(value, Unassigned):
            setattr(instance, name, None)
    return instance


def _is_valid_path(path: str, path_type: Literal["File", "Directory", "Any"] = "Any") -> bool:
    """Check if the path is a valid local path.

    Args:
        path (str): Local path to validate
        path_type (Optional(Literal["File", "Directory", "Any"])): The type of the path to validate.
            Defaults to "Any".

    Returns:
        bool: True if the path is a valid local path, False otherwise
    """
    if not os.path.exists(path):
        return False

    if path_type == "File":
        return os.path.isfile(path)
    if path_type == "Directory":
        return os.path.isdir(path)

    return path_type == "Any"


def _is_valid_s3_uri(path: str, path_type: Literal["File", "Directory", "Any"] = "Any") -> bool:
    """Check if the path is a valid S3 URI.

    This method checks if the path is a valid S3 URI. If the path_type is specified,
    it will also check if the path is a file or a directory.
    This method does not check if the S3 bucket or object exists.

    Args:
        path (str): S3 URI to validate
        path_type (Optional(Literal["File", "Directory", "Any"])): The type of the path to validate.
            Defaults to "Any".

    Returns:
        bool: True if the path is a valid S3 URI, False otherwise
    """
    # Check if the path is a valid S3 URI
    if not path.startswith("s3://"):
        return False

    if path_type == "File":
        # If it's a file, it should not end with a slash
        return not path.endswith("/")
    if path_type == "Directory":
        # If it's a directory, it should end with a slash
        return path.endswith("/")

    return path_type == "Any"
