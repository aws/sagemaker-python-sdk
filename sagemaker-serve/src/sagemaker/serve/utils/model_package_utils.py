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
"""Utilities for Restricted Model Package support."""
from __future__ import absolute_import

from typing import Optional


def is_restricted_model_package(model_package) -> bool:
    """Detect if a model package is a Restricted Model Package.

    Args:
        model_package: A ModelPackage resource object.

    Returns:
        True if the model package is restricted, False otherwise.
    """
    if not model_package:
        return False

    managed_storage_type = getattr(model_package, "managed_storage_type", None)
    return managed_storage_type == "Restricted"


def get_s3_uri_from_inference_spec(inference_specification) -> Optional[str]:
    """Extract s3_uri from the first container's model_data_source.

    Args:
        inference_specification: The inference_specification from a ModelPackage.

    Returns:
        The s3_uri string, or None if not available.
    """
    if not inference_specification:
        return None
    containers = getattr(inference_specification, "containers", None)
    if not containers:
        return None
    container = containers[0]
    data_source = getattr(container, "model_data_source", None)
    if not data_source:
        return None
    s3_data_source = getattr(data_source, "s3_data_source", None)
    if not s3_data_source:
        return None
    return getattr(s3_data_source, "s3_uri", None)
