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

"""
Unit test package for sagemaker-serve.

This package contains unit tests and test utilities for the sagemaker-serve module.
"""

from __future__ import absolute_import

# Make test constants easily accessible
from .constants import (
    MOCK_IMAGE_CONFIG,
    MOCK_VPC_CONFIG,
    DEPLOYMENT_CONFIGS,
    NON_OPTIMIZED_DEPLOYMENT_CONFIG,
    OPTIMIZED_DEPLOYMENT_CONFIG_WITH_GATED_DRAFT_MODEL,
    GATED_DRAFT_MODEL_CONFIG,
    NON_GATED_DRAFT_MODEL_CONFIG,
    CAMEL_CASE_ADDTL_DRAFT_MODEL_DATA_SOURCES,
)

__all__ = [
    "MOCK_IMAGE_CONFIG",
    "MOCK_VPC_CONFIG", 
    "DEPLOYMENT_CONFIGS",
    "NON_OPTIMIZED_DEPLOYMENT_CONFIG",
    "OPTIMIZED_DEPLOYMENT_CONFIG_WITH_GATED_DRAFT_MODEL",
    "GATED_DRAFT_MODEL_CONFIG",
    "NON_GATED_DRAFT_MODEL_CONFIG",
    "CAMEL_CASE_ADDTL_DRAFT_MODEL_DATA_SOURCES",
]
