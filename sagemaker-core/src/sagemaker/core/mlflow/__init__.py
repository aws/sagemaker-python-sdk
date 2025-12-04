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
"""MLflow integration module - stub.

This module provides stubs for MLflow integration functionality.
MLflow integration is an optional feature that requires the mlflow package.

NOTE: This is a stub module. Full MLflow integration will be implemented
in a future release.
"""
from __future__ import absolute_import

__all__ = ["forward_sagemaker_metrics"]


def forward_sagemaker_metrics(*args, **kwargs):
    """Stub for MLflow metrics forwarding.

    This function is not yet implemented. MLflow integration is an optional
    feature that will be added in a future release.

    Raises:
        NotImplementedError: Always raised as this is a stub.
    """
    raise NotImplementedError(
        "MLflow integration is not yet implemented. "
        "This feature will be added in a future release."
    )
