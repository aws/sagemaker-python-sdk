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
"""Exceptions for the AI inference recommender module."""
from __future__ import absolute_import

from sagemaker.core.utils.exceptions import SageMakerCoreError, ValidationError

from sagemaker.serve.ai_inference_recommender._constants import FEATURE_GATING_RUNBOOK_URL


class FeatureGatedError(SageMakerCoreError):
    """Raised when the AI inference recommender feature is not enabled for the account."""

    fmt = (
        "The AI inference recommender feature is not enabled for this account. "
        "{message} See {runbook_url} for enrollment information."
    )

    def __init__(self, message: str = "", runbook_url: str = FEATURE_GATING_RUNBOOK_URL):
        super().__init__(message=message, runbook_url=runbook_url)
        self.runbook_url = runbook_url


class WorkloadValidationError(ValidationError):
    """Raised when the server rejects a workload spec."""

    fmt = "Server rejected workload: {message}"
