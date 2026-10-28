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
"""Exception taxonomy for DataPreparer."""
from __future__ import absolute_import

from typing import List, Optional

from sagemaker.ai_registry.data_preparer.constants import JobCategory


class DataPreparerError(Exception):
    """Base for all DataPreparer-originated errors."""


class DataPreparerConfigurationError(DataPreparerError):
    """Pre-submit configuration incomplete or invalid."""


class DataPreparerSchemaError(DataPreparerError):
    """Schema-related error base."""

    def __init__(
        self,
        message: str,
        category: Optional[JobCategory] = None,
        schema_version: Optional[str] = None,
    ):
        super().__init__(message)
        self.category = category
        self.schema_version = schema_version


class DataPreparerSchemaNotFoundError(DataPreparerSchemaError):
    """No Hub JobConfigSchema for the requested (category, version)."""

    def __init__(
        self,
        category: JobCategory,
        schema_version: str,
        available_versions: List[str],
        transient: bool = False,
    ):
        if transient:
            message = (
                f"No JobConfigSchema for category={category.value} "
                f"version={schema_version}: could not reach the schema service. "
                f"Retry the call, or pin a schema_version."
            )
        elif not available_versions:
            message = (
                f"No JobConfigSchema for category={category.value} "
                f"version={schema_version}: no versions are published for "
                f"{category.value}. If this is unexpected, pin a schema_version "
                f"once one becomes available."
            )
        else:
            message = (
                f"No JobConfigSchema for category={category.value} "
                f"version={schema_version}. "
                f"Available versions: {available_versions}. "
                f"Pin a schema_version to one of these."
            )
        super().__init__(
            message,
            category=category,
            schema_version=schema_version,
        )
        self.available_versions = available_versions
        self.transient = transient


class DataPreparerSchemaValidationError(DataPreparerSchemaError):
    """User payload fails dynamic Pydantic model validation."""

    def __init__(
        self,
        category: JobCategory,
        schema_version: str,
        pydantic_errors: list,
    ):
        super().__init__(
            f"JobConfigDocument failed schema validation for "
            f"category={category.value} version={schema_version}",
            category=category,
            schema_version=schema_version,
        )
        self.errors = pydantic_errors


class DataPreparerJobError(DataPreparerError):
    """Job-lifecycle error base."""

    def __init__(
        self,
        message: str,
        job_arn: str,
        job_name: Optional[str] = None,
        category: Optional[JobCategory] = None,
    ):
        super().__init__(message)
        self.job_arn = job_arn
        self.job_name = job_name
        self.category = category


class DataPreparerJobFailedError(DataPreparerJobError):
    """Service reported JobStatus=Failed."""

    def __init__(
        self,
        job_arn: str,
        failure_reason: Optional[str],
        secondary_status: Optional[str] = None,
        job_name: Optional[str] = None,
        category: Optional[JobCategory] = None,
    ):
        super().__init__(
            f"Job {job_arn} failed: {failure_reason or '<no reason>'}",
            job_arn=job_arn,
            job_name=job_name,
            category=category,
        )
        self.failure_reason = failure_reason
        self.secondary_status = secondary_status


class DataPreparerJobStoppedError(DataPreparerJobError):
    """Service reported JobStatus=Stopped."""

    def __init__(
        self,
        job_arn: str,
        job_name: Optional[str] = None,
        category: Optional[JobCategory] = None,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Job {job_arn} was stopped",
            job_arn=job_arn,
            job_name=job_name,
            category=category,
        )


class DataPreparerJobTimeoutError(DataPreparerJobError):
    """sagemaker-core waiter timeout."""

    def __init__(
        self,
        job_arn: str,
        elapsed_seconds: float,
        job_name: Optional[str] = None,
    ):
        super().__init__(
            f"Job {job_arn} did not reach a terminal state within {elapsed_seconds:.0f}s",
            job_arn=job_arn,
            job_name=job_name,
        )
        self.elapsed_seconds = elapsed_seconds
