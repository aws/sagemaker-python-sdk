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
"""Unit tests for sagemaker.ai_registry.data_preparer.exceptions."""
from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerConfigurationError,
    DataPreparerError,
    DataPreparerJobError,
    DataPreparerJobFailedError,
    DataPreparerJobStoppedError,
    DataPreparerJobTimeoutError,
    DataPreparerSchemaError,
    DataPreparerSchemaNotFoundError,
    DataPreparerSchemaValidationError,
)


class TestSubclassRelationships:
    def test_configuration_error_is_data_preparer_error(self):
        err = DataPreparerConfigurationError("bad config")
        assert isinstance(err, DataPreparerError)

    def test_schema_error_is_data_preparer_error(self):
        err = DataPreparerSchemaError("schema issue")
        assert isinstance(err, DataPreparerError)

    def test_schema_not_found_is_schema_error(self):
        err = DataPreparerSchemaNotFoundError(
            JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", ["0.9.0"]
        )
        assert isinstance(err, DataPreparerSchemaError)
        assert isinstance(err, DataPreparerError)

    def test_schema_validation_is_schema_error(self):
        err = DataPreparerSchemaValidationError(
            JobCategory.DATA_QUALITY_EVALUATION, "1.0.0", [{"loc": ("x",), "msg": "required"}]
        )
        assert isinstance(err, DataPreparerSchemaError)
        assert isinstance(err, DataPreparerError)

    def test_job_error_is_data_preparer_error(self):
        err = DataPreparerJobError("job issue", job_arn="arn:aws:sagemaker:us-west-2:123:job/x")
        assert isinstance(err, DataPreparerError)

    def test_job_failed_is_job_error(self):
        err = DataPreparerJobFailedError(
            job_arn="arn:aws:sagemaker:us-west-2:123:job/x",
            failure_reason="OOM",
        )
        assert isinstance(err, DataPreparerJobError)
        assert isinstance(err, DataPreparerError)

    def test_job_stopped_is_job_error(self):
        err = DataPreparerJobStoppedError(
            job_arn="arn:aws:sagemaker:us-west-2:123:job/x"
        )
        assert isinstance(err, DataPreparerJobError)

    def test_job_timeout_is_job_error(self):
        err = DataPreparerJobTimeoutError(
            job_arn="arn:aws:sagemaker:us-west-2:123:job/x", elapsed_seconds=3600.0
        )
        assert isinstance(err, DataPreparerJobError)
        assert isinstance(err, DataPreparerError)


class TestFieldPropagation:
    def test_schema_error_fields(self):
        err = DataPreparerSchemaError(
            "msg", category=JobCategory.SYNTHETIC_DATA_GENERATION, schema_version="2.0.0"
        )
        assert err.category == JobCategory.SYNTHETIC_DATA_GENERATION
        assert err.schema_version == "2.0.0"

    def test_schema_not_found_fields(self):
        err = DataPreparerSchemaNotFoundError(
            JobCategory.DATA_QUALITY_EVALUATION, "99.0.0", ["1.0.0", "2.0.0"]
        )
        assert err.category == JobCategory.DATA_QUALITY_EVALUATION
        assert err.schema_version == "99.0.0"
        assert err.available_versions == ["1.0.0", "2.0.0"]

    def test_schema_validation_fields(self):
        errors = [{"loc": ("field_a",), "msg": "required", "type": "missing"}]
        err = DataPreparerSchemaValidationError(
            JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", errors
        )
        assert err.category == JobCategory.SYNTHETIC_DATA_GENERATION
        assert err.schema_version == "1.0.0"
        assert err.errors == errors

    def test_job_error_fields(self):
        err = DataPreparerJobError(
            "msg",
            job_arn="arn:job",
            job_name="my-job",
            category=JobCategory.SYNTHETIC_DATA_GENERATION,
        )
        assert err.job_arn == "arn:job"
        assert err.job_name == "my-job"
        assert err.category == JobCategory.SYNTHETIC_DATA_GENERATION

    def test_job_failed_fields(self):
        err = DataPreparerJobFailedError(
            job_arn="arn:job",
            failure_reason="HUB_FINALIZATION_FAILED",
            secondary_status="GENERATING",
            job_name="my-job",
            category=JobCategory.SYNTHETIC_DATA_GENERATION,
        )
        assert err.job_arn == "arn:job"
        assert err.failure_reason == "HUB_FINALIZATION_FAILED"
        assert err.secondary_status == "GENERATING"
        assert err.job_name == "my-job"
        assert err.category == JobCategory.SYNTHETIC_DATA_GENERATION

    def test_job_timeout_fields(self):
        err = DataPreparerJobTimeoutError(
            job_arn="arn:job", elapsed_seconds=7200.5, job_name="my-job"
        )
        assert err.job_arn == "arn:job"
        assert err.elapsed_seconds == 7200.5
        assert err.job_name == "my-job"


class TestMessageFormat:
    def test_schema_not_found_message_includes_available_versions(self):
        err = DataPreparerSchemaNotFoundError(
            JobCategory.SYNTHETIC_DATA_GENERATION, "5.0.0", ["1.0.0", "2.0.0", "3.0.0"]
        )
        msg = str(err)
        assert "5.0.0" in msg
        assert "SyntheticDataGeneration" in msg
        assert "['1.0.0', '2.0.0', '3.0.0']" in msg

    def test_schema_validation_message_includes_category_and_version(self):
        err = DataPreparerSchemaValidationError(
            JobCategory.DATA_QUALITY_EVALUATION,
            "1.2.0",
            [{"loc": ("x",), "msg": "required"}],
        )
        msg = str(err)
        assert "DataQualityEvaluation" in msg
        assert "1.2.0" in msg

    def test_job_failed_message_includes_reason(self):
        err = DataPreparerJobFailedError(job_arn="arn:job", failure_reason="OOM")
        msg = str(err)
        assert "arn:job" in msg
        assert "OOM" in msg

    def test_job_failed_no_reason(self):
        err = DataPreparerJobFailedError(job_arn="arn:job", failure_reason=None)
        msg = str(err)
        assert "<no reason>" in msg

    def test_job_timeout_message_includes_elapsed(self):
        err = DataPreparerJobTimeoutError(job_arn="arn:job", elapsed_seconds=3600.0)
        msg = str(err)
        assert "3600" in msg
        assert "arn:job" in msg


class TestRaiseCatch:
    def test_catch_schema_not_found_as_schema_error(self):
        with pytest.raises(DataPreparerSchemaError):
            raise DataPreparerSchemaNotFoundError(
                JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", []
            )

    def test_catch_job_failed_as_job_error(self):
        with pytest.raises(DataPreparerJobError):
            raise DataPreparerJobFailedError(job_arn="arn:x", failure_reason="boom")

    def test_catch_any_as_base(self):
        with pytest.raises(DataPreparerError):
            raise DataPreparerJobTimeoutError(job_arn="arn:x", elapsed_seconds=10.0)


class TestSchemaNotFoundA2:
    """A2 (P0): DataPreparerSchemaNotFoundError must give actionable fix guidance
    and never render a dangling 'Available: []' regardless of path.
    """

    def test_transient_message_says_retry_or_pin(self):
        """transient=True → message tells user to retry or pin schema_version."""
        err = DataPreparerSchemaNotFoundError(
            JobCategory.SYNTHETIC_DATA_GENERATION,
            "latest",
            [],
            transient=True,
        )
        msg = str(err)
        assert "retry" in msg.lower() or "Retry" in msg, (
            "Transient message must mention retry"
        )
        assert "schema_version" in msg, (
            "Transient message must mention pinning schema_version"
        )
        # Must NOT look like a permanent "no versions published" message
        assert "no versions are published" not in msg.lower()

    def test_empty_available_versions_message_says_none_published(self):
        """Non-transient empty list → 'no versions are published' guidance."""
        err = DataPreparerSchemaNotFoundError(
            JobCategory.SYNTHETIC_DATA_GENERATION,
            "latest",
            [],
            transient=False,
        )
        msg = str(err)
        assert "no versions are published" in msg.lower() or "no versions" in msg.lower(), (
            "Empty non-transient message must say no versions are published"
        )
        # Must NOT render a dangling bracket literal
        assert "Available: []" not in msg, (
            "Must never render dangling 'Available: []'"
        )
        assert "[]" not in msg, "Must never render empty-list literal '[]'"

    def test_no_bracket_literal_when_empty(self):
        """Neither transient nor non-transient should ever show '[]' in the message."""
        for transient in (True, False):
            err = DataPreparerSchemaNotFoundError(
                JobCategory.DATA_QUALITY_EVALUATION,
                "latest",
                [],
                transient=transient,
            )
            assert "[]" not in str(err), (
                f"transient={transient}: '[]' must not appear in message"
            )

    def test_nonempty_versions_message_lists_them_with_pin_guidance(self):
        """Non-empty available_versions → list them and give pin guidance."""
        err = DataPreparerSchemaNotFoundError(
            JobCategory.DATA_QUALITY_EVALUATION,
            "5.0.0",
            ["1.0.0", "2.0.0", "3.0.0"],
        )
        msg = str(err)
        assert "1.0.0" in msg
        assert "2.0.0" in msg
        assert "3.0.0" in msg
        # Must include actionable "pin" guidance
        assert "pin" in msg.lower() or "schema_version" in msg.lower(), (
            "Message with available versions must include pin guidance"
        )

    def test_transient_vs_none_published_messages_differ(self):
        """Transient and non-transient empty-list messages must be distinct."""
        transient_err = DataPreparerSchemaNotFoundError(
            JobCategory.SYNTHETIC_DATA_GENERATION, "latest", [], transient=True
        )
        nonpublished_err = DataPreparerSchemaNotFoundError(
            JobCategory.SYNTHETIC_DATA_GENERATION, "latest", [], transient=False
        )
        assert str(transient_err) != str(nonpublished_err), (
            "Transient and non-transient empty-list messages must be distinguishable"
        )

    def test_transient_attribute_stored(self):
        """transient attribute is accessible on the exception object."""
        err_t = DataPreparerSchemaNotFoundError(
            JobCategory.SYNTHETIC_DATA_GENERATION, "latest", [], transient=True
        )
        err_f = DataPreparerSchemaNotFoundError(
            JobCategory.SYNTHETIC_DATA_GENERATION, "latest", [], transient=False
        )
        assert err_t.transient is True
        assert err_f.transient is False

    def test_available_versions_attribute_preserved_transient(self):
        """available_versions attribute is accessible even in the transient path."""
        err = DataPreparerSchemaNotFoundError(
            JobCategory.SYNTHETIC_DATA_GENERATION, "latest", [], transient=True
        )
        assert err.available_versions == []
