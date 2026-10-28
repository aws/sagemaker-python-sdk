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
"""Integration tests: schema not found and unknown field with fuzzy suggestion."""
from __future__ import absolute_import

import pytest

from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerSchemaNotFoundError,
    DataPreparerSchemaValidationError,
)


@pytest.mark.integ
class TestSchemaNotFound:
    """18.5: Non-existent schema version raises DataPreparerSchemaNotFoundError."""

    def test_nonexistent_version_raises(self, sagemaker_session, test_role, s3_fixtures):
        dataset = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            schema_version="99.99.99",
            spec_s3_uri=s3_fixtures["task_description.txt"],
        )
        preparer = DataPreparer(
            sagemaker_session=sagemaker_session,
            role=test_role,
        )
        with pytest.raises(DataPreparerSchemaNotFoundError) as exc_info:
            preparer.generate(dataset, wait=False, logs=False)

        assert exc_info.value.schema_version == "99.99.99"

        # A2: schema-not-found error carries corrective guidance and is not
        # marked as transient (the schema service is reachable; the version
        # simply does not exist).
        exc = exc_info.value
        assert exc.transient is False, (
            f"A2: SchemaNotFoundError for non-existent version should have "
            f"transient=False; got transient={exc.transient}"
        )
        msg = str(exc)
        assert "pin a schema_version" in msg.lower(), (
            f"A2: error message lacks version-pinning guidance; got: {msg!r}"
        )
        assert "[]" not in msg, (
            "A2: error message contains an empty available_versions list '[]'; "
            f"the resolver must return the published versions — got: {msg!r}"
        )

    # NOTE: the former `test_latest_version_not_found_raises` test was removed
    # in the migration to ``ListJobSchemaVersions`` / ``DescribeJobSchemaVersion``.
    # Under the AIRHub-based resolver (now superseded), the SDK shipped a
    # hardcoded fallback schema so "no schemas published at all" was a real
    # path; asserting ``DataPreparerSchemaNotFoundError(version='latest')``
    # protected against accidental silent fallback. With the direct SageMaker
    # API as the authoritative source, and v1.0.0 schemas always published for
    # both categories, this scenario is no longer reachable in production.
    # Req 3.6 ("empty list → DataPreparerSchemaNotFoundError") is covered by
    # the unit test ``test_empty_versions_raises_not_found``.


@pytest.mark.integ
class TestUnknownFieldSuggestion:
    """18.6: Typo field gets fuzzy suggestion in error message.

    NOTE: This test requires a JobConfigSchema to be published in the Hub.
    Skip if schema is not available.
    """

    def test_typo_field_gets_suggestion(self, sagemaker_session, test_role):
        dataset = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            spec_s3_url="s3://bucket/spec.txt",  # typo: should be spec_s3_uri
            num_requested_samples=5,
        )
        preparer = DataPreparer(
            sagemaker_session=sagemaker_session,
            role=test_role,
        )
        try:
            preparer.generate(dataset, wait=False, logs=False)
            pytest.fail("Expected DataPreparerSchemaValidationError or DataPreparerSchemaNotFoundError")
        except DataPreparerSchemaValidationError as e:
            msg = str(e)
            assert "print_schema" in msg
        except DataPreparerSchemaNotFoundError:
            pytest.skip("JobConfigSchema not published in Hub yet — cannot test validation")
