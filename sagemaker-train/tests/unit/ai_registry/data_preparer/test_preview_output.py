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
"""Unit tests for DataPreparer._preview_output (pandas-based)."""

from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.preparer import DataPreparer


@pytest.fixture
def preparer():
    session = MagicMock()
    session.boto_region_name = "us-west-2"
    p = DataPreparer(sagemaker_session=session)
    return p


@pytest.fixture
def dataset_sdg():
    ds = MagicMock()
    ds.source = "s3://bucket/jobs/sdg-123/output/"
    return ds


@pytest.fixture
def dataset_dq():
    ds = MagicMock()
    ds.source = "s3://bucket/jobs/dq-456/output/"
    return ds


class TestSDGPathConstruction:
    """SDG path: source.rstrip('/') + '/output_samples.jsonl'."""

    @patch("boto3.client")
    def test_sdg_path_trailing_slash(self, mock_boto3, preparer, dataset_sdg):
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        body = MagicMock()
        body.read.return_value = b'{"messages":[{"role":"user","content":"hi"}]}\n'
        mock_s3.get_object.return_value = {"Body": body}

        preparer._preview_output(dataset_sdg, JobCategory.SYNTHETIC_DATA_GENERATION)

        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket",
            Key="jobs/sdg-123/output/output_samples.jsonl",
        )


class TestDQPathConstruction:
    """DQ path: strip /output, append /record_level/evaluated_data.jsonl."""

    @patch("boto3.client")
    def test_dq_path_strips_output(self, mock_boto3, preparer, dataset_dq):
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        body = MagicMock()
        body.read.return_value = b'{"score":0.9}\n'
        mock_s3.get_object.return_value = {"Body": body}

        preparer._preview_output(dataset_dq, JobCategory.DATA_QUALITY_EVALUATION)

        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket",
            Key="jobs/dq-456/record_level/evaluated_data.jsonl",
        )


class TestNCapping:
    """n is capped at 20."""

    @patch("boto3.client")
    def test_n_capped_at_20(self, mock_boto3, preparer, dataset_sdg):
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        lines = "\n".join(['{"i":' + str(i) + "}" for i in range(30)])
        body = MagicMock()
        body.read.return_value = lines.encode()
        mock_s3.get_object.return_value = {"Body": body}

        preparer._preview_output(dataset_sdg, JobCategory.SYNTHETIC_DATA_GENERATION, n=25)

        # pandas nrows=20 (capped from 25)
        # We verify it doesn't crash and the call was made
        mock_s3.get_object.assert_called_once()


class TestGracefulFailure:
    """Errors return without raising."""

    @patch("boto3.client")
    def test_read_error_no_propagation(self, mock_boto3, preparer, dataset_sdg):
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        mock_s3.get_object.side_effect = Exception("NoSuchKey")

        result = preparer._preview_output(dataset_sdg, JobCategory.SYNTHETIC_DATA_GENERATION)
        assert result is None


class TestEmptySource:
    """None or empty string source returns early."""

    def test_none_source(self, preparer):
        ds = MagicMock()
        ds.source = None
        result = preparer._preview_output(ds, JobCategory.SYNTHETIC_DATA_GENERATION)
        assert result is None

    def test_empty_string_source(self, preparer):
        ds = MagicMock()
        ds.source = ""
        result = preparer._preview_output(ds, JobCategory.SYNTHETIC_DATA_GENERATION)
        assert result is None


class TestNonS3Source:
    """Non-S3 source returns early without calling boto3."""

    @patch("boto3.client")
    def test_non_s3_source(self, mock_boto3, preparer):
        ds = MagicMock()
        ds.source = "https://example.com/data.jsonl"

        result = preparer._preview_output(ds, JobCategory.SYNTHETIC_DATA_GENERATION)

        assert result is None
        mock_boto3.assert_not_called()


# ---------------------------------------------------------------------------
# A8 — DQ output preview
# ---------------------------------------------------------------------------


def _make_dq_job(metadata_s3_uri=None) -> MagicMock:
    """Return a minimal mock DQ job; a completed job's config document carries ``MetadataS3Uri``."""
    import json

    document = {
        "InputDataConfig": [{"DatasetArn": "arn:aws:sagemaker:us-west-2:123:hub-content/h/d/n/1"}],
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/dq-output/"},
    }
    if metadata_s3_uri:
        document["MetadataS3Uri"] = metadata_s3_uri
    job = MagicMock()
    job.job_name = "my-dq-job"
    job.job_config_schema_version = "3.0.0"
    job.job_config_document = json.dumps(document)
    return job


class TestDQOutputPath:
    """DQ: preview reads the records file in the job's own output folder, not next to the input."""

    @patch("boto3.client")
    def test_dq_reads_the_records_next_to_metadata_s3_uri(self, mock_boto3, preparer, dataset_dq):
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        body = MagicMock()
        body.read.return_value = b'{"score":0.9}\n'
        mock_s3.get_object.return_value = {"Body": body}
        folder = "dq-output/data-quality-evaluation/my-dq-job/2026-10-27T192836.911415410Z"
        job = _make_dq_job(f"s3://bucket/{folder}/metadata.json")

        preparer._preview_output(dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job=job)

        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket", Key=f"{folder}/record_level/evaluated_data.jsonl"
        )

    @patch("boto3.client")
    def test_dq_without_metadata_s3_uri_falls_back_to_legacy(
        self, mock_boto3, preparer, dataset_dq
    ):
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        body = MagicMock()
        body.read.return_value = b'{"score":0.8}\n'
        mock_s3.get_object.return_value = {"Body": body}

        preparer._preview_output(
            dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job=_make_dq_job()
        )

        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket",
            Key="jobs/dq-456/record_level/evaluated_data.jsonl",
        )


class TestSDGUnchangedWithJob:
    """SDG preview uses source-based path regardless of job parameter."""

    @patch("boto3.client")
    def test_sdg_ignores_job_parameter(self, mock_boto3, preparer, dataset_sdg):
        """Passing a job to SDG preview does not change the path construction."""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        body = MagicMock()
        body.read.return_value = b'{"messages":[{"role":"user","content":"hi"}]}\n'
        mock_s3.get_object.return_value = {"Body": body}

        job = MagicMock()
        job.job_name = "my-sdg-job"

        preparer._preview_output(dataset_sdg, JobCategory.SYNTHETIC_DATA_GENERATION, job=job)

        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket",
            Key="jobs/sdg-123/output/output_samples.jsonl",
        )
        mock_s3.list_objects_v2.assert_not_called()
