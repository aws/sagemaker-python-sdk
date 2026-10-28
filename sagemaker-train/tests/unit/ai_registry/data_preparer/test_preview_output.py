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
        lines = "\n".join(['{"i":' + str(i) + '}' for i in range(30)])
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
# A8 — DQ v2 output preview fix (added by PR4)
# ---------------------------------------------------------------------------


def _make_v2_job(job_name: str, output_path: str) -> MagicMock:
    """Return a minimal mock job for a v2 DQ (schema 0.2.0) completed run."""
    import json

    job = MagicMock()
    job.job_name = job_name
    job.job_config_schema_version = "0.2.0"
    job.job_config_document = json.dumps(
        {
            "InputDataConfig": [{"DatasetArn": "arn:aws:sagemaker:us-west-2:123:hub-content/h/d/n/1"}],
            "OutputDataConfig": {"S3OutputPath": output_path},
        }
    )
    return job


class TestDQV2OutputPath:
    """DQ v2: preview resolves the file from S3OutputPath, not the input source."""

    @patch("boto3.client")
    def test_v2_dq_finds_file_under_s3output_path(self, mock_boto3, preparer, dataset_dq):
        """v2 DQ preview lists under S3OutputPath and reads the matching key."""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": (
                        "dq-output/my-dq-job-20260101T000000/"
                        "record_level/evaluated_data.jsonl"
                    )
                },
            ]
        }
        body = MagicMock()
        body.read.return_value = b'{"score":0.9}\n'
        mock_s3.get_object.return_value = {"Body": body}

        job = _make_v2_job("my-dq-job", "s3://bucket/dq-output/")

        preparer._preview_output(dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job=job)

        mock_s3.list_objects_v2.assert_called_once_with(
            Bucket="bucket",
            Prefix="dq-output/my-dq-job",
        )
        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket",
            Key=(
                "dq-output/my-dq-job-20260101T000000/"
                "record_level/evaluated_data.jsonl"
            ),
        )

    @patch("boto3.client")
    def test_v2_dq_picks_key_matching_job_name(self, mock_boto3, preparer, dataset_dq):
        """When multiple entries share the prefix, picks the key containing the job name."""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": (
                        "dq-output/unrelated-job-20260101T000000/"
                        "record_level/evaluated_data.jsonl"
                    )
                },
                {
                    "Key": (
                        "dq-output/my-dq-job-20260102T120000/"
                        "record_level/evaluated_data.jsonl"
                    )
                },
            ]
        }
        body = MagicMock()
        body.read.return_value = b'{"score":0.9}\n'
        mock_s3.get_object.return_value = {"Body": body}

        job = _make_v2_job("my-dq-job", "s3://bucket/dq-output/")

        preparer._preview_output(dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job=job)

        # Only the key whose path contains "my-dq-job" should be read
        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket",
            Key=(
                "dq-output/my-dq-job-20260102T120000/"
                "record_level/evaluated_data.jsonl"
            ),
        )

    @patch("boto3.client")
    def test_v2_dq_falls_back_to_legacy_when_no_candidates(
        self, mock_boto3, preparer, dataset_dq
    ):
        """When listing returns no matching key, the legacy source-based path is used."""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        # No keys matching the job name
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": (
                        "dq-output/unrelated-job-20260101T000000/"
                        "record_level/evaluated_data.jsonl"
                    )
                },
            ]
        }
        body = MagicMock()
        body.read.return_value = b'{"score":0.8}\n'
        mock_s3.get_object.return_value = {"Body": body}

        job = _make_v2_job("my-dq-job", "s3://bucket/dq-output/")

        preparer._preview_output(dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job=job)

        # Falls back to legacy: strip /output, append /record_level/evaluated_data.jsonl
        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket",
            Key="jobs/dq-456/record_level/evaluated_data.jsonl",
        )

    @patch("boto3.client")
    def test_v2_dq_list_error_falls_back_to_legacy(self, mock_boto3, preparer, dataset_dq):
        """S3 errors during listing silently fall back to the legacy path."""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        mock_s3.list_objects_v2.side_effect = Exception("AccessDenied")
        body = MagicMock()
        body.read.return_value = b'{"score":0.8}\n'
        mock_s3.get_object.return_value = {"Body": body}

        job = _make_v2_job("my-dq-job", "s3://bucket/dq-output/")

        result = preparer._preview_output(
            dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job=job
        )

        # Never raises — and falls back to legacy path
        assert result is None
        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket",
            Key="jobs/dq-456/record_level/evaluated_data.jsonl",
        )


def _paged_s3(keys, page_size=1000):
    """Mock S3 client whose list_objects_v2 behaves like S3: sorted keys, prefix
    filtering, ``page_size`` keys per page, continuation tokens."""
    keys = sorted(keys)
    s3 = MagicMock()

    def list_objects_v2(Bucket, Prefix="", ContinuationToken=None, MaxKeys=page_size):
        matching = [k for k in keys if k.startswith(Prefix)]
        start = int(ContinuationToken) if ContinuationToken else 0
        page = matching[start : start + min(MaxKeys, page_size)]
        end = start + len(page)
        response = {"Contents": [{"Key": k} for k in page], "IsTruncated": end < len(matching)}
        if end < len(matching):
            response["NextContinuationToken"] = str(end)
        return response

    s3.list_objects_v2.side_effect = list_objects_v2
    body = MagicMock()
    body.read.return_value = b'{"score":0.9}\n'
    s3.get_object.return_value = {"Body": body}
    return s3


class TestDQOutputPathReusedAcrossJobs:
    """A shared S3OutputPath that holds many earlier jobs must not hide this job's records."""

    _TARGET = "dq-output/my-dq-job-2026-10-28-01-25-39-2026-10-28T012541Z/record_level/evaluated_data.jsonl"

    def _earlier_job_keys(self, count):
        # Earlier runs sort before today's job, as timestamped job names do.
        return [
            f"dq-output/aaa-earlier-job-{i:04d}-2026-10-01T000000Z/{suffix}"
            for i in range(count)
            for suffix in (
                "aggregate/quality_report.json",
                "metadata.json",
                "record_level/evaluated_data.jsonl",
                "token_usage_report.json",
            )
        ]

    @patch("boto3.client")
    def test_finds_job_records_behind_more_than_one_page_of_other_jobs(
        self, mock_boto3, preparer, dataset_dq
    ):
        """Reproduces the live failure: >100 keys from earlier jobs under the shared path."""
        mock_boto3.return_value = _paged_s3(self._earlier_job_keys(30) + [self._TARGET], page_size=100)
        job = _make_v2_job("my-dq-job-2026-10-28-01-25-39", "s3://bucket/dq-output/")

        uri = preparer._resolve_output_records_uri(
            dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job
        )

        assert uri == f"s3://bucket/{self._TARGET}"

    @patch("boto3.client")
    def test_follows_continuation_tokens_within_the_job_folder(
        self, mock_boto3, preparer, dataset_dq
    ):
        """If the job's own folder spans several pages, later pages are still read."""
        folder = "dq-output/my-dq-job-2026-10-28-01-25-39-2026-10-28T012541Z/"
        filler = [f"{folder}parts/part-{i:05d}.json" for i in range(250)]
        mock_boto3.return_value = _paged_s3(filler + [self._TARGET], page_size=100)
        job = _make_v2_job("my-dq-job-2026-10-28-01-25-39", "s3://bucket/dq-output/")

        uri = preparer._resolve_output_records_uri(
            dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job
        )

        assert uri == f"s3://bucket/{self._TARGET}"

    @patch("boto3.client")
    def test_output_path_without_trailing_slash_lists_under_job_folder(
        self, mock_boto3, preparer, dataset_dq
    ):
        """S3OutputPath given without a trailing slash still lists "<path>/<job_name>"."""
        mock_s3 = _paged_s3([self._TARGET])
        mock_boto3.return_value = mock_s3
        job = _make_v2_job("my-dq-job-2026-10-28-01-25-39", "s3://bucket/dq-output")

        uri = preparer._resolve_output_records_uri(
            dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job
        )

        assert uri == f"s3://bucket/{self._TARGET}"
        assert mock_s3.list_objects_v2.call_args_list[0].kwargs["Prefix"] == (
            "dq-output/my-dq-job-2026-10-28-01-25-39"
        )

    @patch("boto3.client")
    def test_listing_is_bounded(self, mock_boto3, preparer, dataset_dq):
        """A pathological folder never triggers unbounded listing calls."""
        folder = "dq-output/my-dq-job-2026-10-28-01-25-39-x/"
        mock_s3 = _paged_s3([f"{folder}f-{i:06d}" for i in range(5000)], page_size=100)
        mock_boto3.return_value = mock_s3
        job = _make_v2_job("my-dq-job-2026-10-28-01-25-39", "s3://bucket/dq-output/")

        preparer._resolve_output_records_uri(dataset_dq, JobCategory.DATA_QUALITY_EVALUATION, job)

        assert mock_s3.list_objects_v2.call_count == 10


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
