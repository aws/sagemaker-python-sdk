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
from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer._job_resource import Job


@pytest.fixture
def session_with_region():
    s = MagicMock()
    s.boto_region_name = "us-west-2"
    s.account_id.return_value = "123456789012"
    return s


@pytest.fixture
def job_with_progress(session_with_region):
    return Job(
        job_name="golden-sdg-abc-2026-05-08-21-02-47-594-1",
        job_category="SyntheticDataGeneration",
        session=session_with_region,
        expected_samples=10,
        progress_output_s3_uri="s3://my-bucket/sdg-out/",
    )


def _get_object_response(count: int):
    """Build a fake GetObject response with a JSONL body containing `count` lines."""
    body_bytes = b"".join(json.dumps({"i": i}).encode("utf-8") + b"\n" for i in range(count))

    class _Body:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

        def iter_chunks(self, chunk_size=None):
            """Yield the body as a single chunk for streaming line-count tests."""
            if self._data:
                yield self._data

    return {"Body": _Body(body_bytes)}


def _wire_paginator(s3_mock):
    paginator = MagicMock()

    def _paginate(*_args, **_kwargs):
        # Bump list_objects_v2's call counter so existing assertions on
        # `s3.list_objects_v2.call_count` continue to reflect "discovery passes".
        page = s3_mock.list_objects_v2()
        return iter([page])

    paginator.paginate.side_effect = _paginate
    s3_mock.get_paginator.return_value = paginator


class TestCountSamplesDisambiguation:
    """The resolver locks onto `{JobName}-{iso-timestamp}/…` objects ONLY."""

    @patch("boto3.client")
    def test_locks_onto_correct_subdir(self, mock_boto_client, job_with_progress):
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        job_name = job_with_progress.job_name
        correct_subdir = f"{job_name}-2026-05-08T210250.953723324Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{correct_subdir}/job_metadata/verified_samples.jsonl"},
                {"Key": f"sdg-out/{correct_subdir}/output/output_samples.jsonl"},
            ]
        }
        s3.get_object.return_value = _get_object_response(7)
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job_with_progress._count_samples()
        assert count == 7
        # Prefers verified_samples (streaming) over output_samples (finalized).
        selected_key = s3.get_object.call_args.kwargs["Key"]
        assert selected_key.endswith("verified_samples.jsonl")
        # And cached it for future polls.
        assert job_with_progress._progress_target_key == selected_key

    @patch("boto3.client")
    def test_rejects_sibling_job_whose_name_extends_ours(self, mock_boto_client, job_with_progress):
        """A concurrent sibling with JobName='{ours}-extra' must NOT match."""
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        job_name = job_with_progress.job_name
        # Only the sibling's data is in S3 — nothing from our job yet.
        sibling_subdir = f"{job_name}-extra-2026-05-08T210250.999999999Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{sibling_subdir}/job_metadata/verified_samples.jsonl"},
                {"Key": f"sdg-out/{sibling_subdir}/output/output_samples.jsonl"},
            ]
        }
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job_with_progress._count_samples()
        # The sibling's path starts with `{ours}-extra-...` — the suffix after
        # stripping `{JobName}-` is `extra-...` which does NOT start with a
        # digit (year) so the is_ours() check rejects it.
        assert count is None
        s3.get_object.assert_not_called()
        assert job_with_progress._progress_target_key is None

    @patch("boto3.client")
    def test_prefers_our_job_when_sibling_data_also_present(
        self, mock_boto_client, job_with_progress
    ):
        """Both ours and a sibling's data landed under overlapping prefixes —
        only ours must be counted.
        """
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        job_name = job_with_progress.job_name
        ours_subdir = f"{job_name}-2026-05-08T210250.953723324Z"
        sibling_subdir = f"{job_name}-extra-2026-05-08T210250.999999999Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{sibling_subdir}/job_metadata/verified_samples.jsonl"},
                {"Key": f"sdg-out/{ours_subdir}/job_metadata/verified_samples.jsonl"},
            ]
        }
        s3.get_object.return_value = _get_object_response(3)
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job_with_progress._count_samples()
        assert count == 3
        selected_key = s3.get_object.call_args.kwargs["Key"]
        assert ours_subdir in selected_key
        assert sibling_subdir not in selected_key

    @patch("boto3.client")
    def test_cached_key_used_on_subsequent_calls(self, mock_boto_client, job_with_progress):
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        job_name = job_with_progress.job_name
        subdir = f"{job_name}-2026-05-08T210250.953723324Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{subdir}/job_metadata/verified_samples.jsonl"},
            ]
        }
        s3.get_object.side_effect = [
            _get_object_response(2),
            _get_object_response(5),
            _get_object_response(10),
        ]
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        a = job_with_progress._count_samples()
        b = job_with_progress._count_samples()
        c = job_with_progress._count_samples()
        assert (a, b, c) == (2, 5, 10)
        # list_objects_v2 was called only on the first tick — subsequent ticks
        # reuse the cached key.
        assert s3.list_objects_v2.call_count == 1
        assert s3.get_object.call_count == 3

    @patch("boto3.client")
    def test_rescan_after_nosuchkey_on_cached_target(self, mock_boto_client, job_with_progress):
        """When the streaming file is finalized/removed, re-list and pick up the
        output_samples.jsonl fallback."""
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        job_name = job_with_progress.job_name
        subdir = f"{job_name}-2026-05-08T210250.953723324Z"

        # First listing: only the streaming file.
        # Second listing (after the streaming file 404s): only the final file.
        s3.list_objects_v2.side_effect = [
            {
                "Contents": [
                    {"Key": f"sdg-out/{subdir}/job_metadata/verified_samples.jsonl"},
                ]
            },
            {
                "Contents": [
                    {"Key": f"sdg-out/{subdir}/output/output_samples.jsonl"},
                ]
            },
        ]
        s3.get_object.side_effect = [
            _get_object_response(3),  # first tick: streaming file → 3 lines
            s3.exceptions.NoSuchKey("gone"),  # second tick: cached key 404s
            _get_object_response(10),  # rescan → output file → 10 lines
        ]
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        first = job_with_progress._count_samples()
        assert first == 3
        # Streaming file cached.
        assert job_with_progress._progress_target_key.endswith("verified_samples.jsonl")

        second = job_with_progress._count_samples()
        # Cached key 404'd → rescanned → now pointing at the output file.
        assert second == 10
        assert job_with_progress._progress_target_key.endswith("output_samples.jsonl")
        assert s3.list_objects_v2.call_count == 2

    def test_no_progress_when_not_configured(self, session_with_region):
        """If DataPreparer didn't set expected_samples / progress_output_s3_uri,
        the count is None and no S3 calls are made.
        """
        job = Job(
            job_name="no-progress-job",
            job_category="DataQualityEvaluation",
            session=session_with_region,
        )
        assert job._count_samples() is None


def _wire_listing(s3_mock, keys):
    """A paginator that, like S3, returns only the keys under the requested Prefix."""
    paginator = MagicMock()

    def _paginate(*_args, Prefix="", **_kwargs):
        return iter([{"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}])

    paginator.paginate.side_effect = _paginate
    s3_mock.get_paginator.return_value = paginator


class TestSdgJobFolder:
    """SDG output lives in {S3OutputPath}/synthetic-data-generation/{JobName}/{createdAt}/."""

    JOB_NAME = "my-sdg-job-1794531357481"
    CREATED_AT = "2026-11-13T00:55:57.543Z"
    FOLDER = f"sdg-out/synthetic-data-generation/{JOB_NAME}/2026-11-13T005557.543Z/"

    def _job(self, session):
        return Job(
            job_name=self.JOB_NAME,
            job_category="SyntheticDataGeneration",
            session=session,
            created_at=self.CREATED_AT,
            expected_samples=10,
            progress_output_s3_uri="s3://my-bucket/sdg-out/",
        )

    def _s3(self, mock_boto_client, keys, count=0):
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        s3.get_object.return_value = _get_object_response(count)
        mock_boto_client.return_value = s3
        _wire_listing(s3, keys)
        return s3

    @patch("boto3.client")
    def test_counts_verified_samples_in_the_job_folder(self, mock_boto_client, session_with_region):
        verified = f"{self.FOLDER}job_metadata/verified_samples.jsonl"
        validation = f"{self.FOLDER}output/data-validation/dqe-val-1/validation/record_level/"
        s3 = self._s3(
            mock_boto_client,
            [
                f"{self.FOLDER}generation_code/generation.py",
                verified,
                f"{self.FOLDER}output/output_samples.jsonl",
                f"{validation}validation_failures.jsonl",
            ],
            count=7,
        )
        job = self._job(session_with_region)

        assert job._count_samples() == 7
        assert s3.get_object.call_args.kwargs["Key"] == verified
        assert job._progress_target_key == verified

    @patch("boto3.client")
    def test_ignores_a_sibling_and_an_earlier_job_with_the_same_name(
        self, mock_boto_client, session_with_region
    ):
        root = "sdg-out/synthetic-data-generation/"
        s3 = self._s3(
            mock_boto_client,
            [
                f"{root}{self.JOB_NAME}-v2/2026-11-13T005557.543Z/job_metadata/verified_samples.jsonl",
                f"{root}{self.JOB_NAME}/2026-11-12T235501.120Z/job_metadata/verified_samples.jsonl",
            ],
        )
        job = self._job(session_with_region)

        assert job._count_samples() is None
        s3.get_object.assert_not_called()
        assert job._progress_target_key is None

    @patch("boto3.client")
    def test_finds_a_job_written_in_the_earlier_layout(self, mock_boto_client, session_with_region):
        key = f"sdg-out/{self.JOB_NAME}-2026-11-13T005557.543Z/job_metadata/verified_samples.jsonl"
        s3 = self._s3(mock_boto_client, [key], count=4)
        job = self._job(session_with_region)

        assert job._count_samples() == 4
        assert s3.get_object.call_args.kwargs["Key"] == key


class TestDQProgressTracking:
    """DQ-specific progress: discovers evaluated_data.jsonl and uses correct label."""

    @patch("boto3.client")
    def test_discovers_evaluated_data_jsonl(self, mock_boto_client, session_with_region):
        """DQ jobs lock onto record_level/evaluated_data.jsonl."""
        job = Job(
            job_name="my-dq-eval-2026-05-13-02-31-18-253-3",
            job_category="DataQualityEvaluation",
            session=session_with_region,
            created_at="2026-05-13T02:31:20.528276209Z",
            expected_samples=200,
            progress_output_s3_uri="s3://my-bucket/dq-out/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        subdir = f"{job.job_name}-2026-05-13T023120.528276209Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"dq-out/{subdir}/record_level/evaluated_data.jsonl"},
                {"Key": f"dq-out/{subdir}/aggregate/metrics.json"},
            ]
        }
        s3.get_object.return_value = _get_object_response(45)
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job._count_samples()
        assert count == 45
        assert job._progress_target_key.endswith("evaluated_data.jsonl")

    @patch("boto3.client")
    def test_dq_prefers_evaluated_data_over_output_samples(
        self, mock_boto_client, session_with_region
    ):
        """When both evaluated_data.jsonl and output_samples.jsonl exist, prefer evaluated_data."""
        job = Job(
            job_name="my-dq-eval-2026-05-13-02-31-18-253-3",
            job_category="DataQualityEvaluation",
            session=session_with_region,
            created_at="2026-05-13T02:31:20.528276209Z",
            expected_samples=100,
            progress_output_s3_uri="s3://my-bucket/dq-out/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        subdir = f"{job.job_name}-2026-05-13T023120.528276209Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"dq-out/{subdir}/output/output_samples.jsonl"},
                {"Key": f"dq-out/{subdir}/record_level/evaluated_data.jsonl"},
            ]
        }
        s3.get_object.return_value = _get_object_response(30)
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job._count_samples()
        assert count == 30
        assert "evaluated_data.jsonl" in job._progress_target_key


class TestProgressLabels:
    """Progress display uses category-appropriate labels."""

    def test_dq_job_category_label(self, session_with_region):
        job = Job(
            job_name="dq-test",
            job_category="DataQualityEvaluation",
            session=session_with_region,
            expected_samples=100,
            progress_output_s3_uri="s3://bucket/out/",
        )
        assert job.job_category == "DataQualityEvaluation"

    def test_sdg_job_category_label(self, session_with_region):
        job = Job(
            job_name="sdg-test",
            job_category="SyntheticDataGeneration",
            session=session_with_region,
            expected_samples=10,
            progress_output_s3_uri="s3://bucket/out/",
        )
        assert job.job_category == "SyntheticDataGeneration"


class TestCsvProgressTracking:
    """CSV files are discovered and tracked for tabular SDG jobs."""

    @patch("boto3.client")
    def test_discovers_verified_samples_csv(self, mock_boto_client, session_with_region):
        """verified_samples.csv is discovered and line-counted for tabular jobs."""
        job = Job(
            job_name="tabular-sdg-2026-07-14-19-00-00-000-1",
            job_category="SyntheticDataGeneration",
            session=session_with_region,
            expected_samples=50,
            progress_output_s3_uri="s3://my-bucket/sdg-out/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        subdir = f"{job.job_name}-2026-07-14T190005.123456789Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{subdir}/job_metadata/verified_samples.csv"},
            ]
        }
        s3.get_object.return_value = _get_object_response(12)
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job._count_samples()
        assert count == 12
        assert "verified_samples.csv" in job._progress_target_key

    @patch("boto3.client")
    def test_csv_preferred_over_nothing(self, mock_boto_client, session_with_region):
        """When only .csv exists (no .jsonl), it is discovered correctly."""
        job = Job(
            job_name="tabular-only-2026-07-14-20-00-00-000-1",
            job_category="SyntheticDataGeneration",
            session=session_with_region,
            expected_samples=25,
            progress_output_s3_uri="s3://my-bucket/sdg-out/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        subdir = f"{job.job_name}-2026-07-14T200005.987654321Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{subdir}/output/output_samples.csv"},
            ]
        }
        s3.get_object.return_value = _get_object_response(8)
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job._count_samples()
        assert count == 8
        assert "output_samples.csv" in job._progress_target_key

    @patch("boto3.client")
    def test_jsonl_preferred_over_csv_when_both_exist(self, mock_boto_client, session_with_region):
        """When both verified_samples.jsonl and verified_samples.csv exist, .jsonl wins."""
        job = Job(
            job_name="mixed-format-2026-07-14-21-00-00-000-1",
            job_category="SyntheticDataGeneration",
            session=session_with_region,
            expected_samples=20,
            progress_output_s3_uri="s3://my-bucket/sdg-out/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        subdir = f"{job.job_name}-2026-07-14T210005.111222333Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{subdir}/job_metadata/verified_samples.jsonl"},
                {"Key": f"sdg-out/{subdir}/job_metadata/verified_samples.csv"},
            ]
        }
        s3.get_object.return_value = _get_object_response(15)
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job._count_samples()
        assert count == 15
        # Both match "verified_samples" — first in listing wins (jsonl listed first)
        assert "verified_samples" in job._progress_target_key


class TestParquetExclusion:
    """Binary formats (.parquet) are excluded from line-count progress tracking."""

    @patch("boto3.client")
    def test_parquet_file_not_tracked(self, mock_boto_client, session_with_region):
        """output_samples.parquet is ignored — not line-countable."""
        job = Job(
            job_name="parquet-job-2026-07-14-22-00-00-000-1",
            job_category="SyntheticDataGeneration",
            session=session_with_region,
            expected_samples=100,
            progress_output_s3_uri="s3://my-bucket/sdg-out/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        subdir = f"{job.job_name}-2026-07-14T220005.444555666Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{subdir}/output/output_samples.parquet"},
            ]
        }
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job._count_samples()
        assert count is None
        # get_object should NOT be called for parquet
        s3.get_object.assert_not_called()

    @patch("boto3.client")
    def test_csv_found_when_parquet_also_present(self, mock_boto_client, session_with_region):
        """When both .parquet and .csv exist, .csv is tracked (parquet skipped)."""
        job = Job(
            job_name="mixed-output-2026-07-14-23-00-00-000-1",
            job_category="SyntheticDataGeneration",
            session=session_with_region,
            expected_samples=30,
            progress_output_s3_uri="s3://my-bucket/sdg-out/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        subdir = f"{job.job_name}-2026-07-14T230005.777888999Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{subdir}/output/output_samples.parquet"},
                {"Key": f"sdg-out/{subdir}/output/output_samples.csv"},
            ]
        }
        s3.get_object.return_value = _get_object_response(22)
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job._count_samples()
        assert count == 22
        assert "output_samples.csv" in job._progress_target_key

    @patch("boto3.client")
    def test_sql_file_not_tracked(self, mock_boto_client, session_with_region):
        """output_samples.sql is ignored — not line-countable for progress."""
        job = Job(
            job_name="sql-job-2026-07-14-23-30-00-000-1",
            job_category="SyntheticDataGeneration",
            session=session_with_region,
            expected_samples=50,
            progress_output_s3_uri="s3://my-bucket/sdg-out/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        subdir = f"{job.job_name}-2026-07-14T233005.000111222Z"
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"sdg-out/{subdir}/output/output_samples.sql"},
            ]
        }
        mock_boto_client.return_value = s3
        _wire_paginator(s3)

        count = job._count_samples()
        assert count is None
        s3.get_object.assert_not_called()
