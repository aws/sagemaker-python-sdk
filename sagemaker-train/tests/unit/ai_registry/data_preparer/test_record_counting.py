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
"""Record-count off-by-one fix (ORR EVL-A-4 #8).

Both counters — the SDG progress counter (get_line_count inside
_job_resource.Job._count_samples) and the input-dataset record counter
(_count_input_dataset_records in preparer.DataPreparer) — previously used
``body.count(b'\\n')``, which under-counts by 1 whenever the JSONL file does
not end with a trailing newline.  A 200-record file without a trailing newline
reported 199.

The fix replaces the newline count with ``splitlines()`` + non-blank line
count, which is accurate regardless of trailing newline, \\r\\n endings, or
blank lines interspersed in the file.

Tests in this file exercise both counters with mocked S3 responses.
They ALL fail on the original code when the body lacks a trailing newline
(the primary bug) and pass after the fix.
"""

from __future__ import absolute_import

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch


from sagemaker.ai_registry.data_preparer._job_resource import Job

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(**kwargs):
    """Build a minimal Job instance with the minimum kwargs for _count_samples."""
    s = MagicMock()
    s.boto_region_name = "us-west-2"
    s.sagemaker_client = None
    defaults = dict(job_name="test-job", job_category="SyntheticDataGeneration", session=s)
    defaults.update(kwargs)
    return Job(**defaults)


def _s3_body(content: bytes):
    """Wrap bytes in a minimal S3 GetObject 'Body' mock.

    Configures both ``read()`` (for callers that materialise the whole body)
    and ``iter_chunks()`` (for the streaming line-counter) so the same helper
    works regardless of which API the code-under-test uses.
    """
    stream = MagicMock()
    stream.read.return_value = content
    # iter_chunks must yield the content as one chunk; empty content → no chunks.
    stream.iter_chunks.return_value = [content] if content else []
    return {"Body": stream}


# ---------------------------------------------------------------------------
# get_line_count inside Job._count_samples
# ---------------------------------------------------------------------------


class TestGetLineCountViaMockS3:
    """Exercises the get_line_count nested function inside _count_samples.

    All tests that include 'no_trailing_newline' FAIL on original code
    (body.count(b'\\n') is off by 1); they pass after the fix.
    """

    def _run_count(self, body_bytes: bytes, job_name: str = "test-job") -> int:
        """Exercise _count_samples against a faked S3 key."""
        job = _make_job(
            job_name=job_name,
            progress_output_s3_uri="s3://bucket/prefix/",
        )
        # Pre-seed the locked key so we skip the discovery listing
        job._progress_target_key = (
            f"prefix/{job_name}-2026-01-01T000000Z/job_metadata/verified_samples.jsonl"
        )

        s3_mock = MagicMock()
        s3_mock.get_object.return_value = _s3_body(body_bytes)

        with patch("boto3.client", return_value=s3_mock):
            return job._count_samples()

    # --- Primary bug: file without trailing newline ---

    def test_n_records_without_trailing_newline(self):
        """200 JSONL lines with no trailing \\n → must return 200, not 199.

        FAILS on original code (returns 199 via body.count(b'\\n')).
        """
        body = b"\n".join(b'{"id":' + str(i).encode() + b"}" for i in range(200))
        assert b"\n" not in body[-3:] or not body.endswith(
            b"\n"
        ), "Precondition: body must not end with newline"
        result = self._run_count(body)
        assert result == 200, f"Expected 200 records, got {result}"

    def test_n_records_with_trailing_newline(self):
        """200 JSONL lines WITH trailing \\n → must still return 200."""
        body = b"\n".join(b'{"id":' + str(i).encode() + b"}" for i in range(200)) + b"\n"
        result = self._run_count(body)
        assert result == 200, f"Expected 200 records, got {result}"

    # --- Single-record edge cases ---

    def test_single_record_no_trailing_newline(self):
        """1 record, no trailing \\n → 1. FAILS on original (returns 0)."""
        body = b'{"id": 0}'
        result = self._run_count(body)
        assert result == 1, f"Expected 1 record, got {result}"

    def test_single_record_with_trailing_newline(self):
        """1 record with trailing \\n → 1."""
        body = b'{"id": 0}\n'
        result = self._run_count(body)
        assert result == 1, f"Expected 1 record, got {result}"

    # --- Empty body ---

    def test_empty_body_returns_zero(self):
        """Empty file → 0 records."""
        result = self._run_count(b"")
        assert result == 0

    # --- Blank / whitespace-only lines are skipped ---

    def test_blank_trailing_lines_not_counted(self):
        """Trailing blank lines must not be counted as records."""
        body = b'{"id": 0}\n{"id": 1}\n\n\n'
        result = self._run_count(body)
        assert result == 2, f"Expected 2 records (not counting blank lines), got {result}"

    def test_blank_lines_mixed_in_body_not_counted(self):
        """Blank lines interspersed in the body must not be counted."""
        body = b'{"id": 0}\n\n{"id": 1}\n\n{"id": 2}'
        result = self._run_count(body)
        assert result == 3, f"Expected 3 records, got {result}"

    # --- CRLF line endings ---

    def test_crlf_endings_no_trailing_newline(self):
        r"""\\r\\n line endings, no trailing \\r\\n → correct count.

        FAILS on original code (body.count(b'\\n') handles \\r\\n but
        is still off by 1 without trailing newline).
        """
        body = b'{"id": 0}\r\n{"id": 1}\r\n{"id": 2}'
        result = self._run_count(body)
        assert result == 3, f"Expected 3 records (CRLF), got {result}"

    def test_crlf_endings_with_trailing_newline(self):
        r"""\\r\\n line endings WITH trailing \\r\\n → correct count."""
        body = b'{"id": 0}\r\n{"id": 1}\r\n{"id": 2}\r\n'
        result = self._run_count(body)
        assert result == 3, f"Expected 3 records (CRLF), got {result}"


# ---------------------------------------------------------------------------
# _count_input_dataset_records in DataPreparer
# ---------------------------------------------------------------------------


class TestCountInputDatasetRecords:
    """Exercises _count_input_dataset_records, which uses the same
    splitlines() approach after the fix.

    All tests with 'no_trailing_newline' FAIL on original code.
    """

    def _build_preparer_and_count(self, body_bytes: bytes) -> int:
        """Build a minimal DataPreparer instance and invoke the counter."""
        from sagemaker.ai_registry.data_preparer.preparer import DataPreparer

        session_mock = MagicMock()
        session_mock.boto_region_name = "us-west-2"

        # DataPreparer is a Pydantic BaseModel — use proper construction.
        preparer = DataPreparer(sagemaker_session=session_mock)

        dataset_mock = MagicMock()
        dataset_mock.source = "s3://bucket/prefix/data.jsonl"

        s3_mock = MagicMock()
        listing_response = {"Contents": [{"Key": "prefix/data.jsonl"}]}
        s3_mock.list_objects_v2.return_value = listing_response
        s3_body_mock = MagicMock()
        s3_body_mock.read.return_value = body_bytes
        s3_body_mock.iter_chunks.return_value = [body_bytes] if body_bytes else []
        s3_mock.get_object.return_value = {"Body": s3_body_mock}

        with (
            patch(
                "sagemaker.ai_registry.data_preparer.preparer._parse_dataset_arn",
                return_value=("my-dataset", "my-hub", "1.0.0"),
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                return_value=dataset_mock,
            ),
            patch(
                "boto3.client",
                return_value=s3_mock,
            ),
        ):
            return preparer._count_input_dataset_records(
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/D/1.0.0"
            )

    def test_200_records_no_trailing_newline(self):
        """200 records without trailing \\n → 200. FAILS on original (returns 199)."""
        body = b"\n".join(b'{"id":' + str(i).encode() + b"}" for i in range(200))
        result = self._build_preparer_and_count(body)
        assert result == 200, f"Expected 200, got {result}"

    def test_200_records_with_trailing_newline(self):
        """200 records WITH trailing \\n → 200."""
        body = b"\n".join(b'{"id":' + str(i).encode() + b"}" for i in range(200)) + b"\n"
        result = self._build_preparer_and_count(body)
        assert result == 200, f"Expected 200, got {result}"

    def test_single_record_no_trailing_newline(self):
        """1 record, no trailing \\n → 1. FAILS on original (returns 0)."""
        body = b'{"id": 0}'
        result = self._build_preparer_and_count(body)
        assert result == 1, f"Expected 1, got {result}"

    def test_empty_body_returns_zero(self):
        """Empty file → 0."""
        result = self._build_preparer_and_count(b"")
        assert result == 0

    def test_blank_trailing_lines_not_counted(self):
        """Trailing blank lines must not be counted."""
        body = b'{"a": 1}\n{"b": 2}\n\n'
        result = self._build_preparer_and_count(body)
        assert result == 2, f"Expected 2, got {result}"

    def test_crlf_no_trailing_newline(self):
        r"""\\r\\n endings without trailing newline → correct count.

        FAILS on original code.
        """
        body = b'{"a": 1}\r\n{"b": 2}\r\n{"c": 3}'
        result = self._build_preparer_and_count(body)
        assert result == 3, f"Expected 3, got {result}"


# ---------------------------------------------------------------------------
# DataQualityEvaluation progress lookup
# ---------------------------------------------------------------------------


class TestDqProgressLookup:
    """A DQ job counts only its own evaluated_data.jsonl, found by name and creation time."""

    _NEW = "out/data-quality-evaluation/dq-job/2026-10-27T192836.911415410Z/record_level/evaluated_data.jsonl"
    _OLD = "out/dq-job-2026-10-27T192836.911415410Z/record_level/evaluated_data.jsonl"
    _EARLIER_RUN = "out/dq-job-2026-09-01T080000Z/record_level/evaluated_data.jsonl"

    def _count(self, keys, created_at="2026-10-27T19:28:36.911415410Z"):
        job = _make_job(
            job_name="dq-job",
            job_category="DataQualityEvaluation",
            progress_output_s3_uri="s3://bucket/out/",
            created_at=created_at,
        )
        s3_mock = MagicMock()
        s3_mock.get_paginator.return_value.paginate.side_effect = (
            lambda Bucket, Prefix, PaginationConfig, **kw: [
                {"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}
            ]
        )
        s3_mock.get_object.side_effect = lambda Bucket, Key, **kw: _s3_body(
            b'{"id": 0}\n{"id": 1}\n{"id": 2}\n' if Key != self._EARLIER_RUN else b'{"id": 0}\n'
        )
        with patch("boto3.client", return_value=s3_mock):
            return job._count_samples(), job._progress_target_key

    def test_finds_the_v3_layout(self):
        assert self._count([self._NEW, self._EARLIER_RUN]) == (3, self._NEW)

    def test_finds_the_v2_layout(self):
        assert self._count([self._OLD, self._EARLIER_RUN]) == (3, self._OLD)

    def test_never_counts_an_earlier_job_that_reused_the_name(self):
        assert self._count([self._EARLIER_RUN]) == (None, None)

    def test_accepts_a_describe_job_datetime(self):
        created = datetime(2026, 10, 27, 12, 28, 36, 911415, tzinfo=timezone(timedelta(hours=-7)))
        assert self._count([self._NEW], created_at=created) == (3, self._NEW)

    def test_unknown_creation_time_counts_nothing(self):
        assert self._count([self._NEW, self._EARLIER_RUN], created_at=None) == (None, None)
