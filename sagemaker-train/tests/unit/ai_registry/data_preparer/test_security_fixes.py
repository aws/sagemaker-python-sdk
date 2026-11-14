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
"""Security-fix unit tests (CSE scan cse/r2riv-scan).

Covers:
- W1  ExpectedBucketOwner asserted on S3 calls against the SDK default bucket and
      absent for other buckets (CWE-283/284).
- W2  Bounded Range read in _preview_output (CWE-400/770); streaming line-counter
      in _count_input_dataset_records and Job._count_samples (same).
- W3  timeout forwarded to job.wait() by attach / stop / _run (CWE-400/1088).
- W4  _parse_dataset_arn returns a hub ARN for cross-account inputs (CWE-706/20).

All tests are hermetic: no network, no AWS credentials, no real boto3 calls.
"""

from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer._s3_session import _s3_owner_kwargs
from sagemaker.ai_registry.data_preparer._job_resource import Job
from sagemaker.ai_registry.data_preparer.preparer import DataPreparer, _parse_dataset_arn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(is_default: bool, account: str = "111122223333"):
    """Return a session mock whose default-bucket probe returns *account* when
    *is_default* is True, else None."""
    s = MagicMock()
    s.boto_region_name = "us-west-2"
    s._get_account_id_if_default_bucket.return_value = account if is_default else None
    return s


def _s3_streaming_body(content: bytes):
    """Return a fake S3 Body that supports both read() and iter_chunks()."""
    body = MagicMock()
    body.read.return_value = content
    body.iter_chunks.return_value = [content] if content else []
    return body


# ---------------------------------------------------------------------------
# W1 — _s3_owner_kwargs helper (unit tests for the helper itself)
# ---------------------------------------------------------------------------


class TestS3OwnerKwargs:
    """_s3_owner_kwargs returns the correct dict depending on default-bucket status."""

    def test_returns_expected_owner_for_default_bucket(self):
        session = _session(is_default=True, account="123456789012")
        result = _s3_owner_kwargs(session, "sagemaker-us-west-2-123456789012")
        assert result == {"ExpectedBucketOwner": "123456789012"}

    def test_returns_empty_dict_for_non_default_bucket(self):
        session = _session(is_default=False)
        result = _s3_owner_kwargs(session, "some-other-bucket")
        assert result == {}


# ---------------------------------------------------------------------------
# W1 — ExpectedBucketOwner present / absent at each S3 call site
# ---------------------------------------------------------------------------


class TestOwnerKwargsInOutputHasNoRecords:
    """_output_has_no_records passes ExpectedBucketOwner on get_object when
    the bucket is the session's SDK-generated default, and omits it otherwise."""

    def _run(self, is_default: bool) -> MagicMock:
        session = _session(is_default=is_default)
        preparer = DataPreparer(sagemaker_session=session)
        s3 = MagicMock()
        content = b'{"q": 1}\n'
        s3.head_object.return_value = {"ContentLength": len(content)}
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: content)}
        with patch("boto3.client", return_value=s3):
            preparer._output_has_no_records("s3://mybucket/key.jsonl")
        return s3

    def test_expected_owner_present_for_default_bucket(self):
        s3 = self._run(is_default=True)
        kwargs = s3.get_object.call_args.kwargs
        assert "ExpectedBucketOwner" in kwargs
        assert kwargs["ExpectedBucketOwner"] == "111122223333"

    def test_expected_owner_absent_for_non_default_bucket(self):
        s3 = self._run(is_default=False)
        kwargs = s3.get_object.call_args.kwargs
        assert "ExpectedBucketOwner" not in kwargs


class TestOwnerKwargsInPreviewOutput:
    """_preview_output passes ExpectedBucketOwner on get_object iff default bucket."""

    def _run(self, is_default: bool) -> MagicMock:
        session = _session(is_default=is_default)
        preparer = DataPreparer(sagemaker_session=session)
        dataset = MagicMock()
        dataset.source = "s3://mybucket/jobs/sdg/output/"
        from sagemaker.ai_registry.data_preparer.constants import JobCategory

        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = b'{"a": 1}\n'
        s3.get_object.return_value = {"Body": body}
        with patch("boto3.client", return_value=s3):
            preparer._preview_output(dataset, JobCategory.SYNTHETIC_DATA_GENERATION)
        return s3

    def test_expected_owner_present_for_default_bucket(self):
        s3 = self._run(is_default=True)
        kwargs = s3.get_object.call_args.kwargs
        assert "ExpectedBucketOwner" in kwargs
        assert kwargs["ExpectedBucketOwner"] == "111122223333"

    def test_expected_owner_absent_for_non_default_bucket(self):
        s3 = self._run(is_default=False)
        kwargs = s3.get_object.call_args.kwargs
        assert "ExpectedBucketOwner" not in kwargs


class TestOwnerKwargsInCountInputDatasetRecords:
    """_count_input_dataset_records passes ExpectedBucketOwner on both
    list_objects_v2 and get_object when the bucket is the default."""

    def _run(self, is_default: bool) -> MagicMock:
        session = _session(is_default=is_default)
        preparer = DataPreparer(sagemaker_session=session)
        dataset = MagicMock()
        dataset.source = "s3://mybucket/prefix/data.jsonl"
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"Contents": [{"Key": "prefix/data.jsonl"}]}
        body = _s3_streaming_body(b'{"x": 1}\n')
        s3.get_object.return_value = {"Body": body}
        with (
            patch(
                "sagemaker.ai_registry.data_preparer.preparer._parse_dataset_arn",
                return_value=("ds", "hub", "1.0"),
            ),
            patch("sagemaker.ai_registry.dataset.DataSet.get", return_value=dataset),
            patch("boto3.client", return_value=s3),
        ):
            preparer._count_input_dataset_records(
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/ds/1.0"
            )
        return s3

    def test_list_objects_has_expected_owner_for_default_bucket(self):
        s3 = self._run(is_default=True)
        kw = s3.list_objects_v2.call_args.kwargs
        assert kw.get("ExpectedBucketOwner") == "111122223333"

    def test_get_object_has_expected_owner_for_default_bucket(self):
        s3 = self._run(is_default=True)
        kw = s3.get_object.call_args.kwargs
        assert kw.get("ExpectedBucketOwner") == "111122223333"

    def test_list_objects_no_expected_owner_for_non_default_bucket(self):
        s3 = self._run(is_default=False)
        kw = s3.list_objects_v2.call_args.kwargs
        assert "ExpectedBucketOwner" not in kw

    def test_get_object_no_expected_owner_for_non_default_bucket(self):
        s3 = self._run(is_default=False)
        kw = s3.get_object.call_args.kwargs
        assert "ExpectedBucketOwner" not in kw


class TestOwnerKwargsInCountSamples:
    """Job._count_samples passes ExpectedBucketOwner on list_objects_v2 paginator
    and get_object when the bucket is the session's default."""

    def _make_job(self, is_default: bool) -> tuple:
        session = _session(is_default=is_default)
        job = Job(
            job_name="my-sdg-job",
            job_category="SyntheticDataGeneration",
            session=session,
            progress_output_s3_uri="s3://mybucket/prefix/",
        )
        job._progress_target_key = (
            "prefix/my-sdg-job-2026-01-01T000000Z/job_metadata/verified_samples.jsonl"
        )
        return job, session

    def test_get_object_has_expected_owner_for_default_bucket(self):
        job, _ = self._make_job(is_default=True)
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        body = _s3_streaming_body(b'{"id": 0}\n')
        s3.get_object.return_value = {"Body": body}
        with patch("boto3.client", return_value=s3):
            job._count_samples()
        kw = s3.get_object.call_args.kwargs
        assert kw.get("ExpectedBucketOwner") == "111122223333"

    def test_get_object_no_expected_owner_for_non_default_bucket(self):
        job, _ = self._make_job(is_default=False)
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        body = _s3_streaming_body(b'{"id": 0}\n')
        s3.get_object.return_value = {"Body": body}
        with patch("boto3.client", return_value=s3):
            job._count_samples()
        kw = s3.get_object.call_args.kwargs
        assert "ExpectedBucketOwner" not in kw

    def test_paginator_has_expected_owner_for_default_bucket(self):
        """_list_under passes ExpectedBucketOwner to the paginator on the default bucket."""
        session = _session(is_default=True)
        job = Job(
            job_name="my-sdg-job",
            job_category="SyntheticDataGeneration",
            session=session,
            progress_output_s3_uri="s3://mybucket/prefix/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        # No progress_target_key → discovery triggers _list_under
        paginator = MagicMock()
        paginator.paginate.return_value = iter([{"Contents": []}])
        s3.get_paginator.return_value = paginator
        with patch("boto3.client", return_value=s3):
            job._count_samples()
        kw = paginator.paginate.call_args.kwargs
        assert kw.get("ExpectedBucketOwner") == "111122223333"

    def test_paginator_no_expected_owner_for_non_default_bucket(self):
        session = _session(is_default=False)
        job = Job(
            job_name="my-sdg-job",
            job_category="SyntheticDataGeneration",
            session=session,
            progress_output_s3_uri="s3://mybucket/prefix/",
        )
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        paginator = MagicMock()
        paginator.paginate.return_value = iter([{"Contents": []}])
        s3.get_paginator.return_value = paginator
        with patch("boto3.client", return_value=s3):
            job._count_samples()
        kw = paginator.paginate.call_args.kwargs
        assert "ExpectedBucketOwner" not in kw


# ---------------------------------------------------------------------------
# W2 — bounded Range read in _preview_output
# ---------------------------------------------------------------------------


class TestPreviewOutputBoundedRead:
    """_preview_output fetches only a bounded Range, not the whole object.
    A mid-line truncation (the Range lands mid-record) must still parse cleanly.
    """

    @pytest.fixture
    def preparer(self):
        session = _session(is_default=False)
        return DataPreparer(sagemaker_session=session)

    @pytest.fixture
    def dataset(self):
        ds = MagicMock()
        ds.source = "s3://bucket/jobs/sdg/output/"
        return ds

    @patch("boto3.client")
    def test_range_header_is_sent(self, mock_boto3, preparer, dataset):
        """get_object must receive a Range parameter (bounded fetch)."""
        from sagemaker.ai_registry.data_preparer.constants import JobCategory
        from sagemaker.ai_registry.data_preparer.preparer import _PREVIEW_MAX_BYTES

        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        body = MagicMock()
        body.read.return_value = b'{"a": 1}\n'
        mock_s3.get_object.return_value = {"Body": body}

        preparer._preview_output(dataset, JobCategory.SYNTHETIC_DATA_GENERATION)

        call_kwargs = mock_s3.get_object.call_args.kwargs
        assert "Range" in call_kwargs
        assert call_kwargs["Range"] == f"bytes=0-{_PREVIEW_MAX_BYTES - 1}"

    @patch("boto3.client")
    def test_midline_truncation_does_not_raise(self, mock_boto3, preparer, dataset):
        """If the Range fetch lands mid-line, the truncated last record is
        dropped and the result still parses without error.
        """
        from sagemaker.ai_registry.data_preparer.constants import JobCategory

        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        # Three full lines then a partial (no newline at end) — simulates Range truncation.
        raw = (
            b'{"q": "hello"}\n'
            b'{"q": "world"}\n'
            b'{"q": "foo"}\n'
            b'{"q": "truncat'  # incomplete last record
        )
        body = MagicMock()
        body.read.return_value = raw
        mock_s3.get_object.return_value = {"Body": body}

        # Must not raise even though the last record is truncated.
        preparer._preview_output(dataset, JobCategory.SYNTHETIC_DATA_GENERATION)

    @patch("boto3.client")
    def test_whole_object_is_not_read(self, mock_boto3, preparer, dataset):
        """The method must not issue a plain get_object without Range (which
        would download the full file)."""
        from sagemaker.ai_registry.data_preparer.constants import JobCategory

        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        body = MagicMock()
        body.read.return_value = b'{"a": 1}\n'
        mock_s3.get_object.return_value = {"Body": body}

        preparer._preview_output(dataset, JobCategory.SYNTHETIC_DATA_GENERATION)

        call_kwargs = mock_s3.get_object.call_args.kwargs
        # Range must be present — an unconditional full-file read would omit it.
        assert "Range" in call_kwargs


# ---------------------------------------------------------------------------
# W2 — streaming line counter (get_line_count inside Job._count_samples)
# ---------------------------------------------------------------------------


class TestGetLineCountStreaming:
    """Exercises get_line_count via _count_samples with chunk-split payloads.
    All results must be identical to what a full-body read would have produced.
    """

    def _count(self, chunks: list) -> int:
        """Feed *chunks* to _count_samples via a streaming Body mock."""
        session = _session(is_default=False)
        job = Job(
            job_name="my-sdg",
            job_category="SyntheticDataGeneration",
            session=session,
            progress_output_s3_uri="s3://bucket/prefix/",
        )
        job._progress_target_key = (
            "prefix/my-sdg-2026-01-01T000000Z/job_metadata/verified_samples.jsonl"
        )
        body = MagicMock()
        body.iter_chunks.return_value = iter(chunks)
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        s3.get_object.return_value = {"Body": body}
        with patch("boto3.client", return_value=s3):
            return job._count_samples()

    def test_no_trailing_newline(self):
        """200 records without \\n at end → 200."""
        data = b"\n".join(b'{"id":' + str(i).encode() + b"}" for i in range(200))
        assert self._count([data]) == 200

    def test_with_trailing_newline(self):
        """200 records WITH trailing \\n → 200."""
        data = b"\n".join(b'{"id":' + str(i).encode() + b"}" for i in range(200)) + b"\n"
        assert self._count([data]) == 200

    def test_crlf_endings(self):
        r"""\\r\\n line endings → correct count."""
        data = b'{"a": 1}\r\n{"b": 2}\r\n{"c": 3}'
        assert self._count([data]) == 3

    def test_blank_lines_not_counted(self):
        """Blank and whitespace-only lines must be skipped."""
        data = b'{"a": 1}\n\n  \n{"b": 2}\n\n'
        assert self._count([data]) == 2

    def test_payload_split_across_chunk_boundary(self):
        """A record split across two chunks must be counted exactly once.

        Splitting at an arbitrary byte position simulates a real stream whose
        chunks don't align with line endings.
        """
        line1 = b'{"id": 0}\n'
        line2 = b'{"id": 1}\n'
        line3 = b'{"id": 2}'  # no trailing newline

        full = line1 + line2 + line3
        # Split at a byte that lands mid-record of line2.
        split = len(line1) + 3
        chunk_a = full[:split]
        chunk_b = full[split:]
        assert self._count([chunk_a, chunk_b]) == 3

    def test_empty_body_returns_zero(self):
        assert self._count([]) == 0

    def test_nosuchkey_returns_none(self):
        session = _session(is_default=False)
        job = Job(
            job_name="my-sdg",
            job_category="SyntheticDataGeneration",
            session=session,
            progress_output_s3_uri="s3://bucket/prefix/",
        )
        job._progress_target_key = (
            "prefix/my-sdg-2026-01-01T000000Z/job_metadata/verified_samples.jsonl"
        )
        NoSuchKey = type("NoSuchKey", (Exception,), {})
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = NoSuchKey
        s3.get_object.side_effect = NoSuchKey("gone")
        with patch("boto3.client", return_value=s3):
            assert job._count_samples() is None


# ---------------------------------------------------------------------------
# W2 — streaming line counter (_count_input_dataset_records in DataPreparer)
# ---------------------------------------------------------------------------


class TestCountInputDatasetRecordsStreaming:
    """Streaming line-count in _count_input_dataset_records preserves identical
    semantics to the old full-body read.
    """

    def _count(self, body_bytes: bytes) -> int:
        session = _session(is_default=False)
        preparer = DataPreparer(sagemaker_session=session)
        dataset = MagicMock()
        dataset.source = "s3://bucket/prefix/data.jsonl"
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"Contents": [{"Key": "prefix/data.jsonl"}]}
        s3.get_object.return_value = {"Body": _s3_streaming_body(body_bytes)}
        with (
            patch(
                "sagemaker.ai_registry.data_preparer.preparer._parse_dataset_arn",
                return_value=("ds", "hub", "1.0"),
            ),
            patch("sagemaker.ai_registry.dataset.DataSet.get", return_value=dataset),
            patch("boto3.client", return_value=s3),
        ):
            return preparer._count_input_dataset_records(
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/ds/1.0"
            )

    def test_no_trailing_newline(self):
        data = b"\n".join(b'{"id":' + str(i).encode() + b"}" for i in range(50))
        assert self._count(data) == 50

    def test_with_trailing_newline(self):
        data = b"\n".join(b'{"id":' + str(i).encode() + b"}" for i in range(50)) + b"\n"
        assert self._count(data) == 50

    def test_crlf_endings(self):
        data = b'{"a": 1}\r\n{"b": 2}\r\n{"c": 3}'
        assert self._count(data) == 3

    def test_blank_lines_not_counted(self):
        data = b'{"a": 1}\n\n{"b": 2}\n\n'
        assert self._count(data) == 2

    def test_payload_split_across_chunks(self):
        """A chunk boundary that lands mid-record must still count correctly."""
        session = _session(is_default=False)
        preparer = DataPreparer(sagemaker_session=session)
        dataset = MagicMock()
        dataset.source = "s3://bucket/prefix/data.jsonl"
        # Two chunks: first ends mid-second-record, second finishes it.
        chunk_a = b'{"id": 0}\n{"id'
        chunk_b = b'": 1}\n{"id": 2}'
        body = MagicMock()
        body.iter_chunks.return_value = [chunk_a, chunk_b]
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"Contents": [{"Key": "prefix/data.jsonl"}]}
        s3.get_object.return_value = {"Body": body}
        with (
            patch(
                "sagemaker.ai_registry.data_preparer.preparer._parse_dataset_arn",
                return_value=("ds", "hub", "1.0"),
            ),
            patch("sagemaker.ai_registry.dataset.DataSet.get", return_value=dataset),
            patch("boto3.client", return_value=s3),
        ):
            result = preparer._count_input_dataset_records(
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/ds/1.0"
            )
        assert result == 3

    def test_empty_body_returns_zero(self):
        assert self._count(b"") == 0


# ---------------------------------------------------------------------------
# W3 — timeout forwarded to job.wait()
# ---------------------------------------------------------------------------


class _StubJob:
    """Minimal real class so isinstance(arn_str, _StubJob) stays type-safe.

    DataPreparer.stop uses ``isinstance(job_or_arn, Job)`` to choose between
    the handle path and the ARN path. Using a MagicMock for Job would cause
    ``isinstance()`` to raise TypeError because its second arg must be a type.
    """

    get = MagicMock()


class TestTimeoutForwarding:
    """timeout is forwarded to job.wait() by all three public entry-points.
    The default is None so existing callers are unaffected.
    """

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_attach_forwards_timeout(self, mock_get_session, mock_get_job_cls):
        mock_get_session.return_value = MagicMock()
        job = MagicMock()
        job.job_name = "j"
        job.job_status = "InProgress"
        job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j"
        job.job_config_document = "{}"
        job.job_config_schema_version = "1.0.0"

        def fake_wait(logs=False, timeout=None):
            job.job_status = "Completed"

        job.wait.side_effect = fake_wait
        job.refresh.return_value = None

        MockJob = MagicMock()
        MockJob.get.return_value = job
        mock_get_job_cls.return_value = MockJob

        with patch.object(DataPreparer, "_rehydrate_datasets", return_value=[MagicMock()]):
            with patch.object(DataPreparer, "_preview_output"):
                DataPreparer.attach(
                    "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j",
                    timeout=120,
                )

        job.wait.assert_called_once_with(logs=False, timeout=120)

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_attach_default_timeout_is_none(self, mock_get_session, mock_get_job_cls):
        mock_get_session.return_value = MagicMock()
        job = MagicMock()
        job.job_name = "j"
        job.job_status = "InProgress"
        job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j"
        job.job_config_document = "{}"
        job.job_config_schema_version = "1.0.0"

        def fake_wait(logs=False, timeout=None):
            job.job_status = "Completed"

        job.wait.side_effect = fake_wait
        job.refresh.return_value = None

        MockJob = MagicMock()
        MockJob.get.return_value = job
        mock_get_job_cls.return_value = MockJob

        with patch.object(DataPreparer, "_rehydrate_datasets", return_value=[MagicMock()]):
            with patch.object(DataPreparer, "_preview_output"):
                DataPreparer.attach("arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j")

        job.wait.assert_called_once_with(logs=False, timeout=None)

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_forwards_timeout(self, mock_get_session, mock_get_job_cls):
        mock_get_session.return_value = MagicMock()
        job = _StubJob()
        job.job_name = "j"
        job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j"
        job.job_status = "InProgress"
        job.stop = MagicMock()
        job.refresh = MagicMock()
        job.wait = MagicMock()

        def fake_wait(logs=False, timeout=None):
            job.job_status = "Stopped"

        job.wait.side_effect = fake_wait

        _StubJob.get.reset_mock()
        _StubJob.get.return_value = job
        mock_get_job_cls.return_value = _StubJob

        DataPreparer.stop(
            "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j",
            wait=True,
            timeout=300,
        )

        job.wait.assert_called_once_with(logs=False, timeout=300)

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_default_timeout_is_none(self, mock_get_session, mock_get_job_cls):
        mock_get_session.return_value = MagicMock()
        job = _StubJob()
        job.job_name = "j"
        job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j"
        job.job_status = "InProgress"
        job.stop = MagicMock()
        job.refresh = MagicMock()
        job.wait = MagicMock()

        def fake_wait(logs=False, timeout=None):
            job.job_status = "Stopped"

        job.wait.side_effect = fake_wait

        _StubJob.get.reset_mock()
        _StubJob.get.return_value = job
        mock_get_job_cls.return_value = _StubJob

        DataPreparer.stop(
            "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j", wait=True
        )

        job.wait.assert_called_once_with(logs=False, timeout=None)

    def test_run_forwards_timeout_to_wait(self):
        """_run passes timeout through to job.wait() when set."""
        session = _session(is_default=False)
        session.account_id.return_value = "123"
        preparer = DataPreparer(sagemaker_session=session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j"
        mock_job.job_config_document = "{}"
        mock_job.job_config_schema_version = "1.0.0"
        mock_job.failure_reason = None

        from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
        from sagemaker.ai_registry.data_preparer.constants import JobCategory

        dataset = DatasetJobConfig(job_category=JobCategory.SYNTHETIC_DATA_GENERATION)

        with patch.object(dataset, "_resolve_and_validate"):
            with patch.object(preparer, "_create_job", return_value=mock_job):
                with patch.object(preparer, "_rehydrate_datasets", return_value=[MagicMock()]):
                    with patch.object(preparer, "_preview_output"):
                        preparer._run(
                            dataset,
                            JobCategory.SYNTHETIC_DATA_GENERATION,
                            wait=True,
                            timeout=600,
                        )

        mock_job.wait.assert_called_once_with(logs=True, timeout=600)

    def test_run_default_timeout_is_none(self):
        session = _session(is_default=False)
        session.account_id.return_value = "123"
        preparer = DataPreparer(sagemaker_session=session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/j"
        mock_job.job_config_document = "{}"
        mock_job.job_config_schema_version = "1.0.0"
        mock_job.failure_reason = None

        from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
        from sagemaker.ai_registry.data_preparer.constants import JobCategory

        dataset = DatasetJobConfig(job_category=JobCategory.SYNTHETIC_DATA_GENERATION)

        with patch.object(dataset, "_resolve_and_validate"):
            with patch.object(preparer, "_create_job", return_value=mock_job):
                with patch.object(preparer, "_rehydrate_datasets", return_value=[MagicMock()]):
                    with patch.object(preparer, "_preview_output"):
                        preparer._run(dataset, JobCategory.SYNTHETIC_DATA_GENERATION, wait=True)

        mock_job.wait.assert_called_once_with(logs=True, timeout=None)


# ---------------------------------------------------------------------------
# W4 — _parse_dataset_arn returns a hub ARN for cross-account inputs
# ---------------------------------------------------------------------------


class TestParseDatasetArnHubArn:
    """_parse_dataset_arn must return a fully-qualified hub ARN carrying the
    ARN's own partition/region/account so that cross-account lookups resolve
    against the correct account rather than the local session's account.
    """

    def test_cross_account_arn_returns_hub_arn(self):
        """An ARN owned by a different account yields a hub ARN for that account."""
        content, hub, version = _parse_dataset_arn(
            "arn:aws:sagemaker:us-east-1:999988887777:hub-content/ExtHub/DataSet/ds-name/2.1.0"
        )
        assert content == "ds-name"
        assert version == "2.1.0"
        # The hub ARN must carry the ARN's own account, not the session account.
        assert hub == "arn:aws:sagemaker:us-east-1:999988887777:hub/ExtHub"

    def test_same_account_arn_returns_hub_arn(self):
        """Same-account ARN → hub ARN still built correctly."""
        content, hub, version = _parse_dataset_arn(
            "arn:aws:sagemaker:us-west-2:123456789012:hub-content/MyHub/DataSet/my-ds/3.0.0"
        )
        assert content == "my-ds"
        assert version == "3.0.0"
        assert hub == "arn:aws:sagemaker:us-west-2:123456789012:hub/MyHub"

    def test_hub_arn_contains_arns_own_region(self):
        """Region in hub ARN comes from the dataset ARN, not from anything else."""
        _, hub, _ = _parse_dataset_arn(
            "arn:aws:sagemaker:ap-southeast-1:111222333444:hub-content/H/DataSet/n/1.0.0"
        )
        assert "ap-southeast-1" in hub

    def test_hub_arn_contains_arns_own_account(self):
        """Account in hub ARN comes from the dataset ARN."""
        _, hub, _ = _parse_dataset_arn(
            "arn:aws:sagemaker:us-east-1:555666777888:hub-content/MyHub/DataSet/n/5.0.0"
        )
        assert "555666777888" in hub

    @pytest.mark.parametrize(
        "arn, expected_name",
        [
            # Fallback 1: version segment absent or contains /
            (
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/n/1.0/extra",
                "n",
            ),
            # Fallback 2: not a hub-content ARN
            ("arn:aws:sagemaker:us-west-2:123:hub/H/DataSet/n/1", "n"),
            # Fallback 3: bare name / garbage
            ("some-name/1.0.0", "some-name"),
            ("garbage", "garbage"),
        ],
    )
    def test_existing_fallbacks_return_name_none_none(self, arn, expected_name):
        """All existing fallback paths must still return (name, None, None)."""
        result = _parse_dataset_arn(arn)
        assert result == (expected_name, None, None)


# ---------------------------------------------------------------------------
# D1 — head_object ownership kwargs (reviewer finding: previously missing)
# ---------------------------------------------------------------------------


class TestHeadObjectOwnerKwargsInOutputHasNoRecords:
    """_output_has_no_records must pass ExpectedBucketOwner to head_object (D1),
    not only to get_object.  HeadObject supports ExpectedBucketOwner the same as
    GetObject; omitting it left ownership unasserted on the size probe.
    """

    def _run(self, is_default: bool) -> MagicMock:
        session = _session(is_default=is_default)
        preparer = DataPreparer(sagemaker_session=session)
        s3 = MagicMock()
        content = b'{"q": 1}\n'
        s3.head_object.return_value = {"ContentLength": len(content)}
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: content)}
        with patch("boto3.client", return_value=s3):
            preparer._output_has_no_records("s3://mybucket/key.jsonl")
        return s3

    def test_head_object_has_expected_owner_for_default_bucket(self):
        """head_object must receive ExpectedBucketOwner when the bucket is the
        SDK-generated default bucket."""
        s3 = self._run(is_default=True)
        kw = s3.head_object.call_args.kwargs
        assert (
            "ExpectedBucketOwner" in kw
        ), "head_object was called without ExpectedBucketOwner on the default bucket"
        assert kw["ExpectedBucketOwner"] == "111122223333"

    def test_head_object_no_expected_owner_for_non_default_bucket(self):
        """head_object must NOT receive ExpectedBucketOwner for a non-default bucket."""
        s3 = self._run(is_default=False)
        kw = s3.head_object.call_args.kwargs
        assert "ExpectedBucketOwner" not in kw


# ---------------------------------------------------------------------------
# D4 — _preview_output truncated-no-newline vs small-complete-no-newline
# ---------------------------------------------------------------------------


class TestPreviewOutputTruncatedNoNewline:
    """When a Range read fills _PREVIEW_MAX_BYTES entirely and contains no
    newline, the data is a truncated record — parsing it would fail so the
    preview must be skipped (D4 fix).  A small file whose sole record has no
    trailing newline must still be parsed successfully.
    """

    @pytest.fixture
    def preparer(self):
        session = _session(is_default=False)
        return DataPreparer(sagemaker_session=session)

    @pytest.fixture
    def dataset(self):
        ds = MagicMock()
        ds.source = "s3://bucket/jobs/sdg/output/"
        return ds

    @patch("boto3.client")
    def test_full_window_no_newline_skips_preview(self, mock_boto3, preparer, dataset):
        """When the Range response is exactly _PREVIEW_MAX_BYTES and contains no
        newline, _preview_output must return without calling pd.read_json (the
        record is truncated and would raise on parse)."""
        from sagemaker.ai_registry.data_preparer.constants import JobCategory
        from sagemaker.ai_registry.data_preparer.preparer import _PREVIEW_MAX_BYTES

        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        # Exactly _PREVIEW_MAX_BYTES of data with no newline — truncated record.
        raw = b"x" * _PREVIEW_MAX_BYTES
        body = MagicMock()
        body.read.return_value = raw
        mock_s3.get_object.return_value = {"Body": body}

        with patch("pandas.read_json") as mock_read_json:
            preparer._preview_output(dataset, JobCategory.SYNTHETIC_DATA_GENERATION)

        mock_read_json.assert_not_called()

    @patch("boto3.client")
    def test_small_complete_no_newline_is_parsed(self, mock_boto3, preparer, dataset):
        """A file smaller than _PREVIEW_MAX_BYTES with one complete record and
        no trailing newline must be parsed successfully (content is complete)."""
        from sagemaker.ai_registry.data_preparer.constants import JobCategory

        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        # One complete JSON record, no trailing newline, well below the window.
        raw = b'{"question": "What is 2+2?", "answer": "4"}'
        body = MagicMock()
        body.read.return_value = raw
        mock_s3.get_object.return_value = {"Body": body}

        called_with = {}

        import pandas as pd

        def capture_read_json(buf, **kwargs):
            called_with["buf"] = buf.read() if hasattr(buf, "read") else buf
            called_with["kwargs"] = kwargs
            return pd.DataFrame([{"question": "What is 2+2?", "answer": "4"}])

        with patch("pandas.read_json", side_effect=capture_read_json):
            preparer._preview_output(dataset, JobCategory.SYNTHETIC_DATA_GENERATION)

        assert "buf" in called_with, "pd.read_json was not called for small complete record"
        # The full content (no truncation) must have been passed.
        assert "4" in str(called_with["buf"])

    @patch("boto3.client")
    def test_small_complete_multi_record_no_trailing_newline(self, mock_boto3, preparer, dataset):
        """A small complete file with multiple records and no trailing newline:
        all records including the final one must appear in the preview — the
        final record must NOT be dropped as it was before this fix."""
        from sagemaker.ai_registry.data_preparer.constants import JobCategory

        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        # Two complete records, no trailing newline, well below the window.
        raw = b'{"a":1}\n{"b":2}'
        body = MagicMock()
        body.read.return_value = raw
        mock_s3.get_object.return_value = {"Body": body}

        captured = {}

        import pandas as pd

        def capture_read_json(buf, **kwargs):
            captured["buf"] = buf.read() if hasattr(buf, "read") else buf
            return pd.DataFrame([{"a": 1}, {"b": 2}])

        with patch("pandas.read_json", side_effect=capture_read_json):
            preparer._preview_output(dataset, JobCategory.SYNTHETIC_DATA_GENERATION)

        assert "buf" in captured, "pd.read_json was not called"
        buf_text = captured["buf"] if isinstance(captured["buf"], str) else captured["buf"].decode()
        assert '{"b":2}' in buf_text, "final record was dropped even though file is complete"

    @patch("boto3.client")
    def test_truncated_body_partial_tail_record_not_parsed(self, mock_boto3, preparer, dataset):
        """When the Range response is truncated (== _PREVIEW_MAX_BYTES) and the last
        segment is a partial record, the partial record must be dropped; pd.read_json
        must only receive the complete lines before the last newline, and no exception
        may escape from _preview_output."""
        from sagemaker.ai_registry.data_preparer.constants import JobCategory
        from sagemaker.ai_registry.data_preparer.preparer import _PREVIEW_MAX_BYTES

        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        # Exactly _PREVIEW_MAX_BYTES: one complete record, then a partial record
        # that fills the rest.  The body has a newline so the tail-drop path fires.
        complete = b'{"x": 1}\n'
        partial_prefix = b'{"partial'
        partial = partial_prefix + b"z" * (_PREVIEW_MAX_BYTES - len(complete) - len(partial_prefix))
        raw = complete + partial
        assert len(raw) == _PREVIEW_MAX_BYTES
        body = MagicMock()
        body.read.return_value = raw
        mock_s3.get_object.return_value = {"Body": body}

        captured = {}

        import pandas as pd

        def capture_read_json(buf, **kwargs):
            captured["buf"] = buf.read() if hasattr(buf, "read") else buf
            return pd.DataFrame([{"x": 1}])

        with patch("pandas.read_json", side_effect=capture_read_json):
            preparer._preview_output(dataset, JobCategory.SYNTHETIC_DATA_GENERATION)

        assert "buf" in captured, "pd.read_json was not called; complete prefix was discarded"
        buf_text = captured["buf"] if isinstance(captured["buf"], str) else captured["buf"].decode()
        assert "partial" not in buf_text, "partial tail record was passed to pd.read_json"
