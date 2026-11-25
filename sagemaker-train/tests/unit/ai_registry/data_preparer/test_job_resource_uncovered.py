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
"""Additional unit tests targeting branches not yet covered in _job_resource.py.

Focuses on the pure and near-pure helpers that have no I/O dependencies:
  _output_folder_timestamp, _is_transient_client_error, _extract_log_timestamp,
  _is_jupyter_environment, _coerce_datetime, _pair_sentinels,
  _format_sentinel_row, _format_transition_duration, _cloudwatch_logs_url,
  _output_s3_url, _build_transitions_table, _render_panel (conditional rows),
  _clear_jupyter_output, _emit_backoff_message, create (fallback branch),
  refresh (transitions branch), _process_milestone (no-match branch).

The two wait-loop methods (_wait_panel_mode, _wait_log_tail_mode) are not
attempted here; covering the helpers alone moves the module above 86%.
"""

from __future__ import absolute_import

import types
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from rich.console import Console

from sagemaker.ai_registry.data_preparer._job_resource import (
    Job,
    _extract_log_timestamp,
    _is_jupyter_environment,
    _is_transient_client_error,
    _output_folder_timestamp,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_job(**kwargs):
    """Return a Job instance backed by a MagicMock session."""
    s = MagicMock()
    s.boto_region_name = "us-west-2"
    s.sagemaker_client = MagicMock()
    defaults = dict(job_name="test-job", job_category="SyntheticDataGeneration", session=s)
    defaults.update(kwargs)
    return Job(**defaults)


def _client_error(code="", status=None):
    """Build a minimal ClientError for use in transient-error tests."""
    response = {"Error": {"Code": code, "Message": "msg"}}
    if status is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status}
    return ClientError(response, "DescribeJob")


def _render_to_string(job, is_final=False):
    """Render _render_panel() to a plain string (no ANSI markup)."""
    buf = StringIO()
    console = Console(file=buf, width=140, force_terminal=False, highlight=False, markup=False)
    console.print(job._render_panel(is_final=is_final))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _output_folder_timestamp
# ---------------------------------------------------------------------------


class TestOutputFolderTimestamp:
    """Module-level helper: convert created_at to YYYY-MM-DDTHHMMSS."""

    def test_iso_string_with_z_suffix(self):
        result = _output_folder_timestamp("2024-03-15T14:30:00Z")
        assert result == "2024-03-15T143000"

    def test_iso_string_with_microseconds_and_z(self):
        result = _output_folder_timestamp("2024-03-15T14:30:00.123456Z")
        assert result == "2024-03-15T143000"

    def test_invalid_iso_string_returns_none(self):
        # Exercises the ``except ValueError: return None`` branch (lines 57-58).
        assert _output_folder_timestamp("not-a-date") is None

    def test_integer_input_returns_none(self):
        # Non-str, non-datetime falls through to ``if not isinstance(created_at, datetime)``.
        assert _output_folder_timestamp(42) is None

    def test_none_input_returns_none(self):
        assert _output_folder_timestamp(None) is None

    def test_naive_datetime_assumes_utc(self):
        # Exercises the ``if created_at.tzinfo is None`` branch (line 62).
        naive = datetime(2024, 6, 15, 10, 20, 30)
        result = _output_folder_timestamp(naive)
        assert result == "2024-06-15T102030"

    def test_aware_datetime_converted_to_utc(self):
        # +05:30 → UTC: subtract 5 h 30 m.
        offset = timezone(timedelta(hours=5, minutes=30))
        aware = datetime(2024, 6, 15, 10, 0, 0, tzinfo=offset)
        result = _output_folder_timestamp(aware)
        assert result == "2024-06-15T043000"


# ---------------------------------------------------------------------------
# _is_transient_client_error
# ---------------------------------------------------------------------------


class TestIsTransientClientError:
    """True/False paths for every branch in the transient-error classifier."""

    def test_non_client_error_returns_false(self):
        # First isinstance check rejects non-ClientError exceptions.
        assert _is_transient_client_error(ValueError("oops")) is False
        assert _is_transient_client_error(RuntimeError("fail")) is False

    def test_throttling_exception_code_returns_true(self):
        assert _is_transient_client_error(_client_error(code="ThrottlingException")) is True

    def test_throttling_code_returns_true(self):
        assert _is_transient_client_error(_client_error(code="Throttling")) is True

    def test_internal_server_error_returns_true(self):
        assert _is_transient_client_error(_client_error(code="InternalServerError")) is True

    def test_service_unavailable_returns_true(self):
        assert _is_transient_client_error(_client_error(code="ServiceUnavailable")) is True

    def test_too_many_requests_returns_true(self):
        assert _is_transient_client_error(_client_error(code="TooManyRequestsException")) is True

    def test_500_http_status_returns_true(self):
        # Code is unknown but HTTP 500 → transient (lines 102-103).
        assert _is_transient_client_error(_client_error(status=500)) is True

    def test_503_http_status_returns_true(self):
        assert _is_transient_client_error(_client_error(status=503)) is True

    def test_validation_exception_returns_false(self):
        assert _is_transient_client_error(_client_error(code="ValidationException")) is False

    def test_access_denied_returns_false(self):
        assert _is_transient_client_error(_client_error(code="AccessDeniedException")) is False

    def test_200_http_status_returns_false(self):
        assert _is_transient_client_error(_client_error(status=200)) is False

    def test_non_numeric_status_returns_false(self):
        # HTTPStatusCode cannot be cast to int → TypeError path (lines 104-106).
        err = _client_error(code="")
        err.response["ResponseMetadata"] = {"HTTPStatusCode": "not-a-number"}
        assert _is_transient_client_error(err) is False


# ---------------------------------------------------------------------------
# _extract_log_timestamp
# ---------------------------------------------------------------------------


class TestExtractLogTimestamp:
    """Match and no-match paths for the log-line timestamp extractor."""

    def test_line_with_period_separator_returns_datetime(self):
        # Exercises the match-found path (lines 155-156 in the original numbering).
        line = "2024-03-15 14:30:45.123 - INFO - logger - message"
        result = _extract_log_timestamp(line)
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30
        assert result.second == 45

    def test_line_with_comma_separator_returns_datetime(self):
        # The comma is normalized to a period before strptime.
        line = "2024-03-15 14:30:45,123 - INFO - logger - message"
        result = _extract_log_timestamp(line)
        assert isinstance(result, datetime)
        assert result.microsecond == 123000

    def test_line_without_timestamp_returns_none(self):
        assert _extract_log_timestamp("[MILESTONE] Starting sample generation") is None

    def test_empty_string_returns_none(self):
        assert _extract_log_timestamp("") is None

    def test_partial_timestamp_returns_none(self):
        # Has a date but no time component matching the pattern.
        assert _extract_log_timestamp("2024-03-15 incomplete") is None


# ---------------------------------------------------------------------------
# _is_jupyter_environment
# ---------------------------------------------------------------------------


class TestIsJupyterEnvironment:
    """Four distinct paths: ImportError, general Exception, no kernel, Jupyter kernel."""

    def test_returns_false_when_ipython_not_importable(self):
        # sys.modules[name] = None forces ImportError on ``from IPython import …``.
        with patch.dict("sys.modules", {"IPython": None}):
            result = _is_jupyter_environment()
        assert result is False

    def test_returns_false_when_get_ipython_raises_exception(self):
        # Exercises the ``except Exception: return False`` branch.
        fake_ipython = types.ModuleType("IPython")

        def _explode():
            raise RuntimeError("simulated IPython failure")

        fake_ipython.get_ipython = _explode
        with patch.dict("sys.modules", {"IPython": fake_ipython}):
            result = _is_jupyter_environment()
        assert result is False

    def test_returns_false_when_get_ipython_returns_none(self):
        # In a plain pytest process get_ipython() returns None.
        result = _is_jupyter_environment()
        assert result is False

    def test_returns_true_when_ipkernel_in_config(self):
        # Simulate a live Jupyter kernel with IPKernelApp in config.
        fake_ipython = types.ModuleType("IPython")
        mock_kernel = MagicMock()
        mock_kernel.config = {"IPKernelApp": {"display_name": "Python 3"}}
        fake_ipython.get_ipython = lambda: mock_kernel
        with patch.dict("sys.modules", {"IPython": fake_ipython}):
            result = _is_jupyter_environment()
        assert result is True

    def test_returns_false_when_ipkernel_not_in_config(self):
        fake_ipython = types.ModuleType("IPython")
        mock_kernel = MagicMock()
        mock_kernel.config = {"NotAKernelApp": {}}
        fake_ipython.get_ipython = lambda: mock_kernel
        with patch.dict("sys.modules", {"IPython": fake_ipython}):
            result = _is_jupyter_environment()
        assert result is False


# ---------------------------------------------------------------------------
# Job._coerce_datetime
# ---------------------------------------------------------------------------


class TestCoerceDatetime:
    """Every type-dispatch branch in _coerce_datetime."""

    def test_none_returns_none(self):
        assert Job._coerce_datetime(None) is None

    def test_datetime_passthrough(self):
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = Job._coerce_datetime(dt)
        assert result is dt

    def test_iso_string_with_z_suffix(self):
        result = Job._coerce_datetime("2024-01-15T12:00:00Z")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.tzinfo is not None

    def test_iso_string_without_z_suffix(self):
        result = Job._coerce_datetime("2024-01-15T12:00:00+00:00")
        assert isinstance(result, datetime)

    def test_invalid_string_returns_none(self):
        assert Job._coerce_datetime("not-a-date") is None

    def test_epoch_integer(self):
        # 2024-01-01T00:00:00 UTC = 1704067200 (approximate; exact value
        # depends on local timezone, but year must be 2024).
        result = Job._coerce_datetime(1704067200)
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_epoch_float(self):
        result = Job._coerce_datetime(1704067200.5)
        assert isinstance(result, datetime)

    def test_unknown_type_returns_none(self):
        assert Job._coerce_datetime(["2024-01-01"]) is None
        assert Job._coerce_datetime({"date": "2024"}) is None


# ---------------------------------------------------------------------------
# Job._pair_sentinels
# ---------------------------------------------------------------------------


class TestPairSentinels:
    """Pairing, orphan Completed, free-form, and duplicate-Starting paths."""

    def test_paired_starting_and_completed(self):
        rows = [
            {"phase": "Starting sample generation", "timestamp": None},
            {"phase": "Completed sample generation", "timestamp": None},
        ]
        pairs = Job._pair_sentinels(rows)
        assert len(pairs) == 1
        assert pairs[0]["label"] == "sample generation"
        assert pairs[0]["end"] is None  # end stored as the timestamp value (None here)

    def test_paired_with_real_timestamps(self):
        t_start = datetime(2024, 1, 1, 10, 0, 0)
        t_end = datetime(2024, 1, 1, 10, 5, 0)
        rows = [
            {"phase": "Starting code generation", "timestamp": t_start},
            {"phase": "Completed code generation", "timestamp": t_end},
        ]
        pairs = Job._pair_sentinels(rows)
        assert len(pairs) == 1
        assert pairs[0]["start"] is t_start
        assert pairs[0]["end"] is t_end

    def test_orphaned_starting_keeps_end_as_none(self):
        rows = [{"phase": "Starting code generation", "timestamp": None}]
        pairs = Job._pair_sentinels(rows)
        assert len(pairs) == 1
        assert pairs[0]["label"] == "code generation"
        assert pairs[0]["end"] is None

    def test_completed_before_starting_creates_entry_with_only_end(self):
        """A Completed sentinel with no prior Starting → start=None, end set."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        rows = [{"phase": "Completed code generation", "timestamp": ts}]
        pairs = Job._pair_sentinels(rows)
        assert len(pairs) == 1
        assert pairs[0]["start"] is None
        assert pairs[0]["end"] is ts

    def test_duplicate_starting_updates_start_timestamp(self):
        """A second Starting for the same phase replaces the start timestamp."""
        ts1 = datetime(2024, 1, 1, 12, 0, 0)
        ts2 = datetime(2024, 1, 1, 12, 1, 0)
        rows = [
            {"phase": "Starting sample generation", "timestamp": ts1},
            {"phase": "Starting sample generation", "timestamp": ts2},
        ]
        pairs = Job._pair_sentinels(rows)
        assert len(pairs) == 1
        assert pairs[0]["start"] is ts2

    def test_free_form_phase_becomes_standalone_row(self):
        rows = [{"phase": "Checkpoint saved", "timestamp": None}]
        pairs = Job._pair_sentinels(rows)
        assert len(pairs) == 1
        assert pairs[0]["label"] == "Checkpoint saved"
        assert pairs[0]["start"] is None
        assert pairs[0]["end"] is None

    def test_multiple_phases_preserve_insertion_order(self):
        t_code_start = datetime(2024, 1, 1, 10, 0, 0)
        t_code_end = datetime(2024, 1, 1, 10, 3, 0)
        rows = [
            {"phase": "Starting code generation", "timestamp": t_code_start},
            {"phase": "Completed code generation", "timestamp": t_code_end},
            {"phase": "Starting sample generation", "timestamp": None},
        ]
        pairs = Job._pair_sentinels(rows)
        assert len(pairs) == 2
        assert pairs[0]["label"] == "code generation"
        # Completed sentinel stores ts in 'end'; here it is t_code_end.
        assert pairs[0]["end"] is t_code_end
        assert pairs[1]["label"] == "sample generation"
        assert pairs[1]["end"] is None

    def test_empty_input_returns_empty_list(self):
        assert Job._pair_sentinels([]) == []


# ---------------------------------------------------------------------------
# Job._format_sentinel_row
# ---------------------------------------------------------------------------


class TestFormatSentinelRow:
    """Detail column and duration column for every branch of _format_sentinel_row."""

    def test_completed_non_sample_gen_shows_completed_detail(self):
        job = _make_job()
        phase = {"label": "code generation", "start": None, "end": datetime(2024, 1, 1)}
        glyph, label, detail, _dur = job._format_sentinel_row(phase)
        assert glyph == "✓"
        assert label == "Code Generation"
        assert detail == "Completed"

    def test_in_progress_non_sample_gen_shows_running(self):
        job = _make_job()
        phase = {"label": "code generation", "start": None, "end": None}
        glyph, _label, detail, _dur = job._format_sentinel_row(phase)
        assert glyph == "⋯"
        assert detail == "Running..."

    def test_active_sample_generation_shows_progress_bar(self):
        """In-progress sample generation with expected_samples → progress bar detail."""
        job = _make_job(expected_samples=100)
        job._current_samples = 42
        phase = {"label": "sample generation", "start": None, "end": None}
        _glyph, _label, detail, _dur = job._format_sentinel_row(phase)
        assert "42" in detail
        assert "100" in detail
        assert "42.0%" in detail

    def test_active_sample_generation_zero_current(self):
        """_current_samples=None defaults to 0 for the progress bar."""
        job = _make_job(expected_samples=50)
        job._current_samples = None
        phase = {"label": "sample generation", "start": None, "end": None}
        _glyph, _label, detail, _dur = job._format_sentinel_row(phase)
        assert "0" in detail
        assert "50" in detail

    def test_active_sample_generation_no_expected_shows_running(self):
        """No expected_samples → falls through to Running..."""
        job = _make_job()  # no expected_samples
        phase = {"label": "sample generation", "start": None, "end": None}
        _glyph, _label, detail, _dur = job._format_sentinel_row(phase)
        assert detail == "Running..."

    def test_completed_sample_generation_with_count(self):
        """Completed sample generation with known count → 'N samples'."""
        job = _make_job(expected_samples=100)
        job._current_samples = 95
        end = datetime(2024, 1, 1, 12, 5, 0)
        start = datetime(2024, 1, 1, 12, 0, 0)
        phase = {"label": "sample generation", "start": start, "end": end}
        _glyph, _label, detail, dur = job._format_sentinel_row(phase)
        assert detail == "95 samples"
        assert dur == "300.0s"

    def test_completed_sample_generation_no_count_shows_completed(self):
        """Completed sample generation but _current_samples not set → 'Completed'."""
        job = _make_job(expected_samples=100)
        job._current_samples = None
        phase = {"label": "sample generation", "start": None, "end": datetime(2024, 1, 1)}
        _glyph, _label, detail, _dur = job._format_sentinel_row(phase)
        assert detail == "Completed"

    def test_start_and_end_computes_duration(self):
        job = _make_job()
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 10, 2, 30, tzinfo=timezone.utc)
        phase = {"label": "code generation", "start": start, "end": end}
        _glyph, _label, _detail, dur = job._format_sentinel_row(phase)
        assert dur == "150.0s"

    def test_duration_dash_when_both_timestamps_missing(self):
        job = _make_job()
        phase = {"label": "code generation", "start": None, "end": None}
        _glyph, _label, _detail, dur = job._format_sentinel_row(phase)
        assert dur == "—"

    def test_in_progress_start_only_computes_elapsed(self):
        """An in-progress phase with a start time shows elapsed duration."""
        job = _make_job()
        start = datetime.now(timezone.utc) - timedelta(seconds=60)
        phase = {"label": "code generation", "start": start, "end": None}
        _glyph, _label, _detail, dur = job._format_sentinel_row(phase)
        # Duration is a live-elapsed string like "60.3s".
        assert dur.endswith("s")
        elapsed = float(dur[:-1])
        assert 55 <= elapsed <= 70, f"Expected ~60s elapsed, got {elapsed}s"


# ---------------------------------------------------------------------------
# Job._format_transition_duration
# ---------------------------------------------------------------------------


class TestFormatTransitionDuration:
    """Four resolution tiers: EndTime, next-row start, live elapsed, terminal dash."""

    def test_no_start_time_returns_dash(self):
        assert Job._format_transition_duration({}) == "—"

    def test_uncoerceable_start_returns_dash(self):
        assert Job._format_transition_duration({"StartTime": "not-a-date"}) == "—"

    def test_explicit_end_time_returns_frozen_duration(self):
        """Tier 1: row has an EndTime → use it directly."""
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 10, 1, 30, tzinfo=timezone.utc)
        result = Job._format_transition_duration({"StartTime": start, "EndTime": end})
        assert result == "90.0s"

    def test_uncoerceable_end_time_returns_dash(self):
        result = Job._format_transition_duration(
            {"StartTime": datetime(2024, 1, 1, tzinfo=timezone.utc), "EndTime": "bad"}
        )
        assert result == "—"

    def test_next_row_start_gives_frozen_duration(self):
        """Tier 2: use next row's StartTime when this row has no EndTime."""
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        next_start = datetime(2024, 1, 1, 10, 0, 45, tzinfo=timezone.utc)
        result = Job._format_transition_duration(
            {"StartTime": start},
            next_row={"StartTime": next_start},
        )
        assert result == "45.0s"

    def test_in_progress_no_next_row_returns_live_elapsed(self):
        """Tier 3: no EndTime, no next row, job not terminal → live elapsed."""
        start = datetime.now(timezone.utc) - timedelta(seconds=30)
        result = Job._format_transition_duration({"StartTime": start}, is_terminal=False)
        assert result.endswith("s")
        elapsed = float(result[:-1])
        assert 25 <= elapsed <= 40, f"Expected ~30s, got {elapsed}s"

    def test_terminal_no_next_row_returns_dash(self):
        """Tier 4: terminal job, trailing marker row → '—' (not a duration-bearing step)."""
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        result = Job._format_transition_duration({"StartTime": start}, is_terminal=True)
        assert result == "—"


# ---------------------------------------------------------------------------
# Job._cloudwatch_logs_url
# ---------------------------------------------------------------------------


class TestCloudwatchLogsUrl:
    """URL construction branches for the CloudWatch console link."""

    def test_returns_none_when_no_stream_name(self):
        job = _make_job()
        assert job._cloudwatch_logs_url() is None

    def test_returns_correct_url_with_region_and_stream(self):
        job = _make_job()
        job._log_stream_name = "test-job/generation-2024-01-01T000000"
        url = job._cloudwatch_logs_url()
        assert url is not None
        assert "us-west-2" in url
        assert "cloudwatch" in url
        # The log-group path includes the job category.
        assert "SyntheticDataGeneration" in url
        # Stream name appears (URL-encoded) in the URL.
        assert "test-job" in url

    def test_region_exception_produces_url_with_empty_region(self):
        """When boto_region_name raises, the URL is built with region=''."""

        class _BadSession:
            sagemaker_client = None

            @property
            def boto_region_name(self):
                raise Exception("no region attribute")

        job = Job(job_name="j", job_category="SyntheticDataGeneration", session=_BadSession())
        job._log_stream_name = "stream/name"
        url = job._cloudwatch_logs_url()
        assert url is not None
        # The URL is still valid but the region segment is empty.
        assert "region=" in url


# ---------------------------------------------------------------------------
# Job._output_s3_url — additional branches
# ---------------------------------------------------------------------------


class TestOutputS3UrlAdditional:
    """Branches not covered by the existing TestOutputS3Url in test_job_resource_helpers.py."""

    def test_bucket_only_uri_produces_empty_prefix(self):
        """s3://bucket with no path component → prefix='' in the console URL."""
        job = _make_job(progress_output_s3_uri="s3://my-bucket")
        url = job._output_s3_url()
        assert url is not None
        assert "my-bucket" in url
        # Assert the host exactly rather than as a substring: a substring check
        # reads as a (bypassable) URL validation, which CodeQL flags.
        assert urlparse(url).netloc == "s3.console.aws.amazon.com"

    def test_region_exception_produces_url_with_empty_region(self):
        """When boto_region_name raises, region falls back to '' in the URL."""

        class _BadSession:
            sagemaker_client = None

            @property
            def boto_region_name(self):
                raise Exception("no region")

        job = Job(
            job_name="j",
            job_category="SyntheticDataGeneration",
            session=_BadSession(),
            progress_output_s3_uri="s3://bucket/prefix/",
        )
        url = job._output_s3_url()
        assert url is not None
        assert "s3.console.aws.amazon.com/s3/buckets/bucket" in url
        assert "region=" in url

    def test_tier1_exception_falls_back_to_tier2(self):
        """Tier 1 exception (bucket/prefix = None, None) falls through to tier 2."""
        job = _make_job(progress_output_s3_uri="s3://bucket/pfx/")
        # Set a target key so tier 1 is attempted.  _progress_output_s3_uri
        # is set after construction to None so tier 1 raises when it tries
        # to parse the s3:// URI (finds None → not startswith "s3://").
        job._progress_target_key = "pfx/job-ts/metadata/file.jsonl"
        job._progress_output_s3_uri = None  # tier 1 try→except, then tier 2 returns None
        url = job._output_s3_url()
        assert url is None  # tier 2 also returns None when uri is None


# ---------------------------------------------------------------------------
# Job._extract_dataset_arn — non-dict JSON branch
# ---------------------------------------------------------------------------


class TestExtractDatasetArnNonDict:
    """Line 462: valid JSON that is not a dict is treated as missing."""

    def test_json_array_returns_none(self):
        job = _make_job()
        job.job_config_document = "[1, 2, 3]"
        assert job._extract_dataset_arn() is None

    def test_json_string_literal_returns_none(self):
        job = _make_job()
        job.job_config_document = '"just a string"'
        assert job._extract_dataset_arn() is None


# ---------------------------------------------------------------------------
# Job.refresh — transitions branch
# ---------------------------------------------------------------------------


class TestRefreshWithTransitions:
    """refresh() branch where SecondaryStatusTransitions is non-empty (line 321)."""

    def test_status_message_set_from_last_transition(self):
        job = _make_job()
        job._client.describe_job.return_value = {
            "JobStatus": "InProgress",
            "SecondaryStatus": "Executing",
            "SecondaryStatusTransitions": [
                {"Status": "Validating", "StatusMessage": "first"},
                {"Status": "Executing", "StatusMessage": "stream=my-job/gen-ts"},
            ],
        }
        job.refresh()
        assert job._status_message == "stream=my-job/gen-ts"
        assert len(job._secondary_status_transitions) == 2


# ---------------------------------------------------------------------------
# Job._process_milestone — no-match branch
# ---------------------------------------------------------------------------


class TestProcessMilestoneNoMatch:
    """_process_milestone silently ignores lines without a [MILESTONE] token."""

    def test_non_milestone_line_does_not_append_transition(self):
        job = _make_job()
        before = len(job._milestone_transitions)
        job._process_milestone("2024-01-01 12:00:00.000 - INFO - logger - plain message")
        assert len(job._milestone_transitions) == before

    def test_empty_line_does_not_append_transition(self):
        job = _make_job()
        before = len(job._milestone_transitions)
        job._process_milestone("")
        assert len(job._milestone_transitions) == before


# ---------------------------------------------------------------------------
# Job._build_transitions_table — sentinel-interaction branches
# ---------------------------------------------------------------------------


class TestBuildTransitionsTableSentinelInteraction:
    """Executing-with-sentinels (blank detail) and no-executing-with-sentinels (top-level) paths."""

    @staticmethod
    def _get_detail_cells(table):
        return list(table.columns[2]._cells)

    @staticmethod
    def _get_step_cells(table):
        return list(table.columns[1]._cells)

    def test_executing_row_with_sentinels_has_blank_api_detail(self):
        """When sentinel sub-rows are present, the Executing row's detail is suppressed."""
        job = _make_job()
        now = datetime.now(timezone.utc)
        job._secondary_status_transitions = [
            {
                "Status": "Executing",
                "StartTime": now - timedelta(seconds=60),
                "EndTime": None,
                "StatusMessage": "stream=job/gen-ts",
            },
        ]
        job._milestone_transitions = [
            {"phase": "Starting code generation", "timestamp": None},
        ]
        job.job_status = "InProgress"

        table = job._build_transitions_table()
        details = self._get_detail_cells(table)
        # First cell is the Executing row — its detail is blank.
        assert details[0] == ""

    def test_no_executing_row_sentinels_rendered_as_top_level_unindented(self):
        """Sentinels without an Executing API row appear without the ↳ indent prefix."""
        job = _make_job()
        now = datetime.now(timezone.utc)
        job._secondary_status_transitions = [
            {
                "Status": "Validating",
                "StartTime": now - timedelta(seconds=30),
                "EndTime": None,
                "StatusMessage": "Validating request",
            },
        ]
        job._milestone_transitions = [
            {"phase": "Starting code generation", "timestamp": None},
        ]
        job.job_status = "InProgress"

        table = job._build_transitions_table()
        step_cells = self._get_step_cells(table)
        # Find the sentinel-sourced row (Code Generation).
        sentinel_rows = [c for c in step_cells if "code generation" in c.lower()]
        assert len(sentinel_rows) == 1
        assert not sentinel_rows[0].startswith("↳ "), "Top-level sentinels must not be indented"


# ---------------------------------------------------------------------------
# Job._render_panel — conditional header rows
# ---------------------------------------------------------------------------


class TestRenderPanelConditionalRows:
    """job_arn, dataset_arn, and cw_url conditional rows in _render_panel."""

    def test_job_arn_row_appears_when_set(self):
        job = _make_job(job_arn="arn:aws:sagemaker:us-west-2:111:job/test-job")
        text = _render_to_string(job)
        assert "ARN" in text

    def test_dataset_arn_row_appears_when_set(self):
        job = _make_job()
        job.job_config_document = (
            '{"DatasetArn": "arn:aws:sagemaker:us-west-2:111:hub-content/X/DataSet/foo/0.0.1"}'
        )
        text = _render_to_string(job)
        assert "Dataset ARN" in text

    def test_cloudwatch_link_is_clickable_when_stream_name_known(self):
        """When the log stream name is known, the panel renders a CloudWatch link."""
        job = _make_job()
        job._log_stream_name = "test-job/generation-2024-01-01T000000"
        text = _render_to_string(job)
        # The word "CloudWatch" must appear somewhere in the rendered output.
        assert "CloudWatch" in text
        # The "pending" placeholder must NOT appear when the stream is known.
        assert "pending" not in text


# ---------------------------------------------------------------------------
# Job._clear_jupyter_output
# ---------------------------------------------------------------------------


class TestClearJupyterOutput:
    """clear_output called when IPython.display importable; swallows failures."""

    def test_calls_clear_output_with_wait_arg(self):
        """clear_output(wait=True) is called when IPython.display is importable."""
        mock_clear_output = MagicMock()
        fake_display = types.ModuleType("IPython.display")
        fake_display.clear_output = mock_clear_output
        with patch.dict("sys.modules", {"IPython.display": fake_display}):
            Job._clear_jupyter_output(wait=True)
        mock_clear_output.assert_called_once_with(wait=True)

    def test_calls_clear_output_with_wait_false(self):
        mock_clear_output = MagicMock()
        fake_display = types.ModuleType("IPython.display")
        fake_display.clear_output = mock_clear_output
        with patch.dict("sys.modules", {"IPython.display": fake_display}):
            Job._clear_jupyter_output(wait=False)
        mock_clear_output.assert_called_once_with(wait=False)

    def test_silently_swallows_import_error(self):
        """When IPython.display is unavailable, _clear_jupyter_output must not raise."""
        with patch.dict("sys.modules", {"IPython.display": None}):
            Job._clear_jupyter_output(wait=True)  # must not raise

    def test_silently_swallows_clear_output_exception(self):
        """A RuntimeError from clear_output must be swallowed."""
        fake_display = types.ModuleType("IPython.display")
        fake_display.clear_output = MagicMock(side_effect=RuntimeError("rendering failed"))
        with patch.dict("sys.modules", {"IPython.display": fake_display}):
            Job._clear_jupyter_output(wait=True)  # must not raise


# ---------------------------------------------------------------------------
# Job._emit_backoff_message
# ---------------------------------------------------------------------------


class TestEmitBackoffMessage:
    """Content assertions and console-failure fallback for _emit_backoff_message."""

    def _make_console(self):
        buf = StringIO()
        return Console(file=buf, force_terminal=False, width=120), buf

    def test_message_contains_still_running(self):
        job = _make_job()
        console, buf = self._make_console()
        job._emit_backoff_message(console=console, elapsed_quiet_seconds=360)
        assert "still running" in buf.getvalue()

    def test_message_contains_elapsed_duration(self):
        """130 seconds → '2m 10s' in the message."""
        job = _make_job()
        console, buf = self._make_console()
        job._emit_backoff_message(console=console, elapsed_quiet_seconds=130)
        text = buf.getvalue()
        assert "2m" in text
        assert "10s" in text

    def test_message_contains_stream_name_when_known(self):
        job = _make_job()
        job._log_stream_name = "my-job/generation-2024"
        console, buf = self._make_console()
        job._emit_backoff_message(console=console, elapsed_quiet_seconds=300)
        assert "my-job/generation-2024" in buf.getvalue()

    def test_message_uses_pending_placeholder_when_no_stream(self):
        job = _make_job()
        console, buf = self._make_console()
        job._emit_backoff_message(console=console, elapsed_quiet_seconds=300)
        assert "<stream pending>" in buf.getvalue()

    def test_console_failure_falls_back_to_logger_without_raising(self):
        """When console.print raises, _emit_backoff_message must not propagate."""
        job = _make_job()
        broken_console = MagicMock()
        broken_console.print.side_effect = RuntimeError("rendering failed")
        # Must not raise:
        job._emit_backoff_message(console=broken_console, elapsed_quiet_seconds=300)


# ---------------------------------------------------------------------------
# Job.create — optional-field and introspection-failure branches
# ---------------------------------------------------------------------------


class TestCreateOptionalFields:
    """create() classmethod: job_arn, customer_details, and introspection fallback."""

    def _make_session_with_model(self, accepted_keys):
        s = MagicMock()
        s.boto_region_name = "us-west-2"
        s.sagemaker_client = MagicMock()
        s.sagemaker_client.create_job.return_value = {"JobArn": "arn:aws:sm:us-west-2:111:job/j"}
        op_model = MagicMock()
        op_model.input_shape.members.keys.return_value = accepted_keys
        s.sagemaker_client.meta.service_model.operation_model.return_value = op_model
        return s

    def test_job_arn_forwarded_when_model_accepts_it(self):
        s = self._make_session_with_model(
            [
                "JobName",
                "RoleArn",
                "JobCategory",
                "JobConfigSchemaVersion",
                "JobConfigDocument",
                "JobArn",
            ]
        )
        Job.create(
            job_name="my-job",
            role_arn="arn:aws:iam::111:role/role",
            job_category="SyntheticDataGeneration",
            job_config_schema_version="0.1.0",
            job_config_document='{"key": "value"}',
            session=s,
            job_arn="arn:aws:sm:us-west-2:111:job/my-job",
        )
        call_kwargs = s.sagemaker_client.create_job.call_args[1]
        assert "JobArn" in call_kwargs

    def test_customer_details_forwarded_when_model_accepts_it(self):
        s = self._make_session_with_model(
            [
                "JobName",
                "RoleArn",
                "JobCategory",
                "JobConfigSchemaVersion",
                "JobConfigDocument",
                "CustomerDetails",
            ]
        )
        Job.create(
            job_name="my-job",
            role_arn="arn:aws:iam::111:role/role",
            job_category="SyntheticDataGeneration",
            job_config_schema_version="0.1.0",
            job_config_document='{"key": "value"}',
            session=s,
            customer_details={"foo": "bar"},
        )
        call_kwargs = s.sagemaker_client.create_job.call_args[1]
        assert call_kwargs.get("CustomerDetails") == {"foo": "bar"}

    def test_introspection_failure_falls_back_to_non_none_filter(self):
        """op_model introspection raises → all non-None fields are forwarded."""
        s = MagicMock()
        s.boto_region_name = "us-west-2"
        s.sagemaker_client = MagicMock()
        s.sagemaker_client.create_job.return_value = {"JobArn": "arn:aws:sm:us-west-2:111:job/j"}
        s.sagemaker_client.meta.service_model.operation_model.side_effect = Exception("no model")

        Job.create(
            job_name="my-job",
            role_arn="arn:aws:iam::111:role/role",
            job_category="SyntheticDataGeneration",
            job_config_schema_version="0.1.0",
            job_config_document='{"key": "value"}',
            session=s,
        )
        call_kwargs = s.sagemaker_client.create_job.call_args[1]
        # All required non-None fields must be present after the fallback filter.
        assert call_kwargs["JobName"] == "my-job"
        assert call_kwargs["RoleArn"] == "arn:aws:iam::111:role/role"
        assert call_kwargs["JobCategory"] == "SyntheticDataGeneration"
        assert call_kwargs["JobConfigSchemaVersion"] == "0.1.0"
