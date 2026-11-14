"""Unit tests for Job helper methods: _extract_log_stream_name, _output_s3_url, _extract_dataset_arn."""

from unittest.mock import MagicMock

from sagemaker.ai_registry.data_preparer._job_resource import Job


def _make_job(**kwargs):
    s = MagicMock()
    s.boto_region_name = "us-west-2"
    s.sagemaker_client = None
    defaults = dict(job_name="test-job", job_category="SyntheticDataGeneration", session=s)
    defaults.update(kwargs)
    return Job(**defaults)


class TestExtractLogStreamName:
    """Stream-name cache persists after Executing phase ends."""

    def test_returns_none_when_no_message(self):
        job = _make_job()
        assert job._extract_log_stream_name() is None

    def test_parses_from_status_message(self):
        job = _make_job()
        job._status_message = "stream=my-job/generation-2026-01-01T00-00-00"
        assert job._extract_log_stream_name() == "my-job/generation-2026-01-01T00-00-00"

    def test_caches_after_first_observation(self):
        job = _make_job()
        job._status_message = "stream=my-job/gen-ts"
        assert job._extract_log_stream_name() == "my-job/gen-ts"
        # Overwrite _status_message (simulates Completed phase)
        job._status_message = "Generation completed"
        # Cache should still return the original
        assert job._extract_log_stream_name() == "my-job/gen-ts"

    def test_fallback_scans_transitions(self):
        job = _make_job()
        job._status_message = "Generation completed"
        job._secondary_status_transitions = [
            {"Status": "Validating", "StatusMessage": "ok"},
            {"Status": "Executing", "StatusMessage": "stream=my-job/gen-ts"},
            {"Status": "Completed", "StatusMessage": "Generation completed"},
        ]
        assert job._extract_log_stream_name() == "my-job/gen-ts"


class TestOutputS3Url:
    """S3 console URL for the output directory."""

    def test_returns_none_when_not_configured(self):
        job = _make_job()
        assert job._output_s3_url() is None

    def test_pre_discovery_uses_customer_prefix(self):
        job = _make_job(progress_output_s3_uri="s3://bucket/output/prefix/")
        url = job._output_s3_url()
        assert url.startswith("https://s3.console.aws.amazon.com/s3/buckets/bucket")
        assert "prefix=output%2Fprefix%2F" in url
        assert "region=us-west-2" in url

    def test_post_discovery_uses_job_subdir(self):
        job = _make_job(progress_output_s3_uri="s3://bucket/output/prefix/")
        job._progress_target_key = (
            "output/prefix/my-job-2026-01-01T000000Z/job_metadata/verified_samples.jsonl"
        )
        url = job._output_s3_url()
        assert "my-job-2026-01-01T000000Z%2F" in url


class TestExtractDatasetArn:
    """DatasetArn extraction from JobConfigDocument."""

    def test_returns_none_when_no_document(self):
        job = _make_job()
        assert job._extract_dataset_arn() is None

    def test_returns_none_for_malformed_json(self):
        job = _make_job()
        job.job_config_document = "not json"
        assert job._extract_dataset_arn() is None

    def test_returns_none_when_key_missing(self):
        job = _make_job()
        job.job_config_document = '{"OutputDataConfig": {}}'
        assert job._extract_dataset_arn() is None

    def test_returns_arn_when_present(self):
        job = _make_job()
        job.job_config_document = (
            '{"DatasetArn": "arn:aws:sagemaker:us-west-2:111:hub-content/X/DataSet/foo/0.0.1"}'
        )
        assert (
            job._extract_dataset_arn()
            == "arn:aws:sagemaker:us-west-2:111:hub-content/X/DataSet/foo/0.0.1"
        )

    def test_handles_already_parsed_dict(self):
        job = _make_job()
        job.job_config_document = {"DatasetArn": "arn:test"}
        assert job._extract_dataset_arn() == "arn:test"


class TestApplyCompletionFallback:
    """Completion fallback snaps sample count on success."""

    def test_snaps_on_completed_with_zero_count(self):
        """zero count → file found but empty; must NOT snap (A3 fix).

        FAILS on original code (old code snapped on 0 too); passes after fix.
        """
        job = _make_job(expected_samples=100)
        job.job_status = "Completed"
        job._current_samples = 0
        job._apply_completion_fallback()
        assert (
            job._current_samples == 0
        ), "Found-but-empty file (count=0) must not be snapped to expected"

    def test_snaps_on_completed_with_none_count(self):
        job = _make_job(expected_samples=50)
        job.job_status = "Completed"
        job._current_samples = None
        job._apply_completion_fallback()
        assert job._current_samples == 50

    def test_no_snap_on_completed_when_file_found_empty(self):
        """A completed job whose output file exists but has 0 records must not snap.

        Regression guard for bug-bash SDG #9: job Completed, empty output,
        panel showed N/N (masked the empty-output condition) before this fix.
        """
        job = _make_job(expected_samples=100)
        job.job_status = "Completed"
        job._current_samples = 0  # file was found, but it has zero records
        job._apply_completion_fallback()
        assert job._current_samples == 0

    def test_no_snap_on_failed(self):
        job = _make_job(expected_samples=100)
        job.job_status = "Failed"
        job._current_samples = 0
        job._apply_completion_fallback()
        assert job._current_samples == 0

    def test_no_snap_when_count_already_positive(self):
        job = _make_job(expected_samples=100)
        job.job_status = "Completed"
        job._current_samples = 42
        job._apply_completion_fallback()
        assert job._current_samples == 42


class TestBuildTransitionsTableGlyphs:
    """Check mark (✓) vs ellipsis (⋯) glyph assignment in status panel.

    Completed steps should show ✓; only the currently active step shows ⋯.
    """

    @staticmethod
    def _get_glyphs(table):
        """Extract the glyph column values from a Rich Table."""
        return list(table.columns[0]._cells)

    def test_past_steps_show_check_mark_when_no_end_time(self):
        """Steps before the last one show ✓ even without EndTime from the API."""
        from datetime import datetime, timezone, timedelta

        job = _make_job()
        now = datetime.now(timezone.utc)
        job._secondary_status_transitions = [
            {
                "Status": "Validating",
                "StartTime": now - timedelta(seconds=100),
                "EndTime": None,
                "StatusMessage": "Validating generation request",
            },
            {
                "Status": "Preparing",
                "StartTime": now - timedelta(seconds=90),
                "EndTime": None,
                "StatusMessage": "Preparing generation artifacts",
            },
            {
                "Status": "Executing",
                "StartTime": now - timedelta(seconds=80),
                "EndTime": None,
                "StatusMessage": "",
            },
        ]
        job._milestone_transitions = []
        job.job_status = "InProgress"

        table = job._build_transitions_table()
        glyphs = self._get_glyphs(table)
        # First two rows (Validating, Preparing) should have ✓
        assert glyphs[0] == "✓"
        assert glyphs[1] == "✓"
        # Last row (Executing) is the active step — should have ⋯
        assert glyphs[2] == "⋯"

    def test_all_steps_show_check_mark_when_job_completed(self):
        """All steps show ✓ when the job has reached a terminal state."""
        from datetime import datetime, timezone, timedelta

        job = _make_job()
        now = datetime.now(timezone.utc)
        job._secondary_status_transitions = [
            {
                "Status": "Validating",
                "StartTime": now - timedelta(seconds=3675),
                "EndTime": None,
                "StatusMessage": "Validating generation request",
            },
            {
                "Status": "Preparing",
                "StartTime": now - timedelta(seconds=3666),
                "EndTime": None,
                "StatusMessage": "Preparing generation artifacts",
            },
            {
                "Status": "Executing",
                "StartTime": now - timedelta(seconds=3661),
                "EndTime": None,
                "StatusMessage": "",
            },
            {
                "Status": "Publishing",
                "StartTime": now - timedelta(seconds=34),
                "EndTime": None,
                "StatusMessage": "Publishing generation results",
            },
            {
                "Status": "Finalizing",
                "StartTime": now - timedelta(seconds=29),
                "EndTime": None,
                "StatusMessage": "Finalizing generation request",
            },
            {
                "Status": "Completed",
                "StartTime": now - timedelta(seconds=25),
                "EndTime": None,
                "StatusMessage": "Generation completed",
            },
        ]
        job._milestone_transitions = []
        job.job_status = "Completed"

        table = job._build_transitions_table()
        glyphs = self._get_glyphs(table)
        step_names = list(table.columns[1]._cells)
        for i, glyph in enumerate(glyphs):
            assert glyph == "✓", f"Row {i} ({step_names[i]}) should show ✓"

    def test_step_with_end_time_shows_check_mark(self):
        """Steps with EndTime still show ✓ (original behavior preserved)."""
        from datetime import datetime, timezone, timedelta

        job = _make_job()
        now = datetime.now(timezone.utc)
        job._secondary_status_transitions = [
            {
                "Status": "Validating",
                "StartTime": now - timedelta(seconds=100),
                "EndTime": now - timedelta(seconds=90),
                "StatusMessage": "Validating generation request",
            },
            {
                "Status": "Executing",
                "StartTime": now - timedelta(seconds=90),
                "EndTime": None,
                "StatusMessage": "",
            },
        ]
        job._milestone_transitions = []
        job.job_status = "InProgress"

        table = job._build_transitions_table()
        glyphs = self._get_glyphs(table)
        assert glyphs[0] == "✓"
        assert glyphs[1] == "⋯"

    def test_single_active_step_shows_ellipsis(self):
        """A single transition with no EndTime and job still running shows ⋯."""
        from datetime import datetime, timezone, timedelta

        job = _make_job()
        now = datetime.now(timezone.utc)
        job._secondary_status_transitions = [
            {
                "Status": "Validating",
                "StartTime": now - timedelta(seconds=10),
                "EndTime": None,
                "StatusMessage": "Validating generation request",
            },
        ]
        job._milestone_transitions = []
        job.job_status = "InProgress"

        table = job._build_transitions_table()
        glyphs = self._get_glyphs(table)
        assert len(glyphs) == 1
        assert glyphs[0] == "⋯"

    def test_failed_job_all_steps_show_check_mark(self):
        """All steps show ✓ when the job has failed (terminal state)."""
        from datetime import datetime, timezone, timedelta

        job = _make_job()
        now = datetime.now(timezone.utc)
        job._secondary_status_transitions = [
            {
                "Status": "Validating",
                "StartTime": now - timedelta(seconds=50),
                "EndTime": None,
                "StatusMessage": "Validating generation request",
            },
            {
                "Status": "Executing",
                "StartTime": now - timedelta(seconds=40),
                "EndTime": None,
                "StatusMessage": "",
            },
        ]
        job._milestone_transitions = []
        job.job_status = "Failed"

        table = job._build_transitions_table()
        glyphs = self._get_glyphs(table)
        assert glyphs[0] == "✓"
        assert glyphs[1] == "✓"
