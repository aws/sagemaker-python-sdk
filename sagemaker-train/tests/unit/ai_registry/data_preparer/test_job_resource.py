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
"""Unit tests for Job resource helpers and wait() behavior.

Coverage areas:
* Helpers and state: ``_scan_for_milestones``, ``_process_milestone``,
  ``_log_next_token`` persistence, ``_milestone_transitions`` population.
* Panel rendering: ``_render_panel`` field rules.
* Panel mode wait(): ``logs=False`` path, terminal-on-entry, initial panel,
  transient describe failures, quiet-period backoff, terminal panel, timeout.
* Log tail mode wait(): ``logs=True`` path, line printing, initial panel,
  post-quiet-period CW polling.

Mocking conventions follow the sibling ``test_job_resource_progress.py`` file:
``MagicMock`` for the SageMaker session, ``boto3.client`` patched at module
scope, ``time.sleep`` patched to keep tests fast.
"""

from __future__ import absolute_import

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.ai_registry.data_preparer._job_resource import Job
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerJobTimeoutError,
)


class _frozen_now:
    """Context manager that controls ``_job_resource.datetime.now()`` without
    replacing the ``datetime`` class itself.

    Patching the whole ``datetime`` symbol breaks ``_coerce_datetime``'s
    ``isinstance(value, datetime)`` check (the name becomes a MagicMock).
    Instead we swap in a real ``datetime`` subclass whose ``now()`` returns a
    caller-controlled instant, so ``isinstance`` and ``fromtimestamp`` keep
    working. ``set(epoch)`` advances the frozen clock mid-test.
    """

    def __init__(self, epoch: float):
        self._epoch = epoch
        self._patcher = None

    def set(self, epoch: float) -> None:
        self._epoch = epoch

    def __enter__(self):
        outer = self

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.fromtimestamp(outer._epoch, tz)

        self._patcher = patch(
            "sagemaker.ai_registry.data_preparer._job_resource.datetime",
            _FrozenDateTime,
        )
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        self._patcher.stop()
        return False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_with_region():
    s = MagicMock()
    s.boto_region_name = "us-west-2"
    s.account_id.return_value = "123456789012"
    return s


@pytest.fixture
def sdg_job(session_with_region):
    """A minimal SDG ``Job`` with no S3 progress configured."""
    return Job(
        job_name="my-sdg-job-001",
        job_category="SyntheticDataGeneration",
        session=session_with_region,
    )


@pytest.fixture
def sdg_job_with_progress(session_with_region):
    """A SDG ``Job`` with sample-progress tracking configured."""
    return Job(
        job_name="my-sdg-job-001",
        job_category="SyntheticDataGeneration",
        session=session_with_region,
        expected_samples=10,
        progress_output_s3_uri="s3://my-bucket/sdg-out/",
    )


@pytest.fixture(autouse=True)
def _no_real_sleep():
    """Don't actually sleep during tests — keeps the suite fast and allows
    panel-mode polling loops to spin through ticks deterministically."""
    with patch(
        "sagemaker.ai_registry.data_preparer._job_resource.time.sleep",
        return_value=None,
    ):
        yield


# ---------------------------------------------------------------------------
# JobConfigSchemaVersion plumbing
# ---------------------------------------------------------------------------


class TestJobConfigSchemaVersion:
    """The resource must carry ``JobConfigSchemaVersion`` end-to-end.

    ``DataPreparer._rehydrate_datasets`` dispatches ``DqShape.for_version``
    on ``job.job_config_schema_version``. Before this attribute existed,
    every completed DQ v2 ("2.0.0", formerly the "0.2.0" preview) job fell
    back to the v1 parser and failed rehydration with 'Completed job missing
    DatasetArn in JobConfigDocument'.
    """

    def test_init_defaults_to_none(self, sdg_job):
        assert sdg_job.job_config_schema_version is None

    def test_init_stores_kwarg(self, session_with_region):
        job = Job(
            job_name="dq-1",
            job_category="DataQualityEvaluation",
            session=session_with_region,
            job_config_schema_version="2.0.0",
        )
        assert job.job_config_schema_version == "2.0.0"

    def test_refresh_populates_schema_version_from_describe_job(self, session_with_region):
        job = Job(
            job_name="dq-1",
            job_category="DataQualityEvaluation",
            session=session_with_region,
        )
        session_with_region.sagemaker_client.describe_job.return_value = {
            "JobArn": "arn:aws:sagemaker:us-west-2:123456789012:job/DataQualityEvaluation/dq-1",
            "JobStatus": "Completed",
            "JobConfigSchemaVersion": "2.0.0",
            "JobConfigDocument": "{}",
        }
        job.refresh()
        assert job.job_config_schema_version == "2.0.0"

    def test_create_carries_schema_version_onto_instance(self, session_with_region):
        session_with_region.sagemaker_client.create_job.return_value = {
            "JobArn": "arn:aws:sagemaker:us-west-2:123456789012:job/DataQualityEvaluation/dq-1",
        }
        job = Job.create(
            job_name="dq-1",
            role_arn="arn:aws:iam::123456789012:role/my-role",
            job_category="DataQualityEvaluation",
            job_config_schema_version="2.0.0",
            job_config_document="{}",
            session=session_with_region,
        )
        assert job.job_config_schema_version == "2.0.0"


class TestJobGet:
    """Job.get re-attaches to an existing job by name and hydrates it via
    DescribeJob — the companion to fire-and-forget (wait=False) submission."""

    def test_get_hydrates_from_describe_job(self, session_with_region):
        session_with_region.sagemaker_client.describe_job.return_value = {
            "JobArn": "arn:aws:sagemaker:us-west-2:123456789012:job/DataQualityEvaluation/dq-1",
            "JobStatus": "Completed",
            "JobConfigSchemaVersion": "2.0.0",
            "JobConfigDocument": "{}",
        }
        job = Job.get(
            job_name="dq-1",
            job_category="DataQualityEvaluation",
            session=session_with_region,
        )
        session_with_region.sagemaker_client.describe_job.assert_called_once_with(
            JobName="dq-1", JobCategory="DataQualityEvaluation"
        )
        assert job.job_name == "dq-1"
        assert job.job_status == "Completed"
        assert job.job_config_schema_version == "2.0.0"


# ---------------------------------------------------------------------------
# Helpers and state
# ---------------------------------------------------------------------------


class TestIsPrintableLogLine:
    """``_is_printable_log_line`` decides which raw CW lines reach the cell.

    The two-tier model means INFO/WARNING/ERROR/CRITICAL and ``[MILESTONE]``
    lines are customer-relevant; bare DEBUG and EMF JSON blobs are noise.
    Lines with no recognizable level token (e.g. traceback continuations)
    are printed so error output is never hidden.
    """

    @pytest.mark.parametrize(
        "level_token",
        [" - INFO - ", " - WARNING - ", " - ERROR - ", " - CRITICAL - "],
    )
    def test_is_printable_log_line_info(self, level_token):
        line = f"2026-05-28 22:10:47,670{level_token}orchestrator - did a thing"
        assert Job._is_printable_log_line(line) is True

    def test_is_printable_log_line_milestone(self):
        line = (
            "2026-05-28 22:10:47,670 - INFO - ecs_entrypoint - "
            "[MILESTONE] Starting code generation"
        )
        assert Job._is_printable_log_line(line) is True

    def test_is_printable_log_line_rejects_debug(self):
        debug_line = "2026-05-28 22:10:47,670 - DEBUG - clients - Bedrock invoke begin"
        emf_blob = (
            '{"_aws":{"Timestamp":1716934247670,"CloudWatchMetrics":[{"Namespace":'
            '"DataPreparer","Dimensions":[["JobName"]],"Metrics":[{"Name":"foo"}]}]}}'
        )
        assert Job._is_printable_log_line(debug_line) is False
        assert Job._is_printable_log_line(emf_blob) is False

    @pytest.mark.parametrize(
        "line",
        [
            "Traceback (most recent call last):",
            '  File "/opt/ml/code/transformer.py", line 12, in transform',
            "KeyError: 'text'",
            "plain stdout from a user transformer",
        ],
    )
    def test_is_printable_log_line_keeps_unstructured_lines(self, line):
        assert Job._is_printable_log_line(line) is True

    @pytest.mark.parametrize("line", ["", "   ", None])
    def test_is_printable_log_line_rejects_blank(self, line):
        assert Job._is_printable_log_line(line) is False

    def test_debug_token_inside_message_text_is_not_suppressed(self):
        # Only the level column decides; the word DEBUG elsewhere is content.
        line = "2026-05-28 22:10:47,670 - INFO - orchestrator - set DEBUG=false"
        assert Job._is_printable_log_line(line) is True


class TestScanForMilestones:
    """Tests that ``_scan_for_milestones`` advances ``_log_next_token`` so
    successive calls never replay the same CW events."""

    def test_scan_for_milestones_advances_token(self, sdg_job):
        # Two consecutive calls should each get the previous call's token.
        # First call: starts with None, returns token 'tok-1' + a list of lines.
        # Second call: must pass 'tok-1' back to _tail_logs.
        with patch.object(
            sdg_job,
            "_tail_logs",
            side_effect=[
                ("tok-1", ["[MILESTONE] Starting sample generation"]),
                ("tok-2", []),
            ],
        ) as mock_tail:
            sdg_job._scan_for_milestones()
            sdg_job._scan_for_milestones()

        # First call: token=None
        first_call_token = mock_tail.call_args_list[0].args[0]
        # Second call: token=tok-1 (the value returned by the first call)
        second_call_token = mock_tail.call_args_list[1].args[0]
        assert first_call_token is None
        assert second_call_token == "tok-1"
        # And the instance has the most recent token after both calls.
        assert sdg_job._log_next_token == "tok-2"


class TestProcessMilestone:
    """``_process_milestone`` extracts the phase description and timestamp,
    appending a ``{phase, timestamp}`` entry to ``_milestone_transitions``.
    Failure to parse the timestamp prefix MUST NOT raise — the entry is still
    appended with ``timestamp=None``."""

    def test_process_milestone_populates_transitions(self, sdg_job):
        line = (
            "2026-05-28 22:10:47,670 - INFO - ecs_entrypoint - "
            "[MILESTONE] Starting code generation"
        )
        sdg_job._process_milestone(line)

        assert len(sdg_job._milestone_transitions) == 1
        entry = sdg_job._milestone_transitions[0]
        assert entry["phase"] == "Starting code generation"
        assert entry["timestamp"] is not None
        # The parsed timestamp should be a datetime corresponding to the
        # log-line prefix; we don't assert tz here because the prefix omits one.
        assert isinstance(entry["timestamp"], datetime)

    def test_process_milestone_handles_unparseable_timestamp(self, sdg_job):
        # No parseable timestamp prefix — just the [MILESTONE] payload.
        line = "no-timestamp-here [MILESTONE] Completed sample generation"
        sdg_job._process_milestone(line)  # must not raise

        assert len(sdg_job._milestone_transitions) == 1
        entry = sdg_job._milestone_transitions[0]
        assert entry["phase"] == "Completed sample generation"
        assert entry["timestamp"] is None


class TestTokenPersistedAcrossWaitCalls:
    """``self._log_next_token`` is an *instance* attribute (not a local) so
    a second ``wait()`` call resumes CW polling from where the first left
    off. This guards against replaying old events when a customer calls
    ``wait()`` repeatedly on the same Job (e.g. ``logs=False`` then
    ``logs=True``)."""

    def test_token_persisted_across_wait_calls(self, sdg_job):
        # Configure refresh() to make the job appear terminal on the first
        # tick, so wait() exits cleanly.
        def fake_refresh():
            sdg_job.job_status = "Completed"
            sdg_job.secondary_status = "Completed"
            return sdg_job

        with (
            patch.object(sdg_job, "refresh", side_effect=fake_refresh),
            patch.object(
                sdg_job,
                "_tail_logs",
                side_effect=[
                    ("tok-after-first-wait", []),
                    ("tok-after-second-wait", []),
                ],
            ) as mock_tail,
        ):
            sdg_job.wait(poll=0, describe_interval=0, logs=False)
            assert sdg_job._log_next_token == "tok-after-first-wait"
            sdg_job.wait(poll=0, describe_interval=0, logs=False)

        # The second wait() must have called _tail_logs with the token left
        # by the first wait().
        second_call_token = mock_tail.call_args_list[-1].args[0]
        assert second_call_token == "tok-after-first-wait"
        assert sdg_job._log_next_token == "tok-after-second-wait"


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------


class TestRenderPanel:
    """``_render_panel`` produces a ``rich.Panel`` whose body fields obey
    the always-present fields + conditional fields rules."""

    def _capture_panel_text(self, panel):
        """Render a ``rich.Panel`` to plain text and return it."""
        from rich.console import Console

        buffer_console = Console(record=True, width=120, force_terminal=True)
        buffer_console.print(panel)
        return buffer_console.export_text()

    def test_render_panel_shows_job_name(self, sdg_job):
        panel = sdg_job._render_panel()
        assert sdg_job.job_name in self._capture_panel_text(panel)

    def test_render_panel_shows_sample_progress_when_expected_set(self, sdg_job_with_progress):
        sdg_job_with_progress._current_samples = 4
        panel = sdg_job_with_progress._render_panel()
        text = self._capture_panel_text(panel)
        assert "4 / 10" in text

    def test_render_panel_omits_sample_progress_when_not_set(self, sdg_job):
        # No _expected_samples on a plain SDG job — no sample progress bar.
        sdg_job._current_samples = None
        panel = sdg_job._render_panel()
        text = self._capture_panel_text(panel)
        # The "N / M" pattern is the canonical render of the progress field.
        # If _expected_samples is None, no slash between two integers should
        # appear in the body. We use a relaxed check: no "/" within a
        # numeric context.
        import re

        assert re.search(r"\d+\s*/\s*\d+", text) is None

    def test_render_panel_shows_failure_reason(self, sdg_job):
        sdg_job.job_status = "Failed"
        sdg_job.failure_reason = "container exited with non-zero status (137)"
        panel = sdg_job._render_panel()
        text = self._capture_panel_text(panel)
        assert "container exited with non-zero status (137)" in text


# ---------------------------------------------------------------------------
# Transition Duration column (F17)
# ---------------------------------------------------------------------------


class TestTransitionDuration:
    """Regression coverage for F17 — a completed step's Duration must be the
    step's *duration*, not an ever-incrementing age.

    The service never populates ``EndTime`` on ``SecondaryStatusTransitions``
    (not even on a Completed job), so a completed step's end is recovered from
    the *next* transition's ``StartTime``. Only the genuinely in-progress step
    (no successor, job not terminal) should tick live; the trailing terminal
    marker row (no successor, job terminal) has no phase to measure and renders
    ``—``.
    """

    def _durations(self, panel):
        """Extract the Duration cell from each transition row in a panel."""
        from rich.console import Console
        import re

        buf = Console(record=True, width=120, force_terminal=True)
        buf.print(panel)
        text = buf.export_text()
        out = {}
        for step in (
            "Validating",
            "Preparing",
            "Executing",
            "Publishing",
            "Finalizing",
            "Completed",
        ):
            for line in text.splitlines():
                # Match the row whose Step cell is this phase (require a glyph
                # so "Completed" doesn't match the Job Status line). The
                # Duration is the last number/dash token before the panel
                # border, so search within the line rather than anchoring EOL.
                if re.search(rf"(✓|⋯)\s+{step}\b", line):
                    m = re.findall(r"(\d+\.\d+s|—)", line)
                    if m:
                        out[step] = m[-1]
                    break
        return out

    # Real DescribeJob payload shape: epoch-float StartTimes, no EndTime.
    _TRANSITIONS = [
        {"Status": "Validating", "StartTime": 1778889964.464, "StatusMessage": "Validating"},
        {"Status": "Preparing", "StartTime": 1778889974.558, "StatusMessage": "Preparing"},
        {"Status": "Executing", "StartTime": 1778889980.614, "StatusMessage": "Executing"},
        {"Status": "Publishing", "StartTime": 1778894689.309, "StatusMessage": "Publishing"},
        {"Status": "Finalizing", "StartTime": 1778894692.355, "StatusMessage": "Finalizing"},
        {"Status": "Completed", "StartTime": 1778894692.464, "StatusMessage": "Completed"},
    ]

    def test_completed_steps_show_frozen_duration(self, sdg_job):
        """Each completed work step's Duration is next.StartTime - StartTime."""
        sdg_job.job_status = "Completed"
        sdg_job._secondary_status_transitions = list(self._TRANSITIONS)
        durations = self._durations(sdg_job._render_panel(is_final=True))
        assert durations["Validating"] == "10.1s"
        assert durations["Preparing"] == "6.1s"
        assert durations["Executing"] == "4708.7s"
        assert durations["Publishing"] == "3.0s"
        assert durations["Finalizing"] == "0.1s"

    def test_terminal_marker_row_omits_duration(self, sdg_job):
        """The trailing Completed marker row has no successor and no phase to
        measure — it renders ``—`` rather than a live age."""
        sdg_job.job_status = "Completed"
        sdg_job._secondary_status_transitions = list(self._TRANSITIONS)
        durations = self._durations(sdg_job._render_panel(is_final=True))
        assert durations["Completed"] == "—"

    def test_completed_duration_does_not_increment_over_time(self, sdg_job):
        """The core F17 assertion: a completed step's Duration is identical
        across two renders at different wall-clock times (it must not tick)."""
        sdg_job.job_status = "Completed"
        sdg_job._secondary_status_transitions = list(self._TRANSITIONS)

        with _frozen_now(1778894700.0) as clock:
            first = self._durations(sdg_job._render_panel(is_final=True))
            # Advance the clock by an hour and re-render.
            clock.set(1778898300.0)
            second = self._durations(sdg_job._render_panel(is_final=True))

        for step in ("Validating", "Preparing", "Executing", "Publishing", "Finalizing"):
            assert first[step] == second[step], (
                f"{step} duration changed between renders: " f"{first[step]} -> {second[step]}"
            )

    def test_in_progress_step_ticks(self, sdg_job):
        """The active step (no successor, job not terminal) still ticks live —
        its Duration reflects now - StartTime and advances between renders."""
        sdg_job.job_status = "InProgress"
        # Up to Executing; Executing is the running tail with no successor.
        sdg_job._secondary_status_transitions = list(self._TRANSITIONS[:3])

        with _frozen_now(1778889990.614) as clock:
            first = self._durations(sdg_job._render_panel())
            clock.set(1778890000.614)
            second = self._durations(sdg_job._render_panel())

        # Completed work steps stay frozen...
        assert first["Validating"] == second["Validating"] == "10.1s"
        # ...but the active Executing row advances by the 10s clock delta.
        assert first["Executing"] == "10.0s"
        assert second["Executing"] == "20.0s"


# ---------------------------------------------------------------------------
# Panel mode wait()
# ---------------------------------------------------------------------------


class TestWaitPanelMode:
    """``wait(logs=False)`` is the new default — panel-only, no log tail."""

    def test_wait_already_terminal_returns_immediately(self, sdg_job):
        """Job is terminal on the first DescribeJob — final panel emitted
        via ``force_terminal_console`` and the polling loop is skipped."""
        assert sdg_job._log_next_token is None

        # Pre-set the terminal state and stub refresh() to a no-op so the
        # first describe tick observes Completed.
        sdg_job.job_status = "Completed"

        def fake_refresh():
            sdg_job.job_status = "Completed"
            return sdg_job

        with (
            patch("rich.console.Console") as mock_console_cls,
            patch.object(sdg_job, "refresh", side_effect=fake_refresh) as mock_refresh,
            patch.object(sdg_job, "_tail_logs", return_value=(None, [])),
        ):
            sdg_job.wait(poll=0, describe_interval=0, logs=False)

        # refresh() was called exactly once for the initial state check.
        assert mock_refresh.call_count == 1
        # A Console with force_terminal=True must have been used to emit
        # the final panel (so it serializes to .ipynb output).
        ft_calls = [
            c for c in mock_console_cls.call_args_list if c.kwargs.get("force_terminal") is True
        ]
        assert ft_calls, "force_terminal_console was not constructed"

    def test_wait_initial_panel_printed_before_first_describe(self, sdg_job):
        """At least one ``Console.print`` call happens before the first
        ``refresh()`` returns from the polling loop — the initial panel
        is printed up-front."""
        call_order = []
        printed_console = MagicMock()

        def record_print(*args, **kwargs):
            call_order.append("print")

        printed_console.print.side_effect = record_print

        def fake_refresh():
            call_order.append("refresh")
            sdg_job.job_status = "Completed"
            return sdg_job

        with (
            patch(
                "rich.console.Console",
                return_value=printed_console,
            ),
            patch.object(sdg_job, "refresh", side_effect=fake_refresh),
            patch.object(sdg_job, "_tail_logs", return_value=(None, [])),
        ):
            sdg_job.wait(poll=0, describe_interval=0, logs=False)

        # The first event in call_order MUST be a print (the initial panel).
        assert call_order, "no console.print() or refresh() was observed"
        assert call_order[0] == "print"

    def test_wait_logs_false_does_not_call_get_log_events(self, sdg_job):
        """``wait(logs=False)`` must not call ``get_log_events``
        directly — all CW access flows through ``_tail_logs``."""
        assert sdg_job._milestone_transitions == []

        def fake_refresh():
            sdg_job.job_status = "Completed"
            return sdg_job

        with (
            patch("boto3.client") as mock_boto,
            patch.object(sdg_job, "refresh", side_effect=fake_refresh),
        ):
            cw = MagicMock()
            mock_boto.return_value = cw
            sdg_job.wait(poll=0, describe_interval=0, logs=False)

            cw.get_log_events.assert_not_called()

    def test_quiet_period_backoff_emitted_once(self, sdg_job):
        """The backoff message is emitted at most once per
        ``wait()`` invocation, regardless of how long the job runs after
        the quiet-period threshold expires."""
        # Drive elapsed time forward across many ticks; never reach
        # terminal until well past the quiet period.
        # Step size 100s and range to 100_000s so the quiet period (300s)
        # is crossed quickly while we still have enough headroom to allow
        # rate-limited DescribeJob calls (gated below by describe_interval)
        # to terminate the loop.
        ticks = iter(range(0, 100_000, 100))
        printed_console = MagicMock()

        def fake_refresh():
            # Stay in 'InProgress' until we've spun through enough ticks
            # to cross the quiet period (5 min default).
            sdg_job.job_status = "InProgress"
            return sdg_job

        # Eventually terminate so the test exits.
        terminate_after = {"count": 0}

        def fake_refresh_then_terminate():
            terminate_after["count"] += 1
            if terminate_after["count"] >= 5:
                sdg_job.job_status = "Completed"
            else:
                sdg_job.job_status = "InProgress"
            return sdg_job

        backoff_substring = "still running"  # backoff text

        def get_time():
            try:
                return next(ticks)
            except StopIteration:
                return 1_000_000

        with (
            patch(
                "rich.console.Console",
                return_value=printed_console,
            ),
            patch(
                "sagemaker.ai_registry.data_preparer._job_resource.time.time",
                side_effect=get_time,
            ),
            patch.object(sdg_job, "refresh", side_effect=fake_refresh_then_terminate),
            patch.object(sdg_job, "_tail_logs", return_value=(None, [])),
        ):
            # describe_interval is set well past the quiet period (300s) so
            # the new tick-driven panel redraw — which resets
            # last_panel_time — cannot interfere with the quiet-period
            # timer firing. The redraw clause is also explicitly gated
            # off once _quiet_period_notified flips, so once the backoff
            # fires it will never fire again on subsequent ticks.
            sdg_job.wait(poll=0, describe_interval=400, logs=False)

        # Count print calls whose first positional arg contains the backoff
        # marker. Across many quiet ticks, this should happen at most once.
        backoff_prints = 0
        for c in printed_console.print.call_args_list:
            args = c.args or ()
            for a in args:
                if isinstance(a, str) and backoff_substring in a.lower():
                    backoff_prints += 1
                    break
        assert (
            backoff_prints == 1
        ), f"backoff message should be emitted exactly once; got {backoff_prints}"

    def test_describe_transient_failure_retried(self, sdg_job):
        """Transient ``ClientError`` from ``refresh()`` is retried; the loop
        does not raise on the first failure(s) and continues polling."""
        throttle = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "rate"}},
            "DescribeJob",
        )
        call_count = {"n": 0}

        def fake_refresh():
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise throttle
            sdg_job.job_status = "Completed"
            return sdg_job

        with (
            patch.object(sdg_job, "refresh", side_effect=fake_refresh),
            patch.object(sdg_job, "_tail_logs", return_value=(None, [])),
        ):
            # No exception should escape — wait() retries past 2 transient
            # failures and observes Completed on the third tick.
            sdg_job.wait(poll=0, describe_interval=0, logs=False)

        assert call_count["n"] >= 3

    def test_final_panel_uses_force_terminal_console(self, sdg_job):
        """The terminal panel goes through a
        ``force_terminal=True`` console so it serializes to .ipynb."""
        assert sdg_job._quiet_period_notified is False

        # Track which Console instances see a print() call after terminal.
        force_terminal_console = MagicMock(name="force_terminal_console")
        force_jupyter_console = MagicMock(name="force_jupyter_console")

        def console_factory(*args, **kwargs):
            if kwargs.get("force_terminal"):
                return force_terminal_console
            return force_jupyter_console

        def fake_refresh():
            sdg_job.job_status = "Completed"
            return sdg_job

        with (
            patch(
                "rich.console.Console",
                side_effect=console_factory,
            ),
            patch.object(sdg_job, "refresh", side_effect=fake_refresh),
            patch.object(sdg_job, "_tail_logs", return_value=(None, [])),
        ):
            sdg_job.wait(poll=0, describe_interval=0, logs=False)

        # The LAST print call across both consoles must be on the
        # force_terminal one — that's the terminal-state panel.
        last_force_terminal_print = force_terminal_console.print.call_count
        assert (
            last_force_terminal_print >= 1
        ), "final panel was not printed via force_terminal_console"

    def test_timeout_prints_panel_before_raising(self, sdg_job):
        """``DataPreparerJobTimeoutError`` is only raised after
        a panel reflecting the current state has been printed."""
        printed_console = MagicMock()
        events = []

        def record_print(*args, **kwargs):
            events.append("print")

        printed_console.print.side_effect = record_print

        def fake_refresh():
            # Never terminate; rely on timeout=0 to bail out immediately.
            sdg_job.job_status = "InProgress"
            return sdg_job

        with (
            patch(
                "rich.console.Console",
                return_value=printed_console,
            ),
            patch.object(sdg_job, "refresh", side_effect=fake_refresh),
            patch.object(sdg_job, "_tail_logs", return_value=(None, [])),
        ):
            with pytest.raises(DataPreparerJobTimeoutError):
                sdg_job.wait(poll=0, describe_interval=0, timeout=0, logs=False)

        # At least one print() must have happened before the raise.
        assert (
            "print" in events
        ), "no panel was printed before DataPreparerJobTimeoutError was raised"


# ---------------------------------------------------------------------------
# Log tail mode wait()
# ---------------------------------------------------------------------------


class TestWaitLogTailMode:
    """``wait(logs=True)`` activates pure log-tailing mode: append-only
    printing of INFO+ CW lines with no panels rendered mid-loop. A simple
    status summary is printed at the end."""

    def test_wait_logs_true_prints_info_lines(self, sdg_job):
        assert sdg_job._log_next_token is None

        info_line = "2026-05-28 22:10:47,670 - INFO - orchestrator - Job attempt 1 of 3."
        printed_console = MagicMock()

        def fake_refresh():
            sdg_job.job_status = "Completed"
            return sdg_job

        with (
            patch(
                "rich.console.Console",
                return_value=printed_console,
            ),
            patch.object(sdg_job, "refresh", side_effect=fake_refresh),
            patch.object(sdg_job, "_tail_logs", return_value=("tok-1", [info_line])),
        ):
            sdg_job.wait(poll=0, describe_interval=0, logs=True)

        # The INFO line must have been printed verbatim to the console.
        printed = [
            c.args[0]
            for c in printed_console.print.call_args_list
            if c.args and isinstance(c.args[0], str)
        ]
        assert any(info_line == p for p in printed), f"INFO line was not printed; printed={printed}"

    def test_wait_logs_true_suppresses_debug_lines(self, sdg_job):
        debug_line = "2026-05-28 22:10:47,670 - DEBUG - clients - Bedrock invoke begin"
        printed_console = MagicMock()

        def fake_refresh():
            sdg_job.job_status = "Completed"
            return sdg_job

        with (
            patch(
                "rich.console.Console",
                return_value=printed_console,
            ),
            patch.object(sdg_job, "refresh", side_effect=fake_refresh),
            patch.object(sdg_job, "_tail_logs", return_value=("tok-1", [debug_line])),
        ):
            sdg_job.wait(poll=0, describe_interval=0, logs=True)

        printed_strings = [
            c.args[0]
            for c in printed_console.print.call_args_list
            if c.args and isinstance(c.args[0], str)
        ]
        assert (
            debug_line not in printed_strings
        ), f"DEBUG line should be suppressed; printed={printed_strings}"

    def test_wait_logs_true_quiet_period_continues_tailing(self, sdg_job):
        """In log-tailing mode, after the quiet-period backoff is emitted the
        ``_tail_logs`` calls MUST continue on subsequent ticks (unlike
        ``logs=False`` which stops scanning)."""
        assert sdg_job._quiet_period_notified is False
        ticks = iter(range(0, 10_000, 10))
        printed_console = MagicMock()
        terminate_after = {"count": 0}
        tail_call_offsets = []

        def fake_refresh():
            terminate_after["count"] += 1
            if terminate_after["count"] >= 200:
                sdg_job.job_status = "Completed"
            else:
                sdg_job.job_status = "InProgress"
            return sdg_job

        def fake_tail(token):
            tail_call_offsets.append(terminate_after["count"])
            return (token or "tok-x", [])

        def get_time():
            try:
                return next(ticks)
            except StopIteration:
                return 1_000_000

        with (
            patch(
                "rich.console.Console",
                return_value=printed_console,
            ),
            patch(
                "sagemaker.ai_registry.data_preparer._job_resource.time.time",
                side_effect=get_time,
            ),
            patch.object(sdg_job, "refresh", side_effect=fake_refresh),
            patch.object(sdg_job, "_tail_logs", side_effect=fake_tail),
        ):
            sdg_job.wait(poll=0, describe_interval=0, logs=True)

        # The quiet period (5 min default) crosses around tick ~30. We
        # demand that _tail_logs was called both BEFORE and AFTER that
        # crossing — i.e. tail keeps running in logs=True mode.
        early_calls = [c for c in tail_call_offsets if c < 30]
        late_calls = [c for c in tail_call_offsets if c >= 50]
        assert early_calls and late_calls, (
            "tail should keep running after the quiet period in logs=True mode; "
            f"tail_call_offsets={tail_call_offsets}"
        )
