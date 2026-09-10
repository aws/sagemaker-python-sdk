"""Unit tests for LogStreamer utility."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.train.common_utils.log_streamer import (
    LogStreamer,
    _resolve_start_time_ms,
    _validate_poll,
    stream_log_loop,
)


class TestResolveStartTimeMs:
    def test_datetime_converts_to_ms(self):
        dt = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
        result = _resolve_start_time_ms(dt)
        assert result == int(dt.timestamp() * 1000)

    def test_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="start_time must be datetime or int"):
            _resolve_start_time_ms("2023-11-14")


class TestValidatePoll:
    def test_poll_too_low(self):
        with pytest.raises(ValueError, match="poll must be between 1 and 300"):
            _validate_poll(0)

    def test_poll_too_high(self):
        with pytest.raises(ValueError, match="poll must be between 1 and 300"):
            _validate_poll(301)

    def test_poll_not_int(self):
        with pytest.raises(ValueError, match="poll must be between 1 and 300"):
            _validate_poll(5.0)


def _make_mock_session():
    session = MagicMock()
    session.boto_session.region_name = "us-west-2"
    return session


def _make_client_error(code, message="error"):
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "operation_name",
    )


class TestLogStreamerStreamMode:
    def test_poll_once_returns_events(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        # describe_log_streams returns one stream
        logs_client.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [{"logStreamName": "my-job/algo-1"}]}
        ]
        # get_log_events returns events
        logs_client.get_log_events.return_value = {
            "events": [
                {"timestamp": 1700000000000, "message": "Training started\n"},
                {"timestamp": 1700000001000, "message": "Epoch 1\n"},
            ],
            "nextForwardToken": "token-2",
        }

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="my-job",
            sagemaker_session=session,
        )
        events = streamer.poll_once()

        assert len(events) == 2
        assert events[0] == (1700000000000, "Training started")
        assert events[1] == (1700000001000, "Epoch 1")

    def test_poll_once_returns_empty_when_caught_up(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        logs_client.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [{"logStreamName": "my-job/algo-1"}]}
        ]
        # First call returns events
        logs_client.get_log_events.return_value = {
            "events": [{"timestamp": 1700000000000, "message": "line1\n"}],
            "nextForwardToken": "token-A",
        }

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="my-job",
            sagemaker_session=session,
        )
        streamer.poll_once()

        # Second call: same token means caught up
        logs_client.get_log_events.return_value = {
            "events": [],
            "nextForwardToken": "token-A",
        }
        events = streamer.poll_once()
        assert events == []

    def test_poll_once_returns_empty_when_no_streams(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        logs_client.get_paginator.return_value.paginate.return_value = [
            {"logStreams": []}
        ]

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="my-job",
            sagemaker_session=session,
        )
        events = streamer.poll_once()
        assert events == []

    def test_start_time_ms_passed_to_first_call(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        logs_client.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [{"logStreamName": "my-job/algo-1"}]}
        ]
        logs_client.get_log_events.return_value = {
            "events": [],
            "nextForwardToken": "token-1",
        }

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="my-job",
            sagemaker_session=session,
            start_time_ms=1700000000000,
        )
        streamer.poll_once()

        call_kwargs = logs_client.get_log_events.call_args[1]
        assert call_kwargs["startTime"] == 1700000000000


class TestLogStreamerFilterMode:
    def test_poll_once_with_filter_pattern(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        logs_client.filter_log_events.return_value = {
            "events": [
                {"eventId": "e1", "timestamp": 1700000000000, "message": "log line 1\n"},
                {"eventId": "e2", "timestamp": 1700000001000, "message": "log line 2\n"},
            ]
        }

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Clusters/my-cluster/abc123",
            job_name="my-hp-job",
            sagemaker_session=session,
            filter_pattern='"my-hp-job"',
            start_time_ms=1700000000000,
        )
        events = streamer.poll_once()

        assert len(events) == 2
        assert events[0] == (1700000000000, "log line 1")
        assert events[1] == (1700000001000, "log line 2")

        # Verify filter_log_events was called with correct params
        call_kwargs = logs_client.filter_log_events.call_args[1]
        assert call_kwargs["logGroupName"] == "/aws/sagemaker/Clusters/my-cluster/abc123"
        assert call_kwargs["filterPattern"] == '"my-hp-job"'
        assert call_kwargs["startTime"] == 1700000000000
        assert call_kwargs["logStreamNamePrefix"] == "SagemakerHyperPodTrainingJob"

    def test_dedup_within_same_millisecond(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        # First poll
        logs_client.filter_log_events.return_value = {
            "events": [
                {"eventId": "e1", "timestamp": 1700000000000, "message": "line 1\n"},
            ]
        }
        streamer = LogStreamer(
            log_group="/aws/sagemaker/Clusters/c/id",
            job_name="job",
            sagemaker_session=session,
            filter_pattern='"job"',
        )
        events = streamer.poll_once()
        assert len(events) == 1

        # Second poll returns same event again
        logs_client.filter_log_events.return_value = {
            "events": [
                {"eventId": "e1", "timestamp": 1700000000000, "message": "line 1\n"},
                {"eventId": "e2", "timestamp": 1700000000000, "message": "line 2\n"},
            ]
        }
        events = streamer.poll_once()
        # e1 should be deduped, only e2 is new
        assert len(events) == 1
        assert events[0][1] == "line 2"

    def test_dedup_clears_on_timestamp_advance(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Clusters/c/id",
            job_name="job",
            sagemaker_session=session,
            filter_pattern='"job"',
        )

        # First poll
        logs_client.filter_log_events.return_value = {
            "events": [
                {"eventId": "e1", "timestamp": 1000, "message": "a\n"},
            ]
        }
        streamer.poll_once()

        # Second poll: new timestamp
        logs_client.filter_log_events.return_value = {
            "events": [
                {"eventId": "e2", "timestamp": 2000, "message": "b\n"},
            ]
        }
        events = streamer.poll_once()
        assert len(events) == 1
        assert events[0] == (2000, "b")


class TestStreamLogLoop:
    """Tests for the shared stream_log_loop function."""

    def test_early_exit_when_already_terminal(self):
        """stream_log_loop returns immediately if status_fn reports terminal."""

        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value
        logs_client.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [{"logStreamName": "job/algo-1"}]}
        ]
        logs_client.get_log_events.return_value = {
            "events": [{"timestamp": 1000, "message": "done\n"}],
            "nextForwardToken": "t1",
        }

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="job",
            sagemaker_session=session,
        )

        call_count = [0]

        def _status():
            call_count[0] += 1
            return "Completed"

        stream_log_loop(streamer, poll=1, status_fn=_status)
        # status_fn called exactly once (the upfront check)
        assert call_count[0] == 1

    def test_exits_when_status_becomes_terminal(self):
        """stream_log_loop exits after status transitions to terminal."""

        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value
        logs_client.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [{"logStreamName": "job/algo-1"}]}
        ]
        # Return events then empty (caught up)
        logs_client.get_log_events.side_effect = [
            {"events": [{"timestamp": 1000, "message": "training\n"}], "nextForwardToken": "t1"},
            {"events": [], "nextForwardToken": "t1"},
        ]

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="job",
            sagemaker_session=session,
        )

        statuses = iter(["Training", "Completed"])

        with patch("time.sleep"):
            stream_log_loop(streamer, poll=1, status_fn=lambda: next(statuses))

    def test_empty_cycles_feedback(self):
        """stream_log_loop logs 'No log events yet' after ~30s of empty polls."""

        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value
        logs_client.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [{"logStreamName": "job/algo-1"}]}
        ]
        logs_client.get_log_events.return_value = {
            "events": [], "nextForwardToken": "t1",
        }

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="job",
            sagemaker_session=session,
        )

        call_count = [0]

        def _status():
            call_count[0] += 1
            # With poll=5, warn_cycle=6. Become terminal after 8 calls.
            return "Completed" if call_count[0] >= 8 else "Training"

        with patch("time.sleep"):
            with patch("sagemaker.train.common_utils.log_streamer.logger") as mock_logger:
                stream_log_loop(streamer, poll=5, status_fn=_status)
                info_calls = [str(c) for c in mock_logger.info.call_args_list]
                assert any("No log events yet" in c for c in info_calls)

    def test_resource_not_found_with_running_job_retries(self):
        """stream_log_loop retries on ResourceNotFoundException while job runs."""

        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value
        # First call raises ResourceNotFound, second returns events
        logs_client.get_paginator.return_value.paginate.side_effect = [
            _make_client_error("ResourceNotFoundException"),
            [{"logStreams": [{"logStreamName": "job/algo-1"}]}],
        ]
        logs_client.get_log_events.return_value = {
            "events": [{"timestamp": 1000, "message": "hi\n"}],
            "nextForwardToken": "t1",
        }

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="job",
            sagemaker_session=session,
        )

        statuses = iter(["Training", "Training", "Completed"])

        with patch("time.sleep"):
            stream_log_loop(streamer, poll=1, status_fn=lambda: next(statuses))

    def test_access_denied_propagates(self):
        """stream_log_loop raises AccessDeniedException."""

        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value
        logs_client.get_paginator.return_value.paginate.side_effect = _make_client_error(
            "AccessDeniedException"
        )

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="job",
            sagemaker_session=session,
        )

        with pytest.raises(ClientError):
            stream_log_loop(streamer, poll=1, status_fn=lambda: "Training")


class TestLogStreamerErrorHandling:
    def test_resource_not_found_propagates(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        logs_client.get_paginator.return_value.paginate.side_effect = _make_client_error(
            "ResourceNotFoundException"
        )

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="my-job",
            sagemaker_session=session,
        )
        with pytest.raises(ClientError) as exc_info:
            streamer.poll_once()
        assert exc_info.value.response["Error"]["Code"] == "ResourceNotFoundException"

    def test_access_denied_propagates(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        logs_client.get_paginator.return_value.paginate.side_effect = _make_client_error(
            "AccessDeniedException"
        )

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="my-job",
            sagemaker_session=session,
        )
        with pytest.raises(ClientError) as exc_info:
            streamer.poll_once()
        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"

    def test_throttling_returns_empty(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        logs_client.get_paginator.return_value.paginate.side_effect = _make_client_error(
            "ThrottlingException"
        )

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Job/AgentRFT",
            job_name="my-job",
            sagemaker_session=session,
        )
        events = streamer.poll_once()
        assert events == []

    def test_filter_mode_access_denied_propagates(self):
        session = _make_mock_session()
        logs_client = session.boto_session.client.return_value

        logs_client.filter_log_events.side_effect = _make_client_error(
            "AccessDeniedException"
        )

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Clusters/c/id",
            job_name="job",
            sagemaker_session=session,
            filter_pattern='"job"',
        )
        with pytest.raises(ClientError):
            streamer.poll_once()


class TestPollTailStreamMode:
    """Unit tests for LogStreamer.poll_tail() in stream mode (SMTJ)."""

    def test_poll_tail_returns_last_n_events(self):
        """poll_tail(n) returns the last N events in chronological order."""
        session = _make_mock_session()
        mock_logs = session.boto_session.client.return_value

        # describe_log_streams returns one stream
        mock_logs.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [{"logStreamName": "job/algo-1-123"}]}
        ]

        # get_log_events: first call (startFromHead=False) returns events via backward pagination
        mock_logs.get_log_events.side_effect = [
            # First call: startFromHead=False, no token → returns empty + backward token
            {"events": [], "nextBackwardToken": "btoken1"},
            # Second call: with backward token + limit → returns last N events
            {
                "events": [
                    {"timestamp": 300, "message": "third"},
                    {"timestamp": 200, "message": "second"},
                    {"timestamp": 100, "message": "first"},
                ],
                "nextBackwardToken": "btoken1",
            },
        ]

        streamer = LogStreamer(
            log_group="/aws/sagemaker/TrainingJobs",
            job_name="job",
            sagemaker_session=session,
        )

        result = streamer.poll_tail(3)

        assert len(result) == 3
        # Should be sorted by timestamp (chronological)
        assert result[0] == (100, "first")
        assert result[1] == (200, "second")
        assert result[2] == (300, "third")

    def test_poll_tail_multi_stream_merges_by_timestamp(self):
        """poll_tail merges events across multiple streams and returns globally last N."""
        session = _make_mock_session()
        mock_logs = session.boto_session.client.return_value

        mock_logs.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [
                {"logStreamName": "job/algo-1-123"},
                {"logStreamName": "job/algo-2-456"},
            ]}
        ]

        # Stream 1: events at ts 100, 300
        # Stream 2: events at ts 200, 400
        mock_logs.get_log_events.side_effect = [
            # Stream 1: first call (startFromHead=False)
            {"events": [], "nextBackwardToken": "bt1"},
            # Stream 1: second call with token
            {
                "events": [
                    {"timestamp": 300, "message": "s1-late"},
                    {"timestamp": 100, "message": "s1-early"},
                ],
                "nextBackwardToken": "bt1",
            },
            # Stream 2: first call (startFromHead=False)
            {"events": [], "nextBackwardToken": "bt2"},
            # Stream 2: second call with token
            {
                "events": [
                    {"timestamp": 400, "message": "s2-latest"},
                    {"timestamp": 200, "message": "s2-mid"},
                ],
                "nextBackwardToken": "bt2",
            },
        ]

        streamer = LogStreamer(
            log_group="/aws/sagemaker/TrainingJobs",
            job_name="job",
            sagemaker_session=session,
        )

        result = streamer.poll_tail(3)

        # Should take globally last 3 by timestamp: 200, 300, 400
        assert len(result) == 3
        assert result[0] == (200, "s2-mid")
        assert result[1] == (300, "s1-late")
        assert result[2] == (400, "s2-latest")


class TestPollTailFilterMode:
    """Unit tests for LogStreamer.poll_tail() in filter mode (SMHP)."""

    def test_poll_tail_filter_mode_returns_events(self):
        """poll_tail in filter mode paginates with startFromHead=False."""
        session = _make_mock_session()
        mock_logs = session.boto_session.client.return_value

        # First page: 0 events (API scanning streams), second page: events found
        mock_logs.filter_log_events.side_effect = [
            {"events": [], "nextToken": "page2"},
            {
                "events": [
                    {"timestamp": 300, "message": "newest"},
                    {"timestamp": 200, "message": "middle"},
                    {"timestamp": 100, "message": "oldest"},
                ],
            },
        ]

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Clusters/c/id",
            job_name="job",
            sagemaker_session=session,
            filter_pattern='"job"',
            start_time_ms=1784833626000,  # Must be on or after 2024-01-01
        )

        result = streamer.poll_tail(3)

        # Events come reverse-chronological, should be reversed to chronological
        assert len(result) == 3
        assert result[0] == (100, "oldest")
        assert result[1] == (200, "middle")
        assert result[2] == (300, "newest")

        # Verify startFromHead=False was passed
        call_args = mock_logs.filter_log_events.call_args_list[0]
        assert call_args[1]["startFromHead"] is False

    def test_poll_tail_filter_mode_warns_without_start_time(self):
        """poll_tail logs a warning when no start_time is set."""
        session = _make_mock_session()
        mock_logs = session.boto_session.client.return_value

        mock_logs.filter_log_events.return_value = {
            "events": [{"timestamp": 100, "message": "msg"}],
        }

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Clusters/c/id",
            job_name="job",
            sagemaker_session=session,
            filter_pattern='"job"',
            start_time_ms=None,  # No start time
        )

        with patch("sagemaker.train.common_utils.log_streamer.logger") as mock_logger:
            streamer.poll_tail(1)
            mock_logger.warning.assert_called_once()
            assert "start_time" in mock_logger.warning.call_args[0][0]


class TestStreamLogLoopTailLines:
    """Unit tests for stream_log_loop with tail_lines parameter."""

    def test_tail_lines_calls_poll_tail_and_returns(self, capsys):
        """When tail_lines is set, stream_log_loop calls poll_tail and prints."""
        streamer = MagicMock()
        streamer.poll_tail.return_value = [
            (1000, "line one"),
            (2000, "line two"),
            (3000, "line three"),
        ]
        status_fn = MagicMock(return_value="Completed")

        stream_log_loop(streamer, poll=5, status_fn=status_fn, tail_lines=3)

        streamer.poll_tail.assert_called_once_with(3)
        # status_fn should NOT be called — tail_lines returns immediately
        status_fn.assert_not_called()

        captured = capsys.readouterr()
        assert "line one" in captured.out
        assert "line two" in captured.out
        assert "line three" in captured.out

    def test_tail_lines_none_does_not_call_poll_tail(self):
        """When tail_lines is None, stream_log_loop uses normal streaming."""
        streamer = MagicMock()
        streamer.poll_once.return_value = []
        status_fn = MagicMock(return_value="Completed")

        stream_log_loop(streamer, poll=5, status_fn=status_fn, tail_lines=None)

        streamer.poll_tail.assert_not_called()
        status_fn.assert_called()


    def test_poll_tail_filter_mode_raises_for_pre_2024_start_time(self):
        """poll_tail raises ValueError when start_time is before 2024-01-01."""
        session = _make_mock_session()

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Clusters/c/id",
            job_name="job",
            sagemaker_session=session,
            filter_pattern='"job"',
            start_time_ms=1000,  # Way before 2024-01-01
        )

        with pytest.raises(ValueError, match="before 2024-01-01"):
            streamer.poll_tail(5)

    def test_poll_tail_filter_mode_paginates_multiple_pages(self):
        """poll_tail paginates until enough events are found."""
        session = _make_mock_session()
        mock_logs = session.boto_session.client.return_value

        # Simulate: first 3 pages return empty (scanning streams), 4th has events
        mock_logs.filter_log_events.side_effect = [
            {"events": [], "nextToken": "page2"},
            {"events": [], "nextToken": "page3"},
            {"events": [], "nextToken": "page4"},
            {
                "events": [
                    {"timestamp": 300, "message": "third"},
                    {"timestamp": 200, "message": "second"},
                    {"timestamp": 100, "message": "first"},
                ],
            },
        ]

        streamer = LogStreamer(
            log_group="/aws/sagemaker/Clusters/c/id",
            job_name="job",
            sagemaker_session=session,
            filter_pattern='"job"',
            start_time_ms=1784833626000,
        )

        result = streamer.poll_tail(3)

        assert len(result) == 3
        # Should be reversed to chronological order
        assert result[0] == (100, "first")
        assert result[1] == (200, "second")
        assert result[2] == (300, "third")
        # Should have made 4 API calls
        assert mock_logs.filter_log_events.call_count == 4
