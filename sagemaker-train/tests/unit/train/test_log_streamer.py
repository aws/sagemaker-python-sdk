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
