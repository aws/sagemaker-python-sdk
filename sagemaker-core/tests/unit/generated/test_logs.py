import botocore
import pytest
from unittest.mock import patch, MagicMock
from sagemaker.core.utils.logs import LogStreamHandler, MultiLogStreamHandler


def test_single_stream_handler_get_latest():
    mock_log_events = [
        {
            "nextForwardToken": "nextToken1",
            "events": [
                {"ingestionTime": 123456789, "message": "test message", "timestamp": 123456789}
            ],
        },
        {"nextForwardToken": "nextToken2", "events": []},
    ]

    log_stream_handler = LogStreamHandler("logGroupName", "logStreamName", 0)

    with patch.object(log_stream_handler, "cw_client") as mock_cw_client:
        mock_cw_client.get_log_events.side_effect = mock_log_events
        events = log_stream_handler.get_latest_log_events()

        result = next(events)

        assert result == (
            "logStreamName",
            {"ingestionTime": 123456789, "message": "test message", "timestamp": 123456789},
        )

        mock_cw_client.get_log_events.assert_called_once_with(
            logGroupName="logGroupName", logStreamName="logStreamName", startFromHead=True
        )

        with pytest.raises(StopIteration):
            next(events)


@patch("sagemaker.core.utils.logs.MultiLogStreamHandler.ready", autospec=True)
def test_multi_stream_handler_get_latest(mock_ready):
    mock_ready.return_value = True

    mock_stream = MagicMock(spec=LogStreamHandler)
    mock_stream.get_latest_log_events.return_value = iter(
        [
            (
                "streamName",
                {"ingestionTime": 123456789, "message": "test message", "timestamp": 123456789},
            )
        ]
    )

    mulit_log_stream_handler = MultiLogStreamHandler("log_group_name", "training_job_name", 1)
    mulit_log_stream_handler.streams = [mock_stream]

    events = mulit_log_stream_handler.get_latest_log_events()

    result = next(events)

    assert result == (
        "streamName",
        {"ingestionTime": 123456789, "message": "test message", "timestamp": 123456789},
    )

    with pytest.raises(StopIteration):
        next(events)


def test_ready():
    mock_streams = {
        "logStreams": [{"logStreamName": "streamName"}],
        "nextToken": None,
    }

    multi_log_stream_handler = MultiLogStreamHandler("logGroupName", "logStreamNamePrefix", 1)
    with patch.object(multi_log_stream_handler, "cw_client") as mock_cw_client:
        mock_cw_client.describe_log_streams.return_value = mock_streams

        result = multi_log_stream_handler.ready()

        assert result == True
        mock_cw_client.describe_log_streams.assert_called_once()


def test_ready_streams_set():
    log_stream = LogStreamHandler("logGroupName", "logStreamName", 0)
    multi_log_stream_handler = MultiLogStreamHandler("logGroupName", "logStreamNamePrefix", 1)
    multi_log_stream_handler.streams = [log_stream]

    with patch.object(multi_log_stream_handler, "cw_client") as mock_cw_client:
        result = multi_log_stream_handler.ready()

        assert result == True
        mock_cw_client.describe_log_streams.assert_not_called()


def test_not_ready():
    mock_streams = {"logStreams": [], "nextToken": None}

    multi_log_stream_handler = MultiLogStreamHandler("logGroupName", "logStreamNamePrefix", 1)
    with patch.object(multi_log_stream_handler, "cw_client") as mock_cw_client:
        mock_cw_client.describe_log_streams.return_value = mock_streams

        result = multi_log_stream_handler.ready()

        assert result == False
        mock_cw_client.describe_log_streams.assert_called_once()


def test_ready_resource_not_found():

    multi_log_stream_handler = MultiLogStreamHandler("logGroupName", "logStreamNamePrefix", 1)
    with patch.object(multi_log_stream_handler, "cw_client") as mock_cw_client:
        mock_cw_client.describe_log_streams.side_effect = botocore.exceptions.ClientError(
            error_response={"Error": {"Code": "ResourceNotFoundException"}}, operation_name="test"
        )

        result = multi_log_stream_handler.ready()

        assert result == False
        mock_cw_client.describe_log_streams.assert_called_once()
