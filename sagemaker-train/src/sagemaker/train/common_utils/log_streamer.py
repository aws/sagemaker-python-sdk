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
"""CloudWatch log streaming utility for SageMaker training and evaluation jobs."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from botocore.exceptions import ClientError

from sagemaker.train.defaults import TrainDefaults

logger = logging.getLogger(__name__)

JOB_LOG_GROUP_PREFIX = "/aws/sagemaker/Job"
AGENT_RFT_LOG_GROUP = f"{JOB_LOG_GROUP_PREFIX}/AgentRFT"
AGENT_RFT_EVAL_LOG_GROUP = f"{JOB_LOG_GROUP_PREFIX}/AgentRFTEvaluation"
TERMINAL_STATUSES = frozenset({"Completed", "Succeeded", "Failed", "Stopped"})

_MIN_POLL = 1
_MAX_POLL = 300
_SMHP_STREAM_PREFIX = "SagemakerHyperPodTrainingJob"


def _resolve_start_time_ms(start_time: datetime | int | None) -> int | None:
    """Convert start_time to epoch milliseconds.

    :param start_time: datetime, epoch milliseconds int, or None.
    :returns: Epoch milliseconds or None.
    :raises TypeError: If start_time is not a supported type.
    """
    if start_time is None:
        return None
    if isinstance(start_time, datetime):
        return int(start_time.timestamp() * 1000)
    if isinstance(start_time, int):
        return start_time
    raise TypeError(
        f"start_time must be datetime or int (epoch ms), got: {type(start_time).__name__}"
    )


def _validate_poll(poll: int) -> None:
    """Validate poll interval.

    :raises ValueError: If poll is out of range.
    """
    if not isinstance(poll, int) or poll < _MIN_POLL or poll > _MAX_POLL:
        raise ValueError(f"poll must be between {_MIN_POLL} and {_MAX_POLL} seconds, got: {poll!r}")


def _format_timestamp(ts_ms: int) -> str:
    """Format epoch milliseconds to ISO timestamp string."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


class LogStreamer:
    """Fetches new CloudWatch log events since last call.

    This is a low-level utility that does NOT own the polling loop,
    does NOT check job status, and does NOT handle timeouts. The caller
    is responsible for all of that.

    :param log_group: CloudWatch log group name.
    :param job_name: Job name used as log stream prefix (stream mode)
        or in the filter pattern (filter mode).
    :param sagemaker_session: SageMaker session for obtaining CW client.
    :param filter_pattern: If provided, uses filter_log_events API (HyperPod).
        If None, uses describe_log_streams + get_log_events (stream mode).
    :param start_time_ms: Initial start time as epoch milliseconds.
    """

    def __init__(
        self,
        log_group: str,
        job_name: str,
        sagemaker_session=None,
        filter_pattern: str | None = None,
        start_time_ms: int | None = None,
    ):
        self._log_group = log_group
        self._job_name = job_name
        self._filter_pattern = filter_pattern
        self._start_time_ms = start_time_ms

        session = sagemaker_session or TrainDefaults.get_sagemaker_session()
        region = session.boto_session.region_name
        self._logs_client = session.boto_session.client("logs", region_name=region)

        # Stream mode state
        self._stream_handlers: list[dict] | None = None

        # Filter mode state
        self._last_timestamp_ms: int | None = start_time_ms
        self._last_event_ids: set[str] = set()

    def poll_once(self) -> list[tuple[int, str]]:
        """Fetch and return new log event messages since last poll.

        :returns: List of (timestamp_ms, message) tuples. May be empty.
        :raises: ClientError with ResourceNotFoundException or
            AccessDeniedException propagates to caller. Other ClientErrors
            are caught internally and result in an empty list.
        """
        try:
            if self._filter_pattern is not None:
                return self._poll_filter_mode()
            return self._poll_stream_mode()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ResourceNotFoundException", "AccessDeniedException"):
                raise
            logger.debug("Transient CloudWatch error: %s", e)
            return []

    def poll_tail(self, n: int) -> list[tuple[int, str]]:
        """Fetch the last N log events in chronological order.

        Behaves like ``tail -n`` or ``kubectl logs --tail=N``.

        For stream mode (SMTJ): uses get_log_events backward pagination to
        fetch the last N events per stream, merges by timestamp, returns the
        globally last N.

        For filter mode (SMHP): uses filter_log_events with startFromHead=False
        to fetch recent events. Requires startTime to be set for CloudWatch to
        scope the search efficiently.

        :param n: Number of most recent log events to return.
        :returns: List of (timestamp_ms, message) tuples in chronological order.
        """
        if self._filter_pattern is not None:
            return self._tail_filter_mode(n)
        return self._tail_stream_mode(n)

    def _tail_stream_mode(self, n: int) -> list[tuple[int, str]]:
        """Get last N events via get_log_events backward pagination.

        For multi-stream jobs, fetches last N from each stream, merges by
        timestamp, and returns the globally last N events in chronological order.
        """
        if self._stream_handlers is None:
            self._stream_handlers = self._discover_streams()
            if not self._stream_handlers:
                return []

        all_results = []
        for handler in self._stream_handlers:
            events = []
            next_token = None

            while len(events) < n:
                kwargs = {
                    "logGroupName": self._log_group,
                    "logStreamName": handler["stream_name"],
                    "limit": n - len(events),
                    "startFromHead": False,
                }
                if next_token:
                    kwargs["nextToken"] = next_token

                response = self._logs_client.get_log_events(**kwargs)
                if response.get("events"):
                    events.extend(response["events"])

                backward_token = response.get("nextBackwardToken")
                if backward_token and backward_token != next_token:
                    next_token = backward_token
                else:
                    break

            for event in events:
                message = event.get("message", "").rstrip()
                ts = event.get("timestamp", 0)
                if message:
                    all_results.append((ts, message))

        # Sort by timestamp across all streams, take the last N globally
        all_results.sort(key=lambda x: x[0])
        return all_results[-n:]

    def _tail_filter_mode(self, n: int) -> list[tuple[int, str]]:
        """Get last N events via filter_log_events with startFromHead=False.

        Uses reverse-chronological order to get the most recent events first.
        Requires startTime to be set for CloudWatch to scope the search.
        Paginates without limit (faster scanning), then slices client-side.

        Note: startFromHead=False with logStreamNamePrefix may require several
        pagination calls before CloudWatch locates the matching streams.
        """
        # CloudWatch requires startTime on or after 2024-01-01 for
        # startFromHead=False with filter_log_events.
        _JAN_1_2024_MS = 1704067200000
        if self._last_timestamp_ms and self._last_timestamp_ms < _JAN_1_2024_MS:
            raise ValueError(
                "stream_logs does not support tail_lines when start_time is before 2024-01-01."
            )
        params = {
            "logGroupName": self._log_group,
            "logStreamNamePrefix": _SMHP_STREAM_PREFIX,
            "filterPattern": self._filter_pattern,
            "startFromHead": False,
        }
        if self._last_timestamp_ms is not None:
            params["startTime"] = self._last_timestamp_ms
        else:
            logger.warning(
                "No start_time provided for tail_lines. Scanning without time "
                "bounds may take a while to identify matching log streams."
            )

        results = []
        next_token = None

        # filter_log_events bounds pages by scan volume, not result count.
        # Must follow nextToken until N matching events are collected.
        while True:
            if next_token:
                params["nextToken"] = next_token

            response = self._logs_client.filter_log_events(**params)
            for event in response.get("events", []):
                message = event.get("message", "").rstrip()
                ts = event.get("timestamp", 0)
                if message:
                    results.append((ts, message))

            # Stop once we have enough events
            if len(results) >= n:
                break

            next_token = response.get("nextToken")
            if not next_token:
                break

        # Events come in reverse chronological order; take first N and reverse
        results = results[:n]
        results.reverse()
        return results

    def _poll_filter_mode(self) -> list[tuple[int, str]]:
        """Poll using filter_log_events (HyperPod style)."""
        params = {
            "logGroupName": self._log_group,
            "logStreamNamePrefix": _SMHP_STREAM_PREFIX,
            "filterPattern": self._filter_pattern,
        }
        if self._last_timestamp_ms is not None:
            params["startTime"] = self._last_timestamp_ms

        response = self._logs_client.filter_log_events(**params)
        events = response.get("events", [])

        results = []
        for event in events:
            event_id = event.get("eventId", "")
            ts = event.get("timestamp", 0)

            # Dedup: skip events from same millisecond already seen
            if ts == self._last_timestamp_ms and event_id in self._last_event_ids:
                continue

            message = event.get("message", "").rstrip()
            if message:
                results.append((ts, message))

            # Advance cursor
            if ts > (self._last_timestamp_ms or 0):
                self._last_timestamp_ms = ts
                self._last_event_ids = {event_id}
            elif ts == self._last_timestamp_ms:
                self._last_event_ids.add(event_id)

        return results

    def _poll_stream_mode(self) -> list[tuple[int, str]]:
        """Poll using describe_log_streams + get_log_events per stream."""
        if self._stream_handlers is None:
            self._stream_handlers = self._discover_streams()
            if not self._stream_handlers:
                return []

        results = []
        for handler in self._stream_handlers:
            events = self._get_events_for_stream(handler)
            results.extend(events)

        return results

    def _discover_streams(self) -> list[dict]:
        """Discover log streams matching the job name prefix."""
        handlers = []
        kwargs = {
            "logGroupName": self._log_group,
            "logStreamNamePrefix": self._job_name,
        }
        paginator = self._logs_client.get_paginator("describe_log_streams")
        for page in paginator.paginate(**kwargs):
            for stream in page.get("logStreams", []):
                handlers.append({
                    "stream_name": stream["logStreamName"],
                    "next_token": None,
                    "started": False,
                })
        return handlers

    def _get_events_for_stream(self, handler: dict) -> list[tuple[int, str]]:
        """Get new events from a single log stream."""
        kwargs = {
            "logGroupName": self._log_group,
            "logStreamName": handler["stream_name"],
            "startFromHead": True,
        }
        if handler["next_token"]:
            kwargs["nextToken"] = handler["next_token"]
        elif not handler["started"] and self._start_time_ms is not None:
            kwargs["startTime"] = self._start_time_ms

        response = self._logs_client.get_log_events(**kwargs)
        new_token = response.get("nextForwardToken")

        # Same token means caught up — no new events
        if new_token == handler["next_token"]:
            return []

        handler["next_token"] = new_token
        handler["started"] = True

        results = []
        for event in response.get("events", []):
            message = event.get("message", "").rstrip()
            ts = event.get("timestamp", 0)
            if message:
                results.append((ts, message))

        return results


def stream_log_loop(
    streamer: LogStreamer,
    poll: int,
    status_fn: Callable[[], str],
    tail_lines: Optional[int] = None,
) -> None:
    """Run the standard log streaming loop.

    Polls the streamer, prints events, checks job status, and exits
    when terminal. Handles ResourceNotFoundException with escalating
    feedback and propagates AccessDeniedException.

    :param streamer: A configured LogStreamer instance.
    :param poll: Seconds between polls.
    :param status_fn: Callable that returns the current job status string.
    :param tail_lines: Optional number of most recent log events to return.
        Fetches the last N events (like ``tail -n`` or ``kubectl logs --tail``),
        regardless of whether the job is still running or completed.
        If not provided, streams all logs until the job completes.
    """
    _CW_PREFIX = "[CloudWatch] "

    def _print_event(ts_ms: int, message: str):
        """Print a formatted CloudWatch log event."""
        print(f"{_CW_PREFIX}[{_format_timestamp(ts_ms)}] {message}")

    # When tail_lines is set, fetch the last N events and return immediately.
    if tail_lines:
        events = streamer.poll_tail(tail_lines)
        for ts_ms, message in events:
            _print_event(ts_ms, message)
        return

    status = status_fn()
    if status in TERMINAL_STATUSES:
        logger.info("Job already in terminal state: %s", status)
        try:
            while True:
                events = streamer.poll_once()
                if not events:
                    break
                for ts_ms, message in events:
                    _print_event(ts_ms, message)
        except ClientError:
            pass
        logger.info("Job finished with status: %s", status)
        return

    empty_cycles = 0
    max_empty_cycles = max(300 // poll, 1)  # ~5 minutes
    warn_cycle = max(30 // poll, 1)  # ~30 seconds

    while True:
        try:
            events = streamer.poll_once()
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "AccessDeniedException":
                raise
            if error_code == "ResourceNotFoundException":
                status = status_fn()
                if status in TERMINAL_STATUSES:
                    logger.info("Job finished before producing logs.")
                    return
                empty_cycles += 1
                if empty_cycles == 1:
                    logger.info("Waiting for log group to become available...")
                elif empty_cycles >= max_empty_cycles:
                    raise RuntimeError(
                        f"Log group not found after {empty_cycles * poll}s. "
                        "Check IAM permissions for logs:GetLogEvents and "
                        "logs:DescribeLogStreams."
                    )
                time.sleep(poll)
                continue
            raise

        if events:
            empty_cycles = 0
            for ts_ms, message in events:
                _print_event(ts_ms, message)
        else:
            empty_cycles += 1
            if empty_cycles == warn_cycle:
                logger.info("No log events yet, still waiting...")
            elif empty_cycles >= max_empty_cycles:
                logger.warning("No log events found after 5 minutes.")
                return

        status = status_fn()
        if status in TERMINAL_STATUSES:
            for ts_ms, message in streamer.poll_once():
                _print_event(ts_ms, message)
            logger.info("Job finished with status: %s", status)
            return

        try:
            time.sleep(poll)
        except KeyboardInterrupt:
            logger.info("Streaming stopped by user.")
            return
