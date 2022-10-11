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
"""Contains classes to manage metrics for Sagemaker Experiment"""
from __future__ import absolute_import

import datetime
import json
import logging
import os
import time

import dateutil.tz
from botocore.config import Config
from sagemaker.apiutils import _utils


METRICS_DIR = os.environ.get("SAGEMAKER_METRICS_DIRECTORY", ".")
METRIC_TS_LOWER_BOUND_TO_NOW = 1209600  # on seconds
METRIC_TS_UPPER_BOUND_FROM_NOW = 7200  # on seconds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: remove this _SageMakerFileMetricsWriter class
# when _MetricsManager is fully ready
class _SageMakerFileMetricsWriter(object):
    """Write metric data to file."""

    def __init__(self, metrics_file_path=None):
        """Construct a `_SageMakerFileMetricsWriter` object"""
        self._metrics_file_path = metrics_file_path
        self._file = None
        self._closed = False

    def log_metric(self, metric_name, value, timestamp=None, step=None):
        """Write a metric to file.

        Args:
            metric_name (str): The name of the metric.
            value (float): The value of the metric.
            timestamp (datetime.datetime): Timestamp of the metric.
                If not specified, the current UTC time will be used.
            step (int):  Iteration number of the metric (default: None).

        Raises:
            SageMakerMetricsWriterException: If the metrics file is closed.
            AttributeError: If file has been initialized and the writer hasn't been closed.
        """
        raw_metric_data = _RawMetricData(
            metric_name=metric_name, value=value, timestamp=timestamp, step=step
        )
        try:
            logger.debug("Writing metric: %s", raw_metric_data)
            self._file.write(json.dumps(raw_metric_data.to_record()))
            self._file.write("\n")
        except AttributeError as attr_err:
            if self._closed:
                raise SageMakerMetricsWriterException("log_metric called on a closed writer")
            if not self._file:
                self._file = open(self._get_metrics_file_path(), "a", buffering=1)
                self._file.write(json.dumps(raw_metric_data.to_record()))
                self._file.write("\n")
            else:
                raise attr_err

    def close(self):
        """Closes the metric file."""
        if not self._closed and self._file:
            self._file.close()
            self._file = None  # invalidate reference, causing subsequent log_metric to fail.
        self._closed = True

    def __enter__(self):
        """Return self"""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Execute self.close()"""
        self.close()

    def __del__(self):
        """Execute self.close()"""
        self.close()

    def _get_metrics_file_path(self):
        """Get file path to store metrics"""
        pid_filename = "{}.json".format(str(os.getpid()))
        metrics_file_path = self._metrics_file_path or os.path.join(METRICS_DIR, pid_filename)
        logger.debug("metrics_file_path = %s", metrics_file_path)
        return metrics_file_path


class SageMakerMetricsWriterException(Exception):
    """SageMakerMetricsWriterException"""

    def __init__(self, message, errors=None):
        """Construct a `SageMakerMetricsWriterException` instance"""
        super().__init__(message)
        if errors:
            self.errors = errors


class _RawMetricData(object):
    """A Raw Metric Data Object"""

    MetricName = None
    Value = None
    Timestamp = None
    Step = None

    def __init__(self, metric_name, value, timestamp=None, step=None):
        """Construct a `_RawMetricData` instance.

        Args:
            metric_name (str): The name of the metric.
            value (float): The value of the metric.
            timestamp (datetime.datetime or float or str): Timestamp of the metric.
                If not specified, the current UTC time will be used.
            step (int):  Iteration number of the metric (default: None).
        """
        if timestamp is None:
            timestamp = time.time()
        elif isinstance(timestamp, datetime.datetime):
            # If the input is a datetime then convert it to UTC time.
            # Assume a naive datetime is in local timezone
            if not timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=dateutil.tz.tzlocal())
            timestamp = (timestamp - timestamp.utcoffset()).replace(tzinfo=datetime.timezone.utc)
            timestamp = timestamp.timestamp()
        else:
            timestamp = float(timestamp)

        if timestamp < (time.time() - METRIC_TS_LOWER_BOUND_TO_NOW) or timestamp > (
            time.time() + METRIC_TS_UPPER_BOUND_FROM_NOW
        ):
            raise ValueError(
                "Supplied timestamp %f is invalid."
                " Timestamps must be between two weeks before and two hours from now." % timestamp
            )
        value = float(value)

        self.MetricName = metric_name
        self.Value = float(value)
        # Update timestamp to milliseconds
        # to be compatible with the metrics service
        self.Timestamp = timestamp * 1000
        if step is not None:
            if not isinstance(step, int):
                raise ValueError("step must be int.")
            self.Step = step

    def to_record(self):
        """Convert the `_RawMetricData` object to dict"""
        return self.__dict__

    def __str__(self):
        """String representation of the `_RawMetricData` object."""
        return repr(self)

    def __repr__(self):
        """Return a string representation of this _RawMetricData` object."""
        return "{}({})".format(
            type(self).__name__,
            ",".join(["{}={}".format(k, repr(v)) for k, v in vars(self).items()]),
        )


class _MetricsManager(object):
    """Collects metrics and sends them directly to metrics service.

    Note this is a draft implementation for beta and will change significantly prior to launch.
    """

    _BATCH_SIZE = 10

    def __init__(self, resource_arn, sagemaker_session=None) -> None:
        """Initiate a `_MetricsManager` instance

        Args:
            resource_arn (str): The ARN of a Trial Component to log metrics.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
        """
        self._resource_arn = resource_arn
        self.sagemaker_session = sagemaker_session or _utils.default_session()
        self._buffer = []
        # this client instantiation will need to go into Session
        config = Config(retries={"max_attempts": 10, "mode": "adaptive"})
        stage = "prod"
        region = self.sagemaker_session.boto_session.region_name
        endpoint = f"https://training-metrics.{stage}.{region}.ml-platform.aws.a2z.com"
        self.metrics_service_client = self.sagemaker_session.boto_session.client(
            "sagemaker-metrics", config=config, endpoint_url=endpoint
        )

    def log_metric(self, metric_name, value, timestamp=None, step=None):
        """Sends a metric to metrics service.

        Args:
            metric_name (str): The name of the metric.
            value (float): The value of the metric.
            timestamp (datetime.datetime): Timestamp of the metric.
                If not specified, the current UTC time will be used.
            step (int):  Iteration number of the metric (default: None).
        """
        metric_data = _RawMetricData(
            metric_name=metric_name, value=value, timestamp=timestamp, step=step
        )
        # this is a simplistic solution which calls BatchPutMetrics
        # on the same thread as the client code
        self._buffer.append(metric_data)
        self._drain()

    def _drain(self, close=False):
        """Pops off all metrics in the buffer and starts sending them to metrics service.

        Args:
            close (bool): Indicates if this method is invoked within the `close` function
                (default: False). If invoked in the `close` function, the remaining logged
                metrics in the buffer will be all sent out to the Metrics Service.
                Otherwise, the metrics will only be sent out if the number of them reaches the
                batch size.
        """
        # no metrics to send
        if not self._buffer:
            return

        if len(self._buffer) < self._BATCH_SIZE and not close:
            return

        # pop all the available metrics
        available_metrics, self._buffer = self._buffer, []

        self._send_metrics(available_metrics)

    def _send_metrics(self, metrics):
        """Calls BatchPutMetrics directly on the metrics service.

        Args:
            metrics (list[_RawMetricData]): A list of `_RawMetricData` objects.
        """
        while metrics:
            batch, metrics = metrics[: self._BATCH_SIZE], metrics[self._BATCH_SIZE :]
            request = self._construct_batch_put_metrics_request(batch)
            self.metrics_service_client.batch_put_metrics(**request)

    def _construct_batch_put_metrics_request(self, batch):
        """Creates dictionary object used as request to metrics service.

        Args:
            batch (list[_RawMetricData]): A list of `_RawMetricData` objects,
                whose length is within the batch size limitation.
        """
        return {
            "ResourceArn": self._resource_arn,
            "MetricData": list(map(self._to_raw_metric_data, batch)),
        }

    @staticmethod
    def _to_raw_metric_data(metric_data):
        """Transform a RawMetricData item to a list item for BatchPutMetrics request.

        Args:
            metric_data (_RawMetricData): The `_RawMetricData` object to be transformed.
        """
        item = {
            "MetricName": metric_data.MetricName,
            "Timestamp": int(metric_data.Timestamp),
            "Value": metric_data.Value,
        }

        if metric_data.Step is not None:
            item["IterationNumber"] = metric_data.Step

        return item

    def close(self):
        """Drain the metrics buffer and send metrics to Metrics Service."""
        self._drain(close=True)

    def __enter__(self):
        """Return self"""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Execute self.close() to send out metrics in the buffer.

        Args:
            exc_type (str): The exception type.
            exc_value (str): The exception value.
            exc_traceback (str): The stack trace of the exception.
        """
        self.close()
