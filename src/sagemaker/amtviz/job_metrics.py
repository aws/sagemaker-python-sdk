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
"""Helper functions to retrieve job metrics from CloudWatch."""
from __future__ import absolute_import

from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple, Dict, Any
import hashlib
import os
from pathlib import Path

import logging
import pandas as pd
import numpy as np
import boto3

logger = logging.getLogger(__name__)

cw = boto3.client("cloudwatch")
sm = boto3.client("sagemaker")


def disk_cache(outer: Callable) -> Callable:
    """A decorator that implements disk-based caching for CloudWatch metrics data.

    This decorator caches the output of the wrapped function to disk in JSON Lines format.
    It creates a cache key using MD5 hash of the function arguments and stores the data
    in the user's home directory under .amtviz/cw_metrics_cache/.

    Args:
        outer (Callable): The function to be wrapped. Must return a pandas DataFrame
            containing CloudWatch metrics data.

    Returns:
        Callable: A wrapper function that implements the caching logic.
    """

    def inner(*args: Any, **kwargs: Any) -> pd.DataFrame:
        key_input = str(args) + str(kwargs)
        # nosec b303 - Not used for cryptography, but to create lookup key
        key = hashlib.md5(key_input.encode("utf-8")).hexdigest()
        cache_dir = Path.home().joinpath(".amtviz/cw_metrics_cache")
        fn = f"{cache_dir}/req_{key}.jsonl.gz"
        if Path(fn).exists():
            try:
                df = pd.read_json(fn, lines=True)
                logger.debug("H", end="")
                df["ts"] = pd.to_datetime(df["ts"])
                df["ts"] = df["ts"].dt.tz_localize(None)
                # pyright: ignore [reportIndexIssue, reportOptionalSubscript]
                df["rel_ts"] = pd.to_datetime(df["rel_ts"])
                df["rel_ts"] = df["rel_ts"].dt.tz_localize(None)
                return df
            except KeyError:
                # Empty file leads to empty df, hence no df['ts'] possible
                pass
            # nosec b110 - doesn't matter why we could not load it.
            except BaseException as e:
                logger.error("\nException: %s - %s", type(e), e)

        logger.debug("M", end="")
        df = outer(*args, **kwargs)
        assert isinstance(df, pd.DataFrame), "Only caching Pandas DataFrames."

        os.makedirs(cache_dir, exist_ok=True)
        df.to_json(fn, orient="records", date_format="iso", lines=True)

        return df

    return inner


def _metric_data_query_tpl(metric_name: str, dim_name: str, dim_value: str) -> Dict[str, Any]:
    """Returns a CloudWatch metric data query template."""
    return {
        "Id": metric_name.lower().replace(":", "_").replace("-", "_"),
        "MetricStat": {
            "Stat": "Average",
            "Metric": {
                "Namespace": "/aws/sagemaker/TrainingJobs",
                "MetricName": metric_name,
                "Dimensions": [
                    {"Name": dim_name, "Value": dim_value},
                ],
            },
            "Period": 60,
        },
        "ReturnData": True,
    }


def _get_metric_data(
    queries: List[Dict[str, Any]], start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """Fetches CloudWatch metrics between timestamps, returns a DataFrame with selected columns."""
    start_time = start_time - timedelta(hours=1)
    end_time = end_time + timedelta(hours=1)
    response = cw.get_metric_data(MetricDataQueries=queries, StartTime=start_time, EndTime=end_time)

    df = pd.DataFrame()
    if "MetricDataResults" not in response:
        return df

    for metric_data in response["MetricDataResults"]:
        values = metric_data["Values"]
        ts = np.array(metric_data["Timestamps"], dtype=np.datetime64)
        labels = [metric_data["Label"]] * len(values)

        df = pd.concat([df, pd.DataFrame({"value": values, "ts": ts, "label": labels})])

    # We now calculate the relative time based on the first actual observed
    # time stamps, not the potentially start time that we used to scope our CW
    # API call. The difference could be for example startup times or waiting
    # for Spot.
    if not df.empty:
        df["rel_ts"] = datetime.fromtimestamp(1) + (df["ts"] - df["ts"].min())  # pyright: ignore
    return df


@disk_cache
def _collect_metrics(
    dimensions: List[Tuple[str, str]], start_time: datetime, end_time: Optional[datetime]
) -> pd.DataFrame:
    """Collects SageMaker training job metrics from CloudWatch for dimensions and time range."""
    df = pd.DataFrame()
    for dim_name, dim_value in dimensions:
        response = cw.list_metrics(
            Namespace="/aws/sagemaker/TrainingJobs",
            Dimensions=[
                {"Name": dim_name, "Value": dim_value},
            ],
        )
        if not response["Metrics"]:
            continue
        metric_names = [metric["MetricName"] for metric in response["Metrics"]]
        if not metric_names:
            # No metric data yet, or not any longer, because the data were aged out
            continue
        metric_data_queries = [
            _metric_data_query_tpl(metric_name, dim_name, dim_value) for metric_name in metric_names
        ]
        df = pd.concat([df, _get_metric_data(metric_data_queries, start_time, end_time)])

    return df


def get_cw_job_metrics(
    job_name: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
) -> pd.DataFrame:
    """Retrieves CloudWatch metrics for a SageMaker training job.

    Args:
        job_name (str): Name of the SageMaker training job.
        start_time (datetime, optional): Start time for metrics collection.
            Defaults to now - 4 hours.
        end_time (datetime, optional): End time for metrics collection.
            Defaults to start_time + 4 hours.

    Returns:
        pd.DataFrame: Metrics data with columns for value, timestamp, and metric name.
            Results are cached to disk for improved performance.
    """
    dimensions = [
        ("TrainingJobName", job_name),
        ("Host", job_name + "/algo-1"),
    ]
    # If not given, use reasonable defaults for start and end time
    start_time = start_time or datetime.now() - timedelta(hours=4)
    end_time = end_time or start_time + timedelta(hours=4)
    return _collect_metrics(dimensions, start_time, end_time)
