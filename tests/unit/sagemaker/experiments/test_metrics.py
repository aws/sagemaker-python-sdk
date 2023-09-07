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
from __future__ import absolute_import

import os
import pytest
import tempfile
import shutil
import datetime
import dateutil
import time

from sagemaker.experiments._metrics import _RawMetricData


@pytest.fixture
def tempdir():
    dir = tempfile.mkdtemp()
    yield dir
    shutil.rmtree(dir)


@pytest.fixture
def filepath(tempdir):
    return os.path.join(tempdir, "foo.json")


@pytest.fixture
def timestamp():
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)


def test_raw_metric_data_utc_timestamp():
    utcnow = datetime.datetime.now(datetime.timezone.utc)
    assert utcnow.tzinfo
    metric = _RawMetricData(metric_name="foo", value=1.0, timestamp=utcnow)
    assert utcnow.timestamp() == metric.Timestamp


def test_raw_metric_data_utc_():
    utcnow = datetime.datetime.now(datetime.timezone.utc)
    assert utcnow.tzinfo
    metric = _RawMetricData(metric_name="foo", value=1.0, timestamp=utcnow)
    assert utcnow.timestamp() == metric.Timestamp


def test_raw_metric_data_aware_timestamp():
    aware_datetime = datetime.datetime.now(dateutil.tz.gettz("America/Chicago"))
    assert aware_datetime.tzinfo
    metric = _RawMetricData(metric_name="foo", value=1.0, timestamp=aware_datetime)
    assert (aware_datetime - aware_datetime.utcoffset()).replace(
        tzinfo=datetime.timezone.utc
    ).timestamp() == metric.Timestamp


def test_raw_metric_data_naive_timestamp():
    naive_datetime = datetime.datetime.now()
    assert naive_datetime.tzinfo is None
    metric = _RawMetricData(metric_name="foo", value=1.0, timestamp=naive_datetime)
    local_datetime = naive_datetime.replace(tzinfo=dateutil.tz.tzlocal())
    assert (local_datetime - local_datetime.utcoffset()).replace(
        tzinfo=datetime.timezone.utc
    ).timestamp() == metric.Timestamp


def test_raw_metric_data_number_timestamp():
    time_now = time.time()
    metric = _RawMetricData(metric_name="foo", value=1.0, timestamp=time_now)
    assert time_now == metric.Timestamp


def test_raw_metric_data_request_item():
    time_now = time.time()
    metric = _RawMetricData(metric_name="foo", value=1.0, timestamp=time_now, step=10)
    expected = {
        "MetricName": "foo",
        "Value": 1.0,
        "Timestamp": str(int(time_now)),
        "Step": 10,
    }
    assert expected == metric.to_raw_metric_data()


def test_raw_metric_data_invalid_timestamp():
    with pytest.raises(ValueError) as error1:
        _RawMetricData(metric_name="IFail", value=100, timestamp=time.time() - 2000000)
    assert "Timestamps must be between two weeks before and two hours from now" in str(error1)

    with pytest.raises(ValueError) as error2:
        _RawMetricData(metric_name="IFail", value=100, timestamp=time.time() + 10000)
    assert "Timestamps must be between two weeks before and two hours from now" in str(error2)
