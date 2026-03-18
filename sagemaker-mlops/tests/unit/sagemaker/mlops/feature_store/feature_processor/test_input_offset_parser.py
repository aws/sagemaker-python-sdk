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

from sagemaker.mlops.feature_store.feature_processor._input_offset_parser import (
    InputOffsetParser,
)
from sagemaker.mlops.feature_store.feature_processor._constants import (
    EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT,
)
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest


@pytest.fixture
def input_offset_parser():
    time_spec = dict(year=2023, month=5, day=10, hour=17, minute=30, second=20)
    return InputOffsetParser(now=datetime(**time_spec))


@pytest.mark.parametrize(
    "param",
    [
        (None, None),
        ("1 hour", "2023-05-10T16:30:20Z"),
        ("1 day", "2023-05-09T17:30:20Z"),
        ("1 month", "2023-04-10T17:30:20Z"),
        ("1 year", "2022-05-10T17:30:20Z"),
    ],
)
def test_get_iso_format_offset_date(param, input_offset_parser):
    input_offset, expected_offset_date = param
    output_offset_date = input_offset_parser.get_iso_format_offset_date(input_offset)

    assert output_offset_date == expected_offset_date


@pytest.mark.parametrize(
    "param",
    [
        (None, None),
        (
            "1 hour",
            datetime.strptime("2023-05-10T16:30:20Z", EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT),
        ),
        (
            "1 day",
            datetime.strptime("2023-05-09T17:30:20Z", EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT),
        ),
        (
            "1 month",
            datetime.strptime("2023-04-10T17:30:20Z", EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT),
        ),
        (
            "1 year",
            datetime.strptime("2022-05-10T17:30:20Z", EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT),
        ),
    ],
)
def test_get_offset_datetime(param, input_offset_parser):
    input_offset, expected_offset_datetime = param
    output_offet_datetime = input_offset_parser.get_offset_datetime(input_offset)

    assert output_offet_datetime == expected_offset_datetime


@pytest.mark.parametrize(
    "param",
    [
        (None, (None, None, None, None)),
        ("1 hour", ("2023", "05", "10", "16")),
        ("1 day", ("2023", "05", "09", "17")),
        ("1 month", ("2023", "04", "10", "17")),
        ("1 year", ("2022", "05", "10", "17")),
    ],
)
def test_get_offset_date_year_month_day_hour(param, input_offset_parser):
    input_offset, expected_date_tuple = param
    output_date_tuple = input_offset_parser.get_offset_date_year_month_day_hour(input_offset)

    assert output_date_tuple == expected_date_tuple


@pytest.mark.parametrize(
    "param",
    [
        (None, None),
        ("1 hour", relativedelta(hours=-1)),
        ("20 hours", relativedelta(hours=-20)),
        ("1 day", relativedelta(days=-1)),
        ("20 days", relativedelta(days=-20)),
        ("1 month", relativedelta(months=-1)),
        ("20 months", relativedelta(months=-20)),
        ("1 year", relativedelta(years=-1)),
        ("20 years", relativedelta(years=-20)),
    ],
)
def test_parse_offset_to_timedelta(param, input_offset_parser):
    input_offset, expected_deltatime = param
    output_deltatime = input_offset_parser.parse_offset_to_timedelta(input_offset)

    assert output_deltatime == expected_deltatime


@pytest.mark.parametrize(
    "param",
    [
        (
            "random invalid string",
            "[random invalid string] is not in a valid offset format. Please pass a valid offset e.g '1 day'.",
        ),
        (
            "1 invalid string",
            "[1 invalid string] is not in a valid offset format. Please pass a valid offset e.g '1 day'.",
        ),
        (
            "2 days invalid string",
            "[2 days invalid string] is not in a valid offset format. Please pass a valid offset e.g '1 day'.",
        ),
        (
            "1 second",
            "[second] is not a valid offset unit. Supported units: ['hour', 'day', 'week', 'month', 'year']",
        ),
    ],
)
def test_parse_offset_to_timedelta_negative(param, input_offset_parser):
    input_offset, expected_error_message = param

    with pytest.raises(ValueError) as e:
        input_offset_parser.parse_offset_to_timedelta(input_offset)

    assert str(e.value) == expected_error_message
