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

from sagemaker.model_monitor import CronExpressionGenerator


def test_cron_expression_generator_hourly_returns_expected_value():
    assert CronExpressionGenerator.hourly() == "cron(0 * ? * * *)"


def test_cron_expression_generator_daily_returns_expected_value_when_called_with_no_parameters():
    assert CronExpressionGenerator.daily() == "cron(0 0 ? * * *)"


def test_cron_expression_generator_daily_returns_expected_value_when_called_with_parameters():
    assert CronExpressionGenerator.daily(hour=5) == "cron(0 5 ? * * *)"


def test_cron_expression_generator_daily_every_x_hours_returns_expected_value_when_called_without_customizations():
    assert CronExpressionGenerator.daily_every_x_hours(hour_interval=6) == "cron(0 0/6 ? * * *)"


def test_cron_expression_generator_daily_every_x_hours_returns_expected_value_when_called_with_customizations():
    assert (
        CronExpressionGenerator.daily_every_x_hours(hour_interval=7, starting_hour=8)
        == "cron(0 8/7 ? * * *)"
    )


def test_cron_expression_generator_now():
    assert CronExpressionGenerator.now() == "NOW"
