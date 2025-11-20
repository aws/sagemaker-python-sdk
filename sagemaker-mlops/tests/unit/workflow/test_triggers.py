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
"""Unit tests for workflow triggers."""
from __future__ import absolute_import

import pytest
from datetime import datetime

from sagemaker.mlops.workflow.triggers import PipelineSchedule


def test_pipeline_schedule_init():
    schedule = PipelineSchedule(
        name="test-schedule",
        at="rate(1 hour)"
    )
    assert schedule.name == "test-schedule"
    assert schedule.at == "rate(1 hour)"


def test_pipeline_schedule_with_cron():
    schedule = PipelineSchedule(
        name="test-schedule",
        at="cron(0 12 * * ? *)"
    )
    assert "cron" in schedule.at


def test_pipeline_schedule_resolve_trigger_name():
    schedule = PipelineSchedule(name="test-schedule", at="rate(1 hour)")
    trigger_name = schedule.resolve_trigger_name("my-pipeline")
    assert "test-schedule" in trigger_name or "my-pipeline" in trigger_name
