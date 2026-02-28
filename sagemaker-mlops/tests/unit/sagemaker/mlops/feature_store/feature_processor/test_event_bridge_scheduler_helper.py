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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from datetime import datetime
import pytest
from botocore.exceptions import ClientError

from sagemaker.mlops.feature_store.feature_processor._event_bridge_scheduler_helper import (
    EventBridgeSchedulerHelper,
)
from mock import Mock

from sagemaker.core.helper.session_helper import Session

SCHEDULE_NAME = "test_schedule"
SCHEDULE_ARN = "test_schedule_arn"
NEW_SCHEDULE_ARN = "test_new_schedule_arn"
TARGET_ARN = "test_arn"
CRON_SCHEDULE = "test_cron"
STATE = "ENABLED"
ROLE = "test_role"
START_DATE = datetime.now()


@pytest.fixture
def sagemaker_session():
    boto_session = Mock()
    boto_session.client("scheduler").return_value = Mock()
    return Mock(Session, boto_session=boto_session)


@pytest.fixture
def event_bridge_scheduler_helper(sagemaker_session):
    return EventBridgeSchedulerHelper(
        sagemaker_session, sagemaker_session.boto_session.client("scheduler")
    )


def test_upsert_schedule_already_exists(event_bridge_scheduler_helper):
    event_bridge_scheduler_helper.event_bridge_scheduler_client.update_schedule.return_value = (
        SCHEDULE_ARN
    )
    schedule_arn = event_bridge_scheduler_helper.upsert_schedule(
        schedule_name=SCHEDULE_NAME,
        pipeline_arn=TARGET_ARN,
        schedule_expression=CRON_SCHEDULE,
        state=STATE,
        start_date=START_DATE,
        role=ROLE,
    )
    assert schedule_arn == SCHEDULE_ARN
    event_bridge_scheduler_helper.event_bridge_scheduler_client.create_schedule.assert_not_called()


def test_upsert_schedule_not_exists(event_bridge_scheduler_helper):
    event_bridge_scheduler_helper.event_bridge_scheduler_client.update_schedule.side_effect = (
        ClientError(
            error_response={"Error": {"Code": "ResourceNotFoundException"}},
            operation_name="update_schedule",
        )
    )
    event_bridge_scheduler_helper.event_bridge_scheduler_client.create_schedule.return_value = (
        NEW_SCHEDULE_ARN
    )

    schedule_arn = event_bridge_scheduler_helper.upsert_schedule(
        schedule_name=SCHEDULE_NAME,
        pipeline_arn=TARGET_ARN,
        schedule_expression=CRON_SCHEDULE,
        state=STATE,
        start_date=START_DATE,
        role=ROLE,
    )
    assert schedule_arn == NEW_SCHEDULE_ARN
    event_bridge_scheduler_helper.event_bridge_scheduler_client.create_schedule.assert_called_once()


def test_delete_schedule(event_bridge_scheduler_helper):
    event_bridge_scheduler_helper.sagemaker_session.boto_session = Mock()
    event_bridge_scheduler_helper.sagemaker_session.sagemaker_client = Mock()
    event_bridge_scheduler_helper.delete_schedule(schedule_name=TARGET_ARN)
    event_bridge_scheduler_helper.event_bridge_scheduler_client.delete_schedule.assert_called_with(
        Name=TARGET_ARN
    )
