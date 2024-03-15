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
"""Contains classes for EventBridge Schedule management for a SageMaker Pipeline."""
from __future__ import absolute_import

import logging
from datetime import datetime
from typing import Dict, Any

import attr
from botocore.exceptions import ClientError

logger = logging.getLogger("sagemaker")

EXECUTION_TIME_PIPELINE_PARAMETER = "scheduled_time"
EVENT_BRIDGE_INVOCATION_TIME = "<aws.scheduler.scheduled-time>"
NO_FLEXIBLE_TIME_WINDOW = dict(Mode="OFF")
RESOURCE_NOT_FOUND = "ResourceNotFound"
RESOURCE_NOT_FOUND_EXCEPTION = "ResourceNotFoundException"
EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # 2023-01-01T07:00:00Z


@attr.s
class EventBridgeSchedulerHelper:
    """Contains helper methods for scheduling targets on EventBridge"""

    event_bridge_scheduler_client = attr.ib()

    def upsert_schedule(
        self,
        schedule_name: str,
        pipeline_arn: str,
        schedule_expression: str,
        state: str,
        start_date: datetime,
        role: str,
    ) -> Dict:
        """Creates or updates a Schedule for the given pipeline_arn and schedule_expression.

        Args:
            schedule_name: The name of the scheduled target pipeline.
            pipeline_arn: The ARN of the sagemaker pipeline that needs to scheduled.
            schedule_expression: The schedule expression that EventBridge expects.
            state: Specifies whether the schedule is enabled or disabled. Can only
                be ENABLED or DISABLED. Default ENABLED.
            start_date: The date, in UTC, after which the schedule can begin invoking its target.
            role: The RoleArn used to execute the scheduled events. This role must have EventBridge
            permissions.

        Returns:
            Dict: The arn of the schedule that was successfully created/updated.
        """

        create_or_update_schedule_request_dict = dict(
            Name=schedule_name,
            ScheduleExpression=schedule_expression,
            FlexibleTimeWindow=NO_FLEXIBLE_TIME_WINDOW,
            Target=dict(
                Arn=pipeline_arn,
                SageMakerPipelineParameters={},  # required for Target resolution
                RoleArn=role,
            ),
            State=state,
            StartDate=start_date,
        )

        try:
            return self.event_bridge_scheduler_client.update_schedule(
                **create_or_update_schedule_request_dict
            )
        except ClientError as e:
            if RESOURCE_NOT_FOUND_EXCEPTION == e.response["Error"]["Code"]:
                return self.event_bridge_scheduler_client.create_schedule(
                    **create_or_update_schedule_request_dict
                )
            raise e

    def delete_schedule(self, schedule_name: str) -> None:
        """Deletes an EventBridge Schedule of a given pipeline if there is one.

        Args:
            schedule_name: The name of the EventBridge Schedule.
        """
        delete_request_dict = dict(Name=schedule_name)
        self.event_bridge_scheduler_client.delete_schedule(**delete_request_dict)

    def describe_schedule(self, schedule_name) -> Dict[str, Any]:
        """Describe the EventBridge Schedule ARN corresponding to a SageMaker Pipeline.

        Args:
            schedule_name: The name of the EventBridge Schedule.
        Returns:
            Dict[str, str] : Describe EventBridge Schedule response
        """
        describe_request_dict = dict(Name=schedule_name)
        return self.event_bridge_scheduler_client.get_schedule(**describe_request_dict)
