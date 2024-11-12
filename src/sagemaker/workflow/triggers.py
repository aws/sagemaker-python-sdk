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
"""The Trigger entity for EventBridge Integration."""

from __future__ import absolute_import

from datetime import datetime
import logging

from typing import Optional, Sequence
import attr
import pytz

from sagemaker.workflow.parameters import Parameter

logger = logging.getLogger("sagemaker")

ENABLED_TRIGGER_STATE = "ENABLED"
DISABLED_TRIGGER_STATE = "DISABLED"

UTC = pytz.utc


@attr.s
class Trigger:
    """Abstract class representing a Pipeline Trigger

    Attributes:
        name (str): The name of the trigger, default to pipeline_name.
        enabled (boolean): The state of the schedule, default True resolves to 'ENABLED'.
    """

    name: Optional[str] = attr.ib(default=None)
    enabled: Optional[bool] = attr.ib(default=True)

    def resolve_trigger_name(self, pipeline_name: str) -> str:
        """Resolve the schedule name given a parent pipeline.

        Args:
            pipeline_name (str): Parent pipeline name

        Returns:
            str: Resolved schedule name.
        """
        if self.name is None:
            self.name = pipeline_name  # default to pipeline name
        return self.name

    def resolve_trigger_state(self) -> str:
        """Helper method for Enabled/Disabled Resolution on Trigger States

        Returns:
            (str): ENABLED/DISABLED string literal
        """
        return ENABLED_TRIGGER_STATE if self.enabled else DISABLED_TRIGGER_STATE


@attr.s
class PipelineSchedule(Trigger):
    """Pipeline Schedule trigger type used to create EventBridge Schedules for SageMaker Pipelines.

    To create a pipeline schedule, specify a single type using the ``at``, ``rate``, or ``cron``
    parameters. For more information about EventBridge syntax, see
    `Schedule types on EventBridge Scheduler
    <https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html>`_.

    Args:
        start_date (datetime): The start date of the schedule. Default is ``time.now()``.
        at (datetime): An "At" EventBridge expression. Defaults to UTC timezone. Note
            that if you use ``datetime.now()``, the result is a snapshot of your current local
            time. Eventbridge requires a time in UTC format. You can convert the
            result of ``datetime.now()`` to UTC by using ``datetime.utcnow()`` or
            ``datetime.now(tz=pytz.utc)``. For example, you can create a time two minutes from now
            with the expression ``datetime.now(tz=pytz.utc) + timedelta(0, 120)``.
        rate (tuple): A "Rate" EventBridge expression. Format is (value, unit).
        cron (str): A "Cron" EventBridge expression. Format is "minutes hours
            day-of-month month day-of-week year".
        name (str): The schedule name. Default is ``None``.
        enabled (boolean): If the schedule is enabled. Defaults to ``True``.
    """

    start_date: Optional[datetime] = attr.ib(default=None)

    at: Optional[datetime] = attr.ib(default=None)
    rate: Optional[tuple] = attr.ib(default=None)
    cron: Optional[str] = attr.ib(default=None)

    def resolve_schedule_expression(self) -> str:
        """Resolve schedule expression

        Format schedule expression for an EventBridge client call from the specified
            at, rate, or cron parameter. After resolution, if there are any othererrors
            in the syntax, this will throw an expected ValidationException from EventBridge.

        Returns:
            schedule_expression: Correctly string formatted schedule expression based on type.
        """

        if len([x for x in [self.at, self.rate, self.cron] if x is not None]) > 1:
            raise TypeError(
                "Too many types specified for PipelineSchedule. Please specify a single type "
                "in [at, rate, or cron] to successfully create the EventBridge Schedule."
            )

        # "at(yyyy-mm-ddThh:mm:ss)"
        if self.at:
            if isinstance(self.at, datetime):
                utc_dt = self.at  # no tz, assume UTC
                if self.at.tzinfo:
                    utc_dt = self.at.astimezone(tz=UTC)  # convert to UTC
                return f"at({utc_dt.strftime('%Y-%m-%dT%H:%M:%S')})"
            raise TypeError("Incorrect type specified for at= schedule.")

        # "rate(value unit)"
        if self.rate:
            if isinstance(self.rate, tuple):
                val, unit = self.rate
                resolved_str = f"rate({val} {unit})"
                return resolved_str
            raise TypeError("Incorrect type specified for rate= schedule.")

        # "cron(minutes hours day-of-month month day-of-week year)"
        if self.cron:
            if isinstance(self.cron, str):
                return f"cron({self.cron})"
            raise TypeError("Incorrect type specified for cron= schedule.")

        raise ValueError(
            "No schedule type specified. Please specify a single type "
            "in [at, rate, or cron] to successfully create the EventBridge Schedule."
        )


def validate_default_parameters_for_schedules(parameters: Optional[Sequence[Parameter]]):
    """Validate that pipeline parameters have defaults if it will interact with EventBridge.

    Currently, we are not allowing scheduled executions to override pipeline parameters.
    This means that we must fail fast in the case that no default is specified.

    Args:
        parameters (Optional[Sequence[Parameter]]): A list of Pipeline Parameters
    """
    if parameters:
        no_defaults = [
            param.name
            for param in parameters
            if isinstance(param, Parameter) and param.default_value is None
        ]
        if no_defaults:
            raise ValueError(
                f"When using pipeline triggers, please specify default values for all Pipeline "
                f"Parameters. Currently, they are not overridable at runtime as inputs to a "
                f"scheduled pipeline execution. The current parameters don't have defaults "
                f"{no_defaults}."
            )
