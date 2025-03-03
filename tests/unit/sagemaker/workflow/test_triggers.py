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

from datetime import datetime

import pytest
import pytz
from mock.mock import ANY
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterBoolean

from sagemaker import Session
from sagemaker.workflow import ParameterString
from sagemaker.workflow._event_bridge_client_helper import EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT
from sagemaker.workflow.pipeline import Pipeline
from mock import Mock, patch

from sagemaker.workflow.triggers import PipelineSchedule, validate_default_parameters_for_schedules
from tests.unit.sagemaker.workflow.helpers import CustomStep

EXPECTED_AT_EXPRESSION = "at(2023-10-10T17:44:55)"  # UTC
EXPECTED_AT_EXPRESSION_DST = "at(2023-12-10T18:44:55)"
EXPECTED_RATE_EXPRESSION = "rate(5 minutes)"
EXPECTED_CRON_EXPRESSION = "cron(15 10 ? * 6L 2022-2023)"

EXECUTION_ROLE_ARN = "arn:aws:iam::123456789012:role/my_dummy_execution_role"

SCHEDULE_NAME = "my_schedule"
SCHEDULE_ARN = "arn:schedule/my_schedule"
DEFAULT_STATE = "ENABLED"

INPUT_AT_DATETIME_PST = datetime(2023, 10, 10, 10, 44, 55)  # +7UTC
INPUT_AT_DATETIME_PDT = datetime(2023, 12, 10, 10, 44, 55)  # +8UTC
INPUT_AT_DATETIME_UTC = datetime(2023, 10, 10, 17, 44, 55)
START_DATE = datetime.now()

PIPELINE_ARN = "arn:pipeline/TestSchedulerPipeline"
PIPELINE_NAME = "TestSchedulerPipeline"

PACIFIC = pytz.timezone("US/Pacific")


@pytest.fixture
def role_arn():
    return "arn:role"


@pytest.fixture
def sagemaker_session_mock():
    session_mock = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session_mock.local_mode = False
    session_mock._append_sagemaker_config_tags = Mock(
        name="_append_sagemaker_config_tags", side_effect=lambda tags, config_path_to_tags: tags
    )
    session_mock.sagemaker_client.describe_pipeline = Mock(
        return_value={"PipelineArn": f"{PIPELINE_ARN}", "CreationTime": datetime.now()}
    )
    session_mock.sagemaker_config = {
        "SchemaVersion": "1.0",
        "SageMaker": {"Pipeline": {"RoleArn": EXECUTION_ROLE_ARN}},
    }
    return session_mock


def mock_event_bridge_scheduler_helper():
    eb_helper = Mock()
    eb_helper.upsert_schedule.return_value = dict(ScheduleArn=SCHEDULE_ARN)
    eb_helper.delete_schedule.return_value = None
    eb_helper.describe_schedule.return_value = {
        "Arn": SCHEDULE_ARN,
        "ScheduleExpression": EXPECTED_AT_EXPRESSION,
        "State": DEFAULT_STATE,
        "StartDate": START_DATE,
        "Target": {
            "RoleArn": EXECUTION_ROLE_ARN,
        },
    }

    return eb_helper


@pytest.mark.parametrize(
    "inputs",
    [
        (PipelineSchedule(at=INPUT_AT_DATETIME_UTC), EXPECTED_AT_EXPRESSION),
        (PipelineSchedule(rate=(5, "minutes")), EXPECTED_RATE_EXPRESSION),
        (PipelineSchedule(cron="15 10 ? * 6L 2022-2023"), EXPECTED_CRON_EXPRESSION),
    ],
)
def test_resolve_schedule_expressions_common_cases(inputs):
    schedule, expected_expression = inputs
    assert schedule.resolve_schedule_expression() == expected_expression


@pytest.mark.parametrize(
    "input_dt, expected_dt",
    [
        (PACIFIC.localize(INPUT_AT_DATETIME_PST), EXPECTED_AT_EXPRESSION),  # PST -7UTC)
        (PACIFIC.localize(INPUT_AT_DATETIME_PDT), EXPECTED_AT_EXPRESSION_DST),  # PDT -8UTC
        (INPUT_AT_DATETIME_UTC, EXPECTED_AT_EXPRESSION),  # UTC
    ],
)
def test_resolve_at_schedule_expression_timezone(input_dt, expected_dt):
    schedule = PipelineSchedule(at=input_dt)
    assert schedule.resolve_schedule_expression() == expected_dt


@pytest.mark.parametrize(
    "schedule, expected",
    [
        (
            PipelineSchedule(at=datetime(2023, 12, 12, 10, 00, 00), rate=(5, "minutes")),
            (TypeError, "Too many types specified for PipelineSchedule"),
        ),
        (
            PipelineSchedule(name="schedule-no-type-specified"),
            (ValueError, "No schedule type specified"),
        ),
        (
            PipelineSchedule(at="at(2023-11-22T03:44:55)"),
            (TypeError, "Incorrect type specified for at= schedule."),
        ),
        (
            PipelineSchedule(rate="rate(5, minutes)"),
            (TypeError, "Incorrect type specified for rate= schedule."),
        ),
        (
            PipelineSchedule(cron=[1, 2, 3]),
            (TypeError, "Incorrect type specified for cron= schedule."),
        ),
    ],
)
def test_resolve_schedule_expression_exception_types(schedule, expected):
    err, msg = expected
    if err is TypeError:
        with pytest.raises(TypeError) as err:
            schedule.resolve_schedule_expression()
        assert str(err.value).startswith(msg)
    if err is ValueError:
        with pytest.raises(ValueError) as err:
            schedule.resolve_schedule_expression()
        assert str(err.value).startswith(msg)


@patch(
    "sagemaker.workflow.pipeline.EventBridgeSchedulerHelper",
    return_value=mock_event_bridge_scheduler_helper(),
)
def test_schedule_trigger_functionality(eb_helper, sagemaker_session_mock, role_arn):
    step1 = CustomStep(name="MyStep1")
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[],
        steps=[step1],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.create(role_arn=role_arn)
    sagemaker_session_mock.sagemaker_client.create_pipeline.assert_called_with(
        PipelineName=PIPELINE_NAME, PipelineDefinition=pipeline.definition(), RoleArn=role_arn
    )
    # put
    schedule = PipelineSchedule(name=SCHEDULE_NAME, at=INPUT_AT_DATETIME_UTC)
    pipeline.put_triggers(triggers=[schedule])
    eb_helper().upsert_schedule.assert_called_once_with(
        schedule_name=SCHEDULE_NAME,
        pipeline_arn=PIPELINE_ARN,
        schedule_expression=EXPECTED_AT_EXPRESSION,
        state=DEFAULT_STATE,
        start_date=ANY,  # since start_time defaults to datetime.now, will not match exactly at runtime
        role=EXECUTION_ROLE_ARN,
    )
    # describe
    trigger = pipeline.describe_trigger(trigger_name=schedule.name)
    eb_helper().describe_schedule.assert_called_once_with(schedule_name=schedule.name)
    assert trigger["Schedule_Arn"] == SCHEDULE_ARN
    assert trigger["Schedule_State"] == DEFAULT_STATE
    assert trigger["Schedule_Start_Date"] == START_DATE.strftime(
        EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT
    )
    assert trigger["Schedule_Expression"] == EXPECTED_AT_EXPRESSION
    assert trigger["Schedule_Role"] == EXECUTION_ROLE_ARN
    # delete
    pipeline.delete_triggers(trigger_names=[schedule.name])
    eb_helper().delete_schedule.assert_called_once_with(schedule_name=schedule.name)
    # delete pipeline
    pipeline.delete()
    assert sagemaker_session_mock.sagemaker_client.delete_pipeline.assert_called_once_with(
        PipelineName=PIPELINE_NAME,
    )


@patch(
    "sagemaker.workflow.pipeline.EventBridgeSchedulerHelper",
    return_value=mock_event_bridge_scheduler_helper(),
)
def test_default_schedule_name_assignment(eb_helper, sagemaker_session_mock, role_arn):
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.create(role_arn=role_arn)
    # No name
    schedule = PipelineSchedule(at=INPUT_AT_DATETIME_UTC)
    assert not schedule.name
    # Default to pipeline name for EventBridge create_schedule call
    pipeline.put_triggers(triggers=[schedule])
    assert schedule.name == PIPELINE_NAME
    schedule = PipelineSchedule(name="new-name", at=INPUT_AT_DATETIME_UTC)
    assert schedule.name == "new-name"


@pytest.mark.parametrize(
    "inputs",
    [
        (PipelineSchedule(rate=(5, "minutes")), "ENABLED"),
        (PipelineSchedule(rate=(5, "minutes"), enabled=False), "DISABLED"),
    ],
)
def test_resolve_trigger_state(inputs):
    schedule, expected_state = inputs
    assert schedule.resolve_trigger_state() == expected_state


@pytest.mark.parametrize(
    "params, no_defaults",
    [
        ([], None),
        ("My-String-Parameter", None),
        ([ParameterString(name="MyString", default_value="MyValue")], None),
        ([ParameterString(name="MyString")], "['MyString']"),
        (
            [
                ParameterString(name="MyString"),
                ParameterFloat(name="MyFloat"),
                ParameterBoolean(name="MyBool"),
                ParameterInteger(name="MyInt", default_value=1),
            ],
            "['MyString', 'MyFloat', 'MyBool']",
        ),
    ],
)
def test_validate_default_parameters_for_schedules(params, no_defaults):
    if no_defaults is not None:
        with pytest.raises(ValueError) as err:
            validate_default_parameters_for_schedules(params)
            assert no_defaults in str(err)
    else:
        validate_default_parameters_for_schedules(params)
