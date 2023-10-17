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

from mock import Mock
from sagemaker.session import Session
from sagemaker.feature_store.feature_processor._event_bridge_rule_helper import (
    EventBridgeRuleHelper,
)
from botocore.exceptions import ClientError
from sagemaker.feature_store.feature_processor._feature_processor_pipeline_events import (
    FeatureProcessorPipelineEvents,
    FeatureProcessorPipelineExecutionStatus,
)
import pytest


@pytest.fixture
def sagemaker_session():
    boto_session = Mock()
    boto_session.client("events").return_value = Mock()
    return Mock(Session, boto_session=boto_session, sagemaker_client=Mock())


@pytest.fixture
def event_bridge_rule_helper(sagemaker_session):
    return EventBridgeRuleHelper(sagemaker_session, sagemaker_session.boto_session.client("events"))


def test_put_rule_without_event_pattern(event_bridge_rule_helper):
    source_pipeline_events = [
        FeatureProcessorPipelineEvents(
            pipeline_name="test_pipeline",
            pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.SUCCEEDED],
        )
    ]

    event_bridge_rule_helper._generate_pipeline_arn_and_name = Mock(
        return_value=dict(pipeline_arn="pipeline_arn", pipeline_name="pipeline_name")
    )
    event_bridge_rule_helper.event_bridge_rule_client.put_rule = Mock(
        return_value=dict(RuleArn="rule_arn")
    )
    event_bridge_rule_helper.put_rule(
        source_pipeline_events=source_pipeline_events,
        target_pipeline="target_pipeline",
        event_pattern=None,
        state="Disabled",
    )

    event_bridge_rule_helper.event_bridge_rule_client.put_rule.assert_called_with(
        Name="target_pipeline",
        EventPattern=(
            '{"detail-type": ["SageMaker Model Building Pipeline Execution Status Change"], '
            '"source": ["aws.sagemaker"], "detail": {"currentPipelineExecutionStatus": '
            '["Succeeded"], "pipelineArn": ["pipeline_arn"]}}'
        ),
        State="Disabled",
    )


def test_put_rule_with_event_pattern(event_bridge_rule_helper):
    source_pipeline_events = [
        FeatureProcessorPipelineEvents(
            pipeline_name="test_pipeline",
            pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.SUCCEEDED],
        )
    ]

    event_bridge_rule_helper._generate_pipeline_arn_and_name = Mock(
        return_value=dict(pipeline_arn="pipeline_arn", pipeline_name="pipeline_name")
    )
    event_bridge_rule_helper.event_bridge_rule_client.put_rule = Mock(
        return_value=dict(RuleArn="rule_arn")
    )
    event_bridge_rule_helper.put_rule(
        source_pipeline_events=source_pipeline_events,
        target_pipeline="target_pipeline",
        event_pattern="event_pattern",
        state="Disabled",
    )

    event_bridge_rule_helper.event_bridge_rule_client.put_rule.assert_called_with(
        Name="target_pipeline",
        EventPattern="event_pattern",
        State="Disabled",
    )


def test_put_targets_success(event_bridge_rule_helper):
    event_bridge_rule_helper._generate_pipeline_arn_and_name = Mock(
        return_value=dict(pipeline_arn="pipeline_arn", pipeline_name="pipeline_name")
    )
    event_bridge_rule_helper.event_bridge_rule_client.put_targets = Mock(
        return_value=dict(FailedEntryCount=0)
    )
    event_bridge_rule_helper.put_target(
        rule_name="rule_name",
        target_pipeline="target_pipeline",
        target_pipeline_parameters={"param": "value"},
        role_arn="role_arn",
    )

    event_bridge_rule_helper.event_bridge_rule_client.put_targets.assert_called_with(
        Rule="rule_name",
        Targets=[
            {
                "Id": "pipeline_name",
                "Arn": "pipeline_arn",
                "RoleArn": "role_arn",
                "SageMakerPipelineParameters": {"PipelineParameterList": {"param": "value"}},
            }
        ],
    )


def test_put_targets_failure(event_bridge_rule_helper):
    event_bridge_rule_helper._generate_pipeline_arn_and_name = Mock(
        return_value=dict(pipeline_arn="pipeline_arn", pipeline_name="pipeline_name")
    )
    event_bridge_rule_helper.event_bridge_rule_client.put_targets = Mock(
        return_value=dict(
            FailedEntryCount=1,
            FailedEntries=[dict(ErrorMessage="test_error_message")],
        )
    )
    with pytest.raises(
        Exception, match="Failed to add target pipeline to rule. Failure reason: test_error_message"
    ):
        event_bridge_rule_helper.put_target(
            rule_name="rule_name",
            target_pipeline="target_pipeline",
            target_pipeline_parameters={"param": "value"},
            role_arn="role_arn",
        )


def test_delete_rule(event_bridge_rule_helper):
    event_bridge_rule_helper.event_bridge_rule_client.delete_rule = Mock()
    event_bridge_rule_helper.delete_rule("rule_name")

    event_bridge_rule_helper.event_bridge_rule_client.delete_rule.assert_called_with(
        Name="rule_name"
    )


def test_describe_rule_success(event_bridge_rule_helper):
    mock_describe_response = dict(State="ENABLED", RuleName="rule_name")
    event_bridge_rule_helper.event_bridge_rule_client.describe_rule = Mock(
        return_value=mock_describe_response
    )
    assert event_bridge_rule_helper.describe_rule("rule_name") == mock_describe_response


def test_describe_rule_non_existent(event_bridge_rule_helper):
    mock_describe_response = dict(State="ENABLED", RuleName="rule_name")
    event_bridge_rule_helper.event_bridge_rule_client.describe_rule = Mock(
        return_value=mock_describe_response,
        side_effect=ClientError(
            error_response={"Error": {"Code": "ResourceNotFoundException"}},
            operation_name="describe_rule",
        ),
    )
    assert event_bridge_rule_helper.describe_rule("rule_name") is None


def test_remove_targets(event_bridge_rule_helper):
    event_bridge_rule_helper.event_bridge_rule_client.remove_targets = Mock()
    event_bridge_rule_helper.remove_targets(rule_name="rule_name", ids=["target_pipeline"])
    event_bridge_rule_helper.event_bridge_rule_client.remove_targets.assert_called_with(
        Rule="rule_name",
        Ids=["target_pipeline"],
    )


def test_enable_rule(event_bridge_rule_helper):
    event_bridge_rule_helper.event_bridge_rule_client.enable_rule = Mock()
    event_bridge_rule_helper.enable_rule("rule_name")

    event_bridge_rule_helper.event_bridge_rule_client.enable_rule.assert_called_with(
        Name="rule_name"
    )


def test_disable_rule(event_bridge_rule_helper):
    event_bridge_rule_helper.event_bridge_rule_client.disable_rule = Mock()
    event_bridge_rule_helper.disable_rule("rule_name")

    event_bridge_rule_helper.event_bridge_rule_client.disable_rule.assert_called_with(
        Name="rule_name"
    )


def test_add_tags(event_bridge_rule_helper):
    event_bridge_rule_helper.event_bridge_rule_client.tag_resource = Mock()
    event_bridge_rule_helper.add_tags("rule_arn", [{"key": "value"}])

    event_bridge_rule_helper.event_bridge_rule_client.tag_resource.assert_called_with(
        ResourceARN="rule_arn", Tags=[{"key": "value"}]
    )


def test_generate_event_pattern_from_feature_processor_pipeline_events(event_bridge_rule_helper):
    event_bridge_rule_helper._generate_pipeline_arn_and_name = Mock(
        return_value=dict(pipeline_arn="pipeline_arn", pipeline_name="pipeline_name")
    )
    event_pattern = (
        event_bridge_rule_helper._generate_event_pattern_from_feature_processor_pipeline_events(
            [
                FeatureProcessorPipelineEvents(
                    pipeline_name="test_pipeline_1",
                    pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.SUCCEEDED],
                ),
                FeatureProcessorPipelineEvents(
                    pipeline_name="test_pipeline_2",
                    pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.FAILED],
                ),
            ]
        )
    )

    assert (
        event_pattern
        == '{"detail-type": ["SageMaker Model Building Pipeline Execution Status Change"], '
        '"$or": [{"source": ["aws.sagemaker"], "detail": {"currentPipelineExecutionStatus": '
        '["Failed"], "pipelineArn": ["pipeline_arn"]}}, {"source": ["aws.sagemaker"], "detail": '
        '{"currentPipelineExecutionStatus": ["Failed"], "pipelineArn": ["pipeline_arn"]}}]}'
    )


def test_validate_feature_processor_pipeline_events(event_bridge_rule_helper):
    events = [
        FeatureProcessorPipelineEvents(
            pipeline_name="test_pipeline_1",
            pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.SUCCEEDED],
        ),
        FeatureProcessorPipelineEvents(
            pipeline_name="test_pipeline_1",
            pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.FAILED],
        ),
    ]

    with pytest.raises(ValueError, match="Pipeline names in pipeline_events must be unique."):
        event_bridge_rule_helper._validate_feature_processor_pipeline_events(events)


def test_aggregate_pipeline_events_with_same_desired_status(event_bridge_rule_helper):
    events = [
        FeatureProcessorPipelineEvents(
            pipeline_name="test_pipeline_1",
            pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.FAILED],
        ),
        FeatureProcessorPipelineEvents(
            pipeline_name="test_pipeline_2",
            pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.FAILED],
        ),
    ]

    assert event_bridge_rule_helper._aggregate_pipeline_events_with_same_desired_status(events) == {
        (FeatureProcessorPipelineExecutionStatus.FAILED,): [
            "test_pipeline_1",
            "test_pipeline_2",
        ]
    }


@pytest.mark.parametrize(
    "pipeline_uri,expected_result",
    [
        (
            "arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline",
            dict(
                pipeline_arn="arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline",
                pipeline_name="test-pipeline",
            ),
        ),
        (
            "test-pipeline",
            dict(
                pipeline_arn="test-pipeline-arn",
                pipeline_name="test-pipeline",
            ),
        ),
    ],
)
def test_generate_pipeline_arn_and_name(event_bridge_rule_helper, pipeline_uri, expected_result):
    event_bridge_rule_helper.sagemaker_session.sagemaker_client.describe_pipeline = Mock(
        return_value=dict(PipelineArn="test-pipeline-arn")
    )
    assert event_bridge_rule_helper._generate_pipeline_arn_and_name(pipeline_uri) == expected_result
