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

import datetime
import unittest.mock

import pytest
from sagemaker.lineage import context, _api_types
from sagemaker.lineage.action import Action
from sagemaker.lineage.lineage_trial_component import LineageTrialComponent
from sagemaker.lineage.query import LineageQueryDirectionEnum


@pytest.fixture
def sagemaker_session():
    return unittest.mock.Mock()


def test_create(sagemaker_session):
    sagemaker_session.sagemaker_client.create_context.return_value = {
        "ContextArn": "bazz",
    }
    obj = context.Context.create(
        context_name="foo",
        source_uri="bar",
        source_type="test-source-type",
        context_type="test-context-type",
        description="test-description",
        properties={"k1": "v1", "k2": "v2"},
        sagemaker_session=sagemaker_session,
    )
    sagemaker_session.sagemaker_client.create_context.assert_called_with(
        ContextName="foo",
        Source={"SourceUri": "bar", "SourceType": "test-source-type"},
        ContextType="test-context-type",
        Description="test-description",
        Properties={"k1": "v1", "k2": "v2"},
    )
    assert "foo" == obj.context_name
    assert "bar" == obj.source.source_uri
    assert "test-source-type" == obj.source.source_type
    assert "test-context-type" == obj.context_type
    assert "test-description" == obj.description
    assert {"k1": "v1", "k2": "v2"} == obj.properties
    assert "bazz" == obj.context_arn


def test_create_with_tags(sagemaker_session):
    sagemaker_session.sagemaker_client.create_context.return_value = {
        "ContextArn": "bazz",
    }

    tags = [{"Key": "foo", "Value": "bar"}]
    context.Context.create(
        context_name="foo",
        source_uri="bar",
        tags=tags,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.create_context.assert_called_with(
        ContextName="foo", Source={"SourceUri": "bar"}, Tags=[{"Key": "foo", "Value": "bar"}]
    )


def test_set_tag(sagemaker_session):
    sagemaker_session.sagemaker_client.create_context.return_value = {
        "ContextArn": "context-arn",
    }
    obj = context.Context.create(
        context_name="foo",
        source_uri="bar",
        sagemaker_session=sagemaker_session,
    )
    tag = {"Key": "foo", "Value": "bar"}
    sagemaker_session.sagemaker_client.add_tags.return_value = {"Tags": [tag]}

    obj.set_tag(tag=tag)

    sagemaker_session.sagemaker_client.add_tags.assert_called_with = {
        "ResourceArn": "context-arn",
        "Tags": [{"Key": "foo", "Value": "bar"}],
    }


def test_set_tags(sagemaker_session):
    sagemaker_session.sagemaker_client.create_context.return_value = {
        "ContextArn": "context-arn",
    }
    obj = context.Context.create(
        context_name="foo",
        source_uri="bar",
        sagemaker_session=sagemaker_session,
    )
    tags = [{"Key": "foo1", "Value": "bar1"}, {"Key": "foo2", "Value": "bar2"}]
    sagemaker_session.sagemaker_client.add_tags.return_value = {"Tags": tags}
    obj.set_tags(tags=tags)

    sagemaker_session.sagemaker_client.add_tags.assert_called_with = {
        "ResourceArn": "context-arn",
        "Tags": [{"Key": "foo1", "Value": "bar1"}, {"Key": "foo2", "Value": "bar2"}],
    }


def test_load(sagemaker_session):
    now = datetime.datetime.now(datetime.timezone.utc)

    sagemaker_session.sagemaker_client.describe_context.return_value = {
        "ContextName": "A",
        "Source": {"SourceUri": "B", "SourceType": "test-source-type"},
        "ContextArn": "D",
        "ContextType": "E",
        "Properties": {"F": "FValue", "G": "GValue"},
        "CreationTime": now,
        "CreatedBy": {},
        "LastModifiedTime": now,
        "LastModifiedBy": {},
    }
    obj = context.Context.load(context_name="A", sagemaker_session=sagemaker_session)
    sagemaker_session.sagemaker_client.describe_context.assert_called_with(ContextName="A")
    assert "A" == obj.context_name
    assert "B" == obj.source.source_uri
    assert "test-source-type" == obj.source.source_type
    assert "D" == obj.context_arn
    assert "E" == obj.context_type
    assert {"F": "FValue", "G": "GValue"} == obj.properties


def test_list(sagemaker_session):
    creation_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=3)
    last_modified_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=4)

    sagemaker_session.sagemaker_client.list_contexts.side_effect = [
        {
            "ContextSummaries": [
                {
                    "ContextName": "A" + str(i),
                    "ContextArn": "B" + str(i),
                    "Source": {
                        "SourceUri": "test-source-uri" + str(i),
                        "SourceType": "test-source-type" + str(i),
                    },
                    "ContextType": "test-context-type",
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "LastModifiedTime": last_modified_time + datetime.timedelta(hours=i),
                    "LastModifiedBy": {},
                }
                for i in range(10)
            ],
            "NextToken": "100",
        },
        {
            "ContextSummaries": [
                {
                    "ContextName": "A" + str(i),
                    "ContextArn": "B" + str(i),
                    "Source": {
                        "SourceUri": "test-source-uri" + str(i),
                        "SourceType": "test-source-type" + str(i),
                    },
                    "ContextType": "test-context-type",
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "LastModifiedTime": last_modified_time + datetime.timedelta(hours=i),
                    "LastModifiedBy": {},
                }
                for i in range(10, 20)
            ]
        },
    ]

    expected = [
        _api_types.ContextSummary(
            context_name="A" + str(i),
            context_arn="B" + str(i),
            source=_api_types.ContextSource(
                source_uri="test-source-uri" + str(i), source_type="test-source-type" + str(i)
            ),
            context_type="test-context-type",
            creation_time=creation_time + datetime.timedelta(hours=i),
            last_modified_time=last_modified_time + datetime.timedelta(hours=i),
            last_modified_by={},
        )
        for i in range(20)
    ]
    result = list(
        context.Context.list(
            source_uri="foo",
            sagemaker_session=sagemaker_session,
            sort_by="CreationTime",
            sort_order="Ascending",
        )
    )

    assert expected == result
    expected_calls = [
        unittest.mock.call(SortBy="CreationTime", SortOrder="Ascending", SourceUri="foo"),
        unittest.mock.call(
            NextToken="100", SortBy="CreationTime", SortOrder="Ascending", SourceUri="foo"
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_contexts.mock_calls


def test_list_empty(sagemaker_session):
    sagemaker_session.sagemaker_client.list_contexts.return_value = {"ContextSummaries": []}
    assert [] == list(context.Context.list(sagemaker_session=sagemaker_session))


def test_list_context_call_args(sagemaker_session):
    created_before = datetime.datetime(1999, 10, 12, 0, 0, 0)
    created_after = datetime.datetime(1990, 10, 12, 0, 0, 0)
    source_uri = "test-source-uri"
    context_type = "foo-context-type"
    next_token = "thetoken"
    max_results = 99

    sagemaker_session.sagemaker_client.list_contexts.return_value = {}
    assert [] == list(
        context.Context.list(
            sagemaker_session=sagemaker_session,
            source_uri=source_uri,
            context_type=context_type,
            created_before=created_before,
            created_after=created_after,
            next_token=next_token,
            max_results=max_results,
            sort_by="CreationTime",
            sort_order="Ascending",
        )
    )

    expected_calls = [
        unittest.mock.call(
            SourceUri="test-source-uri",
            ContextType="foo-context-type",
            CreatedBefore=created_before,
            CreatedAfter=created_after,
            SortBy="CreationTime",
            SortOrder="Ascending",
            NextToken="thetoken",
            MaxResults=99,
        )
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_contexts.mock_calls


def test_save(sagemaker_session):
    obj = context.Context(
        sagemaker_session,
        context_name="foo",
        description="test-description",
        properties={"k1": "v1", "k2": "v2"},
        properties_to_remove=["E"],
    )
    sagemaker_session.sagemaker_client.update_context.return_value = {}
    obj.save()

    sagemaker_session.sagemaker_client.update_context.assert_called_with(
        ContextName="foo",
        Description="test-description",
        Properties={"k1": "v1", "k2": "v2"},
        PropertiesToRemove=["E"],
    )


def test_delete(sagemaker_session):
    obj = context.Context(sagemaker_session, context_name="foo")
    sagemaker_session.sagemaker_client.delete_context.return_value = {}
    obj.delete()
    sagemaker_session.sagemaker_client.delete_context.assert_called_with(ContextName="foo")


def test_create_delete_with_association(sagemaker_session):
    obj = context.Context(sagemaker_session, context_name="foo", context_arn="foo")

    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": obj.context_arn,
                    "SourceName": "X" + str(i),
                    "DestinationArn": "B" + str(i),
                    "DestinationName": "Y" + str(i),
                    "SourceType": "C" + str(i),
                    "DestinationType": "D" + str(i),
                    "AssociationType": "E" + str(i),
                    "CreationTime": None,
                    "CreatedBy": {},
                }
                for i in range(1)
            ],
        },
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "A" + str(i),
                    "SourceName": "X" + str(i),
                    "DestinationArn": obj.context_arn,
                    "DestinationName": "Y" + str(i),
                    "SourceType": "C" + str(i),
                    "DestinationType": "D" + str(i),
                    "AssociationType": "E" + str(i),
                    "CreationTime": None,
                    "CreatedBy": {},
                }
                for i in range(1, 2)
            ]
        },
    ]
    sagemaker_session.sagemaker_client.delete_association.return_value = {}
    sagemaker_session.sagemaker_client.delete_context.return_value = {}

    obj.delete(disassociate=True)

    delete_with_association_expected_calls = [
        unittest.mock.call(SourceArn=obj.context_arn, DestinationArn="B0"),
        unittest.mock.call(SourceArn="A1", DestinationArn=obj.context_arn),
    ]
    assert (
        delete_with_association_expected_calls
        == sagemaker_session.sagemaker_client.delete_association.mock_calls
    )


def test_actions(sagemaker_session):
    context_arn = "arn:aws:sagemaker:us-west-2:123456789012:context/lineage-unit-3b05f017-0d87-4c37"
    action_arn = "arn:aws:sagemaker:us-west-2:123456789012:action/lineage-unit-3b05f017-0d87-4c37"
    obj = context.EndpointContext(sagemaker_session, context_name="foo", context_arn=context_arn)

    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": action_arn, "Type": "Approval", "LineageType": "Action"},
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }

    sagemaker_session.sagemaker_client.describe_action.return_value = {
        "ActionName": "MyAction",
        "ActionArn": action_arn,
    }

    action_list = obj.actions(direction=LineageQueryDirectionEnum.DESCENDANTS)

    expected_calls = [
        unittest.mock.call(
            Direction="Descendants",
            Filters={"LineageTypes": ["Action"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=[context_arn],
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.query_lineage.mock_calls

    expected_action_list = [
        Action(
            action_arn=action_arn,
            action_name="MyAction",
        )
    ]

    assert expected_action_list[0].action_arn == action_list[0].action_arn
    assert expected_action_list[0].action_name == action_list[0].action_name


def test_processing_jobs(sagemaker_session):
    context_arn = "arn:aws:sagemaker:us-west-2:123456789012:context/lineage-unit-3b05f017-0d87-4c37"
    processing_job_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:trial_component/lineage-unit-3b05f017-0d87-4c37"
    )
    obj = context.EndpointContext(sagemaker_session, context_name="foo", context_arn=context_arn)

    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": processing_job_arn, "Type": "ProcessingJob", "LineageType": "TrialComponent"},
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }
    sagemaker_session.sagemaker_client.describe_trial_component.return_value = {
        "TrialComponentName": "MyProcessingJob",
        "TrialComponentArn": processing_job_arn,
    }

    trial_component_list = obj.processing_jobs(direction=LineageQueryDirectionEnum.ASCENDANTS)
    expected_calls = [
        unittest.mock.call(
            Direction="Ascendants",
            Filters={"Types": ["ProcessingJob"], "LineageTypes": ["TrialComponent"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=[context_arn],
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.query_lineage.mock_calls
    expected_trial_component_list = [
        LineageTrialComponent(
            trial_component_name="MyProcessingJob",
            trial_component_arn=processing_job_arn,
        )
    ]

    assert (
        expected_trial_component_list[0].trial_component_arn
        == trial_component_list[0].trial_component_arn
    )
    assert (
        expected_trial_component_list[0].trial_component_name
        == trial_component_list[0].trial_component_name
    )


def test_transform_jobs(sagemaker_session):
    context_arn = "arn:aws:sagemaker:us-west-2:123456789012:context/lineage-unit-3b05f017-0d87-4c37"
    transform_job_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:trial_component/lineage-unit-3b05f017-0d87-4c37"
    )
    obj = context.EndpointContext(sagemaker_session, context_name="foo", context_arn=context_arn)

    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": transform_job_arn, "Type": "TransformJob", "LineageType": "TrialComponent"},
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }
    sagemaker_session.sagemaker_client.describe_trial_component.return_value = {
        "TrialComponentName": "MyTransformJob",
        "TrialComponentArn": transform_job_arn,
    }

    trial_component_list = obj.transform_jobs(direction=LineageQueryDirectionEnum.ASCENDANTS)
    expected_calls = [
        unittest.mock.call(
            Direction="Ascendants",
            Filters={"Types": ["TransformJob"], "LineageTypes": ["TrialComponent"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=[context_arn],
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.query_lineage.mock_calls
    expected_trial_component_list = [
        LineageTrialComponent(
            trial_component_name="MyTransformJob",
            trial_component_arn=transform_job_arn,
        )
    ]

    assert (
        expected_trial_component_list[0].trial_component_arn
        == trial_component_list[0].trial_component_arn
    )
    assert (
        expected_trial_component_list[0].trial_component_name
        == trial_component_list[0].trial_component_name
    )


def test_trial_components(sagemaker_session):
    context_arn = "arn:aws:sagemaker:us-west-2:123456789012:context/lineage-unit-3b05f017-0d87-4c37"
    trial_component_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:trial_component/lineage-unit-3b05f017-0d87-4c37"
    )
    obj = context.EndpointContext(sagemaker_session, context_name="foo", context_arn=context_arn)

    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": trial_component_arn, "Type": "TransformJob", "LineageType": "TrialComponent"},
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }
    sagemaker_session.sagemaker_client.describe_trial_component.return_value = {
        "TrialComponentName": "MyTransformJob",
        "TrialComponentArn": trial_component_arn,
    }

    trial_component_list = obj.trial_components(direction=LineageQueryDirectionEnum.ASCENDANTS)
    expected_calls = [
        unittest.mock.call(
            Direction="Ascendants",
            Filters={"LineageTypes": ["TrialComponent"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=[context_arn],
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.query_lineage.mock_calls
    expected_trial_component_list = [
        LineageTrialComponent(
            trial_component_name="MyTransformJob",
            trial_component_arn=trial_component_arn,
        )
    ]

    assert (
        expected_trial_component_list[0].trial_component_arn
        == trial_component_list[0].trial_component_arn
    )
    assert (
        expected_trial_component_list[0].trial_component_name
        == trial_component_list[0].trial_component_name
    )
