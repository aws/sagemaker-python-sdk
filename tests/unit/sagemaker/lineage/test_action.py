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

from sagemaker.lineage import action, _api_types
from sagemaker.lineage._api_types import ActionSource


def test_create(sagemaker_session):
    sagemaker_session.sagemaker_client.create_action.return_value = {
        "ActionArn": "bazz",
    }
    obj = action.Action.create(
        action_name="foo",
        source_uri="bar",
        sagemaker_session=sagemaker_session,
    )
    sagemaker_session.sagemaker_client.create_action.assert_called_with(
        ActionName="foo", Source={"SourceUri": "bar"}
    )
    assert "foo" == obj.action_name
    assert "bar" == obj.source.source_uri
    assert "bazz" == obj.action_arn


def test_set_tag(sagemaker_session):
    sagemaker_session.sagemaker_client.create_action.return_value = {
        "ActionArn": "action-arn",
    }
    obj = action.Action.create(
        action_name="foo",
        source_uri="bar",
        sagemaker_session=sagemaker_session,
    )
    tag = {"Key": "foo", "Value": "bar"}
    sagemaker_session.sagemaker_client.add_tags.return_value = {"Tags": [tag]}
    obj.set_tag(tag=tag)

    sagemaker_session.sagemaker_client.add_tags.assert_called_with = {
        "ResourceArn": "action-arn",
        "Tags": [{"Key": "foo", "Value": "bar"}],
    }


def test_set_tags(sagemaker_session):
    sagemaker_session.sagemaker_client.create_action.return_value = {
        "ActionArn": "action-arn",
    }
    obj = action.Action.create(
        action_name="foo",
        source_uri="bar",
        sagemaker_session=sagemaker_session,
    )
    tags = [{"Key": "foo1", "Value": "bar1"}, {"Key": "foo2", "Value": "bar2"}]

    sagemaker_session.sagemaker_client.add_tags.return_value = {"Tags": tags}
    obj.set_tags(tags=tags)

    sagemaker_session.sagemaker_client.add_tags.assert_called_with = {
        "ResourceArn": "artifact-arn",
        "Tags": [{"Key": "foo1", "Value": "bar1"}, {"Key": "foo2", "Value": "bar2"}],
    }


def test_create_with_tags(sagemaker_session):
    sagemaker_session.sagemaker_client.create_action.return_value = {
        "ActionArn": "bazz",
    }

    tags = [{"Key": "foo", "Value": "bar"}]
    obj = action.Action.create(
        action_name="foo",
        source_uri="bar",
        tags=tags,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.create_action.assert_called_with(
        ActionName="foo", Source={"SourceUri": "bar"}, Tags=[{"Key": "foo", "Value": "bar"}]
    )

    assert "foo" == obj.action_name
    assert "bar" == obj.source.source_uri
    assert "bazz" == obj.action_arn


def test_load(sagemaker_session):
    now = datetime.datetime.now(datetime.timezone.utc)

    sagemaker_session.sagemaker_client.describe_action.return_value = {
        "ActionName": "A",
        "Source": {"SourceUri": "B", "SourceTypes": ["C1", "C2", "C3"]},
        "ActionArn": "D",
        "ActionType": "E",
        "Properties": {"F": "FValue", "G": "GValue"},
        "CreationTime": now,
        "CreatedBy": {},
        "LastModifiedTime": now,
        "LastModifiedBy": {},
    }
    obj = action.Action.load(action_name="A", sagemaker_session=sagemaker_session)
    sagemaker_session.sagemaker_client.describe_action.assert_called_with(ActionName="A")
    assert "A" == obj.action_name
    assert "B" == obj.source.source_uri
    assert "C1" == obj.source.source_types[0]
    assert "C2" == obj.source.source_types[1]
    assert "C3" == obj.source.source_types[2]
    assert "D" == obj.action_arn
    assert "E" == obj.action_type
    assert {"F": "FValue", "G": "GValue"} == obj.properties


def test_list(sagemaker_session):
    start_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
    end_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2)
    creation_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=3)
    last_modified_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=4)

    sagemaker_session.sagemaker_client.list_actions.side_effect = [
        {
            "ActionSummaries": [
                {
                    "ActionName": "A" + str(i),
                    "Action" "Arn": "B" + str(i),
                    "DisplayName": "C" + str(i),
                    "Source": {"SourceUri": "D" + str(i)},
                    "StartTime": start_time + datetime.timedelta(hours=i),
                    "EndTime": end_time + datetime.timedelta(hours=i),
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "LastModifiedTime": last_modified_time + datetime.timedelta(hours=i),
                    "LastModifiedBy": {},
                }
                for i in range(10)
            ],
            "NextToken": "100",
        },
        {
            "ActionSummaries": [
                {
                    "ActionName": "A" + str(i),
                    "ActionArn": "B" + str(i),
                    "DisplayName": "C" + str(i),
                    "Source": {"SourceUri": "D" + str(i)},
                    "StartTime": start_time + datetime.timedelta(hours=i),
                    "EndTime": end_time + datetime.timedelta(hours=i),
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "LastModifiedTime": last_modified_time + datetime.timedelta(hours=i),
                    "LastModifiedBy": {},
                }
                for i in range(10, 20)
            ]
        },
    ]

    expected = [
        _api_types.ActionSummary(
            action_name="A" + str(i),
            action_arn="B" + str(i),
            display_name="C" + str(i),
            source=_api_types.ActionSource(source_uri="D" + str(i)),
            start_time=start_time + datetime.timedelta(hours=i),
            end_time=end_time + datetime.timedelta(hours=i),
            creation_time=creation_time + datetime.timedelta(hours=i),
            last_modified_time=last_modified_time + datetime.timedelta(hours=i),
            last_modified_by={},
        )
        for i in range(20)
    ]
    result = list(
        action.Action.list(
            sagemaker_session=sagemaker_session,
            source_uri="foo",
            sort_by="CreationTime",
            sort_order="Ascending",
        )
    )

    assert expected == result
    expected_calls = [
        unittest.mock.call(
            SortBy="CreationTime",
            SortOrder="Ascending",
            SourceUri="foo",
        ),
        unittest.mock.call(
            NextToken="100",
            SortBy="CreationTime",
            SortOrder="Ascending",
            SourceUri="foo",
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_actions.mock_calls


def test_list_empty(sagemaker_session):
    sagemaker_session.sagemaker_client.list_actions.return_value = {"ActionSummaries": []}
    assert [] == list(action.Action.list(sagemaker_session=sagemaker_session))


def test_list_actions_call_args(sagemaker_session):
    created_before = datetime.datetime(1999, 10, 12, 0, 0, 0)
    created_after = datetime.datetime(1990, 10, 12, 0, 0, 0)
    source_uri = "foo-source-uri"
    action_type = "foo-action-type"
    next_token = "thetoken"
    max_results = 99

    sagemaker_session.sagemaker_client.list_actions.return_value = {}
    assert [] == list(
        action.Action.list(
            sagemaker_session=sagemaker_session,
            source_uri=source_uri,
            action_type=action_type,
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
            SourceUri="foo-source-uri",
            ActionType="foo-action-type",
            CreatedBefore=created_before,
            CreatedAfter=created_after,
            SortBy="CreationTime",
            SortOrder="Ascending",
            NextToken="thetoken",
            MaxResults=99,
        )
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_actions.mock_calls


def test_save(sagemaker_session):
    obj = action.Action(
        sagemaker_session,
        action_name="foo",
        description="updated-description",
        status="updated-status",
        properties={"k1": "v1"},
        properties_to_remove=["k2"],
    )
    sagemaker_session.sagemaker_client.update_action.return_value = {}
    obj.save()

    sagemaker_session.sagemaker_client.update_action.assert_called_with(
        ActionName="foo",
        Description="updated-description",
        Status="updated-status",
        Properties={"k1": "v1"},
        PropertiesToRemove=["k2"],
    )


def test_delete(sagemaker_session):
    obj = action.Action(sagemaker_session, action_name="foo", source_uri="bar")
    sagemaker_session.sagemaker_client.delete_action.return_value = {}
    obj.delete()
    sagemaker_session.sagemaker_client.delete_action.assert_called_with(ActionName="foo")


def test_create_delete_with_association(sagemaker_session):
    obj = action.Action(
        sagemaker_session,
        action_name="foo",
        action_arn="a1",
        description="updated-description",
        status="updated-status",
        properties={"k1": "v1"},
        properties_to_remove=["k2"],
    )

    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": obj.action_arn,
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
                    "DestinationArn": obj.action_arn,
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
    sagemaker_session.sagemaker_client.delete_action.return_value = {}

    obj.delete(disassociate=True)

    delete_with_association_expected_calls = [
        unittest.mock.call(SourceArn=obj.action_arn, DestinationArn="B0"),
        unittest.mock.call(SourceArn="A1", DestinationArn=obj.action_arn),
    ]
    assert (
        delete_with_association_expected_calls
        == sagemaker_session.sagemaker_client.delete_association.mock_calls
    )


def test_model_package(sagemaker_session):
    obj = action.ModelPackageApprovalAction(
        sagemaker_session,
        action_name="abcd-aws-model-package",
        source=ActionSource(
            source_uri="arn:aws:sagemaker:us-west-2:123456789012:model-package/pipeline88modelpackage/1",
            source_type="ARN",
        ),
        status="updated-status",
        properties={"k1": "v1"},
        properties_to_remove=["k2"],
    )
    sagemaker_session.sagemaker_client.describe_model_package.return_value = {}
    obj.model_package()

    sagemaker_session.sagemaker_client.describe_model_package.assert_called_with(
        ModelPackageName="pipeline88modelpackage",
    )
