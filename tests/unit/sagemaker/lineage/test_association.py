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

from sagemaker.lineage import association, _api_types


def test_create(sagemaker_session):
    sagemaker_session.sagemaker_client.add_association.return_value = {
        "AssociationArn": "bazz",
    }
    obj = association.Association.create(
        source_arn="foo",
        destination_arn="bar",
        association_type="test-type",
        sagemaker_session=sagemaker_session,
    )
    sagemaker_session.sagemaker_client.add_association.assert_called_with(
        SourceArn="foo",
        DestinationArn="bar",
        AssociationType="test-type",
    )
    assert "foo" == obj.source_arn
    assert "bar" == obj.destination_arn
    assert "test-type" == obj.association_type


def test_set_tag(sagemaker_session):
    sagemaker_session.sagemaker_client.add_association.return_value = {
        "AssociationArn": "association-arn",
    }
    obj = association.Association.create(
        source_arn="foo",
        destination_arn="bar",
        association_type="test-type",
        sagemaker_session=sagemaker_session,
    )
    tag = {"Key": "foo", "Value": "bar"}
    sagemaker_session.sagemaker_client.add_tags.return_value = {"Tags": tag}
    obj.set_tag(tag)

    sagemaker_session.sagemaker_client.add_tags.assert_called_with = {
        "ResourceArn": "association-arn",
        "Tags": [{"Key": "foo", "Value": "bar"}],
    }


def test_set_tags(sagemaker_session):
    sagemaker_session.sagemaker_client.add_association.return_value = {
        "AssociationArn": "association-arn",
    }
    obj = association.Association.create(
        source_arn="foo",
        destination_arn="bar",
        association_type="test-type",
        sagemaker_session=sagemaker_session,
    )
    tags = [{"Key": "foo1", "Value": "bar1"}, {"Key": "foo2", "Value": "bar2"}]
    sagemaker_session.sagemaker_client.add_tags.return_value = {"Tags": tags}
    obj.set_tags(tags=tags)
    sagemaker_session.sagemaker_client.add_tags.assert_called_with = {
        "ResourceArn": "association-arn",
        "Tags": [{"Key": "foo1", "Value": "bar1"}, {"Key": "foo2", "Value": "bar2"}],
    }


def test_list(sagemaker_session):
    creation_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=3)

    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "A" + str(i),
                    "SourceName": "X" + str(i),
                    "DestinationArn": "B" + str(i),
                    "DestinationName": "Y" + str(i),
                    "SourceType": "C" + str(i),
                    "DestinationType": "D" + str(i),
                    "AssociationType": "E" + str(i),
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "CreatedBy": {
                        "UserProfileArn": "profileArn",
                        "UserProfileName": "profileName",
                        "DomainId": "domainId",
                    },
                }
                for i in range(10)
            ],
            "NextToken": "100",
        },
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "A" + str(i),
                    "SourceName": "X" + str(i),
                    "DestinationArn": "B" + str(i),
                    "DestinationName": "Y" + str(i),
                    "SourceType": "C" + str(i),
                    "DestinationType": "D" + str(i),
                    "AssociationType": "E" + str(i),
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "CreatedBy": {
                        "UserProfileArn": "profileArn",
                        "UserProfileName": "profileName",
                        "DomainId": "domainId",
                    },
                }
                for i in range(10, 20)
            ]
        },
    ]

    expected = [
        _api_types.AssociationSummary(
            source_arn="A" + str(i),
            source_name="X" + str(i),
            destination_arn="B" + str(i),
            destination_name="Y" + str(i),
            source_type="C" + str(i),
            destination_type="D" + str(i),
            association_type="E" + str(i),
            creation_time=creation_time + datetime.timedelta(hours=i),
            created_by=_api_types.UserContext(
                user_profile_arn="profileArn",
                user_profile_name="profileName",
                domain_id="domainId",
            ),
        )
        for i in range(20)
    ]
    result = list(
        association.Association.list(
            sagemaker_session=sagemaker_session,
            source_arn="foo",
            destination_arn="bar",
            sort_by="CreationTime",
            sort_order="Ascending",
        )
    )

    assert expected == result
    expected_calls = [
        unittest.mock.call(
            SortBy="CreationTime",
            SortOrder="Ascending",
            SourceArn="foo",
            DestinationArn="bar",
        ),
        unittest.mock.call(
            NextToken="100",
            SortBy="CreationTime",
            SortOrder="Ascending",
            SourceArn="foo",
            DestinationArn="bar",
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls


def test_list_empty(sagemaker_session):
    sagemaker_session.sagemaker_client.list_associations.return_value = {"AssociationSummaries": []}
    assert [] == list(association.Association.list(sagemaker_session=sagemaker_session))


def test_list_associations_call_args(sagemaker_session):
    created_before = datetime.datetime(1999, 10, 12, 0, 0, 0)
    created_after = datetime.datetime(1990, 10, 12, 0, 0, 0)
    source_arn = "foo-arn"
    destination_arn = "bar-arn"
    next_token = "thetoken"
    max_results = 99

    sagemaker_session.sagemaker_client.list_associations.return_value = {}
    assert [] == list(
        association.Association.list(
            sagemaker_session=sagemaker_session,
            source_arn=source_arn,
            destination_arn=destination_arn,
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
            SourceArn="foo-arn",
            DestinationArn="bar-arn",
            CreatedBefore=created_before,
            CreatedAfter=created_after,
            SortBy="CreationTime",
            SortOrder="Ascending",
            NextToken="thetoken",
            MaxResults=99,
        )
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls


def test_delete(sagemaker_session):
    obj = association.Association(sagemaker_session, source_arn="foo", destination_arn="bar")
    sagemaker_session.sagemaker_client.delete_association.return_value = {}
    obj.delete()
    sagemaker_session.sagemaker_client.delete_association.assert_called_with(
        SourceArn="foo",
        DestinationArn="bar",
    )
