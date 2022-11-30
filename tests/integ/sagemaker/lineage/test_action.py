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
"""This module contains code to test SageMaker ``Actions``"""
from __future__ import absolute_import

import datetime
import logging
import time

import pytest

from sagemaker.lineage import action
from sagemaker.lineage.query import LineageQueryDirectionEnum


def test_create_delete(action_obj):
    # fixture does create and then delete, this test ensures it happens at least once
    assert action_obj.action_arn


def test_create_delete_with_association(action_obj_with_association):
    # fixture does create and then delete, this test ensures it happens at least once
    assert action_obj_with_association.action_arn


def test_save(action_obj, sagemaker_session):
    action_obj.description = "updated integration test description"
    action_obj.status = "Completed"
    action_obj.properties = {"k3": "v3"}
    action_obj.properties_to_remove = ["k1"]

    action_obj.save()

    loaded = action.Action.load(
        action_name=action_obj.action_name, sagemaker_session=sagemaker_session
    )

    assert "updated integration test description" == loaded.description
    assert "Completed" == loaded.status
    assert {"k3": "v3"} == loaded.properties


def test_load(action_obj, sagemaker_session):
    assert action_obj.action_name
    logging.info(f"loading {action_obj.action_name}")
    loaded = action.Action.load(
        action_name=action_obj.action_name, sagemaker_session=sagemaker_session
    )
    assert action_obj.action_arn == loaded.action_arn


def test_list(action_objs, sagemaker_session):
    slack = datetime.timedelta(minutes=1)
    now = datetime.datetime.now(datetime.timezone.utc)
    action_names = [actn.action_name for actn in action_objs]

    for sort_order in ["Ascending", "Descending"]:
        action_names_listed = [
            action_listed.action_name
            for action_listed in action.Action.list(
                created_after=now - slack,
                created_before=now + slack,
                sort_by="CreationTime",
                sort_order=sort_order,
                sagemaker_session=sagemaker_session,
            )
            if action_listed.action_name in action_names
        ]

    if sort_order == "Descending":
        action_names_listed = action_names_listed[::-1]
    assert action_names == action_names_listed
    # sanity check
    assert action_names


@pytest.mark.timeout(30)
def test_tag(action_obj, sagemaker_session):
    tag = {"Key": "foo", "Value": "bar"}
    action_obj.set_tag(tag)

    while True:
        actual_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=action_obj.action_arn
        )["Tags"]
        if actual_tags:
            break
        time.sleep(5)
    # When sagemaker-client-config endpoint-url is passed as argument to hit some endpoints,
    # length of actual tags will be greater than 1
    assert len(actual_tags) > 0
    assert actual_tags[0] == tag


@pytest.mark.timeout(30)
def test_tags(action_obj, sagemaker_session):
    tags = [{"Key": "foo1", "Value": "bar1"}]
    action_obj.set_tags(tags)

    while True:
        actual_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=action_obj.action_arn
        )["Tags"]
        if actual_tags:
            break
        time.sleep(5)
    # When sagemaker-client-config endpoint-url is passed as argument to hit some endpoints,
    # length of actual tags will be greater than 1
    assert len(actual_tags) > 0
    assert [actual_tags[-1]] == tags


@pytest.mark.skip("data inconsistency P61661075")
def test_upstream_artifacts(static_model_deployment_action):
    artifacts_from_query = static_model_deployment_action.artifacts(
        direction=LineageQueryDirectionEnum.ASCENDANTS
    )
    assert len(artifacts_from_query) > 0
    for artifact in artifacts_from_query:
        assert "artifact" in artifact.artifact_arn


@pytest.mark.skip("data inconsistency P61661075")
def test_downstream_artifacts(static_approval_action):
    artifacts_from_query = static_approval_action.artifacts(
        direction=LineageQueryDirectionEnum.DESCENDANTS
    )
    assert len(artifacts_from_query) > 0
    for artifact in artifacts_from_query:
        assert "artifact" in artifact.artifact_arn


@pytest.mark.skip("data inconsistency P61661075")
def test_datasets(static_approval_action, static_dataset_artifact, sagemaker_session):
    try:
        sagemaker_session.sagemaker_client.add_association(
            SourceArn=static_dataset_artifact.artifact_arn,
            DestinationArn=static_approval_action.action_arn,
            AssociationType="ContributedTo",
        )
    except Exception:
        print("Source and Destination association already exists.")

    time.sleep(3)
    artifacts_from_query = static_approval_action.datasets()

    assert len(artifacts_from_query) > 0
    for artifact in artifacts_from_query:
        assert "artifact" in artifact.artifact_arn
        assert artifact.artifact_type == "DataSet"

    try:
        sagemaker_session.sagemaker_client.delete_association(
            SourceArn=static_dataset_artifact.artifact_arn,
            DestinationArn=static_approval_action.action_arn,
        )
    except Exception:
        pass


@pytest.mark.skip("data inconsistency P61661075")
def test_endpoints(static_approval_action):
    endpoint_contexts_from_query = static_approval_action.endpoints()
    assert len(endpoint_contexts_from_query) > 0
    for endpoint in endpoint_contexts_from_query:
        assert endpoint.context_type == "Endpoint"
        assert "endpoint" in endpoint.context_arn
