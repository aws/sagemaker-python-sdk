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
