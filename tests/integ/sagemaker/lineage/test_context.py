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
"""This module contains code to test SageMaker ``Contexts``"""
from __future__ import absolute_import

import datetime
import logging
import time

import pytest

from sagemaker.lineage import context
from sagemaker.lineage.query import LineageQueryDirectionEnum


def test_create_delete(context_obj):
    # fixture does create and then delete, this test ensures it happens at least once
    assert context_obj.context_arn


def test_create_delete_with_association(context_obj_with_association):
    # fixture does create and then delete, this test ensures it happens at least once
    assert context_obj_with_association.context_arn


def test_action(static_endpoint_context, sagemaker_session):
    actions_from_query = static_endpoint_context.actions(
        direction=LineageQueryDirectionEnum.ASCENDANTS
    )

    assert len(actions_from_query) > 0
    for action in actions_from_query:
        assert "action" in action.action_arn


def test_save(context_obj, sagemaker_session):
    context_obj.description = "updated description"
    context_obj.properties = {"k3": "v3"}
    context_obj.properties_to_remove = ["k1"]

    context_obj.save()

    loaded = context.Context.load(
        context_name=context_obj.context_name, sagemaker_session=sagemaker_session
    )

    assert "updated description" == loaded.description
    assert {"k3": "v3"} == loaded.properties


def test_load(context_obj, sagemaker_session):
    assert context_obj.context_name
    logging.info(f"loading {context_obj.context_name}")
    loaded = context.Context.load(
        context_name=context_obj.context_name, sagemaker_session=sagemaker_session
    )
    assert context_obj.context_arn == loaded.context_arn


def test_list(context_objs, sagemaker_session):
    slack = datetime.timedelta(minutes=1)
    now = datetime.datetime.now(datetime.timezone.utc)
    context_names = [context_obj.context_name for context_obj in context_objs]

    for sort_order in ["Ascending", "Descending"]:
        context_names_listed = [
            context_listed.context_name
            for context_listed in context.Context.list(
                created_after=now - slack,
                created_before=now + slack,
                sort_by="CreationTime",
                sort_order=sort_order,
                sagemaker_session=sagemaker_session,
            )
            if context_listed.context_name in context_names
        ]

    if sort_order == "Descending":
        context_names_listed = context_names_listed[::-1]
    assert context_names == context_names_listed
    # sanity check
    assert context_names


@pytest.mark.timeout(30)
def test_tag(context_obj, sagemaker_session):
    tag = {"Key": "foo", "Value": "bar"}
    context_obj.set_tag(tag)

    while True:
        actual_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=context_obj.context_arn
        )["Tags"]
        if actual_tags:
            break
        time.sleep(5)
    # When sagemaker-client-config endpoint-url is passed as argument to hit some endpoints,
    # length of actual tags will be greater than 1
    assert len(actual_tags) > 0
    assert actual_tags[0] == tag


@pytest.mark.timeout(30)
def test_tags(context_obj, sagemaker_session):
    tags = [{"Key": "foo1", "Value": "bar1"}]
    context_obj.set_tags(tags)

    while True:
        actual_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=context_obj.context_arn
        )["Tags"]
        if actual_tags:
            break
        time.sleep(5)
    # When sagemaker-client-config endpoint-url is passed as argument to hit some endpoints,
    # length of actual tags will be greater than 1
    assert len(actual_tags) > 0
    assert [actual_tags[-1]] == tags
