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
"""This module contains code to test SageMaker ``Associations``"""
from __future__ import absolute_import

import datetime
import time

import pytest

from sagemaker.lineage import association


def test_create_delete(association_obj):
    # fixture does create and then delete, this test ensures it happens at least once
    assert association_obj.source_arn


def test_list(association_objs, sagemaker_session):
    slack = datetime.timedelta(minutes=1)
    now = datetime.datetime.now(datetime.timezone.utc)
    association_keys = [
        assoc_obj.source_arn + ":" + assoc_obj.destination_arn for assoc_obj in association_objs
    ]

    for sort_order in ["Ascending", "Descending"]:
        association_keys_listed = []
        source_arn = [assoc_obj.source_arn for assoc_obj in association_objs][0]
        listed = association.Association.list(
            source_arn=source_arn,
            created_after=now - slack,
            created_before=now + slack,
            sort_by="CreationTime",
            sort_order=sort_order,
            sagemaker_session=sagemaker_session,
        )

        for assoc in listed:
            key = assoc.source_arn + ":" + assoc.destination_arn
            if key in association_keys:
                association_keys_listed.append(key)

    if sort_order == "Descending":
        association_names_listed = association_keys_listed[::-1]
    assert association_keys[::-1] == association_names_listed
    # sanity check
    assert association_keys_listed

    for listed_asscn in listed:
        assert listed_asscn.created_by is None


@pytest.mark.timeout(30)
def test_set_tag(association_obj, sagemaker_session):
    tag = {"Key": "foo", "Value": "bar"}
    association_obj.set_tag(tag)

    while True:
        actual_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=association_obj.source_arn
        )["Tags"]
        if actual_tags:
            break
        time.sleep(5)
    # When sagemaker-client-config endpoint-url is passed as argument to hit some endpoints,
    # length of actual tags will be greater than 1
    assert len(actual_tags) > 0
    assert actual_tags[0] == tag


@pytest.mark.timeout(30)
def test_tags(association_obj, sagemaker_session):
    tags = [{"Key": "foo1", "Value": "bar1"}]
    association_obj.set_tags(tags)

    while True:
        actual_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=association_obj.source_arn
        )["Tags"]
        if actual_tags:
            break
        time.sleep(5)
    # When sagemaker-client-config endpoint-url is passed as argument to hit some endpoints,
    # length of actual tags will be greater than 1
    assert len(actual_tags) > 0
    assert [actual_tags[-1]] == tags
