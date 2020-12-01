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

import pytest
from sagemaker.lineage import association


@pytest.mark.skip(reason="Not in CMH yet")
def test_create_delete(association_obj):
    # fixture does create and then delete, this test ensures it happens at least once
    assert association_obj.source_arn


@pytest.mark.skip(reason="Not in CMH yet")
def test_list(association_objs, sagemaker_session):
    slack = datetime.timedelta(minutes=1)
    now = datetime.datetime.now(datetime.timezone.utc)
    association_keys = [
        assoc_obj.source_arn + ":" + assoc_obj.destination_arn for assoc_obj in association_objs
    ]

    for sort_order in ["Ascending", "Descending"]:
        association_keys_listed = []
        listed = association.Association.list(
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
    assert association_keys == association_names_listed
    # sanity check
    assert association_keys_listed


@pytest.mark.skip(reason="Not in CMH yet")
def test_set_tag(association_obj, sagemaker_session):
    tag = {"Key": "foo", "Value": "bar"}
    association_obj.set_tag(tag)
    assert association_obj.get_tag() == tag
