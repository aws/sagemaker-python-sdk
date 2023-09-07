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

import logging

from sagemaker.experiments import trial
from src.sagemaker.utils import retry_with_backoff


def test_create_delete(trial_obj):
    # Fixture creates / deletes, just ensure used at least once.
    assert trial_obj.trial_name


def test_create_tags(trial_obj, sagemaker_session):
    client = sagemaker_session.sagemaker_client
    while True:
        actual_tags = client.list_tags(ResourceArn=trial_obj.trial_arn)["Tags"]
        if actual_tags:
            break
    for tag in actual_tags:
        if "aws:tag" in tag.get("Key"):
            actual_tags.remove(tag)
    assert actual_tags == trial_obj.tags


def test_save_load(trial_obj, sagemaker_session):
    trial_obj.display_name = "foo"
    trial_obj.save()
    assert (
        "foo"
        == trial._Trial.load(
            trial_name=trial_obj.trial_name,
            sagemaker_session=sagemaker_session,
        ).display_name
    )


def test_add_remove_trial_component(trial_obj, trial_component_obj):
    trial_obj.add_trial_component(trial_component_obj)
    logging.info(
        f"Added trial component {trial_component_obj.trial_component_name} to trial {trial_obj.trial_name}"
    )

    def validate_add():
        trial_components = list(trial_obj.list_trial_components())
        assert 1 == len(
            trial_components
        ), "Expected trial component to be included in trials list of TC"

    retry_with_backoff(validate_add)

    trial_obj.remove_trial_component(trial_component_obj)
    logging.info(
        f"Removed trial component {trial_component_obj.trial_component_name} from trial {trial_obj.trial_name}"
    )

    def validate_remove():
        trial_components = list(trial_obj.list_trial_components())
        assert 0 == len(
            trial_components
        ), "Expected trial component to be removed from trials list of TC"

    retry_with_backoff(validate_remove)
