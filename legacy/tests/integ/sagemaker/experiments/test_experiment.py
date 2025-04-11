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

from sagemaker.experiments import experiment
from tests.integ.sagemaker.experiments.helpers import name


def test_create_delete(experiment_obj):
    # The fixture creates deletes, just ensure fixture is used at least once
    assert experiment_obj.experiment_name


def test_create_tags(experiment_obj, sagemaker_session):
    client = sagemaker_session.sagemaker_client
    while True:
        actual_tags = client.list_tags(ResourceArn=experiment_obj.experiment_arn)["Tags"]
        if actual_tags:
            break
    for tag in actual_tags:
        if "aws:tag" in tag.get("Key"):
            actual_tags.remove(tag)
    assert actual_tags == experiment_obj.tags


def test_save(experiment_obj):
    description = name()
    experiment_obj.description = description
    experiment_obj.save()


def test_save_load(experiment_obj, sagemaker_session):
    experiment_obj_two = experiment.Experiment.load(
        experiment_name=experiment_obj.experiment_name, sagemaker_session=sagemaker_session
    )
    assert experiment_obj.experiment_name == experiment_obj_two.experiment_name
    assert experiment_obj.description == experiment_obj_two.description

    experiment_obj.description = name()
    experiment_obj.display_name = name()
    experiment_obj.save()
    experiment_obj_three = experiment.Experiment.load(
        experiment_name=experiment_obj.experiment_name, sagemaker_session=sagemaker_session
    )
    assert experiment_obj.description == experiment_obj_three.description
    assert experiment_obj.display_name == experiment_obj_three.display_name
