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

import botocore
import pytest

import datetime

from unittest.mock import patch

from sagemaker import Session
from sagemaker.experiments._api_types import TrialSummary
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent


@pytest.fixture
def datetime_obj():
    return datetime.datetime(2017, 6, 16, 15, 55, 0)


def test_load(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.describe_trial.return_value = {"ExperimentName": "experiment-name-value"}
    trial_obj = _Trial.load(trial_name="name-value", sagemaker_session=sagemaker_session)
    assert trial_obj.trial_name == "name-value"
    assert trial_obj.experiment_name == "experiment-name-value"
    client.describe_trial.assert_called_with(TrialName="name-value")


def test_create(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_trial.return_value = {
        "Arn": "arn:aws:1234",
        "TrialName": "name-value",
    }
    trial_obj = _Trial.create(
        trial_name="name-value",
        experiment_name="experiment-name-value",
        sagemaker_session=sagemaker_session,
    )
    assert trial_obj.trial_name == "name-value"
    client.create_trial.assert_called_with(
        TrialName="name-value", ExperimentName="experiment-name-value"
    )


def test_create_with_tags(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_trial.return_value = {
        "Arn": "arn:aws:1234",
        "TrialName": "name-value",
    }
    tags = [{"Key": "foo", "Value": "bar"}]
    trial_obj = _Trial.create(
        trial_name="name-value",
        experiment_name="experiment-name-value",
        sagemaker_session=sagemaker_session,
        tags=tags,
    )
    assert trial_obj.trial_name == "name-value"
    client.create_trial.assert_called_with(
        TrialName="name-value",
        ExperimentName="experiment-name-value",
        Tags=[{"Key": "foo", "Value": "bar"}],
    )


def test_delete(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    obj = _Trial(sagemaker_session, trial_name="foo")
    client.delete_trial.return_value = {}
    obj.delete()
    client.delete_trial.assert_called_with(TrialName="foo")


def test_save(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    obj = _Trial(
        sagemaker_session,
        trial_name="foo",
        experiment_name="whizz",
        display_name="bar",
        tags=[{"Key": "foo", "Value": "bar"}],
    )
    client.update_trial.return_value = {}
    obj.save()

    client.update_trial.assert_called_with(
        TrialName="foo",
        DisplayName="bar",
    )


def test_add_trial_component(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    trial = _Trial(sagemaker_session=sagemaker_session)
    trial.trial_name = "bar"
    trial.add_trial_component("foo")
    client.associate_trial_component.assert_called_with(TrialName="bar", TrialComponentName="foo")

    tc = _TrialComponent(trial_component_name="tc-foo", sagemaker_session=sagemaker_session)
    trial.add_trial_component(tc)
    client.associate_trial_component.assert_called_with(
        TrialName="bar", TrialComponentName=tc.trial_component_name
    )


def test_remove_trial_component(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    trial = _Trial(sagemaker_session=sagemaker_session)
    trial.trial_name = "bar"
    trial.remove_trial_component("foo")
    client.disassociate_trial_component.assert_called_with(
        TrialName="bar", TrialComponentName="foo"
    )

    tc = _TrialComponent(trial_component_name="tc-foo", sagemaker_session=sagemaker_session)
    trial.remove_trial_component(tc)
    client.disassociate_trial_component.assert_called_with(
        TrialName="bar", TrialComponentName=tc.trial_component_name
    )


@patch("sagemaker.experiments.trial._Trial.load")
@patch("sagemaker.experiments.trial._Trial.create")
def test_load_or_create_when_exist(mock_create, mock_load):
    sagemaker_session = Session()
    trial_name = "trial_name"
    exp_name = "exp_name"
    exists_error = botocore.exceptions.ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "Experiment with name (experiment-xyz) already exists.",
            }
        },
        operation_name="foo",
    )
    mock_create.side_effect = exists_error
    # The trial exists and experiment matches
    mock_load.return_value = _Trial(
        trial_name=trial_name,
        experiment_name=exp_name,
        sagemaker_session=sagemaker_session,
    )
    _Trial._load_or_create(
        trial_name=trial_name, experiment_name=exp_name, sagemaker_session=sagemaker_session
    )
    mock_create.assert_called_once_with(
        trial_name=trial_name,
        experiment_name=exp_name,
        display_name=None,
        tags=None,
        sagemaker_session=sagemaker_session,
    )
    mock_load.assert_called_once_with(trial_name, sagemaker_session)

    # The trial exists but experiment does not match
    mock_load.return_value = _Trial(
        trial_name=trial_name,
        exp_name="another_exp_name",
        sagemaker_session=sagemaker_session,
    )
    with pytest.raises(ValueError) as err:
        _Trial._load_or_create(
            trial_name=trial_name, experiment_name=exp_name, sagemaker_session=sagemaker_session
        )
    assert "The given experiment_name {} does not match that in the loaded trial".format(
        exp_name
    ) in str(err)


@patch("sagemaker.experiments.trial._Trial.load")
@patch("sagemaker.experiments.trial._Trial.create")
def test_load_or_create_when_not_exist(mock_create, mock_load):
    sagemaker_session = Session()
    trial_name = "trial_name"
    exp_name = "exp_name"

    _Trial._load_or_create(
        trial_name=trial_name, experiment_name=exp_name, sagemaker_session=sagemaker_session
    )

    mock_create.assert_called_once_with(
        trial_name=trial_name,
        experiment_name=exp_name,
        display_name=None,
        tags=None,
        sagemaker_session=sagemaker_session,
    )
    mock_load.assert_not_called()


def test_list_trials_without_experiment_name(sagemaker_session, datetime_obj):
    client = sagemaker_session.sagemaker_client
    client.list_trials.return_value = {
        "TrialSummaries": [
            {
                "TrialName": "trial-1",
                "CreationTime": datetime_obj,
                "LastModifiedTime": datetime_obj,
            },
            {
                "TrialName": "trial-2",
                "CreationTime": datetime_obj,
                "LastModifiedTime": datetime_obj,
            },
        ]
    }
    expected = [
        TrialSummary(
            trial_name="trial-1", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
        TrialSummary(
            trial_name="trial-2", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
    ]
    assert expected == list(_Trial.list(sagemaker_session=sagemaker_session))
    client.list_trials.assert_called_with(**{})


def test_list_trials_with_experiment_name(sagemaker_session, datetime_obj):
    client = sagemaker_session.sagemaker_client
    client.list_trials.return_value = {
        "TrialSummaries": [
            {
                "TrialName": "trial-1",
                "CreationTime": datetime_obj,
                "LastModifiedTime": datetime_obj,
            },
            {
                "TrialName": "trial-2",
                "CreationTime": datetime_obj,
                "LastModifiedTime": datetime_obj,
            },
        ]
    }
    expected = [
        TrialSummary(
            trial_name="trial-1", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
        TrialSummary(
            trial_name="trial-2", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
    ]
    assert expected == list(_Trial.list(experiment_name="foo", sagemaker_session=sagemaker_session))
    client.list_trials.assert_called_with(ExperimentName="foo")


def test_list_trials_with_trial_component_name(sagemaker_session, datetime_obj):
    client = sagemaker_session.sagemaker_client
    client.list_trials.return_value = {
        "TrialSummaries": [
            {
                "TrialName": "trial-1",
                "CreationTime": datetime_obj,
                "LastModifiedTime": datetime_obj,
            },
            {
                "TrialName": "trial-2",
                "CreationTime": datetime_obj,
                "LastModifiedTime": datetime_obj,
            },
        ]
    }
    expected = [
        TrialSummary(
            trial_name="trial-1", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
        TrialSummary(
            trial_name="trial-2", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
    ]
    assert expected == list(
        _Trial.list(trial_component_name="tc-foo", sagemaker_session=sagemaker_session)
    )
    client.list_trials.assert_called_with(TrialComponentName="tc-foo")
