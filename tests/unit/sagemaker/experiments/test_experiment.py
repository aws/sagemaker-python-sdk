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

import botocore.exceptions
import pytest
import unittest.mock
import datetime

from unittest.mock import patch

from sagemaker import Session
from sagemaker.experiments import experiment
from sagemaker.experiments._api_types import TrialSummary


@pytest.fixture
def datetime_obj():
    return datetime.datetime(2017, 6, 16, 15, 55, 0)


def test_load(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.describe_experiment.return_value = {"Description": "description-value"}
    experiment_obj = experiment.Experiment.load(
        experiment_name="name-value", sagemaker_session=sagemaker_session
    )
    assert experiment_obj.experiment_name == "name-value"
    assert experiment_obj.description == "description-value"

    client.describe_experiment.assert_called_with(ExperimentName="name-value")


def test_create(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_experiment.return_value = {"Arn": "arn:aws:1234"}
    experiment_obj = experiment.Experiment.create(
        experiment_name="name-value", sagemaker_session=sagemaker_session
    )
    assert experiment_obj.experiment_name == "name-value"
    client.create_experiment.assert_called_with(ExperimentName="name-value")


def test_create_with_tags(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_experiment.return_value = {"Arn": "arn:aws:1234"}
    tags = [{"Key": "foo", "Value": "bar"}]
    experiment_obj = experiment.Experiment.create(
        experiment_name="name-value", sagemaker_session=sagemaker_session, tags=tags
    )
    assert experiment_obj.experiment_name == "name-value"
    client.create_experiment.assert_called_with(ExperimentName="name-value", Tags=tags)


def test_save(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    obj = experiment.Experiment(sagemaker_session, experiment_name="foo", description="bar")
    client.update_experiment.return_value = {}
    obj.save()
    client.update_experiment.assert_called_with(ExperimentName="foo", Description="bar")


def test_delete(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    obj = experiment.Experiment(sagemaker_session, experiment_name="foo", description="bar")
    client.delete_experiment.return_value = {}
    obj.delete()
    client.delete_experiment.assert_called_with(ExperimentName="foo")


@patch("sagemaker.experiments.experiment.Experiment.load")
@patch("sagemaker.experiments.experiment.Experiment.create")
def test_load_or_create_when_exist(mock_create, mock_load, sagemaker_session):
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
    experiment.Experiment._load_or_create(
        experiment_name=exp_name, sagemaker_session=sagemaker_session
    )
    mock_create.assert_called_once_with(
        experiment_name=exp_name,
        display_name=None,
        description=None,
        tags=None,
        sagemaker_session=sagemaker_session,
    )
    mock_load.assert_called_once_with(exp_name, sagemaker_session)


@patch("sagemaker.experiments.experiment.Experiment.load")
@patch("sagemaker.experiments.experiment.Experiment.create")
def test_load_or_create_when_not_exist(mock_create, mock_load):
    sagemaker_session = Session()
    exp_name = "exp_name"
    experiment.Experiment._load_or_create(
        experiment_name=exp_name, sagemaker_session=sagemaker_session
    )
    mock_create.assert_called_once_with(
        experiment_name=exp_name,
        display_name=None,
        description=None,
        tags=None,
        sagemaker_session=sagemaker_session,
    )
    mock_load.assert_not_called()


def test_list_trials_empty(sagemaker_session):
    sagemaker_session.sagemaker_client.list_trials.return_value = {"TrialSummaries": []}
    experiment_obj = experiment.Experiment(sagemaker_session=sagemaker_session)
    assert list(experiment_obj.list_trials()) == []


def test_list_trials_single(sagemaker_session, datetime_obj):
    experiment_obj = experiment.Experiment(sagemaker_session=sagemaker_session)
    sagemaker_session.sagemaker_client.list_trials.return_value = {
        "TrialSummaries": [
            {"Name": "trial-foo", "CreationTime": datetime_obj, "LastModifiedTime": datetime_obj}
        ]
    }

    assert list(experiment_obj.list_trials()) == [
        TrialSummary(name="trial-foo", creation_time=datetime_obj, last_modified_time=datetime_obj)
    ]


def test_list_trials_two_values(sagemaker_session, datetime_obj):
    experiment_obj = experiment.Experiment(sagemaker_session=sagemaker_session)
    sagemaker_session.sagemaker_client.list_trials.return_value = {
        "TrialSummaries": [
            {"Name": "trial-foo-1", "CreationTime": datetime_obj, "LastModifiedTime": datetime_obj},
            {"Name": "trial-foo-2", "CreationTime": datetime_obj, "LastModifiedTime": datetime_obj},
        ]
    }

    assert list(experiment_obj.list_trials()) == [
        TrialSummary(
            name="trial-foo-1", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
        TrialSummary(
            name="trial-foo-2", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
    ]


def test_next_token(sagemaker_session, datetime_obj):
    experiment_obj = experiment.Experiment(sagemaker_session)
    client = sagemaker_session.sagemaker_client
    client.list_trials.side_effect = [
        {
            "TrialSummaries": [
                {
                    "Name": "trial-foo-1",
                    "CreationTime": datetime_obj,
                    "LastModifiedTime": datetime_obj,
                },
                {
                    "Name": "trial-foo-2",
                    "CreationTime": datetime_obj,
                    "LastModifiedTime": datetime_obj,
                },
            ],
            "NextToken": "foo",
        },
        {
            "TrialSummaries": [
                {
                    "Name": "trial-foo-3",
                    "CreationTime": datetime_obj,
                    "LastModifiedTime": datetime_obj,
                }
            ]
        },
    ]

    assert list(experiment_obj.list_trials()) == [
        TrialSummary(
            name="trial-foo-1", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
        TrialSummary(
            name="trial-foo-2", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
        TrialSummary(
            name="trial-foo-3", creation_time=datetime_obj, last_modified_time=datetime_obj
        ),
    ]

    client.list_trials.assert_any_call(**{})
    client.list_trials.assert_any_call(NextToken="foo")


def test_list_trials_call_args(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    created_before = datetime.datetime(1999, 10, 12, 0, 0, 0)
    created_after = datetime.datetime(1990, 10, 12, 0, 0, 0)
    experiment_obj = experiment.Experiment(sagemaker_session=sagemaker_session)
    client.list_trials.return_value = {}
    assert [] == list(
        experiment_obj.list_trials(created_after=created_after, created_before=created_before)
    )
    client.list_trials.assert_called_with(CreatedBefore=created_before, CreatedAfter=created_after)


def test_delete_all_with_incorrect_action_name(sagemaker_session):
    obj = experiment.Experiment(sagemaker_session, experiment_name="foo", description="bar")
    with pytest.raises(ValueError) as err:
        obj._delete_all(action="abc")

    assert "Must confirm with string '--force'" in str(err)


def test_delete_all(sagemaker_session):
    obj = experiment.Experiment(sagemaker_session, experiment_name="foo", description="bar")
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
    client.describe_trial.side_effect = [
        {"Trialname": "trial-1", "ExperimentName": "experiment-name-value"},
        {"Trialname": "trial-2", "ExperimentName": "experiment-name-value"},
    ]
    client.list_trial_components.side_effect = [
        {
            "TrialComponentSummaries": [
                {
                    "TrialComponentName": "trial-component-1",
                    "CreationTime": datetime_obj,
                    "LastModifiedTime": datetime_obj,
                },
                {
                    "TrialComponentName": "trial-component-2",
                    "CreationTime": datetime_obj,
                    "LastModifiedTime": datetime_obj,
                },
            ]
        },
        {
            "TrialComponentSummaries": [
                {
                    "TrialComponentName": "trial-component-3",
                    "CreationTime": datetime_obj,
                    "LastModifiedTime": datetime_obj,
                },
                {
                    "TrialComponentName": "trial-component-4",
                    "CreationTime": datetime_obj,
                    "LastModifiedTime": datetime_obj,
                },
            ]
        },
    ]

    client.describe_trial_component.side_effect = [
        {"TrialComponentName": "trial-component-1"},
        {"TrialComponentName": "trial-component-2"},
        {"TrialComponentName": "trial-component-3"},
        {"TrialComponentName": "trial-component-4"},
    ]

    client.delete_trial_component.return_value = {}
    client.delete_trial.return_value = {}
    client.delete_experiment.return_value = {}

    obj._delete_all(action="--force")

    client.delete_experiment.assert_called_with(ExperimentName="foo")

    delete_trial_expected_calls = [
        unittest.mock.call(TrialName="trial-1"),
        unittest.mock.call(TrialName="trial-2"),
    ]
    assert delete_trial_expected_calls == client.delete_trial.mock_calls

    delete_trial_component_expected_calls = [
        unittest.mock.call(TrialComponentName="trial-component-1"),
        unittest.mock.call(TrialComponentName="trial-component-2"),
        unittest.mock.call(TrialComponentName="trial-component-3"),
        unittest.mock.call(TrialComponentName="trial-component-4"),
    ]
    assert delete_trial_component_expected_calls == client.delete_trial_component.mock_calls


def test_delete_all_fail(sagemaker_session):
    obj = experiment.Experiment(sagemaker_session, experiment_name="foo", description="bar")
    sagemaker_session.sagemaker_client.list_trials.side_effect = Exception
    with pytest.raises(Exception) as e:
        obj._delete_all(action="--force")

    assert str(e.value) == "Failed to delete, please try again."
