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

import pytest
import unittest.mock
import datetime

from mock.mock import patch

from sagemaker import Session
from sagemaker.experiments import experiment


@pytest.fixture
def client():
    """Mock client.

    Considerations when appropriate:

        * utilize botocore.stub.Stubber
        * separate runtime client from client
    """
    client_mock = unittest.mock.Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    return client_mock


@pytest.fixture
def sagemaker_session(client):
    return Session(
        sagemaker_client=client,
    )


@pytest.fixture
def datetime_obj():
    return datetime.datetime(2017, 6, 16, 15, 55, 0)


def test_load(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.describe_experiment.return_value = {"Description": "description-value"}
    experiment_obj = experiment._Experiment.load(
        experiment_name="name-value", sagemaker_session=sagemaker_session
    )
    assert experiment_obj.experiment_name == "name-value"
    assert experiment_obj.description == "description-value"

    client.describe_experiment.assert_called_with(ExperimentName="name-value")


def test_create(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_experiment.return_value = {"Arn": "arn:aws:1234"}
    experiment_obj = experiment._Experiment.create(
        experiment_name="name-value", sagemaker_session=sagemaker_session
    )
    assert experiment_obj.experiment_name == "name-value"
    client.create_experiment.assert_called_with(ExperimentName="name-value")


def test_create_with_tags(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_experiment.return_value = {"Arn": "arn:aws:1234"}
    tags = [{"Key": "foo", "Value": "bar"}]
    experiment_obj = experiment._Experiment.create(
        experiment_name="name-value", sagemaker_session=sagemaker_session, tags=tags
    )
    assert experiment_obj.experiment_name == "name-value"
    client.create_experiment.assert_called_with(ExperimentName="name-value", Tags=tags)


def test_save(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    obj = experiment._Experiment(sagemaker_session, experiment_name="foo", description="bar")
    client.update_experiment.return_value = {}
    obj.save()
    client.update_experiment.assert_called_with(ExperimentName="foo", Description="bar")


def test_delete(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    obj = experiment._Experiment(sagemaker_session, experiment_name="foo", description="bar")
    client.delete_experiment.return_value = {}
    obj.delete()
    client.delete_experiment.assert_called_with(ExperimentName="foo")


@patch("sagemaker.experiments.experiment._Experiment.load")
def test_load_or_create_when_exist(mock_load, sagemaker_session):
    exp_name = "exp_name"
    experiment._Experiment._load_or_create(
        experiment_name=exp_name, sagemaker_session=sagemaker_session
    )
    mock_load.assert_called_once_with(exp_name, sagemaker_session)


@patch("sagemaker.experiments.experiment._Experiment.load")
@patch("sagemaker.experiments.experiment._Experiment.create")
def test_load_or_create_when_not_exist(mock_create, mock_load):
    sagemaker_session = Session()
    client = sagemaker_session.sagemaker_client
    exp_name = "exp_name"
    not_found_err = client.exceptions.ResourceNotFound(
        error_response={"Error": {"Code": "ResourceNotFound", "Message": "Not Found"}},
        operation_name="foo",
    )
    mock_load.side_effect = not_found_err

    experiment._Experiment._load_or_create(
        experiment_name=exp_name, sagemaker_session=sagemaker_session
    )

    mock_create.assert_called_once_with(
        experiment_name=exp_name,
        display_name=None,
        description=None,
        tags=None,
        sagemaker_session=sagemaker_session,
    )
