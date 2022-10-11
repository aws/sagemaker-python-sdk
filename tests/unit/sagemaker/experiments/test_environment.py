# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#       http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
import os
import shutil
import tempfile
import unittest.mock

import pytest

from sagemaker import Session
from sagemaker.experiments import _environment
from sagemaker.utils import retry_with_backoff


@pytest.fixture
def tempdir():
    dir = tempfile.mkdtemp()
    yield dir
    shutil.rmtree(dir)


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
def training_job_env():
    old_value = os.environ.get("TRAINING_JOB_ARN")
    os.environ["TRAINING_JOB_ARN"] = "arn:1234"
    yield os.environ
    del os.environ["TRAINING_JOB_ARN"]
    if old_value:
        os.environ["TRAINING_JOB_ARN"] = old_value


def test_processing_job_environment(tempdir):
    config_path = os.path.join(tempdir, "config.json")
    with open(config_path, "w") as f:
        f.write(json.dumps({"ProcessingJobArn": "arn:1234"}))
    environment = _environment._RunEnvironment.load(processing_job_config_path=config_path)

    assert _environment.EnvironmentType.SageMakerProcessingJob == environment.environment_type
    assert "arn:1234" == environment.source_arn


def test_training_job_environment(training_job_env):
    environment = _environment._RunEnvironment.load()
    assert _environment.EnvironmentType.SageMakerTrainingJob == environment.environment_type
    assert "arn:1234" == environment.source_arn


def test_no_environment():
    assert _environment._RunEnvironment.load() is None


def test_resolve_trial_component(training_job_env, sagemaker_session):
    trial_component_name = "foo-bar"
    client = sagemaker_session.sagemaker_client
    client.list_trial_components.return_value = {
        "TrialComponentSummaries": [{"TrialComponentName": trial_component_name}]
    }
    client.describe_trial_component.return_value = {"TrialComponentName": trial_component_name}
    environment = _environment._RunEnvironment.load()
    tc = environment.get_trial_component(sagemaker_session)

    assert trial_component_name == tc.trial_component_name
    client.describe_trial_component.assert_called_with(TrialComponentName=trial_component_name)


@unittest.mock.patch("sagemaker.experiments._environment.retry_with_backoff")
def test_resolve_trial_component_fails(mock_retry, sagemaker_session, training_job_env):
    mock_retry.side_effect = lambda func: retry_with_backoff(func, 2)
    client = sagemaker_session.sagemaker_client
    client.list_trial_components.side_effect = Exception("Failed test")
    environment = _environment._RunEnvironment.load()
    assert environment.get_trial_component(sagemaker_session) is None
