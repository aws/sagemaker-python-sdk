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

from sagemaker.experiments import _environment
from sagemaker.experiments._environment import TRANSFORM_JOB_ARN_ENV, TRAINING_JOB_ARN_ENV
from sagemaker.utils import retry_with_backoff


@pytest.fixture
def tempdir():
    dir = tempfile.mkdtemp()
    yield dir
    shutil.rmtree(dir)


@pytest.fixture
def training_job_env():
    old_value = os.environ.get(TRAINING_JOB_ARN_ENV)
    os.environ[TRAINING_JOB_ARN_ENV] = "arn:1234aBcDe"
    yield os.environ
    del os.environ[TRAINING_JOB_ARN_ENV]
    if old_value:
        os.environ[TRAINING_JOB_ARN_ENV] = old_value


@pytest.fixture
def transform_job_env():
    old_value = os.environ.get(TRANSFORM_JOB_ARN_ENV)
    os.environ[TRANSFORM_JOB_ARN_ENV] = "arn:1234aBcDe"
    yield os.environ
    del os.environ[TRANSFORM_JOB_ARN_ENV]
    if old_value:
        os.environ[TRANSFORM_JOB_ARN_ENV] = old_value


def test_processing_job_environment(tempdir):
    config_path = os.path.join(tempdir, "config.json")
    with open(config_path, "w") as f:
        f.write(json.dumps({"ProcessingJobArn": "arn:1234aBcDe"}))
    environment = _environment._RunEnvironment.load(processing_job_config_path=config_path)

    assert _environment._EnvironmentType.SageMakerProcessingJob == environment.environment_type
    assert "arn:1234aBcDe" == environment.source_arn


def test_training_job_environment(training_job_env):
    environment = _environment._RunEnvironment.load()
    assert _environment._EnvironmentType.SageMakerTrainingJob == environment.environment_type
    assert "arn:1234aBcDe" == environment.source_arn


def test_transform_job_environment(transform_job_env):
    environment = _environment._RunEnvironment.load()
    assert _environment._EnvironmentType.SageMakerTransformJob == environment.environment_type
    assert "arn:1234aBcDe" == environment.source_arn


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
    client.list_trial_components.assert_called_with(SourceArn="arn:1234abcde")


@unittest.mock.patch("sagemaker.experiments._environment.retry_with_backoff")
def test_resolve_trial_component_fails(mock_retry, sagemaker_session, training_job_env):
    mock_retry.side_effect = lambda func: retry_with_backoff(func, 2)
    client = sagemaker_session.sagemaker_client
    client.list_trial_components.side_effect = Exception("Failed test")
    environment = _environment._RunEnvironment.load()
    assert environment.get_trial_component(sagemaker_session) is None
