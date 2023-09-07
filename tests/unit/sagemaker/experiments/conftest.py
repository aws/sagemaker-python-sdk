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

import unittest
from unittest.mock import patch, MagicMock, Mock

import pytest

from sagemaker import Session
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments.run import RUN_NAME_BASE
from sagemaker.experiments import Run
from tests.unit.sagemaker.experiments.helpers import (
    mock_tc_load_or_create_func,
    mock_trial_load_or_create_func,
    TEST_EXP_NAME,
)


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
def run_obj(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.update_trial_component.return_value = {}
    client.associate_trial_component.return_value = {}
    with patch(
        "sagemaker.experiments.run.Experiment._load_or_create",
        MagicMock(
            return_value=Experiment(
                experiment_name=TEST_EXP_NAME, sagemaker_session=sagemaker_session
            )
        ),
    ):
        with patch(
            "sagemaker.experiments.run._TrialComponent._load_or_create",
            MagicMock(side_effect=mock_tc_load_or_create_func),
        ):
            with patch(
                "sagemaker.experiments.run._Trial._load_or_create",
                MagicMock(side_effect=mock_trial_load_or_create_func),
            ):
                sagemaker_session.sagemaker_client.search.return_value = {"Results": []}
                run = Run(
                    experiment_name=TEST_EXP_NAME,
                    sagemaker_session=sagemaker_session,
                )
                run._artifact_uploader = Mock()
                run._lineage_artifact_tracker = Mock()
                run._metrics_manager = Mock()

                assert run.run_name.startswith(RUN_NAME_BASE)
                assert run.run_group_name == Run._generate_trial_name(TEST_EXP_NAME)

                return run
