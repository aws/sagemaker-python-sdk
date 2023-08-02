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
# language governing permissions and limitations under the License.
from __future__ import absolute_import


import pytest
import test_data_helpers as tdh
from mock import Mock

from sagemaker.feature_store.feature_processor._env import EnvironmentHelper
from sagemaker.feature_store.feature_processor._params_loader import (
    ParamsLoader,
    SystemParamsLoader,
)


@pytest.fixture
def system_params_loader_mock():
    system_params_loader = Mock(SystemParamsLoader)
    system_params_loader.get_system_args.return_value = tdh.SYSTEM_PARAMS
    return system_params_loader


@pytest.fixture
def environment_checker():
    environment_checker = Mock(EnvironmentHelper)
    environment_checker.is_training_job.return_value = False
    environment_checker.get_job_scheduled_time = Mock(return_value="2023-05-05T15:22:57Z")
    return environment_checker


@pytest.fixture
def params_loader(system_params_loader_mock):
    return ParamsLoader(system_params_loader_mock)


@pytest.fixture
def system_params_loader(environment_checker):
    return SystemParamsLoader(environment_checker)


def test_get_parameter_args(params_loader, system_params_loader_mock):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.FEATURE_GROUP_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
        parameters=tdh.USER_INPUT_PARAMS,
    )

    params = params_loader.get_parameter_args(fp_config)

    system_params_loader_mock.get_system_args.assert_called_once()
    assert params == {"params": {**tdh.USER_INPUT_PARAMS, **tdh.SYSTEM_PARAMS}}


def test_get_parameter_args_with_no_user_params(params_loader, system_params_loader_mock):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
        parameters=None,
    )

    params = params_loader.get_parameter_args(fp_config)

    system_params_loader_mock.get_system_args.assert_called_once()
    assert params == {"params": {**tdh.SYSTEM_PARAMS}}


def test_get_system_arg_from_pipeline_execution(system_params_loader):
    system_params = system_params_loader.get_system_args()

    assert system_params == {
        "system": {
            "scheduled_time": "2023-05-05T15:22:57Z",
        }
    }
