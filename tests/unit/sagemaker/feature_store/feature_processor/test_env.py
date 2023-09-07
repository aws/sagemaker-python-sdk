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

import json

from mock import mock_open, patch
import pytest
from sagemaker.feature_store.feature_processor._env import EnvironmentHelper

SINGLE_NODE_RESOURCE_CONFIG = {
    "current_host": "algo-1",
    "current_instance_type": "ml.m5.xlarge",
    "current_group_name": "homogeneousCluster",
    "hosts": ["algo-1"],
    "instance_groups": [
        {
            "instance_group_name": "homogeneousCluster",
            "instance_type": "ml.m5.xlarge",
            "hosts": ["algo-1"],
        }
    ],
    "network_interface_name": "eth0",
}
MULTI_NODE_COUNT = 3
MULTI_NODE_RESOURCE_CONFIG = {
    "current_host": "algo-1",
    "current_instance_type": "ml.m5.xlarge",
    "current_group_name": "homogeneousCluster",
    "hosts": ["algo-1", "algo-2", "algo-3"],
    "instance_groups": [
        {
            "instance_group_name": "homogeneousCluster",
            "instance_type": "ml.m5.xlarge",
            "hosts": ["algo-1"],
        },
        {
            "instance_group_name": "homogeneousCluster",
            "instance_type": "ml.m5.xlarge",
            "hosts": ["algo-2"],
        },
        {
            "instance_group_name": "homogeneousCluster",
            "instance_type": "ml.m5.xlarge",
            "hosts": ["algo-3"],
        },
    ],
    "network_interface_name": "eth0",
}


@patch("builtins.open")
def test_is_training_job(mocked_open):
    mocked_open.side_effect = mock_open(read_data=json.dumps(SINGLE_NODE_RESOURCE_CONFIG))

    assert EnvironmentHelper().is_training_job() is True

    mocked_open.assert_called_once_with("/opt/ml/input/config/resourceconfig.json", "r")


@patch("builtins.open")
def test_is_not_training_job(mocked_open):
    mocked_open.side_effect = FileNotFoundError()

    assert EnvironmentHelper().is_training_job() is False


@patch("builtins.open")
def test_get_instance_count_single_node(mocked_open):
    mocked_open.side_effect = mock_open(read_data=json.dumps(SINGLE_NODE_RESOURCE_CONFIG))

    assert EnvironmentHelper().get_instance_count() == 1


@patch("builtins.open")
def test_get_instance_count_multi_node(mocked_open):
    mocked_open.side_effect = mock_open(read_data=json.dumps(MULTI_NODE_RESOURCE_CONFIG))

    assert EnvironmentHelper().get_instance_count() == MULTI_NODE_COUNT


@patch("builtins.open")
def test_load_training_resource_config(mocked_open):
    mocked_open.side_effect = mock_open(read_data=json.dumps(SINGLE_NODE_RESOURCE_CONFIG))

    assert EnvironmentHelper().load_training_resource_config() == SINGLE_NODE_RESOURCE_CONFIG


@patch("builtins.open")
def test_load_training_resource_config_none(mocked_open):
    mocked_open.side_effect = FileNotFoundError()

    assert EnvironmentHelper().load_training_resource_config() is None


@pytest.mark.parametrize(
    "is_training_result",
    [(True), (False)],
)
@patch("datetime.now.strftime", return_value="test_current_time")
@patch("sagemaker.feature_store.feature_processor._env.EnvironmentHelper.is_training_job")
@patch("os.environ", return_value={"scheduled_time": "test_time"})
def get_job_scheduled_time(mock_env, mock_is_training, mock_datetime, is_training_result):

    mock_is_training.return_value = is_training_result
    output_time = EnvironmentHelper().get_job_scheduled_time

    if is_training_result:
        assert output_time == "test_scheduled_time"
    else:
        assert output_time == "test_current_time"
