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
"""Torchrun Driver Unit Tests."""
from __future__ import absolute_import

import os
import sys

from unittest.mock import patch, MagicMock

sys.modules["utils"] = MagicMock()

from sagemaker.modules.train.container_drivers import torchrun_driver  # noqa: E402

DUMMY_SOURCE_CODE = {
    "source_code": "source_code",
    "entry_script": "script.py",
}

DUMMY_distributed = {"_type": "torchrun", "process_count_per_node": 2}


@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.get_python_executable",
    return_value="python3",
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.pytorch_version", return_value=(2, 0)
)
def test_get_base_pytorch_command_torchrun(mock_pytorch_version, mock_get_python_executable):
    assert torchrun_driver.get_base_pytorch_command() == ["torchrun"]


@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.get_python_executable",
    return_value="python3",
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.pytorch_version", return_value=(1, 8)
)
def test_get_base_pytorch_command_torch_distributed_launch(
    mock_pytorch_version, mock_get_python_executable
):
    assert torchrun_driver.get_base_pytorch_command() == (
        ["python3", "-m", "torch.distributed.launch"]
    )


@patch.dict(
    os.environ,
    {
        "SM_CURRENT_INSTANCE_TYPE": "ml.p4d.24xlarge",
        "SM_NETWORK_INTERFACE_NAME": "eth0",
        "SM_HOST_COUNT": "1",
    },
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.USER_CODE_PATH",
    "/opt/ml/input/data/code",
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.get_process_count", return_value=2
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.pytorch_version", return_value=(2, 0)
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.get_base_pytorch_command",
    return_value=["torchrun"],
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.read_source_code_json",
    return_value=DUMMY_SOURCE_CODE,
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.read_distributed_json",
    return_value=DUMMY_distributed,
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.hyperparameters_to_cli_args",
    return_value=[],
)
def test_create_commands_single_node(
    mock_hyperparameters_to_cli_args,
    mock_read_distributed_json,
    mock_read_source_code_json,
    mock_get_base_pytorch_command,
    mock_pytorch_version,
    mock_get_process_count,
):
    expected_command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=2",
        "/opt/ml/input/data/code/script.py",
    ]

    command = torchrun_driver.create_commands()
    assert command == expected_command


@patch.dict(
    os.environ,
    {
        "SM_CURRENT_INSTANCE_TYPE": "ml.p4d.24xlarge",
        "SM_NETWORK_INTERFACE_NAME": "eth0",
        "SM_HOST_COUNT": "2",
        "SM_MASTER_ADDR": "algo-1",
        "SM_MASTER_PORT": "7777",
        "SM_CURRENT_HOST_RANK": "0",
    },
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.USER_CODE_PATH",
    "/opt/ml/input/data/code",
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.get_process_count", return_value=2
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.pytorch_version", return_value=(2, 0)
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.get_base_pytorch_command",
    return_value=["torchrun"],
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.read_source_code_json",
    return_value=DUMMY_SOURCE_CODE,
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.read_distributed_json",
    return_value=DUMMY_distributed,
)
@patch(
    "sagemaker.modules.train.container_drivers.torchrun_driver.hyperparameters_to_cli_args",
    return_value=[],
)
def test_create_commands_multi_node(
    mock_hyperparameters_to_cli_args,
    mock_read_distributed_json,
    mock_read_source_code_json,
    mock_get_base_pytorch_command,
    mock_pytorch_version,
    mock_get_process_count,
):
    expected_command = [
        "torchrun",
        "--nnodes=2",
        "--nproc_per_node=2",
        "--master_addr=algo-1",
        "--master_port=7777",
        "--node_rank=0",
        "/opt/ml/input/data/code/script.py",
    ]

    command = torchrun_driver.create_commands()
    assert command == expected_command
