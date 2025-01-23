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
"""This module contains tests for MPI utility functions."""
from __future__ import absolute_import

import os
from unittest.mock import Mock, patch

import paramiko
import pytest

from sagemaker.modules.train.container_drivers.mpi_utils import (
    CustomHostKeyPolicy,
    _can_connect,
    bootstrap_master_node,
    bootstrap_worker_node,
    get_mpirun_command,
)


def test_custom_host_key_policy_algo_host():
    """Test CustomHostKeyPolicy accepts algo- hosts."""
    policy = CustomHostKeyPolicy()
    mock_client = Mock()
    mock_key = Mock()
    mock_key.get_name.return_value = "ssh-rsa"

    # Should not raise exception for algo- hostname
    policy.missing_host_key(mock_client, "algo-1234", mock_key)

    mock_client.get_host_keys.assert_called_once()
    mock_client.get_host_keys().add.assert_called_once_with("algo-1234", "ssh-rsa", mock_key)


def test_custom_host_key_policy_invalid_host():
    """Test CustomHostKeyPolicy rejects non-algo hosts."""
    policy = CustomHostKeyPolicy()
    mock_client = Mock()
    mock_key = Mock()

    with pytest.raises(paramiko.SSHException) as exc_info:
        policy.missing_host_key(mock_client, "invalid-host", mock_key)

    assert "Unknown host key for invalid-host" in str(exc_info.value)
    mock_client.get_host_keys.assert_not_called()


@patch("paramiko.SSHClient")
def test_can_connect_success(mock_ssh_client):
    """Test successful SSH connection."""
    mock_client = Mock()
    mock_ssh_client.return_value = mock_client

    assert _can_connect("algo-1234") is True
    mock_client.connect.assert_called_once()


@patch("paramiko.SSHClient")
def test_can_connect_failure(mock_ssh_client):
    """Test SSH connection failure."""
    mock_client = Mock()
    mock_ssh_client.return_value = mock_client
    mock_client.connect.side_effect = Exception("Connection failed")

    assert _can_connect("algo-1234") is False


def test_get_mpirun_command():
    """Test MPI command generation."""
    test_network_interface = "eth0"
    test_instance_type = "ml.p4d.24xlarge"

    with patch.dict(
        os.environ,
        {
            "SM_NETWORK_INTERFACE_NAME": test_network_interface,
            "SM_CURRENT_INSTANCE_TYPE": test_instance_type,
        },
    ):
        command = get_mpirun_command(
            host_count=2,
            host_list=["algo-1", "algo-2"],
            num_processes=2,
            additional_options=[],
            entry_script_path="train.py",
        )

        # Basic command structure checks
        assert command[0] == "mpirun"
        assert "--host" in command
        assert "algo-1,algo-2" in command
        assert "-np" in command
        assert "2" in command

        # Network interface check
        expected_nccl_config = f"NCCL_SOCKET_IFNAME={test_network_interface}"
        command_str = " ".join(command)
        assert expected_nccl_config in command_str


@patch("sagemaker.modules.train.container_drivers.mpi_utils._can_connect")
@patch("sagemaker.modules.train.container_drivers.mpi_utils._write_file_to_host")
def test_bootstrap_worker_node(mock_write, mock_connect):
    """Test worker node bootstrapping."""
    mock_connect.return_value = True
    mock_write.return_value = True
    os.environ["SM_CURRENT_HOST"] = "algo-2"

    with pytest.raises(TimeoutError):
        # Should timeout waiting for status file
        bootstrap_worker_node("algo-1", timeout=1)

    mock_connect.assert_called_with("algo-1")
    mock_write.assert_called_with("algo-1", "/tmp/ready.algo-2")


@patch("sagemaker.modules.train.container_drivers.mpi_utils._can_connect")
def test_bootstrap_master_node(mock_connect):
    """Test master node bootstrapping."""
    mock_connect.return_value = True

    with pytest.raises(TimeoutError):
        # Should timeout waiting for ready files
        bootstrap_master_node(["algo-2"], timeout=1)

    mock_connect.assert_called_with("algo-2")


if __name__ == "__main__":
    pytest.main([__file__])
