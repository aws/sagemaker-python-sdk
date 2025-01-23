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
"""MPI Utils Unit Tests."""
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
    write_status_file_to_workers,
)

TEST_HOST = "algo-1"
TEST_WORKER = "algo-2"
TEST_STATUS_FILE = "/tmp/test-status"


def test_custom_host_key_policy_valid_hostname():
    """Test CustomHostKeyPolicy with valid algo- hostname."""
    policy = CustomHostKeyPolicy()
    mock_client = Mock()
    mock_key = Mock()
    mock_key.get_name.return_value = "ssh-rsa"

    policy.missing_host_key(mock_client, "algo-1", mock_key)

    mock_client.get_host_keys.assert_called_once()
    mock_client.get_host_keys().add.assert_called_once_with("algo-1", "ssh-rsa", mock_key)


def test_custom_host_key_policy_invalid_hostname():
    """Test CustomHostKeyPolicy with invalid hostname."""
    policy = CustomHostKeyPolicy()
    mock_client = Mock()
    mock_key = Mock()

    with pytest.raises(paramiko.SSHException) as exc_info:
        policy.missing_host_key(mock_client, "invalid-1", mock_key)

    assert "Unknown host key for invalid-1" in str(exc_info.value)
    mock_client.get_host_keys.assert_not_called()


@patch("paramiko.SSHClient")
def test_can_connect_success(mock_ssh_client):
    """Test successful SSH connection."""
    mock_client = Mock()
    mock_ssh_client.return_value = mock_client

    assert _can_connect(TEST_HOST) is True
    mock_client.connect.assert_called_once_with(TEST_HOST, port=22)


@patch("paramiko.SSHClient")
def test_can_connect_failure(mock_ssh_client):
    """Test SSH connection failure."""
    mock_client = Mock()
    mock_ssh_client.return_value = mock_client
    mock_client.connect.side_effect = Exception("Connection failed")

    assert _can_connect(TEST_HOST) is False


@patch("subprocess.run")
def test_write_status_file_to_workers_success(mock_run):
    """Test successful status file writing to workers."""
    mock_run.return_value = Mock(returncode=0)

    write_status_file_to_workers([TEST_WORKER], TEST_STATUS_FILE)

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args == ["ssh", TEST_WORKER, "touch", TEST_STATUS_FILE]


@patch("subprocess.run")
def test_write_status_file_to_workers_failure(mock_run):
    """Test failed status file writing to workers with retry timeout."""
    mock_run.side_effect = Exception("SSH failed")

    with pytest.raises(TimeoutError) as exc_info:
        write_status_file_to_workers([TEST_WORKER], TEST_STATUS_FILE)

    assert f"Timed out waiting for {TEST_WORKER}" in str(exc_info.value)


def test_get_mpirun_command_basic():
    """Test basic MPI command generation."""
    with patch.dict(
        os.environ,
        {"SM_NETWORK_INTERFACE_NAME": "eth0", "SM_CURRENT_INSTANCE_TYPE": "ml.p3.16xlarge"},
    ):
        command = get_mpirun_command(
            host_count=2,
            host_list=[TEST_HOST, TEST_WORKER],
            num_processes=2,
            additional_options=[],
            entry_script_path="train.py",
        )

        assert command[0] == "mpirun"
        assert "--host" in command
        assert f"{TEST_HOST},{TEST_WORKER}" in command
        assert "-np" in command
        assert "2" in command


def test_get_mpirun_command_efa():
    """Test MPI command generation with EFA instance."""
    with patch.dict(
        os.environ,
        {"SM_NETWORK_INTERFACE_NAME": "eth0", "SM_CURRENT_INSTANCE_TYPE": "ml.p4d.24xlarge"},
    ):
        command = get_mpirun_command(
            host_count=2,
            host_list=[TEST_HOST, TEST_WORKER],
            num_processes=2,
            additional_options=[],
            entry_script_path="train.py",
        )

        command_str = " ".join(command)
        assert "FI_PROVIDER=efa" in command_str
        assert "NCCL_PROTO=simple" in command_str


@patch("sagemaker.modules.train.container_drivers.mpi_utils._can_connect")
@patch("sagemaker.modules.train.container_drivers.mpi_utils._write_file_to_host")
def test_bootstrap_worker_node(mock_write, mock_connect):
    """Test worker node bootstrap process."""
    mock_connect.return_value = True
    mock_write.return_value = True

    with patch.dict(os.environ, {"SM_CURRENT_HOST": TEST_WORKER}):
        with pytest.raises(TimeoutError):
            bootstrap_worker_node(TEST_HOST, timeout=1)

    mock_connect.assert_called_with(TEST_HOST)
    mock_write.assert_called_with(TEST_HOST, f"/tmp/ready.{TEST_WORKER}")


@patch("sagemaker.modules.train.container_drivers.mpi_utils._can_connect")
def test_bootstrap_master_node(mock_connect):
    """Test master node bootstrap process."""
    mock_connect.return_value = True

    with pytest.raises(TimeoutError):
        bootstrap_master_node([TEST_WORKER], timeout=1)

    mock_connect.assert_called_with(TEST_WORKER)


if __name__ == "__main__":
    pytest.main([__file__])
