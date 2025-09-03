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

import subprocess
from unittest.mock import Mock, patch

import paramiko
import pytest

# Mock the utils module before importing mpi_utils
mock_utils = Mock()
mock_utils.logger = Mock()
mock_utils.SM_EFA_NCCL_INSTANCES = []
mock_utils.SM_EFA_RDMA_INSTANCES = []
mock_utils.get_python_executable = Mock(return_value="/usr/bin/python")

with patch.dict("sys.modules", {"utils": mock_utils}):
    from sagemaker.modules.train.container_drivers.distributed_drivers.mpi_utils import (
        CustomHostKeyPolicy,
        _can_connect,
        write_status_file_to_workers,
    )

TEST_HOST = "algo-1"
TEST_WORKER = "algo-2"
TEST_STATUS_FILE = "/tmp/test-status"


def test_custom_host_key_policy_valid_hostname():
    """Test CustomHostKeyPolicy accepts algo- prefixed hostnames."""
    policy = CustomHostKeyPolicy()
    mock_client = Mock()
    mock_key = Mock()
    mock_key.get_name.return_value = "ssh-rsa"

    policy.missing_host_key(mock_client, "algo-1", mock_key)

    mock_client.get_host_keys.assert_called_once()
    mock_client.get_host_keys().add.assert_called_once_with("algo-1", "ssh-rsa", mock_key)


def test_custom_host_key_policy_invalid_hostname():
    """Test CustomHostKeyPolicy rejects non-algo prefixed hostnames."""
    policy = CustomHostKeyPolicy()
    mock_client = Mock()
    mock_key = Mock()

    with pytest.raises(paramiko.SSHException) as exc_info:
        policy.missing_host_key(mock_client, "invalid-1", mock_key)

    assert "Unknown host key for invalid-1" in str(exc_info.value)
    mock_client.get_host_keys.assert_not_called()


@patch("paramiko.SSHClient")
@patch("sagemaker.modules.train.container_drivers.distributed_drivers.mpi_utils.logger")
def test_can_connect_success(mock_logger, mock_ssh_client):
    """Test successful SSH connection."""
    mock_client = Mock()
    mock_ssh_client.return_value.__enter__.return_value = mock_client
    mock_client.connect.return_value = None  # Successful connection

    result = _can_connect(TEST_HOST)

    assert result is True
    mock_client.load_system_host_keys.assert_called_once()
    mock_client.set_missing_host_key_policy.assert_called_once()
    mock_client.connect.assert_called_once_with(TEST_HOST, port=22)


@patch("paramiko.SSHClient")
@patch("sagemaker.modules.train.container_drivers.distributed_drivers.mpi_utils.logger")
def test_can_connect_failure(mock_logger, mock_ssh_client):
    """Test SSH connection failure."""
    mock_client = Mock()
    mock_ssh_client.return_value.__enter__.return_value = mock_client
    mock_client.connect.side_effect = paramiko.SSHException("Connection failed")

    result = _can_connect(TEST_HOST)

    assert result is False
    mock_client.load_system_host_keys.assert_called_once()
    mock_client.set_missing_host_key_policy.assert_called_once()
    mock_client.connect.assert_called_once_with(TEST_HOST, port=22)


@patch("subprocess.run")
@patch("sagemaker.modules.train.container_drivers.distributed_drivers.mpi_utils.logger")
def test_write_status_file_to_workers_failure(mock_logger, mock_run):
    """Test failed status file writing to workers with retry timeout."""
    mock_run.side_effect = subprocess.CalledProcessError(1, "ssh")

    with pytest.raises(TimeoutError) as exc_info:
        write_status_file_to_workers([TEST_WORKER], TEST_STATUS_FILE)

    assert f"Timed out waiting for {TEST_WORKER}" in str(exc_info.value)
    assert mock_run.call_count > 1  # Verifies that retries occurred


if __name__ == "__main__":
    pytest.main([__file__])
