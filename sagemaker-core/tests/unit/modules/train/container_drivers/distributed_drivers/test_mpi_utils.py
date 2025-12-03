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
"""Unit tests for sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils module."""
from __future__ import absolute_import

import pytest
import os
import subprocess
import paramiko
from unittest.mock import Mock, patch, MagicMock, call

from sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils import (
    _write_file_to_host,
    write_status_file_to_workers,
    _wait_for_status_file,
    start_sshd_daemon,
    CustomHostKeyPolicy,
    _can_connect,
    _wait_for_workers,
    _wait_for_master,
    bootstrap_worker_node,
    bootstrap_master_node,
    validate_smddprun,
    validate_smddpmprun,
    write_env_vars_to_file,
    get_mpirun_command,
    FINISHED_STATUS_FILE,
    READY_FILE,
    DEFAULT_SSH_PORT,
)


class TestWriteFileToHost:
    """Test _write_file_to_host function."""

    @patch("subprocess.run")
    def test_write_file_to_host_success(self, mock_run):
        """Test successful file write to host."""
        mock_run.return_value = Mock(returncode=0)

        result = _write_file_to_host("algo-1", "/tmp/test.txt")

        assert result is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_write_file_to_host_failure(self, mock_run):
        """Test failed file write to host."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "ssh")

        result = _write_file_to_host("algo-1", "/tmp/test.txt")

        assert result is False


class TestWriteStatusFileToWorkers:
    """Test write_status_file_to_workers function."""

    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._write_file_to_host"
    )
    def test_write_status_file_to_workers_success(self, mock_write):
        """Test writing status file to workers successfully."""
        mock_write.return_value = True

        write_status_file_to_workers(["algo-1", "algo-2"])

        assert mock_write.call_count == 2

    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._write_file_to_host"
    )
    @patch("time.sleep")
    def test_write_status_file_to_workers_with_retry(self, mock_sleep, mock_write):
        """Test writing status file with retry."""
        mock_write.side_effect = [False, False, True]

        write_status_file_to_workers(["algo-1"])

        assert mock_write.call_count == 3

    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._write_file_to_host"
    )
    @patch("time.sleep")
    def test_write_status_file_to_workers_timeout(self, mock_sleep, mock_write):
        """Test writing status file timeout."""
        mock_write.return_value = False

        with pytest.raises(TimeoutError):
            write_status_file_to_workers(["algo-1"])


class TestWaitForStatusFile:
    """Test _wait_for_status_file function."""

    @patch("os.path.exists")
    @patch("time.sleep")
    def test_wait_for_status_file_exists(self, mock_sleep, mock_exists):
        """Test waiting for status file that exists."""
        mock_exists.return_value = True

        _wait_for_status_file("/tmp/test.txt")

        mock_exists.assert_called_once()

    @patch("os.path.exists")
    @patch("time.sleep")
    def test_wait_for_status_file_eventually_exists(self, mock_sleep, mock_exists):
        """Test waiting for status file that eventually exists."""
        mock_exists.side_effect = [False, False, True]

        _wait_for_status_file("/tmp/test.txt")

        assert mock_exists.call_count == 3


class TestStartSshdDaemon:
    """Test start_sshd_daemon function."""

    @patch("os.path.exists")
    @patch("subprocess.Popen")
    def test_start_sshd_daemon_success(self, mock_popen, mock_exists):
        """Test starting SSH daemon successfully."""
        mock_exists.return_value = True

        start_sshd_daemon()

        mock_popen.assert_called_once_with(["/usr/sbin/sshd", "-D"])

    @patch("os.path.exists")
    def test_start_sshd_daemon_not_found(self, mock_exists):
        """Test starting SSH daemon when not found."""
        mock_exists.return_value = False

        with pytest.raises(RuntimeError, match="SSH daemon not found"):
            start_sshd_daemon()


class TestCustomHostKeyPolicy:
    """Test CustomHostKeyPolicy class."""

    def test_custom_host_key_policy_algo_hostname(self):
        """Test accepting algo-* hostnames."""
        policy = CustomHostKeyPolicy()
        mock_client = Mock()
        mock_client.get_host_keys.return_value = Mock()
        mock_key = Mock()
        mock_key.get_name.return_value = "ssh-rsa"

        # Should not raise exception
        policy.missing_host_key(mock_client, "algo-1234", mock_key)

    def test_custom_host_key_policy_unknown_hostname(self):
        """Test rejecting unknown hostnames."""
        policy = CustomHostKeyPolicy()
        mock_client = Mock()
        mock_key = Mock()

        with pytest.raises(paramiko.SSHException):
            policy.missing_host_key(mock_client, "unknown-host", mock_key)


class TestCanConnect:
    """Test _can_connect function."""

    @patch("paramiko.SSHClient")
    def test_can_connect_success(self, mock_ssh_client):
        """Test successful connection."""
        mock_client_instance = Mock()
        mock_ssh_client.return_value.__enter__.return_value = mock_client_instance

        result = _can_connect("algo-1")

        assert result is True

    @patch("paramiko.SSHClient")
    def test_can_connect_failure(self, mock_ssh_client):
        """Test failed connection."""
        mock_client_instance = Mock()
        mock_client_instance.connect.side_effect = Exception("Connection failed")
        mock_ssh_client.return_value.__enter__.return_value = mock_client_instance

        result = _can_connect("algo-1")

        assert result is False


class TestWaitForWorkers:
    """Test _wait_for_workers function."""

    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._can_connect"
    )
    @patch("os.path.exists")
    def test_wait_for_workers_success(self, mock_exists, mock_connect):
        """Test waiting for workers successfully."""
        mock_connect.return_value = True
        mock_exists.return_value = True

        _wait_for_workers(["algo-1", "algo-2"])

        assert mock_connect.call_count >= 2

    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._can_connect"
    )
    @patch("os.path.exists")
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_workers_timeout(self, mock_time, mock_sleep, mock_exists, mock_connect):
        """Test waiting for workers timeout."""
        mock_connect.return_value = False
        mock_exists.return_value = False
        # Use side_effect with a generator to provide unlimited values
        mock_time.side_effect = (i * 200 for i in range(1000))  # Simulate timeout

        with pytest.raises(TimeoutError):
            _wait_for_workers(["algo-1"])

    def test_wait_for_workers_empty_list(self):
        """Test waiting for workers with empty list."""
        # Should not raise exception
        _wait_for_workers([])


class TestWaitForMaster:
    """Test _wait_for_master function."""

    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._can_connect"
    )
    def test_wait_for_master_success(self, mock_connect):
        """Test waiting for master successfully."""
        mock_connect.return_value = True

        _wait_for_master("algo-1")

        mock_connect.assert_called()

    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._can_connect"
    )
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_master_timeout(self, mock_time, mock_sleep, mock_connect):
        """Test waiting for master timeout."""
        mock_connect.return_value = False
        # Use side_effect with a generator to provide unlimited values
        mock_time.side_effect = (i * 200 for i in range(1000))  # Simulate timeout

        with pytest.raises(TimeoutError):
            _wait_for_master("algo-1")


class TestBootstrapWorkerNode:
    """Test bootstrap_worker_node function."""

    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._wait_for_master"
    )
    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._write_file_to_host"
    )
    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._wait_for_status_file"
    )
    @patch.dict(os.environ, {"SM_CURRENT_HOST": "algo-2"})
    def test_bootstrap_worker_node(self, mock_wait_status, mock_write, mock_wait_master):
        """Test bootstrapping worker node."""
        mock_write.return_value = True

        bootstrap_worker_node("algo-1")

        mock_wait_master.assert_called_once_with("algo-1")
        mock_write.assert_called_once()
        mock_wait_status.assert_called_once()


class TestBootstrapMasterNode:
    """Test bootstrap_master_node function."""

    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils._wait_for_workers"
    )
    def test_bootstrap_master_node(self, mock_wait):
        """Test bootstrapping master node."""
        bootstrap_master_node(["algo-2", "algo-3"])

        mock_wait.assert_called_once_with(["algo-2", "algo-3"])


class TestValidateSmddprun:
    """Test validate_smddprun function."""

    @patch("subprocess.run")
    def test_validate_smddprun_installed(self, mock_run):
        """Test validating smddprun when installed."""
        mock_run.return_value = Mock(stdout="smddprun")

        result = validate_smddprun()

        assert result is True

    @patch("subprocess.run")
    def test_validate_smddprun_not_installed(self, mock_run):
        """Test validating smddprun when not installed."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "which")

        result = validate_smddprun()

        assert result is False


class TestValidateSmddpmprun:
    """Test validate_smddpmprun function."""

    @patch("subprocess.run")
    def test_validate_smddpmprun_installed(self, mock_run):
        """Test validating smddpmprun when installed."""
        mock_run.return_value = Mock(stdout="smddpmprun")

        result = validate_smddpmprun()

        assert result is True

    @patch("subprocess.run")
    def test_validate_smddpmprun_not_installed(self, mock_run):
        """Test validating smddpmprun when not installed."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "which")

        result = validate_smddpmprun()

        assert result is False


class TestWriteEnvVarsToFile:
    """Test write_env_vars_to_file function."""

    @patch("builtins.open", create=True)
    @patch.dict(os.environ, {"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"})
    def test_write_env_vars_to_file(self, mock_open_func):
        """Test writing environment variables to file."""
        mock_file = MagicMock()
        mock_open_func.return_value.__enter__.return_value = mock_file

        write_env_vars_to_file()

        mock_open_func.assert_called_once_with("/etc/environment", "a", encoding="utf-8")
        assert mock_file.write.called


class TestGetMpirunCommand:
    """Test get_mpirun_command function."""

    @patch.dict(
        os.environ,
        {"SM_NETWORK_INTERFACE_NAME": "eth0", "SM_CURRENT_INSTANCE_TYPE": "ml.p3.2xlarge"},
    )
    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils.get_python_executable"
    )
    def test_get_mpirun_command_basic(self, mock_python):
        """Test getting basic mpirun command."""
        mock_python.return_value = "/usr/bin/python3"

        result = get_mpirun_command(
            host_count=2,
            host_list=["algo-1", "algo-2"],
            num_processes=4,
            additional_options=[],
            entry_script_path="/opt/ml/code/train.py",
        )

        assert "mpirun" in result
        assert "--host" in result
        assert "algo-1,algo-2" in result
        assert "-np" in result
        assert "4" in result

    @patch.dict(
        os.environ,
        {
            "SM_NETWORK_INTERFACE_NAME": "eth0",
            "SM_CURRENT_INSTANCE_TYPE": "ml.p4d.24xlarge",
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
        },
    )
    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils.get_python_executable"
    )
    def test_get_mpirun_command_with_efa(self, mock_python):
        """Test getting mpirun command with EFA instance."""
        mock_python.return_value = "/usr/bin/python3"

        result = get_mpirun_command(
            host_count=2,
            host_list=["algo-1", "algo-2"],
            num_processes=4,
            additional_options=[],
            entry_script_path="/opt/ml/code/train.py",
        )

        assert "FI_PROVIDER=efa" in result

    @patch.dict(
        os.environ,
        {"SM_NETWORK_INTERFACE_NAME": "eth0", "SM_CURRENT_INSTANCE_TYPE": "ml.p3.2xlarge"},
    )
    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils.get_python_executable"
    )
    def test_get_mpirun_command_with_additional_options(self, mock_python):
        """Test getting mpirun command with additional options."""
        mock_python.return_value = "/usr/bin/python3"

        result = get_mpirun_command(
            host_count=2,
            host_list=["algo-1", "algo-2"],
            num_processes=4,
            additional_options=["-x", "CUSTOM_VAR"],
            entry_script_path="/opt/ml/code/train.py",
        )

        assert "-x" in result
        assert "CUSTOM_VAR" in result

    @patch.dict(
        os.environ,
        {
            "SM_NETWORK_INTERFACE_NAME": "eth0",
            "SM_CURRENT_INSTANCE_TYPE": "ml.p3.2xlarge",
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_SESSION_TOKEN": "test_token",
        },
    )
    @patch(
        "sagemaker.core.modules.train.container_drivers.distributed_drivers.mpi_utils.get_python_executable"
    )
    def test_get_mpirun_command_with_credentials(self, mock_python):
        """Test getting mpirun command with AWS credentials."""
        mock_python.return_value = "/usr/bin/python3"

        result = get_mpirun_command(
            host_count=2,
            host_list=["algo-1", "algo-2"],
            num_processes=4,
            additional_options=[],
            entry_script_path="/opt/ml/code/train.py",
        )

        assert "AWS_ACCESS_KEY_ID" in result
        assert "AWS_SECRET_ACCESS_KEY" in result
        assert "AWS_SESSION_TOKEN" in result
