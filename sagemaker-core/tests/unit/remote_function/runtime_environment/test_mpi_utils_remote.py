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
"""Tests for mpi_utils_remote module."""
from __future__ import absolute_import

import os
import pytest
import subprocess
import time
from unittest.mock import patch, MagicMock, mock_open, call
import paramiko

from sagemaker.core.remote_function.runtime_environment.mpi_utils_remote import (
    CustomHostKeyPolicy,
    _parse_args,
    _can_connect,
    _write_file_to_host,
    _write_failure_reason_file,
    _wait_for_master,
    _wait_for_status_file,
    _wait_for_workers,
    bootstrap_master_node,
    bootstrap_worker_node,
    start_sshd_daemon,
    write_status_file_to_workers,
    main,
    SUCCESS_EXIT_CODE,
    DEFAULT_FAILURE_CODE,
    FAILURE_REASON_PATH,
    FINISHED_STATUS_FILE,
    READY_FILE,
    DEFAULT_SSH_PORT,
)


class TestCustomHostKeyPolicy:
    """Test CustomHostKeyPolicy class."""

    def test_accepts_algo_hostname(self):
        """Test accepts hostnames starting with algo-."""
        policy = CustomHostKeyPolicy()
        mock_client = MagicMock()
        mock_hostname = "algo-1234"
        mock_key = MagicMock()
        mock_key.get_name.return_value = "ssh-rsa"
        
        # Should not raise exception
        policy.missing_host_key(mock_client, mock_hostname, mock_key)
        
        mock_client.get_host_keys().add.assert_called_once_with(mock_hostname, "ssh-rsa", mock_key)

    def test_rejects_non_algo_hostname(self):
        """Test rejects hostnames not starting with algo-."""
        policy = CustomHostKeyPolicy()
        mock_client = MagicMock()
        mock_hostname = "unknown-host"
        mock_key = MagicMock()
        
        with pytest.raises(paramiko.SSHException):
            policy.missing_host_key(mock_client, mock_hostname, mock_key)


class TestParseArgs:
    """Test _parse_args function."""

    def test_parse_default_args(self):
        """Test parsing with default arguments."""
        args = []
        parsed = _parse_args(args)
        assert parsed.job_ended == "0"

    def test_parse_job_ended_true(self):
        """Test parsing with job_ended set to true."""
        args = ["--job_ended", "1"]
        parsed = _parse_args(args)
        assert parsed.job_ended == "1"

    def test_parse_job_ended_false(self):
        """Test parsing with job_ended set to false."""
        args = ["--job_ended", "0"]
        parsed = _parse_args(args)
        assert parsed.job_ended == "0"


class TestCanConnect:
    """Test _can_connect function."""

    @patch("paramiko.SSHClient")
    def test_can_connect_success(self, mock_ssh_client_class):
        """Test successful connection."""
        mock_client = MagicMock()
        mock_ssh_client_class.return_value.__enter__.return_value = mock_client
        
        result = _can_connect("algo-1", DEFAULT_SSH_PORT)
        
        assert result is True
        mock_client.connect.assert_called_once_with("algo-1", port=DEFAULT_SSH_PORT)

    @patch("paramiko.SSHClient")
    def test_can_connect_failure(self, mock_ssh_client_class):
        """Test failed connection."""
        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception("Connection failed")
        mock_ssh_client_class.return_value.__enter__.return_value = mock_client
        
        result = _can_connect("algo-1", DEFAULT_SSH_PORT)
        
        assert result is False

    @patch("paramiko.SSHClient")
    def test_can_connect_uses_custom_port(self, mock_ssh_client_class):
        """Test connection with custom port."""
        mock_client = MagicMock()
        mock_ssh_client_class.return_value.__enter__.return_value = mock_client
        
        _can_connect("algo-1", 2222)
        
        mock_client.connect.assert_called_once_with("algo-1", port=2222)


class TestWriteFileToHost:
    """Test _write_file_to_host function."""

    @patch("subprocess.run")
    def test_write_file_success(self, mock_run):
        """Test successful file write."""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = _write_file_to_host("algo-1", "/tmp/status")
        
        assert result is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_write_file_failure(self, mock_run):
        """Test failed file write."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "ssh")
        
        result = _write_file_to_host("algo-1", "/tmp/status")
        
        assert result is False


class TestWriteFailureReasonFile:
    """Test _write_failure_reason_file function."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_writes_failure_file(self, mock_exists, mock_file):
        """Test writes failure reason file."""
        mock_exists.return_value = False
        
        _write_failure_reason_file("Test error message")
        
        mock_file.assert_called_once_with(FAILURE_REASON_PATH, "w")
        mock_file().write.assert_called_once_with("RuntimeEnvironmentError: Test error message")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_does_not_write_if_exists(self, mock_exists, mock_file):
        """Test does not write if failure file already exists."""
        mock_exists.return_value = True
        
        _write_failure_reason_file("Test error message")
        
        mock_file.assert_not_called()


class TestWaitForMaster:
    """Test _wait_for_master function."""

    @patch("time.sleep")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    def test_wait_for_master_success(self, mock_can_connect, mock_sleep):
        """Test successful wait for master."""
        mock_can_connect.return_value = True
        
        _wait_for_master("algo-1", DEFAULT_SSH_PORT, timeout=300)
        
        mock_can_connect.assert_called_once_with("algo-1", DEFAULT_SSH_PORT)

    @patch("time.time")
    @patch("time.sleep")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    def test_wait_for_master_timeout(self, mock_can_connect, mock_sleep, mock_time):
        """Test timeout waiting for master."""
        mock_can_connect.return_value = False
        # Need enough values for all time.time() calls in the loop
        mock_time.side_effect = [0] + [i * 5 for i in range(1, 100)]  # Simulate time passing
        
        with pytest.raises(TimeoutError):
            _wait_for_master("algo-1", DEFAULT_SSH_PORT, timeout=300)

    @patch("time.time")
    @patch("time.sleep")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    def test_wait_for_master_retries(self, mock_can_connect, mock_sleep, mock_time):
        """Test retries before successful connection."""
        mock_can_connect.side_effect = [False, False, True]
        # Return value instead of side_effect for time.time()
        mock_time.return_value = 0
        
        _wait_for_master("algo-1", DEFAULT_SSH_PORT, timeout=300)
        
        assert mock_can_connect.call_count == 3


class TestWaitForStatusFile:
    """Test _wait_for_status_file function."""

    @patch("time.sleep")
    @patch("os.path.exists")
    def test_wait_for_status_file_exists(self, mock_exists, mock_sleep):
        """Test wait for status file that exists."""
        mock_exists.return_value = True
        
        _wait_for_status_file("/tmp/status")
        
        mock_exists.assert_called_once_with("/tmp/status")

    @patch("time.sleep")
    @patch("os.path.exists")
    def test_wait_for_status_file_waits(self, mock_exists, mock_sleep):
        """Test waits until status file exists."""
        mock_exists.side_effect = [False, False, True]
        
        _wait_for_status_file("/tmp/status")
        
        assert mock_exists.call_count == 3
        assert mock_sleep.call_count == 2


class TestWaitForWorkers:
    """Test _wait_for_workers function."""

    @patch("os.path.exists")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    def test_wait_for_workers_empty_list(self, mock_can_connect, mock_exists):
        """Test wait for workers with empty list."""
        _wait_for_workers([], DEFAULT_SSH_PORT, timeout=300)
        
        mock_can_connect.assert_not_called()

    @patch("time.sleep")
    @patch("os.path.exists")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    def test_wait_for_workers_success(self, mock_can_connect, mock_exists, mock_sleep):
        """Test successful wait for workers."""
        mock_can_connect.return_value = True
        mock_exists.return_value = True
        
        _wait_for_workers(["algo-2", "algo-3"], DEFAULT_SSH_PORT, timeout=300)
        
        assert mock_can_connect.call_count == 2

    @patch("time.time")
    @patch("time.sleep")
    @patch("os.path.exists")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    def test_wait_for_workers_timeout(self, mock_can_connect, mock_exists, mock_sleep, mock_time):
        """Test timeout waiting for workers."""
        mock_can_connect.return_value = False
        mock_exists.return_value = False
        # Need enough values for all time.time() calls in the loop
        mock_time.side_effect = [0] + [i * 5 for i in range(1, 100)]
        
        with pytest.raises(TimeoutError):
            _wait_for_workers(["algo-2"], DEFAULT_SSH_PORT, timeout=300)


class TestBootstrapMasterNode:
    """Test bootstrap_master_node function."""

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._wait_for_workers")
    def test_bootstrap_master_node(self, mock_wait):
        """Test bootstrap master node."""
        worker_hosts = ["algo-2", "algo-3"]
        
        bootstrap_master_node(worker_hosts)
        
        mock_wait.assert_called_once_with(worker_hosts)


class TestBootstrapWorkerNode:
    """Test bootstrap_worker_node function."""

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._wait_for_status_file")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._write_file_to_host")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._wait_for_master")
    def test_bootstrap_worker_node(self, mock_wait_master, mock_write, mock_wait_status):
        """Test bootstrap worker node."""
        bootstrap_worker_node("algo-1", "algo-2", "/tmp/status")
        
        mock_wait_master.assert_called_once_with("algo-1")
        mock_write.assert_called_once()
        mock_wait_status.assert_called_once_with("/tmp/status")


class TestStartSshdDaemon:
    """Test start_sshd_daemon function."""

    @patch("subprocess.Popen")
    @patch("os.path.exists")
    def test_starts_sshd_successfully(self, mock_exists, mock_popen):
        """Test starts SSH daemon successfully."""
        mock_exists.return_value = True
        
        start_sshd_daemon()
        
        mock_popen.assert_called_once_with(["/usr/sbin/sshd", "-D"])

    @patch("os.path.exists")
    def test_raises_error_if_sshd_not_found(self, mock_exists):
        """Test raises error if SSH daemon not found."""
        mock_exists.return_value = False
        
        with pytest.raises(RuntimeError):
            start_sshd_daemon()


class TestWriteStatusFileToWorkers:
    """Test write_status_file_to_workers function."""

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._write_file_to_host")
    def test_writes_to_all_workers(self, mock_write):
        """Test writes status file to all workers."""
        mock_write.return_value = True
        worker_hosts = ["algo-2", "algo-3"]
        
        write_status_file_to_workers(worker_hosts, "/tmp/status")
        
        assert mock_write.call_count == 2

    @patch("time.sleep")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._write_file_to_host")
    def test_retries_on_failure(self, mock_write, mock_sleep):
        """Test retries writing status file on failure."""
        mock_write.side_effect = [False, False, True]
        worker_hosts = ["algo-2"]
        
        write_status_file_to_workers(worker_hosts, "/tmp/status")
        
        assert mock_write.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._write_file_to_host")
    def test_raises_timeout_after_retries(self, mock_write, mock_sleep):
        """Test raises timeout after max retries."""
        mock_write.return_value = False
        worker_hosts = ["algo-2"]
        
        with pytest.raises(TimeoutError):
            write_status_file_to_workers(worker_hosts, "/tmp/status")


class TestMain:
    """Test main function."""

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.bootstrap_worker_node")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.start_sshd_daemon")
    @patch.dict("os.environ", {"SM_MASTER_ADDR": "algo-1", "SM_CURRENT_HOST": "algo-2"})
    def test_main_worker_node_running(self, mock_start_sshd, mock_bootstrap_worker):
        """Test main function for worker node during job run."""
        args = ["--job_ended", "0"]
        
        main(args)
        
        mock_start_sshd.assert_called_once()
        mock_bootstrap_worker.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.bootstrap_master_node")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.start_sshd_daemon")
    @patch.dict("os.environ", {"SM_MASTER_ADDR": "algo-1", "SM_CURRENT_HOST": "algo-1", "SM_HOSTS": '["algo-1", "algo-2"]'})
    def test_main_master_node_running(self, mock_start_sshd, mock_bootstrap_master):
        """Test main function for master node during job run."""
        args = ["--job_ended", "0"]
        
        main(args)
        
        mock_start_sshd.assert_called_once()
        mock_bootstrap_master.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.write_status_file_to_workers")
    @patch.dict("os.environ", {"SM_MASTER_ADDR": "algo-1", "SM_CURRENT_HOST": "algo-1", "SM_HOSTS": '["algo-1", "algo-2"]'})
    def test_main_master_node_job_ended(self, mock_write_status):
        """Test main function for master node after job ends."""
        args = ["--job_ended", "1"]
        
        main(args)
        
        mock_write_status.assert_called_once()

    @patch.dict("os.environ", {"SM_MASTER_ADDR": "algo-1", "SM_CURRENT_HOST": "algo-2"})
    def test_main_worker_node_job_ended(self):
        """Test main function for worker node after job ends."""
        args = ["--job_ended", "1"]
        
        # Should not raise any exceptions
        main(args)

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._write_failure_reason_file")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.start_sshd_daemon")
    @patch.dict("os.environ", {"SM_MASTER_ADDR": "algo-1", "SM_CURRENT_HOST": "algo-2"})
    def test_main_handles_exception(self, mock_start_sshd, mock_write_failure):
        """Test main function handles exceptions."""
        mock_start_sshd.side_effect = Exception("Test error")
        args = ["--job_ended", "0"]
        
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        
        assert exc_info.value.code == DEFAULT_FAILURE_CODE
        mock_write_failure.assert_called_once()
