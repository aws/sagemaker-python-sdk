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

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import subprocess
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
    FINISHED_STATUS_FILE,
    READY_FILE,
    DEFAULT_SSH_PORT,
)


class TestCustomHostKeyPolicy:
    """Test cases for CustomHostKeyPolicy class"""

    def test_missing_host_key_algo_hostname(self):
        """Test missing_host_key accepts algo-* hostnames"""
        policy = CustomHostKeyPolicy()
        client = Mock()
        client.get_host_keys.return_value = Mock()
        key = Mock()
        key.get_name.return_value = "ssh-rsa"

        # Should not raise exception
        policy.missing_host_key(client, "algo-1", key)

        client.get_host_keys().add.assert_called_once()

    def test_missing_host_key_unknown_hostname(self):
        """Test missing_host_key rejects unknown hostnames"""
        policy = CustomHostKeyPolicy()
        client = Mock()
        key = Mock()

        with pytest.raises(paramiko.SSHException, match="Unknown host key"):
            policy.missing_host_key(client, "unknown-host", key)


class TestConnectionFunctions:
    """Test cases for connection functions"""

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.paramiko.SSHClient")
    def test_can_connect_success(self, mock_ssh_client_class):
        """Test _can_connect when connection succeeds"""
        mock_client = Mock()
        mock_ssh_client_class.return_value.__enter__.return_value = mock_client

        result = _can_connect("algo-1", DEFAULT_SSH_PORT)

        assert result is True
        mock_client.connect.assert_called_once_with("algo-1", port=DEFAULT_SSH_PORT)

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.paramiko.SSHClient")
    def test_can_connect_failure(self, mock_ssh_client_class):
        """Test _can_connect when connection fails"""
        mock_client = Mock()
        mock_client.connect.side_effect = Exception("Connection failed")
        mock_ssh_client_class.return_value.__enter__.return_value = mock_client

        result = _can_connect("algo-1", DEFAULT_SSH_PORT)

        assert result is False

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.subprocess.run")
    def test_write_file_to_host_success(self, mock_run):
        """Test _write_file_to_host when write succeeds"""
        mock_run.return_value = Mock()

        result = _write_file_to_host("algo-1", "/tmp/status")

        assert result is True
        mock_run.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.subprocess.run")
    def test_write_file_to_host_failure(self, mock_run):
        """Test _write_file_to_host when write fails"""
        mock_run.side_effect = subprocess.CalledProcessError(1, "ssh")

        result = _write_file_to_host("algo-1", "/tmp/status")

        assert result is False

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_failure_reason_file(self, mock_file, mock_exists):
        """Test _write_failure_reason_file"""
        mock_exists.return_value = False

        _write_failure_reason_file("Test error")

        mock_file.assert_called_once()
        mock_file().write.assert_called_once_with("RuntimeEnvironmentError: Test error")


class TestWaitFunctions:
    """Test cases for wait functions"""

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.time.sleep")
    def test_wait_for_master_success(self, mock_sleep, mock_can_connect):
        """Test _wait_for_master when master becomes available"""
        mock_can_connect.side_effect = [False, False, True]

        _wait_for_master("algo-1", DEFAULT_SSH_PORT, timeout=300)

        assert mock_can_connect.call_count == 3

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.time.sleep")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.time.time")
    def test_wait_for_master_timeout(self, mock_time, mock_sleep, mock_can_connect):
        """Test _wait_for_master when timeout occurs"""
        mock_can_connect.return_value = False
        mock_time.side_effect = [0, 100, 200, 301, 301]

        with pytest.raises(TimeoutError, match="Timed out waiting for master"):
            _wait_for_master("algo-1", DEFAULT_SSH_PORT, timeout=300)

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.os.path.exists")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.time.sleep")
    def test_wait_for_status_file(self, mock_sleep, mock_exists):
        """Test _wait_for_status_file"""
        mock_exists.side_effect = [False, False, True]

        _wait_for_status_file("/tmp/status")

        assert mock_exists.call_count == 3

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.os.path.exists")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.time.sleep")
    def test_wait_for_workers_success(self, mock_sleep, mock_exists, mock_can_connect):
        """Test _wait_for_workers when all workers become available"""
        mock_can_connect.return_value = True
        mock_exists.return_value = True

        _wait_for_workers(["algo-2", "algo-3"], DEFAULT_SSH_PORT, timeout=300)

        assert mock_can_connect.call_count == 2

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._can_connect")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.time.sleep")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.time.time")
    def test_wait_for_workers_timeout(self, mock_time, mock_sleep, mock_can_connect):
        """Test _wait_for_workers when timeout occurs"""
        mock_can_connect.return_value = False
        mock_time.side_effect = [0, 100, 200, 301, 301]

        with pytest.raises(TimeoutError, match="Timed out waiting for workers"):
            _wait_for_workers(["algo-2"], DEFAULT_SSH_PORT, timeout=300)

    def test_wait_for_workers_no_workers(self):
        """Test _wait_for_workers with no workers"""
        # Should not raise exception
        _wait_for_workers([], DEFAULT_SSH_PORT, timeout=300)


class TestBootstrapFunctions:
    """Test cases for bootstrap functions"""

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._wait_for_workers")
    def test_bootstrap_master_node(self, mock_wait):
        """Test bootstrap_master_node"""
        bootstrap_master_node(["algo-2", "algo-3"])

        mock_wait.assert_called_once_with(["algo-2", "algo-3"])

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._wait_for_master")
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._write_file_to_host"
    )
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._wait_for_status_file"
    )
    def test_bootstrap_worker_node(self, mock_wait_status, mock_write, mock_wait_master):
        """Test bootstrap_worker_node"""
        bootstrap_worker_node("algo-1", "algo-2", "/tmp/status")

        mock_wait_master.assert_called_once_with("algo-1")
        mock_write.assert_called_once()
        mock_wait_status.assert_called_once_with("/tmp/status")

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.os.path.exists")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.subprocess.Popen")
    def test_start_sshd_daemon_success(self, mock_popen, mock_exists):
        """Test start_sshd_daemon when sshd exists"""
        mock_exists.return_value = True

        start_sshd_daemon()

        mock_popen.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.os.path.exists")
    def test_start_sshd_daemon_not_found(self, mock_exists):
        """Test start_sshd_daemon when sshd not found"""
        mock_exists.return_value = False

        with pytest.raises(RuntimeError, match="SSH daemon not found"):
            start_sshd_daemon()

    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._write_file_to_host"
    )
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.time.sleep")
    def test_write_status_file_to_workers_success(self, mock_sleep, mock_write):
        """Test write_status_file_to_workers when writes succeed"""
        mock_write.return_value = True

        write_status_file_to_workers(["algo-2", "algo-3"], "/tmp/status")

        assert mock_write.call_count == 2

    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._write_file_to_host"
    )
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.time.sleep")
    def test_write_status_file_to_workers_timeout(self, mock_sleep, mock_write):
        """Test write_status_file_to_workers when timeout occurs"""
        mock_write.return_value = False

        with pytest.raises(TimeoutError, match="Timed out waiting"):
            write_status_file_to_workers(["algo-2"], "/tmp/status")


class TestParseArgs:
    """Test cases for _parse_args function"""

    def test_parse_args_job_ended_false(self):
        """Test _parse_args with job_ended=0"""
        args = _parse_args(["--job_ended", "0"])

        assert args.job_ended == "0"

    def test_parse_args_job_ended_true(self):
        """Test _parse_args with job_ended=1"""
        args = _parse_args(["--job_ended", "1"])

        assert args.job_ended == "1"

    def test_parse_args_default(self):
        """Test _parse_args with default values"""
        args = _parse_args([])

        assert args.job_ended == "0"


class TestMain:
    """Test cases for main function"""

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._parse_args")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.start_sshd_daemon")
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.bootstrap_worker_node"
    )
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.os.environ",
        {"SM_MASTER_ADDR": "algo-1", "SM_CURRENT_HOST": "algo-2"},
    )
    def test_main_worker_node_job_running(self, mock_bootstrap_worker, mock_start_sshd, mock_parse):
        """Test main for worker node when job is running"""
        mock_args = Mock()
        mock_args.job_ended = "0"
        mock_parse.return_value = mock_args

        main([])

        mock_start_sshd.assert_called_once()
        mock_bootstrap_worker.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._parse_args")
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.start_sshd_daemon")
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.bootstrap_master_node"
    )
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.json.loads")
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.os.environ",
        {
            "SM_MASTER_ADDR": "algo-1",
            "SM_CURRENT_HOST": "algo-1",
            "SM_HOSTS": '["algo-1", "algo-2", "algo-3"]',
        },
    )
    def test_main_master_node_job_running(
        self, mock_json_loads, mock_bootstrap_master, mock_start_sshd, mock_parse
    ):
        """Test main for master node when job is running"""
        mock_args = Mock()
        mock_args.job_ended = "0"
        mock_parse.return_value = mock_args
        mock_json_loads.return_value = ["algo-1", "algo-2", "algo-3"]

        main([])

        mock_start_sshd.assert_called_once()
        mock_bootstrap_master.assert_called_once_with(["algo-2", "algo-3"])

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._parse_args")
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.write_status_file_to_workers"
    )
    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.json.loads")
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.os.environ",
        {
            "SM_MASTER_ADDR": "algo-1",
            "SM_CURRENT_HOST": "algo-1",
            "SM_HOSTS": '["algo-1", "algo-2"]',
        },
    )
    def test_main_master_node_job_ended(self, mock_json_loads, mock_write_status, mock_parse):
        """Test main for master node when job has ended"""
        mock_args = Mock()
        mock_args.job_ended = "1"
        mock_parse.return_value = mock_args
        mock_json_loads.return_value = ["algo-1", "algo-2"]

        main([])

        mock_write_status.assert_called_once_with(["algo-2"])

    @patch("sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._parse_args")
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote._write_failure_reason_file"
    )
    @patch(
        "sagemaker.core.remote_function.runtime_environment.mpi_utils_remote.os.environ",
        {"SM_MASTER_ADDR": "algo-1", "SM_CURRENT_HOST": "algo-2"},
    )
    def test_main_with_exception(self, mock_write_failure, mock_parse):
        """Test main when exception occurs"""
        mock_parse.side_effect = Exception("Test error")

        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code == DEFAULT_FAILURE_CODE
        mock_write_failure.assert_called_once()
