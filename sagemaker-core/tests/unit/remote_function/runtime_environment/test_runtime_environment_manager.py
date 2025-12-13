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
"""Tests for runtime_environment_manager module."""
from __future__ import absolute_import

import json
import os
import subprocess
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open, call

from sagemaker.core.remote_function.runtime_environment.runtime_environment_manager import (
    _DependencySettings,
    RuntimeEnvironmentManager,
    RuntimeEnvironmentError,
    get_logger,
    _run_and_get_output_shell_cmd,
    _run_pre_execution_command_script,
    _run_shell_cmd,
    _log_output,
    _log_error,
    _python_executable,
)


class TestDependencySettings:
    """Test _DependencySettings class."""

    def test_init_with_no_file(self):
        """Test initialization without dependency file."""
        settings = _DependencySettings()
        assert settings.dependency_file is None

    def test_init_with_file(self):
        """Test initialization with dependency file."""
        settings = _DependencySettings(dependency_file="requirements.txt")
        assert settings.dependency_file == "requirements.txt"

    def test_to_string(self):
        """Test converts to JSON string."""
        settings = _DependencySettings(dependency_file="requirements.txt")
        result = settings.to_string()
        assert result == '{"dependency_file": "requirements.txt"}'

    def test_from_string_with_file(self):
        """Test creates from JSON string with file."""
        json_str = '{"dependency_file": "requirements.txt"}'
        settings = _DependencySettings.from_string(json_str)
        assert settings.dependency_file == "requirements.txt"

    def test_from_string_with_none(self):
        """Test creates from None."""
        settings = _DependencySettings.from_string(None)
        assert settings is None

    def test_from_dependency_file_path_with_none(self):
        """Test creates from None file path."""
        settings = _DependencySettings.from_dependency_file_path(None)
        assert settings.dependency_file is None

    def test_from_dependency_file_path_with_auto_capture(self):
        """Test creates from auto_capture."""
        settings = _DependencySettings.from_dependency_file_path("auto_capture")
        assert settings.dependency_file == "env_snapshot.yml"

    def test_from_dependency_file_path_with_path(self):
        """Test creates from file path."""
        settings = _DependencySettings.from_dependency_file_path("/path/to/requirements.txt")
        assert settings.dependency_file == "requirements.txt"


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_logger(self):
        """Test returns logger instance."""
        logger = get_logger()
        assert logger is not None
        assert logger.name == "sagemaker.remote_function"


class TestRuntimeEnvironmentManager:
    """Test RuntimeEnvironmentManager class."""

    def test_init(self):
        """Test initialization."""
        manager = RuntimeEnvironmentManager()
        assert manager is not None

    @patch("os.path.isfile")
    def test_snapshot_returns_none_for_none(self, mock_isfile):
        """Test snapshot returns None when dependencies is None."""
        manager = RuntimeEnvironmentManager()
        result = manager.snapshot(None)
        assert result is None

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._capture_from_local_runtime")
    def test_snapshot_auto_capture(self, mock_capture):
        """Test snapshot with auto_capture."""
        mock_capture.return_value = "/path/to/env_snapshot.yml"
        manager = RuntimeEnvironmentManager()
        result = manager.snapshot("auto_capture")
        assert result == "/path/to/env_snapshot.yml"
        mock_capture.assert_called_once()

    @patch("os.path.isfile")
    def test_snapshot_with_txt_file(self, mock_isfile):
        """Test snapshot with requirements.txt file."""
        mock_isfile.return_value = True
        manager = RuntimeEnvironmentManager()
        result = manager.snapshot("requirements.txt")
        assert result == "requirements.txt"

    @patch("os.path.isfile")
    def test_snapshot_with_yml_file(self, mock_isfile):
        """Test snapshot with conda.yml file."""
        mock_isfile.return_value = True
        manager = RuntimeEnvironmentManager()
        result = manager.snapshot("environment.yml")
        assert result == "environment.yml"

    @patch("os.path.isfile")
    def test_snapshot_raises_error_for_invalid_file(self, mock_isfile):
        """Test snapshot raises error for invalid file."""
        mock_isfile.return_value = False
        manager = RuntimeEnvironmentManager()
        with pytest.raises(ValueError):
            manager.snapshot("requirements.txt")

    def test_snapshot_raises_error_for_invalid_format(self):
        """Test snapshot raises error for invalid format."""
        manager = RuntimeEnvironmentManager()
        with pytest.raises(ValueError):
            manager.snapshot("invalid.json")

    @patch("os.getenv")
    def test_get_active_conda_env_prefix(self, mock_getenv):
        """Test gets active conda environment prefix."""
        mock_getenv.return_value = "/opt/conda/envs/myenv"
        manager = RuntimeEnvironmentManager()
        result = manager._get_active_conda_env_prefix()
        assert result == "/opt/conda/envs/myenv"

    @patch("os.getenv")
    def test_get_active_conda_env_name(self, mock_getenv):
        """Test gets active conda environment name."""
        mock_getenv.return_value = "myenv"
        manager = RuntimeEnvironmentManager()
        result = manager._get_active_conda_env_name()
        assert result == "myenv"

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._export_conda_env_from_prefix")
    @patch("os.getcwd")
    @patch("os.getenv")
    def test_capture_from_local_runtime(self, mock_getenv, mock_getcwd, mock_export):
        """Test captures from local runtime."""
        mock_getenv.side_effect = lambda x: "myenv" if x == "CONDA_DEFAULT_ENV" else "/opt/conda/envs/myenv"
        mock_getcwd.return_value = "/tmp"
        manager = RuntimeEnvironmentManager()
        result = manager._capture_from_local_runtime()
        assert result == "/tmp/env_snapshot.yml"
        mock_export.assert_called_once()

    @patch("os.getenv")
    def test_capture_from_local_runtime_raises_error_no_conda(self, mock_getenv):
        """Test raises error when no conda environment active."""
        mock_getenv.return_value = None
        manager = RuntimeEnvironmentManager()
        with pytest.raises(ValueError):
            manager._capture_from_local_runtime()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._install_requirements_txt")
    def test_bootstrap_with_txt_file_no_conda(self, mock_install):
        """Test bootstrap with requirements.txt without conda."""
        manager = RuntimeEnvironmentManager()
        manager.bootstrap("requirements.txt", "3.8", None)
        mock_install.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._write_conda_env_to_file")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._install_req_txt_in_conda_env")
    def test_bootstrap_with_txt_file_with_conda(self, mock_install, mock_write):
        """Test bootstrap with requirements.txt with conda."""
        manager = RuntimeEnvironmentManager()
        manager.bootstrap("requirements.txt", "3.8", "myenv")
        mock_install.assert_called_once()
        mock_write.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._write_conda_env_to_file")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._update_conda_env")
    def test_bootstrap_with_yml_file_with_conda(self, mock_update, mock_write):
        """Test bootstrap with conda.yml with existing conda env."""
        manager = RuntimeEnvironmentManager()
        manager.bootstrap("environment.yml", "3.8", "myenv")
        mock_update.assert_called_once()
        mock_write.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._write_conda_env_to_file")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._validate_python_version")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._create_conda_env")
    def test_bootstrap_with_yml_file_without_conda(self, mock_create, mock_validate, mock_write):
        """Test bootstrap with conda.yml without existing conda env."""
        manager = RuntimeEnvironmentManager()
        manager.bootstrap("environment.yml", "3.8", None)
        mock_create.assert_called_once()
        mock_validate.assert_called_once()
        mock_write.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_pre_execution_command_script")
    @patch("os.path.isfile")
    def test_run_pre_exec_script_exists(self, mock_isfile, mock_run_script):
        """Test runs pre-execution script when it exists."""
        mock_isfile.return_value = True
        mock_run_script.return_value = (0, "")
        manager = RuntimeEnvironmentManager()
        manager.run_pre_exec_script("/path/to/script.sh")
        mock_run_script.assert_called_once()

    @patch("os.path.isfile")
    def test_run_pre_exec_script_not_exists(self, mock_isfile):
        """Test handles pre-execution script not existing."""
        mock_isfile.return_value = False
        manager = RuntimeEnvironmentManager()
        # Should not raise exception
        manager.run_pre_exec_script("/path/to/script.sh")

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_pre_execution_command_script")
    @patch("os.path.isfile")
    def test_run_pre_exec_script_raises_error_on_failure(self, mock_isfile, mock_run_script):
        """Test raises error when pre-execution script fails."""
        mock_isfile.return_value = True
        mock_run_script.return_value = (1, "Error message")
        manager = RuntimeEnvironmentManager()
        with pytest.raises(RuntimeEnvironmentError):
            manager.run_pre_exec_script("/path/to/script.sh")

    @patch("subprocess.run")
    def test_change_dir_permission_success(self, mock_run):
        """Test changes directory permissions successfully."""
        manager = RuntimeEnvironmentManager()
        manager.change_dir_permission(["/tmp/dir1", "/tmp/dir2"], "777")
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_change_dir_permission_raises_error_on_failure(self, mock_run):
        """Test raises error when permission change fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "chmod", stderr=b"Permission denied")
        manager = RuntimeEnvironmentManager()
        with pytest.raises(RuntimeEnvironmentError):
            manager.change_dir_permission(["/tmp/dir1"], "777")

    @patch("subprocess.run")
    def test_change_dir_permission_raises_error_no_sudo(self, mock_run):
        """Test raises error when sudo not found."""
        mock_run.side_effect = FileNotFoundError("[Errno 2] No such file or directory: 'sudo'")
        manager = RuntimeEnvironmentManager()
        with pytest.raises(RuntimeEnvironmentError):
            manager.change_dir_permission(["/tmp/dir1"], "777")

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_shell_cmd")
    def test_install_requirements_txt(self, mock_run_cmd):
        """Test installs requirements.txt."""
        manager = RuntimeEnvironmentManager()
        manager._install_requirements_txt("/path/to/requirements.txt", "/usr/bin/python")
        mock_run_cmd.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_shell_cmd")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._get_conda_exe")
    def test_create_conda_env(self, mock_get_conda, mock_run_cmd):
        """Test creates conda environment."""
        mock_get_conda.return_value = "conda"
        manager = RuntimeEnvironmentManager()
        manager._create_conda_env("myenv", "/path/to/environment.yml")
        mock_run_cmd.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_shell_cmd")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._get_conda_exe")
    def test_install_req_txt_in_conda_env(self, mock_get_conda, mock_run_cmd):
        """Test installs requirements.txt in conda environment."""
        mock_get_conda.return_value = "conda"
        manager = RuntimeEnvironmentManager()
        manager._install_req_txt_in_conda_env("myenv", "/path/to/requirements.txt")
        mock_run_cmd.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_shell_cmd")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._get_conda_exe")
    def test_update_conda_env(self, mock_get_conda, mock_run_cmd):
        """Test updates conda environment."""
        mock_get_conda.return_value = "conda"
        manager = RuntimeEnvironmentManager()
        manager._update_conda_env("myenv", "/path/to/environment.yml")
        mock_run_cmd.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("subprocess.Popen")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._get_conda_exe")
    def test_export_conda_env_from_prefix(self, mock_get_conda, mock_popen, mock_file):
        """Test exports conda environment."""
        mock_get_conda.return_value = "conda"
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"env output", b"")
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        manager = RuntimeEnvironmentManager()
        manager._export_conda_env_from_prefix("/opt/conda/envs/myenv", "/tmp/env.yml")
        
        mock_popen.assert_called_once()
        mock_file.assert_called_once_with("/tmp/env.yml", "w")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.getcwd")
    def test_write_conda_env_to_file(self, mock_getcwd, mock_file):
        """Test writes conda environment name to file."""
        mock_getcwd.return_value = "/tmp"
        manager = RuntimeEnvironmentManager()
        manager._write_conda_env_to_file("myenv")
        mock_file.assert_called_once_with("/tmp/remote_function_conda_env.txt", "w")
        mock_file().write.assert_called_once_with("myenv")

    @patch("subprocess.Popen")
    def test_get_conda_exe_returns_mamba(self, mock_popen):
        """Test returns mamba when available."""
        mock_popen.return_value.wait.side_effect = [0, 1]  # mamba exists, conda doesn't
        manager = RuntimeEnvironmentManager()
        result = manager._get_conda_exe()
        assert result == "mamba"

    @patch("subprocess.Popen")
    def test_get_conda_exe_returns_conda(self, mock_popen):
        """Test returns conda when mamba not available."""
        mock_popen.return_value.wait.side_effect = [1, 0]  # mamba doesn't exist, conda does
        manager = RuntimeEnvironmentManager()
        result = manager._get_conda_exe()
        assert result == "conda"

    @patch("subprocess.Popen")
    def test_get_conda_exe_raises_error(self, mock_popen):
        """Test raises error when neither conda nor mamba available."""
        mock_popen.return_value.wait.return_value = 1
        manager = RuntimeEnvironmentManager()
        with pytest.raises(ValueError):
            manager._get_conda_exe()

    @patch("subprocess.check_output")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._get_conda_exe")
    def test_python_version_in_conda_env(self, mock_get_conda, mock_check_output):
        """Test gets Python version in conda environment."""
        mock_get_conda.return_value = "conda"
        mock_check_output.return_value = b"Python 3.8.10"
        manager = RuntimeEnvironmentManager()
        result = manager._python_version_in_conda_env("myenv")
        assert result == "3.8"

    @patch("subprocess.check_output")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._get_conda_exe")
    def test_python_version_in_conda_env_raises_error(self, mock_get_conda, mock_check_output):
        """Test raises error when getting Python version fails."""
        mock_get_conda.return_value = "conda"
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "conda", output=b"Error")
        manager = RuntimeEnvironmentManager()
        with pytest.raises(RuntimeEnvironmentError):
            manager._python_version_in_conda_env("myenv")

    def test_current_python_version(self):
        """Test gets current Python version."""
        manager = RuntimeEnvironmentManager()
        result = manager._current_python_version()
        expected = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert result == expected

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._python_version_in_conda_env")
    def test_validate_python_version_with_conda(self, mock_python_version):
        """Test validates Python version with conda environment."""
        mock_python_version.return_value = "3.8"
        manager = RuntimeEnvironmentManager()
        # Should not raise exception
        manager._validate_python_version("3.8", "myenv")

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._python_version_in_conda_env")
    def test_validate_python_version_mismatch_with_conda(self, mock_python_version):
        """Test raises error on Python version mismatch with conda."""
        mock_python_version.return_value = "3.9"
        manager = RuntimeEnvironmentManager()
        with pytest.raises(RuntimeEnvironmentError):
            manager._validate_python_version("3.8", "myenv")

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._current_python_version")
    def test_validate_python_version_without_conda(self, mock_current_version):
        """Test validates Python version without conda environment."""
        mock_current_version.return_value = "3.8"
        manager = RuntimeEnvironmentManager()
        # Should not raise exception
        manager._validate_python_version("3.8", None)

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._current_python_version")
    def test_validate_python_version_mismatch_without_conda(self, mock_current_version):
        """Test raises error on Python version mismatch without conda."""
        mock_current_version.return_value = "3.9"
        manager = RuntimeEnvironmentManager()
        with pytest.raises(RuntimeEnvironmentError):
            manager._validate_python_version("3.8", None)

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._current_sagemaker_pysdk_version")
    def test_validate_sagemaker_pysdk_version_match(self, mock_current_version):
        """Test validates matching SageMaker SDK version."""
        mock_current_version.return_value = "2.100.0"
        manager = RuntimeEnvironmentManager()
        # Should not raise exception or warning
        manager._validate_sagemaker_pysdk_version("2.100.0")

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._current_sagemaker_pysdk_version")
    def test_validate_sagemaker_pysdk_version_mismatch(self, mock_current_version):
        """Test logs warning on SageMaker SDK version mismatch."""
        mock_current_version.return_value = "2.101.0"
        manager = RuntimeEnvironmentManager()
        # Should log warning but not raise exception
        manager._validate_sagemaker_pysdk_version("2.100.0")

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager._current_sagemaker_pysdk_version")
    def test_validate_sagemaker_pysdk_version_none(self, mock_current_version):
        """Test handles None client version."""
        mock_current_version.return_value = "2.100.0"
        manager = RuntimeEnvironmentManager()
        # Should not raise exception
        manager._validate_sagemaker_pysdk_version(None)


class TestRunAndGetOutputShellCmd:
    """Test _run_and_get_output_shell_cmd function."""

    @patch("subprocess.check_output")
    def test_runs_command_successfully(self, mock_check_output):
        """Test runs command and returns output."""
        mock_check_output.return_value = b"command output"
        result = _run_and_get_output_shell_cmd("echo test")
        assert result == "command output"


class TestRunPreExecutionCommandScript:
    """Test _run_pre_execution_command_script function."""

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_error")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_output")
    @patch("subprocess.Popen")
    @patch("os.path.dirname")
    def test_runs_script_successfully(self, mock_dirname, mock_popen, mock_log_output, mock_log_error):
        """Test runs script successfully."""
        mock_dirname.return_value = "/tmp"
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        mock_log_error.return_value = ""
        
        return_code, error_logs = _run_pre_execution_command_script("/tmp/script.sh")
        
        assert return_code == 0
        assert error_logs == ""

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_error")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_output")
    @patch("subprocess.Popen")
    @patch("os.path.dirname")
    def test_runs_script_with_error(self, mock_dirname, mock_popen, mock_log_output, mock_log_error):
        """Test runs script that returns error."""
        mock_dirname.return_value = "/tmp"
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        mock_log_error.return_value = "Error message"
        
        return_code, error_logs = _run_pre_execution_command_script("/tmp/script.sh")
        
        assert return_code == 1
        assert error_logs == "Error message"


class TestRunShellCmd:
    """Test _run_shell_cmd function."""

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_error")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_output")
    @patch("subprocess.Popen")
    def test_runs_command_successfully(self, mock_popen, mock_log_output, mock_log_error):
        """Test runs command successfully."""
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        mock_log_error.return_value = ""
        
        _run_shell_cmd(["echo", "test"])
        
        mock_popen.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_error")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_output")
    @patch("subprocess.Popen")
    def test_runs_command_raises_error_on_failure(self, mock_popen, mock_log_output, mock_log_error):
        """Test raises error when command fails."""
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        mock_log_error.return_value = "Error message"
        
        with pytest.raises(RuntimeEnvironmentError):
            _run_shell_cmd(["false"])


class TestLogOutput:
    """Test _log_output function."""

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.logger")
    def test_logs_output(self, mock_logger):
        """Test logs process output."""
        from io import BytesIO
        mock_process = MagicMock()
        mock_process.stdout = BytesIO(b"line1\nline2\n")
        
        _log_output(mock_process)
        
        assert mock_logger.info.call_count == 2


class TestLogError:
    """Test _log_error function."""

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.logger")
    def test_logs_error(self, mock_logger):
        """Test logs process errors."""
        from io import BytesIO
        mock_process = MagicMock()
        mock_process.stderr = BytesIO(b"ERROR: error message\nwarning message\n")
        
        error_logs = _log_error(mock_process)
        
        assert "ERROR: error message" in error_logs
        assert "warning message" in error_logs


class TestPythonExecutable:
    """Test _python_executable function."""

    def test_returns_python_executable(self):
        """Test returns Python executable path."""
        result = _python_executable()
        assert result == sys.executable

    @patch("sys.executable", None)
    def test_raises_error_if_no_executable(self):
        """Test raises error if no Python executable."""
        with pytest.raises(RuntimeEnvironmentError):
            _python_executable()


class TestRuntimeEnvironmentError:
    """Test RuntimeEnvironmentError class."""

    def test_creates_error_with_message(self):
        """Test creates error with message."""
        error = RuntimeEnvironmentError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
