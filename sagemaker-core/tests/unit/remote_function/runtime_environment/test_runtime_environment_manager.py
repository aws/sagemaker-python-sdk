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
import sys

from sagemaker.core.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentManager,
    RuntimeEnvironmentError,
    _DependencySettings,
    get_logger,
    _run_and_get_output_shell_cmd,
    _run_pre_execution_command_script,
    _run_shell_cmd,
    _log_output,
    _log_error,
    _python_executable,
)


class TestDependencySettings:
    """Test cases for _DependencySettings class"""

    def test_init_with_file(self):
        """Test initialization with dependency file"""
        settings = _DependencySettings(dependency_file="requirements.txt")
        
        assert settings.dependency_file == "requirements.txt"

    def test_init_without_file(self):
        """Test initialization without dependency file"""
        settings = _DependencySettings()
        
        assert settings.dependency_file is None

    def test_to_string(self):
        """Test to_string method"""
        settings = _DependencySettings(dependency_file="requirements.txt")
        
        result = settings.to_string()
        
        assert "requirements.txt" in result

    def test_from_string(self):
        """Test from_string method"""
        json_str = '{"dependency_file": "requirements.txt"}'
        
        settings = _DependencySettings.from_string(json_str)
        
        assert settings.dependency_file == "requirements.txt"

    def test_from_string_none(self):
        """Test from_string with None"""
        settings = _DependencySettings.from_string(None)
        
        assert settings is None

    def test_from_dependency_file_path(self):
        """Test from_dependency_file_path method"""
        settings = _DependencySettings.from_dependency_file_path("/path/to/requirements.txt")
        
        assert settings.dependency_file == "requirements.txt"

    def test_from_dependency_file_path_auto_capture(self):
        """Test from_dependency_file_path with auto_capture"""
        settings = _DependencySettings.from_dependency_file_path("auto_capture")
        
        assert settings.dependency_file == "env_snapshot.yml"

    def test_from_dependency_file_path_none(self):
        """Test from_dependency_file_path with None"""
        settings = _DependencySettings.from_dependency_file_path(None)
        
        assert settings.dependency_file is None


class TestRuntimeEnvironmentManager:
    """Test cases for RuntimeEnvironmentManager class"""

    def test_init(self):
        """Test initialization"""
        manager = RuntimeEnvironmentManager()
        
        assert manager is not None

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.os.path.isfile")
    def test_snapshot_with_requirements_txt(self, mock_isfile):
        """Test snapshot with requirements.txt"""
        mock_isfile.return_value = True
        manager = RuntimeEnvironmentManager()
        
        result = manager.snapshot("requirements.txt")
        
        assert result == "requirements.txt"

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.os.path.isfile")
    def test_snapshot_with_conda_yml(self, mock_isfile):
        """Test snapshot with conda environment.yml"""
        mock_isfile.return_value = True
        manager = RuntimeEnvironmentManager()
        
        result = manager.snapshot("environment.yml")
        
        assert result == "environment.yml"

    @patch.object(RuntimeEnvironmentManager, "_capture_from_local_runtime")
    def test_snapshot_with_auto_capture(self, mock_capture):
        """Test snapshot with auto_capture"""
        mock_capture.return_value = "env_snapshot.yml"
        manager = RuntimeEnvironmentManager()
        
        result = manager.snapshot("auto_capture")
        
        assert result == "env_snapshot.yml"
        mock_capture.assert_called_once()

    def test_snapshot_with_none(self):
        """Test snapshot with None"""
        manager = RuntimeEnvironmentManager()
        
        result = manager.snapshot(None)
        
        assert result is None

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.os.path.isfile")
    def test_snapshot_with_invalid_file(self, mock_isfile):
        """Test snapshot with invalid file"""
        mock_isfile.return_value = False
        manager = RuntimeEnvironmentManager()
        
        with pytest.raises(ValueError, match="No dependencies file named"):
            manager.snapshot("invalid.txt")

    @patch.object(RuntimeEnvironmentManager, "_get_active_conda_env_name")
    @patch.object(RuntimeEnvironmentManager, "_get_active_conda_env_prefix")
    @patch.object(RuntimeEnvironmentManager, "_export_conda_env_from_prefix")
    def test_capture_from_local_runtime_with_conda_env(self, mock_export, mock_prefix, mock_name):
        """Test _capture_from_local_runtime with conda environment"""
        mock_name.return_value = "myenv"
        mock_prefix.return_value = "/opt/conda/envs/myenv"
        manager = RuntimeEnvironmentManager()
        
        result = manager._capture_from_local_runtime()
        
        assert "env_snapshot.yml" in result
        mock_export.assert_called_once()

    @patch.object(RuntimeEnvironmentManager, "_get_active_conda_env_name")
    @patch.object(RuntimeEnvironmentManager, "_get_active_conda_env_prefix")
    def test_capture_from_local_runtime_no_conda_env(self, mock_prefix, mock_name):
        """Test _capture_from_local_runtime without conda environment"""
        mock_name.return_value = None
        mock_prefix.return_value = None
        manager = RuntimeEnvironmentManager()
        
        with pytest.raises(ValueError, match="No conda environment"):
            manager._capture_from_local_runtime()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.os.getenv")
    def test_get_active_conda_env_prefix(self, mock_getenv):
        """Test _get_active_conda_env_prefix"""
        mock_getenv.return_value = "/opt/conda/envs/myenv"
        manager = RuntimeEnvironmentManager()
        
        result = manager._get_active_conda_env_prefix()
        
        assert result == "/opt/conda/envs/myenv"

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.os.getenv")
    def test_get_active_conda_env_name(self, mock_getenv):
        """Test _get_active_conda_env_name"""
        mock_getenv.return_value = "myenv"
        manager = RuntimeEnvironmentManager()
        
        result = manager._get_active_conda_env_name()
        
        assert result == "myenv"

    @patch.object(RuntimeEnvironmentManager, "_install_req_txt_in_conda_env")
    @patch.object(RuntimeEnvironmentManager, "_write_conda_env_to_file")
    def test_bootstrap_with_requirements_txt_and_conda_env(self, mock_write, mock_install):
        """Test bootstrap with requirements.txt and conda environment"""
        manager = RuntimeEnvironmentManager()
        
        manager.bootstrap(
            local_dependencies_file="requirements.txt",
            client_python_version="3.8",
            conda_env="myenv"
        )
        
        mock_install.assert_called_once_with("myenv", "requirements.txt")
        mock_write.assert_called_once_with("myenv")

    @patch.object(RuntimeEnvironmentManager, "_install_requirements_txt")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._python_executable")
    def test_bootstrap_with_requirements_txt_no_conda_env(self, mock_python_exec, mock_install):
        """Test bootstrap with requirements.txt without conda environment"""
        mock_python_exec.return_value = "/usr/bin/python3"
        manager = RuntimeEnvironmentManager()
        
        manager.bootstrap(
            local_dependencies_file="requirements.txt",
            client_python_version="3.8"
        )
        
        mock_install.assert_called_once()

    @patch.object(RuntimeEnvironmentManager, "_update_conda_env")
    @patch.object(RuntimeEnvironmentManager, "_write_conda_env_to_file")
    def test_bootstrap_with_conda_yml_and_conda_env(self, mock_write, mock_update):
        """Test bootstrap with conda yml and existing conda environment"""
        manager = RuntimeEnvironmentManager()
        
        manager.bootstrap(
            local_dependencies_file="environment.yml",
            client_python_version="3.8",
            conda_env="myenv"
        )
        
        mock_update.assert_called_once()
        mock_write.assert_called_once()

    @patch.object(RuntimeEnvironmentManager, "_create_conda_env")
    @patch.object(RuntimeEnvironmentManager, "_validate_python_version")
    @patch.object(RuntimeEnvironmentManager, "_write_conda_env_to_file")
    def test_bootstrap_with_conda_yml_no_conda_env(self, mock_write, mock_validate, mock_create):
        """Test bootstrap with conda yml without existing conda environment"""
        manager = RuntimeEnvironmentManager()
        
        manager.bootstrap(
            local_dependencies_file="environment.yml",
            client_python_version="3.8"
        )
        
        mock_create.assert_called_once()
        mock_validate.assert_called_once()
        mock_write.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.os.path.isfile")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_pre_execution_command_script")
    def test_run_pre_exec_script_exists(self, mock_run_script, mock_isfile):
        """Test run_pre_exec_script when script exists"""
        mock_isfile.return_value = True
        mock_run_script.return_value = (0, "")
        manager = RuntimeEnvironmentManager()
        
        manager.run_pre_exec_script("/path/to/script.sh")
        
        mock_run_script.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.os.path.isfile")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_pre_execution_command_script")
    def test_run_pre_exec_script_fails(self, mock_run_script, mock_isfile):
        """Test run_pre_exec_script when script fails"""
        mock_isfile.return_value = True
        mock_run_script.return_value = (1, "Error message")
        manager = RuntimeEnvironmentManager()
        
        with pytest.raises(RuntimeEnvironmentError, match="Encountered error"):
            manager.run_pre_exec_script("/path/to/script.sh")

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.run")
    def test_change_dir_permission_success(self, mock_run):
        """Test change_dir_permission successfully"""
        manager = RuntimeEnvironmentManager()
        
        manager.change_dir_permission(["/tmp/dir1", "/tmp/dir2"], "777")
        
        mock_run.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.run")
    def test_change_dir_permission_failure(self, mock_run):
        """Test change_dir_permission with failure"""
        mock_run.side_effect = subprocess.CalledProcessError(1, "chmod", stderr=b"Permission denied")
        manager = RuntimeEnvironmentManager()
        
        with pytest.raises(RuntimeEnvironmentError):
            manager.change_dir_permission(["/tmp/dir"], "777")

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_shell_cmd")
    def test_install_requirements_txt(self, mock_run_cmd):
        """Test _install_requirements_txt"""
        manager = RuntimeEnvironmentManager()
        
        manager._install_requirements_txt("/path/to/requirements.txt", "/usr/bin/python3")
        
        mock_run_cmd.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_shell_cmd")
    @patch.object(RuntimeEnvironmentManager, "_get_conda_exe")
    def test_create_conda_env(self, mock_get_conda, mock_run_cmd):
        """Test _create_conda_env"""
        mock_get_conda.return_value = "conda"
        manager = RuntimeEnvironmentManager()
        
        manager._create_conda_env("myenv", "/path/to/environment.yml")
        
        mock_run_cmd.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._run_shell_cmd")
    @patch.object(RuntimeEnvironmentManager, "_get_conda_exe")
    def test_update_conda_env(self, mock_get_conda, mock_run_cmd):
        """Test _update_conda_env"""
        mock_get_conda.return_value = "conda"
        manager = RuntimeEnvironmentManager()
        
        manager._update_conda_env("myenv", "/path/to/environment.yml")
        
        mock_run_cmd.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.Popen")
    def test_get_conda_exe_mamba(self, mock_popen):
        """Test _get_conda_exe returns mamba"""
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        manager = RuntimeEnvironmentManager()
        
        result = manager._get_conda_exe()
        
        assert result == "mamba"

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.Popen")
    def test_get_conda_exe_conda(self, mock_popen):
        """Test _get_conda_exe returns conda"""
        mock_process = Mock()
        mock_process.wait.side_effect = [1, 0]  # mamba not found, conda found
        mock_popen.return_value = mock_process
        manager = RuntimeEnvironmentManager()
        
        result = manager._get_conda_exe()
        
        assert result == "conda"

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.Popen")
    def test_get_conda_exe_not_found(self, mock_popen):
        """Test _get_conda_exe when neither mamba nor conda found"""
        mock_process = Mock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        manager = RuntimeEnvironmentManager()
        
        with pytest.raises(ValueError, match="Neither conda nor mamba"):
            manager._get_conda_exe()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.check_output")
    @patch.object(RuntimeEnvironmentManager, "_get_conda_exe")
    def test_python_version_in_conda_env(self, mock_get_conda, mock_check_output):
        """Test _python_version_in_conda_env"""
        mock_get_conda.return_value = "conda"
        mock_check_output.return_value = b"Python 3.8.10"
        manager = RuntimeEnvironmentManager()
        
        result = manager._python_version_in_conda_env("myenv")
        
        assert result == "3.8"

    def test_current_python_version(self):
        """Test _current_python_version"""
        manager = RuntimeEnvironmentManager()
        
        result = manager._current_python_version()
        
        assert result == f"{sys.version_info.major}.{sys.version_info.minor}"

    @patch.object(RuntimeEnvironmentManager, "_python_version_in_conda_env")
    def test_validate_python_version_match(self, mock_python_version):
        """Test _validate_python_version when versions match"""
        mock_python_version.return_value = "3.8"
        manager = RuntimeEnvironmentManager()
        
        # Should not raise error
        manager._validate_python_version("3.8", conda_env="myenv")

    @patch.object(RuntimeEnvironmentManager, "_python_version_in_conda_env")
    def test_validate_python_version_mismatch(self, mock_python_version):
        """Test _validate_python_version when versions don't match"""
        mock_python_version.return_value = "3.9"
        manager = RuntimeEnvironmentManager()
        
        with pytest.raises(RuntimeEnvironmentError, match="does not match"):
            manager._validate_python_version("3.8", conda_env="myenv")

    @patch.object(RuntimeEnvironmentManager, "_current_sagemaker_pysdk_version")
    def test_validate_sagemaker_pysdk_version_match(self, mock_version):
        """Test _validate_sagemaker_pysdk_version when versions match"""
        mock_version.return_value = "2.0.0"
        manager = RuntimeEnvironmentManager()
        
        # Should not raise error, just log warning
        manager._validate_sagemaker_pysdk_version("2.0.0")

    @patch.object(RuntimeEnvironmentManager, "_current_sagemaker_pysdk_version")
    def test_validate_sagemaker_pysdk_version_mismatch(self, mock_version):
        """Test _validate_sagemaker_pysdk_version when versions don't match"""
        mock_version.return_value = "2.1.0"
        manager = RuntimeEnvironmentManager()
        
        # Should log warning but not raise error
        manager._validate_sagemaker_pysdk_version("2.0.0")


class TestHelperFunctions:
    """Test cases for helper functions"""

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.check_output")
    def test_run_and_get_output_shell_cmd(self, mock_check_output):
        """Test _run_and_get_output_shell_cmd"""
        mock_check_output.return_value = b"output"
        
        result = _run_and_get_output_shell_cmd("echo test")
        
        assert result == "output"

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.Popen")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_output")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_error")
    def test_run_pre_execution_command_script(self, mock_log_error, mock_log_output, mock_popen):
        """Test _run_pre_execution_command_script"""
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        mock_log_error.return_value = ""
        
        return_code, error_logs = _run_pre_execution_command_script("/path/to/script.sh")
        
        assert return_code == 0

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.Popen")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_output")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_error")
    def test_run_shell_cmd_success(self, mock_log_error, mock_log_output, mock_popen):
        """Test _run_shell_cmd with successful command"""
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        mock_log_error.return_value = ""
        
        _run_shell_cmd(["echo", "test"])
        
        mock_popen.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.subprocess.Popen")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_output")
    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager._log_error")
    def test_run_shell_cmd_failure(self, mock_log_error, mock_log_output, mock_popen):
        """Test _run_shell_cmd with failed command"""
        mock_process = Mock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        mock_log_error.return_value = "Error message"
        
        with pytest.raises(RuntimeEnvironmentError, match="Encountered error"):
            _run_shell_cmd(["false"])

    def test_python_executable(self):
        """Test _python_executable"""
        result = _python_executable()
        
        assert result == sys.executable

    @patch("sagemaker.core.remote_function.runtime_environment.runtime_environment_manager.sys.executable", None)
    def test_python_executable_not_found(self):
        """Test _python_executable when not found"""
        with pytest.raises(RuntimeEnvironmentError, match="Failed to retrieve"):
            _python_executable()


class TestRuntimeEnvironmentError:
    """Test cases for RuntimeEnvironmentError exception"""

    def test_init(self):
        """Test initialization"""
        error = RuntimeEnvironmentError("Test error message")
        
        assert error.message == "Test error message"
        assert str(error) == "Test error message"

    def test_raise(self):
        """Test raising the exception"""
        with pytest.raises(RuntimeEnvironmentError, match="Test error"):
            raise RuntimeEnvironmentError("Test error")


class TestGetLogger:
    """Test cases for get_logger function"""

    def test_get_logger(self):
        """Test get_logger returns logger"""
        logger = get_logger()
        
        assert logger is not None
        assert logger.name == "sagemaker.remote_function"
