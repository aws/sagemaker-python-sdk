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
from unittest.mock import Mock, patch, mock_open, MagicMock
import json
import sys

from sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment import (
    _bootstrap_runtime_env_for_remote_function,
    _bootstrap_runtime_env_for_pipeline_step,
    _handle_pre_exec_scripts,
    _install_dependencies,
    _unpack_user_workspace,
    _write_failure_reason_file,
    _parse_args,
    log_key_value,
    log_env_variables,
    mask_sensitive_info,
    num_cpus,
    num_gpus,
    num_neurons,
    safe_serialize,
    set_env,
    main,
    SUCCESS_EXIT_CODE,
    DEFAULT_FAILURE_CODE,
    SENSITIVE_KEYWORDS,
    HIDDEN_VALUE,
)
from sagemaker.core.remote_function.runtime_environment.runtime_environment_manager import (
    _DependencySettings,
)


class TestBootstrapRuntimeEnvironment:
    """Test cases for bootstrap runtime environment functions"""

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment._handle_pre_exec_scripts")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment._install_dependencies")
    def test_bootstrap_runtime_env_for_remote_function(self, mock_install, mock_handle, mock_unpack):
        """Test _bootstrap_runtime_env_for_remote_function"""
        mock_unpack.return_value = "/workspace"
        dependency_settings = _DependencySettings(dependency_file="requirements.txt")
        
        _bootstrap_runtime_env_for_remote_function(
            client_python_version="3.8",
            conda_env="myenv",
            dependency_settings=dependency_settings
        )
        
        mock_unpack.assert_called_once()
        mock_handle.assert_called_once_with("/workspace")
        mock_install.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace")
    def test_bootstrap_runtime_env_for_remote_function_no_workspace(self, mock_unpack):
        """Test _bootstrap_runtime_env_for_remote_function with no workspace"""
        mock_unpack.return_value = None
        
        _bootstrap_runtime_env_for_remote_function(
            client_python_version="3.8"
        )
        
        mock_unpack.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.path.exists")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.mkdir")
    def test_bootstrap_runtime_env_for_pipeline_step(self, mock_mkdir, mock_exists, mock_unpack):
        """Test _bootstrap_runtime_env_for_pipeline_step"""
        mock_unpack.return_value = None
        mock_exists.return_value = False
        
        _bootstrap_runtime_env_for_pipeline_step(
            client_python_version="3.8",
            func_step_workspace="workspace"
        )
        
        mock_mkdir.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.path.isfile")
    def test_handle_pre_exec_scripts_exists(self, mock_isfile, mock_manager_class):
        """Test _handle_pre_exec_scripts when script exists"""
        mock_isfile.return_value = True
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        _handle_pre_exec_scripts("/workspace")
        
        mock_manager.run_pre_exec_script.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.path.isfile")
    def test_handle_pre_exec_scripts_not_exists(self, mock_isfile, mock_manager_class):
        """Test _handle_pre_exec_scripts when script doesn't exist"""
        mock_isfile.return_value = False
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        _handle_pre_exec_scripts("/workspace")
        
        mock_manager.run_pre_exec_script.assert_not_called()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.path.join")
    def test_install_dependencies_with_file(self, mock_join, mock_manager_class):
        """Test _install_dependencies with dependency file"""
        mock_join.return_value = "/workspace/requirements.txt"
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        dependency_settings = _DependencySettings(dependency_file="requirements.txt")
        
        _install_dependencies(
            dependency_file_dir="/workspace",
            conda_env="myenv",
            client_python_version="3.8",
            channel_name="channel",
            dependency_settings=dependency_settings
        )
        
        mock_manager.bootstrap.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    def test_install_dependencies_no_file(self, mock_manager_class):
        """Test _install_dependencies with no dependency file"""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        dependency_settings = _DependencySettings(dependency_file=None)
        
        _install_dependencies(
            dependency_file_dir="/workspace",
            conda_env=None,
            client_python_version="3.8",
            channel_name="channel",
            dependency_settings=dependency_settings
        )
        
        mock_manager.bootstrap.assert_not_called()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.path.exists")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.path.isfile")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.shutil.unpack_archive")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.pathlib.Path")
    def test_unpack_user_workspace_success(self, mock_path, mock_unpack, mock_isfile, mock_exists):
        """Test _unpack_user_workspace successfully unpacks workspace"""
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_path.return_value.absolute.return_value = "/workspace"
        
        result = _unpack_user_workspace()
        
        assert result is not None
        mock_unpack.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.path.exists")
    def test_unpack_user_workspace_no_directory(self, mock_exists):
        """Test _unpack_user_workspace when directory doesn't exist"""
        mock_exists.return_value = False
        
        result = _unpack_user_workspace()
        
        assert result is None

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_failure_reason_file(self, mock_file, mock_exists):
        """Test _write_failure_reason_file"""
        mock_exists.return_value = False
        
        _write_failure_reason_file("Test error message")
        
        mock_file.assert_called_once()
        mock_file().write.assert_called_once_with("RuntimeEnvironmentError: Test error message")

    def test_parse_args(self):
        """Test _parse_args"""
        args = _parse_args([
            "--job_conda_env", "myenv",
            "--client_python_version", "3.8",
            "--dependency_settings", '{"dependency_file": "requirements.txt"}'
        ])
        
        assert args.job_conda_env == "myenv"
        assert args.client_python_version == "3.8"
        assert args.dependency_settings == '{"dependency_file": "requirements.txt"}'


class TestLoggingFunctions:
    """Test cases for logging functions"""

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.logger")
    def test_log_key_value_normal(self, mock_logger):
        """Test log_key_value with normal key"""
        log_key_value("MY_KEY", "my_value")
        
        mock_logger.info.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.logger")
    def test_log_key_value_sensitive(self, mock_logger):
        """Test log_key_value with sensitive key"""
        log_key_value("MY_PASSWORD", "secret123")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0]
        assert HIDDEN_VALUE in str(call_args)

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.logger")
    def test_log_key_value_dict(self, mock_logger):
        """Test log_key_value with dictionary value"""
        log_key_value("MY_CONFIG", {"key": "value"})
        
        mock_logger.info.assert_called_once()

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.logger")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.environ", {"ENV_VAR": "value"})
    def test_log_env_variables(self, mock_logger):
        """Test log_env_variables"""
        log_env_variables({"CUSTOM_VAR": "custom_value"})
        
        assert mock_logger.info.call_count >= 2

    def test_mask_sensitive_info(self):
        """Test mask_sensitive_info"""
        data = {
            "username": "user",
            "password": "secret",
            "nested": {
                "api_key": "key123"
            }
        }
        
        result = mask_sensitive_info(data)
        
        assert result["password"] == HIDDEN_VALUE
        assert result["nested"]["api_key"] == HIDDEN_VALUE
        assert result["username"] == "user"


class TestResourceFunctions:
    """Test cases for resource detection functions"""

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.multiprocessing.cpu_count")
    def test_num_cpus(self, mock_cpu_count):
        """Test num_cpus"""
        mock_cpu_count.return_value = 4
        
        result = num_cpus()
        
        assert result == 4

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.subprocess.check_output")
    def test_num_gpus_with_gpus(self, mock_check_output):
        """Test num_gpus when GPUs are present"""
        mock_check_output.return_value = b"GPU 0: Tesla V100\nGPU 1: Tesla V100\n"
        
        result = num_gpus()
        
        assert result == 2

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.subprocess.check_output")
    def test_num_gpus_no_gpus(self, mock_check_output):
        """Test num_gpus when no GPUs are present"""
        mock_check_output.side_effect = OSError()
        
        result = num_gpus()
        
        assert result == 0

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.subprocess.check_output")
    def test_num_neurons_with_neurons(self, mock_check_output):
        """Test num_neurons when neurons are present"""
        mock_check_output.return_value = b'[{"nc_count": 2}, {"nc_count": 2}]'
        
        result = num_neurons()
        
        assert result == 4

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.subprocess.check_output")
    def test_num_neurons_no_neurons(self, mock_check_output):
        """Test num_neurons when no neurons are present"""
        mock_check_output.side_effect = OSError()
        
        result = num_neurons()
        
        assert result == 0


class TestSerializationFunctions:
    """Test cases for serialization functions"""

    def test_safe_serialize_string(self):
        """Test safe_serialize with string"""
        result = safe_serialize("test_string")
        
        assert result == "test_string"

    def test_safe_serialize_dict(self):
        """Test safe_serialize with dictionary"""
        result = safe_serialize({"key": "value"})
        
        assert result == '{"key": "value"}'

    def test_safe_serialize_list(self):
        """Test safe_serialize with list"""
        result = safe_serialize([1, 2, 3])
        
        assert result == "[1, 2, 3]"

    def test_safe_serialize_non_serializable(self):
        """Test safe_serialize with non-serializable object"""
        class CustomObject:
            def __str__(self):
                return "custom_object"
        
        result = safe_serialize(CustomObject())
        
        assert "custom_object" in result


class TestSetEnv:
    """Test cases for set_env function"""

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.environ", {"TRAINING_JOB_NAME": "test-job"})
    def test_set_env_basic(self, mock_neurons, mock_gpus, mock_cpus, mock_file):
        """Test set_env with basic configuration"""
        mock_cpus.return_value = 4
        mock_gpus.return_value = 0
        mock_neurons.return_value = 0
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.m5.xlarge",
            "hosts": ["algo-1"],
            "network_interface_name": "eth0"
        }
        
        set_env(resource_config)
        
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.environ", {"TRAINING_JOB_NAME": "test-job"})
    def test_set_env_with_torchrun(self, mock_neurons, mock_gpus, mock_cpus, mock_file):
        """Test set_env with torchrun distribution"""
        mock_cpus.return_value = 4
        mock_gpus.return_value = 2
        mock_neurons.return_value = 0
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.p3.2xlarge",
            "hosts": ["algo-1", "algo-2"],
            "network_interface_name": "eth0"
        }
        
        set_env(resource_config, distribution="torchrun")
        
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.environ", {"TRAINING_JOB_NAME": "test-job"})
    def test_set_env_with_mpirun(self, mock_neurons, mock_gpus, mock_cpus, mock_file):
        """Test set_env with mpirun distribution"""
        mock_cpus.return_value = 4
        mock_gpus.return_value = 2
        mock_neurons.return_value = 0
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.p3.2xlarge",
            "hosts": ["algo-1", "algo-2"],
            "network_interface_name": "eth0"
        }
        
        set_env(resource_config, distribution="mpirun")
        
        mock_file.assert_called_once()


class TestMain:
    """Test cases for main function"""

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment._parse_args")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment._bootstrap_runtime_env_for_remote_function")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.getpass.getuser")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment.os.path.exists")
    def test_main_success(self, mock_exists, mock_getuser, mock_manager_class, mock_bootstrap, mock_parse):
        """Test main function successful execution"""
        mock_args = Mock()
        mock_args.client_python_version = "3.8"
        mock_args.client_sagemaker_pysdk_version = "2.0.0"
        mock_args.job_conda_env = None
        mock_args.pipeline_execution_id = None
        mock_args.dependency_settings = None
        mock_args.func_step_s3_dir = None
        mock_args.distribution = None
        mock_args.user_nproc_per_node = None
        mock_parse.return_value = mock_args
        
        mock_getuser.return_value = "root"
        mock_exists.return_value = False
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        with pytest.raises(SystemExit) as exc_info:
            main([])
        
        assert exc_info.value.code == SUCCESS_EXIT_CODE

    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment._parse_args")
    @patch("sagemaker.core.remote_function.runtime_environment.bootstrap_runtime_environment._write_failure_reason_file")
    def test_main_failure(self, mock_write_failure, mock_parse):
        """Test main function with failure"""
        mock_parse.side_effect = Exception("Test error")
        
        with pytest.raises(SystemExit) as exc_info:
            main([])
        
        assert exc_info.value.code == DEFAULT_FAILURE_CODE
        mock_write_failure.assert_called_once()
