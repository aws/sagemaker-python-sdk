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
"""Tests for bootstrap_runtime_environment module."""
from __future__ import absolute_import

import json
import os
import pytest
import subprocess
from unittest.mock import patch, MagicMock, mock_open, call

from sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment import (
    _parse_args,
    _bootstrap_runtime_env_for_remote_function,
    _bootstrap_runtime_env_for_pipeline_step,
    _handle_pre_exec_scripts,
    _install_dependencies,
    _unpack_user_workspace,
    _write_failure_reason_file,
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
    FAILURE_REASON_PATH,
    REMOTE_FUNCTION_WORKSPACE,
    BASE_CHANNEL_PATH,
    JOB_REMOTE_FUNCTION_WORKSPACE,
    SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME,
    SENSITIVE_KEYWORDS,
    HIDDEN_VALUE,
)
from sagemaker.train.remote_function.runtime_environment.runtime_environment_manager import (
    _DependencySettings,
)


class TestParseArgs:
    """Test _parse_args function."""

    def test_parse_required_args(self):
        """Test parsing required arguments."""
        args = [
            "--client_python_version", "3.8",
        ]
        parsed = _parse_args(args)
        assert parsed.client_python_version == "3.8"

    def test_parse_all_args(self):
        """Test parsing all arguments."""
        args = [
            "--job_conda_env", "my-env",
            "--client_python_version", "3.9",
            "--client_sagemaker_pysdk_version", "2.100.0",
            "--pipeline_execution_id", "exec-123",
            "--dependency_settings", '{"dependency_file": "requirements.txt"}',
            "--func_step_s3_dir", "s3://bucket/func",
            "--distribution", "torchrun",
            "--user_nproc_per_node", "4",
        ]
        parsed = _parse_args(args)
        assert parsed.job_conda_env == "my-env"
        assert parsed.client_python_version == "3.9"
        assert parsed.client_sagemaker_pysdk_version == "2.100.0"
        assert parsed.pipeline_execution_id == "exec-123"
        assert parsed.dependency_settings == '{"dependency_file": "requirements.txt"}'
        assert parsed.func_step_s3_dir == "s3://bucket/func"
        assert parsed.distribution == "torchrun"
        assert parsed.user_nproc_per_node == "4"

    def test_parse_default_values(self):
        """Test default values for optional arguments."""
        args = [
            "--client_python_version", "3.8",
        ]
        parsed = _parse_args(args)
        assert parsed.job_conda_env is None
        assert parsed.client_sagemaker_pysdk_version is None
        assert parsed.pipeline_execution_id is None
        assert parsed.dependency_settings is None
        assert parsed.func_step_s3_dir is None
        assert parsed.distribution is None
        assert parsed.user_nproc_per_node is None


class TestLogKeyValue:
    """Test log_key_value function."""

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.logger")
    def test_logs_regular_value(self, mock_logger):
        """Test logs regular key-value pair."""
        log_key_value("my_name", "my_value")
        mock_logger.info.assert_called_once_with("%s=%s", "my_name", "my_value")

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.logger")
    def test_masks_sensitive_key(self, mock_logger):
        """Test masks sensitive keywords."""
        for keyword in ["PASSWORD", "SECRET", "TOKEN", "KEY", "PRIVATE", "CREDENTIALS"]:
            mock_logger.reset_mock()
            log_key_value(f"my_{keyword}", "sensitive_value")
            mock_logger.info.assert_called_once_with("%s=%s", f"my_{keyword}", HIDDEN_VALUE)

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.logger")
    def test_logs_dict_value(self, mock_logger):
        """Test logs dictionary value."""
        value = {"field1": "value1", "field2": "value2"}
        log_key_value("my_config", value)
        mock_logger.info.assert_called_once_with("%s=%s", "my_config", json.dumps(value))

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.logger")
    def test_logs_json_string_value(self, mock_logger):
        """Test logs JSON string value."""
        value = '{"key1": "value1"}'
        log_key_value("my_key", value)
        mock_logger.info.assert_called_once()


class TestLogEnvVariables:
    """Test log_env_variables function."""

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.log_key_value")
    @patch.dict("os.environ", {"ENV_VAR1": "value1", "ENV_VAR2": "value2"})
    def test_logs_env_and_dict_variables(self, mock_log_kv):
        """Test logs both environment and dictionary variables."""
        env_dict = {"DICT_VAR1": "dict_value1", "DICT_VAR2": "dict_value2"}
        log_env_variables(env_dict)
        
        # Should be called for env vars and dict vars
        assert mock_log_kv.call_count >= 4


class TestMaskSensitiveInfo:
    """Test mask_sensitive_info function."""

    def test_masks_sensitive_keys_in_dict(self):
        """Test masks sensitive keys in dictionary."""
        data = {
            "username": "user",
            "password": "secret123",
            "api_key": "key123",
        }
        result = mask_sensitive_info(data)
        assert result["username"] == "user"
        assert result["password"] == HIDDEN_VALUE
        assert result["api_key"] == HIDDEN_VALUE

    def test_masks_nested_dict(self):
        """Test masks sensitive keys in nested dictionary."""
        data = {
            "config": {
                "username": "user",
                "secret": "secret123",
            }
        }
        result = mask_sensitive_info(data)
        assert result["config"]["username"] == "user"
        assert result["config"]["secret"] == HIDDEN_VALUE

    def test_returns_non_dict_unchanged(self):
        """Test returns non-dictionary unchanged."""
        data = "string_value"
        result = mask_sensitive_info(data)
        assert result == "string_value"


class TestNumCpus:
    """Test num_cpus function."""

    @patch("multiprocessing.cpu_count")
    def test_returns_cpu_count(self, mock_cpu_count):
        """Test returns CPU count."""
        mock_cpu_count.return_value = 8
        assert num_cpus() == 8


class TestNumGpus:
    """Test num_gpus function."""

    @patch("subprocess.check_output")
    def test_returns_gpu_count(self, mock_check_output):
        """Test returns GPU count."""
        mock_check_output.return_value = b"GPU 0: Tesla V100\nGPU 1: Tesla V100\n"
        assert num_gpus() == 2

    @patch("subprocess.check_output")
    def test_returns_zero_on_error(self, mock_check_output):
        """Test returns zero when nvidia-smi fails."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")
        assert num_gpus() == 0

    @patch("subprocess.check_output")
    def test_returns_zero_on_os_error(self, mock_check_output):
        """Test returns zero when nvidia-smi not found."""
        mock_check_output.side_effect = OSError()
        assert num_gpus() == 0


class TestNumNeurons:
    """Test num_neurons function."""

    @patch("subprocess.check_output")
    def test_returns_neuron_count(self, mock_check_output):
        """Test returns neuron core count."""
        mock_output = json.dumps([{"nc_count": 2}, {"nc_count": 4}])
        mock_check_output.return_value = mock_output.encode("utf-8")
        assert num_neurons() == 6

    @patch("subprocess.check_output")
    def test_returns_zero_on_os_error(self, mock_check_output):
        """Test returns zero when neuron-ls not found."""
        mock_check_output.side_effect = OSError()
        assert num_neurons() == 0

    @patch("subprocess.check_output")
    def test_returns_zero_on_called_process_error(self, mock_check_output):
        """Test returns zero when neuron-ls fails."""
        error = subprocess.CalledProcessError(1, "neuron-ls")
        error.output = b"error=No neuron devices found"
        mock_check_output.side_effect = error
        assert num_neurons() == 0


class TestSafeSerialize:
    """Test safe_serialize function."""

    def test_returns_string_as_is(self):
        """Test returns string without quotes."""
        assert safe_serialize("test_string") == "test_string"

    def test_serializes_dict(self):
        """Test serializes dictionary."""
        data = {"key": "value"}
        assert safe_serialize(data) == '{"key": "value"}'

    def test_serializes_list(self):
        """Test serializes list."""
        data = [1, 2, 3]
        assert safe_serialize(data) == "[1, 2, 3]"

    def test_returns_str_for_non_serializable(self):
        """Test returns str() for non-serializable objects."""
        class CustomObj:
            def __str__(self):
                return "custom_object"
        
        obj = CustomObj()
        assert safe_serialize(obj) == "custom_object"


class TestSetEnv:
    """Test set_env function."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.log_env_variables")
    @patch.dict("os.environ", {"TRAINING_JOB_NAME": "test-job"})
    def test_sets_basic_env_vars(self, mock_log_env, mock_neurons, mock_gpus, mock_cpus, mock_file):
        """Test sets basic environment variables."""
        mock_cpus.return_value = 8
        mock_gpus.return_value = 2
        mock_neurons.return_value = 0
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.p3.2xlarge",
            "hosts": ["algo-1", "algo-2"],
            "network_interface_name": "eth0",
        }
        
        set_env(resource_config)
        
        mock_file.assert_called_once()
        mock_log_env.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.log_env_variables")
    @patch.dict("os.environ", {"TRAINING_JOB_NAME": "test-job"})
    def test_sets_torchrun_distribution_vars(self, mock_log_env, mock_neurons, mock_gpus, mock_cpus, mock_file):
        """Test sets torchrun distribution environment variables."""
        mock_cpus.return_value = 8
        mock_gpus.return_value = 2
        mock_neurons.return_value = 0
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.p4d.24xlarge",
            "hosts": ["algo-1"],
            "network_interface_name": "eth0",
        }
        
        set_env(resource_config, distribution="torchrun")
        
        # Verify file was written
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.log_env_variables")
    @patch.dict("os.environ", {"TRAINING_JOB_NAME": "test-job"})
    def test_sets_mpirun_distribution_vars(self, mock_log_env, mock_neurons, mock_gpus, mock_cpus, mock_file):
        """Test sets mpirun distribution environment variables."""
        mock_cpus.return_value = 8
        mock_gpus.return_value = 2
        mock_neurons.return_value = 0
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.p3.2xlarge",
            "hosts": ["algo-1", "algo-2"],
            "network_interface_name": "eth0",
        }
        
        set_env(resource_config, distribution="mpirun")
        
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.log_env_variables")
    @patch.dict("os.environ", {"TRAINING_JOB_NAME": "test-job"})
    def test_uses_user_nproc_per_node(self, mock_log_env, mock_neurons, mock_gpus, mock_cpus, mock_file):
        """Test uses user-specified nproc_per_node."""
        mock_cpus.return_value = 8
        mock_gpus.return_value = 2
        mock_neurons.return_value = 0
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.p3.2xlarge",
            "hosts": ["algo-1"],
            "network_interface_name": "eth0",
        }
        
        set_env(resource_config, user_nproc_per_node="4")
        
        mock_file.assert_called_once()


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


class TestUnpackUserWorkspace:
    """Test _unpack_user_workspace function."""

    @patch("os.path.exists")
    def test_returns_none_if_dir_not_exists(self, mock_exists):
        """Test returns None if workspace directory doesn't exist."""
        mock_exists.return_value = False
        
        result = _unpack_user_workspace()
        
        assert result is None

    @patch("os.path.isfile")
    @patch("os.path.exists")
    def test_returns_none_if_archive_not_exists(self, mock_exists, mock_isfile):
        """Test returns None if workspace archive doesn't exist."""
        mock_exists.return_value = True
        mock_isfile.return_value = False
        
        result = _unpack_user_workspace()
        
        assert result is None

    @patch("shutil.unpack_archive")
    @patch("os.path.isfile")
    @patch("os.path.exists")
    @patch("os.getcwd")
    def test_unpacks_workspace_successfully(self, mock_getcwd, mock_exists, mock_isfile, mock_unpack):
        """Test unpacks workspace successfully."""
        mock_getcwd.return_value = "/tmp/workspace"
        mock_exists.return_value = True
        mock_isfile.return_value = True
        
        result = _unpack_user_workspace()
        
        mock_unpack.assert_called_once()
        assert result is not None


class TestHandlePreExecScripts:
    """Test _handle_pre_exec_scripts function."""

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    def test_runs_pre_exec_script(self, mock_manager_class):
        """Test runs pre-execution script."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        _handle_pre_exec_scripts("/tmp/scripts")
        
        mock_manager.run_pre_exec_script.assert_called_once()


class TestInstallDependencies:
    """Test _install_dependencies function."""

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    def test_installs_with_dependency_settings(self, mock_manager_class):
        """Test installs dependencies with dependency settings."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        dep_settings = _DependencySettings(dependency_file="requirements.txt")
        
        _install_dependencies(
            "/tmp/deps",
            "my-env",
            "3.8",
            "channel",
            dep_settings
        )
        
        mock_manager.bootstrap.assert_called_once()

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    def test_skips_if_no_dependency_file(self, mock_manager_class):
        """Test skips installation if no dependency file."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        dep_settings = _DependencySettings(dependency_file=None)
        
        _install_dependencies(
            "/tmp/deps",
            "my-env",
            "3.8",
            "channel",
            dep_settings
        )
        
        mock_manager.bootstrap.assert_not_called()

    @patch("os.listdir")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    def test_finds_dependency_file_legacy(self, mock_manager_class, mock_listdir):
        """Test finds dependency file in legacy mode."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_listdir.return_value = ["requirements.txt", "script.py"]
        
        _install_dependencies(
            "/tmp/deps",
            "my-env",
            "3.8",
            "channel",
            None
        )
        
        mock_manager.bootstrap.assert_called_once()


class TestBootstrapRuntimeEnvForRemoteFunction:
    """Test _bootstrap_runtime_env_for_remote_function function."""

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._install_dependencies")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._handle_pre_exec_scripts")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace")
    def test_bootstraps_successfully(self, mock_unpack, mock_handle_scripts, mock_install):
        """Test bootstraps runtime environment successfully."""
        mock_unpack.return_value = "/tmp/workspace"
        
        _bootstrap_runtime_env_for_remote_function("3.8", "my-env", None)
        
        mock_unpack.assert_called_once()
        mock_handle_scripts.assert_called_once()
        mock_install.assert_called_once()

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace")
    def test_returns_early_if_no_workspace(self, mock_unpack):
        """Test returns early if no workspace to unpack."""
        mock_unpack.return_value = None
        
        _bootstrap_runtime_env_for_remote_function("3.8", "my-env", None)
        
        mock_unpack.assert_called_once()


class TestBootstrapRuntimeEnvForPipelineStep:
    """Test _bootstrap_runtime_env_for_pipeline_step function."""

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._install_dependencies")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._handle_pre_exec_scripts")
    @patch("shutil.copy")
    @patch("os.listdir")
    @patch("os.path.exists")
    @patch("os.mkdir")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace")
    def test_bootstraps_with_workspace(self, mock_unpack, mock_mkdir, mock_exists, mock_listdir, mock_copy, mock_handle_scripts, mock_install):
        """Test bootstraps pipeline step with workspace."""
        mock_unpack.return_value = "/tmp/workspace"
        mock_exists.return_value = True
        mock_listdir.return_value = ["requirements.txt"]
        
        _bootstrap_runtime_env_for_pipeline_step("3.8", "func_step", "my-env", None)
        
        mock_unpack.assert_called_once()
        mock_handle_scripts.assert_called_once()
        mock_install.assert_called_once()

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._install_dependencies")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._handle_pre_exec_scripts")
    @patch("os.path.exists")
    @patch("os.mkdir")
    @patch("os.getcwd")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace")
    def test_creates_workspace_if_none(self, mock_unpack, mock_getcwd, mock_mkdir, mock_exists, mock_handle_scripts, mock_install):
        """Test creates workspace directory if none exists."""
        mock_unpack.return_value = None
        mock_getcwd.return_value = "/tmp"
        mock_exists.return_value = False
        
        _bootstrap_runtime_env_for_pipeline_step("3.8", "func_step", "my-env", None)
        
        mock_mkdir.assert_called_once()


class TestMain:
    """Test main function."""

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.set_env")
    @patch("builtins.open", new_callable=mock_open, read_data='{"current_host": "algo-1", "current_instance_type": "ml.m5.xlarge", "hosts": ["algo-1"], "network_interface_name": "eth0"}')
    @patch("os.path.exists")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._bootstrap_runtime_env_for_remote_function")
    @patch("getpass.getuser")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._parse_args")
    def test_main_success(self, mock_parse_args, mock_getuser, mock_bootstrap, mock_manager_class, mock_exists, mock_file, mock_set_env):
        """Test main function successful execution."""
        mock_getuser.return_value = "root"
        mock_exists.return_value = True
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        # Mock parsed args
        mock_args = MagicMock()
        mock_args.client_python_version = "3.8"
        mock_args.client_sagemaker_pysdk_version = None
        mock_args.job_conda_env = None
        mock_args.pipeline_execution_id = None
        mock_args.dependency_settings = None
        mock_args.func_step_s3_dir = None
        mock_args.distribution = None
        mock_args.user_nproc_per_node = None
        mock_parse_args.return_value = mock_args
        
        args = [
            "--client_python_version", "3.8",
        ]
        
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        
        assert exc_info.value.code == SUCCESS_EXIT_CODE
        mock_bootstrap.assert_called_once()

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._write_failure_reason_file")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    @patch("getpass.getuser")
    def test_main_handles_exception(self, mock_getuser, mock_manager_class, mock_write_failure):
        """Test main function handles exceptions."""
        mock_getuser.return_value = "root"
        mock_manager = MagicMock()
        mock_manager._validate_python_version.side_effect = Exception("Test error")
        mock_manager_class.return_value = mock_manager
        
        args = [
            "--client_python_version", "3.8",
        ]
        
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        
        assert exc_info.value.code == DEFAULT_FAILURE_CODE
        mock_write_failure.assert_called_once()

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.set_env")
    @patch("builtins.open", new_callable=mock_open, read_data='{"current_host": "algo-1", "current_instance_type": "ml.m5.xlarge", "hosts": ["algo-1"], "network_interface_name": "eth0"}')
    @patch("os.path.exists")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._bootstrap_runtime_env_for_pipeline_step")
    @patch("getpass.getuser")
    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment._parse_args")
    def test_main_pipeline_execution(self, mock_parse_args, mock_getuser, mock_bootstrap, mock_manager_class, mock_exists, mock_file, mock_set_env):
        """Test main function for pipeline execution."""
        mock_getuser.return_value = "root"
        mock_exists.return_value = True
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        # Mock parsed args
        mock_args = MagicMock()
        mock_args.client_python_version = "3.8"
        mock_args.client_sagemaker_pysdk_version = None
        mock_args.job_conda_env = None
        mock_args.pipeline_execution_id = "exec-123"
        mock_args.dependency_settings = None
        mock_args.func_step_s3_dir = "s3://bucket/func"
        mock_args.distribution = None
        mock_args.user_nproc_per_node = None
        mock_parse_args.return_value = mock_args
        
        args = [
            "--client_python_version", "3.8",
            "--pipeline_execution_id", "exec-123",
            "--func_step_s3_dir", "s3://bucket/func",
        ]
        
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        
        assert exc_info.value.code == SUCCESS_EXIT_CODE
        mock_bootstrap.assert_called_once()

    @patch("sagemaker.train.remote_function.runtime_environment.bootstrap_runtime_environment.RuntimeEnvironmentManager")
    @patch("getpass.getuser")
    def test_main_non_root_user(self, mock_getuser, mock_manager_class):
        """Test main function with non-root user."""
        mock_getuser.return_value = "ubuntu"
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        args = [
            "--client_python_version", "3.8",
        ]
        
        with pytest.raises(SystemExit):
            main(args)
        
        mock_manager.change_dir_permission.assert_called_once()
