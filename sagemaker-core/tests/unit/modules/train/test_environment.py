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
import json
import os
from unittest.mock import Mock, patch, mock_open, MagicMock
from sagemaker.core.modules.train.container_drivers.scripts.environment import (
    num_cpus,
    num_gpus,
    num_neurons,
    deserialize_hyperparameters,
    set_env,
    mask_sensitive_info,
    log_key_value,
    log_env_variables,
)


class TestEnvironment:
    """Test cases for environment module"""

    def test_num_cpus(self):
        """Test num_cpus returns positive integer"""
        result = num_cpus()
        assert isinstance(result, int)
        assert result > 0

    @patch("subprocess.check_output")
    def test_num_gpus_with_gpus(self, mock_check_output):
        """Test num_gpus when GPUs are available"""
        mock_check_output.return_value = b"GPU 0: Tesla V100\nGPU 1: Tesla V100\n"
        
        result = num_gpus()
        assert result == 2

    @patch("subprocess.check_output")
    def test_num_gpus_no_gpus(self, mock_check_output):
        """Test num_gpus when no GPUs are available"""
        mock_check_output.side_effect = OSError("nvidia-smi not found")
        
        result = num_gpus()
        assert result == 0

    @patch("subprocess.check_output")
    def test_num_gpus_command_error(self, mock_check_output):
        """Test num_gpus when command fails"""
        import subprocess
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")
        
        result = num_gpus()
        assert result == 0

    @patch("subprocess.check_output")
    def test_num_neurons_with_neurons(self, mock_check_output):
        """Test num_neurons when Neuron cores are available"""
        mock_output = json.dumps([
            {"nc_count": 2},
            {"nc_count": 2}
        ])
        mock_check_output.return_value = mock_output.encode()
        
        result = num_neurons()
        assert result == 4

    @patch("subprocess.check_output")
    def test_num_neurons_no_neurons(self, mock_check_output):
        """Test num_neurons when no Neuron cores are available"""
        mock_check_output.side_effect = OSError("neuron-ls not found")
        
        result = num_neurons()
        assert result == 0

    @patch("subprocess.check_output")
    def test_num_neurons_command_error(self, mock_check_output):
        """Test num_neurons when command fails"""
        import subprocess
        error = subprocess.CalledProcessError(1, "neuron-ls")
        error.output = b"error=No Neuron devices found"
        mock_check_output.side_effect = error
        
        result = num_neurons()
        assert result == 0

    @patch("subprocess.check_output")
    def test_num_neurons_command_error_no_output(self, mock_check_output):
        """Test num_neurons when command fails without output"""
        import subprocess
        error = subprocess.CalledProcessError(1, "neuron-ls")
        error.output = None
        mock_check_output.side_effect = error
        
        result = num_neurons()
        assert result == 0

    def test_deserialize_hyperparameters_simple(self):
        """Test deserialize_hyperparameters with simple types"""
        hyperparameters = {
            "learning_rate": "0.001",
            "epochs": "10",
            "batch_size": "32"
        }
        
        result = deserialize_hyperparameters(hyperparameters)
        
        assert result["learning_rate"] == 0.001
        assert result["epochs"] == 10
        assert result["batch_size"] == 32

    def test_deserialize_hyperparameters_complex(self):
        """Test deserialize_hyperparameters with complex types"""
        hyperparameters = {
            "layers": "[128, 64, 32]",
            "config": '{"optimizer": "adam", "loss": "mse"}',
            "enabled": "true"
        }
        
        result = deserialize_hyperparameters(hyperparameters)
        
        assert result["layers"] == [128, 64, 32]
        assert result["config"] == {"optimizer": "adam", "loss": "mse"}
        assert result["enabled"] is True

    def test_mask_sensitive_info_with_password(self):
        """Test mask_sensitive_info masks password fields"""
        data = {
            "username": "user",
            "password": "secret123",
            "api_key": "key123"
        }
        
        result = mask_sensitive_info(data)
        
        assert result["username"] == "user"
        assert result["password"] == "******"
        assert result["api_key"] == "******"

    def test_mask_sensitive_info_nested(self):
        """Test mask_sensitive_info with nested dictionaries"""
        data = {
            "config": {
                "db_password": "secret",
                "db_host": "localhost"
            }
        }
        
        result = mask_sensitive_info(data)
        
        assert result["config"]["db_password"] == "******"
        assert result["config"]["db_host"] == "localhost"

    def test_mask_sensitive_info_case_insensitive(self):
        """Test mask_sensitive_info is case insensitive"""
        data = {
            "API_KEY": "key123",
            "Secret_Token": "token123"
        }
        
        result = mask_sensitive_info(data)
        
        assert result["API_KEY"] == "******"
        assert result["Secret_Token"] == "******"

    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.logger")
    def test_log_key_value_sensitive(self, mock_logger):
        """Test log_key_value masks sensitive values"""
        log_key_value("password", "secret123")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0]
        assert "******" in str(call_args)

    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.logger")
    def test_log_key_value_dict(self, mock_logger):
        """Test log_key_value with dictionary value"""
        log_key_value("config", {"key": "value"})
        
        mock_logger.info.assert_called_once()

    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.logger")
    def test_log_key_value_json_string(self, mock_logger):
        """Test log_key_value with JSON string value"""
        log_key_value("config", '{"key": "value"}')
        
        mock_logger.info.assert_called_once()

    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.logger")
    def test_log_key_value_regular(self, mock_logger):
        """Test log_key_value with regular value"""
        log_key_value("learning_rate", "0.001")
        
        mock_logger.info.assert_called_once()

    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.logger")
    @patch.dict(os.environ, {"TEST_VAR": "test_value"})
    def test_log_env_variables(self, mock_logger):
        """Test log_env_variables logs both environment and dict variables"""
        env_vars_dict = {"CUSTOM_VAR": "custom_value"}
        
        log_env_variables(env_vars_dict)
        
        # Should be called for both os.environ and env_vars_dict
        assert mock_logger.info.call_count > 0

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.read_source_code_json")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.read_distributed_json")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_cpus")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_gpus")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_neurons")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.log_env_variables")
    @patch.dict(os.environ, {"TRAINING_JOB_NAME": "test-job"})
    def test_set_env_minimal(self, mock_log_env, mock_neurons, mock_gpus, mock_cpus, 
                            mock_distributed, mock_source_code, mock_file):
        """Test set_env with minimal configuration"""
        mock_cpus.return_value = 4
        mock_gpus.return_value = 0
        mock_neurons.return_value = 0
        mock_source_code.return_value = None
        mock_distributed.return_value = None
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.m5.xlarge",
            "hosts": ["algo-1"],
            "network_interface_name": "eth0"
        }
        
        input_data_config = {
            "training": {"S3Uri": "s3://bucket/data"}
        }
        
        hyperparameters_config = {
            "learning_rate": "0.001",
            "epochs": "10"
        }
        
        set_env(resource_config, input_data_config, hyperparameters_config)
        
        # Verify file was written
        mock_file.assert_called_once()
        handle = mock_file()
        assert handle.write.called

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.read_source_code_json")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.read_distributed_json")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_cpus")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_gpus")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_neurons")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.log_env_variables")
    @patch.dict(os.environ, {"TRAINING_JOB_NAME": "test-job"})
    def test_set_env_with_source_code(self, mock_log_env, mock_neurons, mock_gpus, mock_cpus,
                                      mock_distributed, mock_source_code, mock_file):
        """Test set_env with source code configuration"""
        mock_cpus.return_value = 4
        mock_gpus.return_value = 1
        mock_neurons.return_value = 0
        mock_source_code.return_value = {"entry_script": "train.py"}
        mock_distributed.return_value = None
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.p3.2xlarge",
            "hosts": ["algo-1", "algo-2"],
            "network_interface_name": "eth0"
        }
        
        input_data_config = {
            "training": {"S3Uri": "s3://bucket/data"}
        }
        
        hyperparameters_config = {
            "learning_rate": "0.001"
        }
        
        set_env(resource_config, input_data_config, hyperparameters_config)
        
        # Verify file was written
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.read_source_code_json")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.read_distributed_json")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_cpus")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_gpus")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_neurons")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.log_env_variables")
    @patch.dict(os.environ, {"TRAINING_JOB_NAME": "test-job"})
    def test_set_env_with_distributed(self, mock_log_env, mock_neurons, mock_gpus, mock_cpus,
                                      mock_distributed, mock_source_code, mock_file):
        """Test set_env with distributed configuration"""
        mock_cpus.return_value = 8
        mock_gpus.return_value = 4
        mock_neurons.return_value = 0
        mock_source_code.return_value = None
        mock_distributed.return_value = {"smdistributed": {"dataparallel": {"enabled": True}}}
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.p3.8xlarge",
            "hosts": ["algo-1", "algo-2", "algo-3"],
            "network_interface_name": "eth0"
        }
        
        input_data_config = {
            "training": {"S3Uri": "s3://bucket/data"},
            "validation": {"S3Uri": "s3://bucket/validation"}
        }
        
        hyperparameters_config = {
            "learning_rate": "0.001",
            "batch_size": "64"
        }
        
        set_env(resource_config, input_data_config, hyperparameters_config)
        
        # Verify file was written
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.read_source_code_json")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.read_distributed_json")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_cpus")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_gpus")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.num_neurons")
    @patch("sagemaker.core.modules.train.container_drivers.scripts.environment.log_env_variables")
    @patch.dict(os.environ, {"TRAINING_JOB_NAME": "test-job"})
    def test_set_env_multiple_channels(self, mock_log_env, mock_neurons, mock_gpus, mock_cpus,
                                       mock_distributed, mock_source_code, mock_file):
        """Test set_env with multiple data channels"""
        mock_cpus.return_value = 4
        mock_gpus.return_value = 0
        mock_neurons.return_value = 0
        mock_source_code.return_value = None
        mock_distributed.return_value = None
        
        resource_config = {
            "current_host": "algo-1",
            "current_instance_type": "ml.m5.xlarge",
            "hosts": ["algo-1"],
            "network_interface_name": "eth0"
        }
        
        input_data_config = {
            "training": {"S3Uri": "s3://bucket/train"},
            "validation": {"S3Uri": "s3://bucket/val"},
            "test": {"S3Uri": "s3://bucket/test"}
        }
        
        hyperparameters_config = {}
        
        set_env(resource_config, input_data_config, hyperparameters_config)
        
        # Verify file was written
        mock_file.assert_called_once()
