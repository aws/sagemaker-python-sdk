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
from __future__ import absolute_import

from sagemaker.core.serverless_inference_config import ServerlessInferenceConfig


def test_serverless_inference_config_default_values():
    """Test ServerlessInferenceConfig with default values."""
    config = ServerlessInferenceConfig()
    
    assert config.memory_size_in_mb == 2048
    assert config.max_concurrency == 5
    assert config.provisioned_concurrency is None


def test_serverless_inference_config_custom_values():
    """Test ServerlessInferenceConfig with custom values."""
    config = ServerlessInferenceConfig(
        memory_size_in_mb=4096,
        max_concurrency=10,
        provisioned_concurrency=2
    )
    
    assert config.memory_size_in_mb == 4096
    assert config.max_concurrency == 10
    assert config.provisioned_concurrency == 2


def test_serverless_inference_config_to_request_dict_without_provisioned():
    """Test _to_request_dict without provisioned_concurrency."""
    config = ServerlessInferenceConfig(
        memory_size_in_mb=3072,
        max_concurrency=8
    )
    
    request_dict = config._to_request_dict()
    
    assert request_dict == {
        "MemorySizeInMB": 3072,
        "MaxConcurrency": 8
    }
    assert "ProvisionedConcurrency" not in request_dict


def test_serverless_inference_config_to_request_dict_with_provisioned():
    """Test _to_request_dict with provisioned_concurrency."""
    config = ServerlessInferenceConfig(
        memory_size_in_mb=5120,
        max_concurrency=15,
        provisioned_concurrency=3
    )
    
    request_dict = config._to_request_dict()
    
    assert request_dict == {
        "MemorySizeInMB": 5120,
        "MaxConcurrency": 15,
        "ProvisionedConcurrency": 3
    }


def test_serverless_inference_config_minimum_memory():
    """Test ServerlessInferenceConfig with minimum memory size."""
    config = ServerlessInferenceConfig(memory_size_in_mb=1024)
    
    assert config.memory_size_in_mb == 1024
    request_dict = config._to_request_dict()
    assert request_dict["MemorySizeInMB"] == 1024


def test_serverless_inference_config_maximum_memory():
    """Test ServerlessInferenceConfig with maximum memory size."""
    config = ServerlessInferenceConfig(memory_size_in_mb=6144)
    
    assert config.memory_size_in_mb == 6144
    request_dict = config._to_request_dict()
    assert request_dict["MemorySizeInMB"] == 6144


def test_serverless_inference_config_max_concurrency_one():
    """Test ServerlessInferenceConfig with max_concurrency of 1."""
    config = ServerlessInferenceConfig(max_concurrency=1)
    
    assert config.max_concurrency == 1
    request_dict = config._to_request_dict()
    assert request_dict["MaxConcurrency"] == 1


def test_serverless_inference_config_provisioned_concurrency_zero():
    """Test ServerlessInferenceConfig with provisioned_concurrency of 0."""
    config = ServerlessInferenceConfig(provisioned_concurrency=0)
    
    assert config.provisioned_concurrency == 0
    request_dict = config._to_request_dict()
    assert request_dict["ProvisionedConcurrency"] == 0


def test_serverless_inference_config_all_parameters():
    """Test ServerlessInferenceConfig with all parameters specified."""
    config = ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=20,
        provisioned_concurrency=5
    )
    
    assert config.memory_size_in_mb == 2048
    assert config.max_concurrency == 20
    assert config.provisioned_concurrency == 5
    
    request_dict = config._to_request_dict()
    assert len(request_dict) == 3
    assert "MemorySizeInMB" in request_dict
    assert "MaxConcurrency" in request_dict
    assert "ProvisionedConcurrency" in request_dict
