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
"""Unit tests for sagemaker.core.image_retriever.image_retriever module."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.core.image_retriever.image_retriever import ImageRetriever


class TestImageRetriever:
    """Test ImageRetriever class."""

    @patch("sagemaker.core.image_retriever.image_retriever._botocore_resolver")
    @patch("sagemaker.core.image_retriever.image_retriever.config_for_framework")
    def test_retrieve_base_python_image_uri(self, mock_config, mock_resolver):
        """Test retrieve_base_python_image_uri method."""
        # Setup mocks
        mock_endpoint = {"hostname": "ecr.us-west-2.amazonaws.com"}
        mock_resolver.return_value.construct_endpoint.return_value = mock_endpoint
        
        mock_config.return_value = {
            "versions": {
                "1.0": {
                    "registries": {"us-west-2": "123456789"},
                    "repository": "sagemaker-base-python"
                }
            }
        }
        
        # Test
        result = ImageRetriever.retrieve_base_python_image_uri("us-west-2", "310")
        
        # Verify
        assert "123456789.dkr.ecr.us-west-2.amazonaws.com" in result
        assert "sagemaker-base-python-310:1.0" in result

    @patch("sagemaker.core.image_retriever.image_retriever._botocore_resolver")
    @patch("sagemaker.core.image_retriever.image_retriever.config_for_framework")
    def test_retrieve_base_python_image_uri_default_py_version(self, mock_config, mock_resolver):
        """Test retrieve_base_python_image_uri with default Python version."""
        mock_endpoint = {"hostname": "ecr.us-west-2.amazonaws.com"}
        mock_resolver.return_value.construct_endpoint.return_value = mock_endpoint
        
        mock_config.return_value = {
            "versions": {
                "1.0": {
                    "registries": {"us-west-2": "123456789"},
                    "repository": "sagemaker-base-python"
                }
            }
        }
        
        result = ImageRetriever.retrieve_base_python_image_uri("us-west-2")
        
        assert "sagemaker-base-python-310:1.0" in result

    @patch("sagemaker.core.image_retriever.image_retriever._retrieve_latest_pytorch_training_uri")
    def test_retrieve_pytorch_uri_all_defaults(self, mock_latest):
        """Test retrieve_pytorch_uri with all default parameters."""
        mock_latest.return_value = "123456789.dkr.ecr.us-west-2.amazonaws.com/pytorch:latest"
        
        result = ImageRetriever.retrieve_pytorch_uri(region="us-west-2")
        
        mock_latest.assert_called_once_with("us-west-2")
        assert result == "123456789.dkr.ecr.us-west-2.amazonaws.com/pytorch:latest"

    @patch("sagemaker.core.image_retriever.image_retriever._botocore_resolver")
    @patch("sagemaker.core.image_retriever.image_retriever._config_for_framework_and_scope")
    @patch("sagemaker.core.image_retriever.image_retriever._validate_version_and_set_if_needed")
    @patch("sagemaker.core.image_retriever.image_retriever._validate_py_version_and_set_if_needed")
    @patch("sagemaker.core.image_retriever.image_retriever._registry_from_region")
    @patch("sagemaker.core.image_retriever.image_retriever._processor")
    @patch("sagemaker.core.image_retriever.image_retriever._get_image_tag")
    def test_retrieve_pytorch_uri_with_params(
        self, mock_tag, mock_processor, mock_registry, mock_py_ver, 
        mock_ver, mock_config, mock_resolver
    ):
        """Test retrieve_pytorch_uri with specific parameters."""
        # Setup mocks
        mock_endpoint = {"hostname": "ecr.us-west-2.amazonaws.com"}
        mock_resolver.return_value.construct_endpoint.return_value = mock_endpoint
        
        mock_config.return_value = {
            "versions": {
                "2.0": {
                    "repository": "pytorch-training",
                    "registries": {"us-west-2": "123456789"}
                }
            }
        }
        mock_ver.return_value = "2.0"
        mock_py_ver.return_value = "py310"
        mock_registry.return_value = "123456789"
        mock_processor.return_value = "gpu"
        mock_tag.return_value = "2.0-gpu-py310"
        
        result = ImageRetriever.retrieve_pytorch_uri(
            region="us-west-2",
            version="2.0",
            py_version="py310",
            instance_type="ml.p3.2xlarge"
        )
        
        assert "123456789.dkr.ecr.us-west-2.amazonaws.com" in result
        assert "pytorch-training:2.0-gpu-py310" in result

    @patch("sagemaker.core.image_retriever.image_retriever.SageMakerConfig")
    @patch("sagemaker.core.image_retriever.image_retriever._botocore_resolver")
    @patch("sagemaker.core.image_retriever.image_retriever._config_for_framework_and_scope")
    @patch("sagemaker.core.image_retriever.image_retriever._get_final_image_scope")
    @patch("sagemaker.core.image_retriever.image_retriever._get_inference_tool")
    @patch("sagemaker.core.image_retriever.image_retriever._validate_for_suppported_frameworks_and_instance_type")
    def test_retrieve_hugging_face_uri_basic(self, mock_validate, mock_inference_tool, mock_final_scope, mock_config, mock_resolver, mock_sagemaker_config):
        """Test retrieve_hugging_face_uri with basic parameters."""
        mock_endpoint = {"hostname": "ecr.us-west-2.amazonaws.com"}
        mock_resolver.return_value.construct_endpoint.return_value = mock_endpoint
        mock_sagemaker_config.resolve_value_from_config.return_value = None
        mock_inference_tool.return_value = None
        mock_final_scope.return_value = "training"
        
        mock_config.return_value = {
            "versions": {
                "4.26": {
                    "version_aliases": {"pytorch2.0": "pytorch2.0"},
                    "pytorch2.0": {
                        "py310": {
                            "repository": "huggingface-pytorch-training",
                            "registries": {"us-west-2": "123456789"}
                        }
                    }
                }
            }
        }
        
        with patch("sagemaker.core.image_retriever.image_retriever._validate_version_and_set_if_needed") as mock_ver:
            with patch("sagemaker.core.image_retriever.image_retriever._validate_py_version_and_set_if_needed") as mock_py:
                with patch("sagemaker.core.image_retriever.image_retriever._registry_from_region") as mock_reg:
                    with patch("sagemaker.core.image_retriever.image_retriever._processor") as mock_proc:
                        with patch("sagemaker.core.image_retriever.image_retriever._get_image_tag") as mock_tag:
                            with patch("sagemaker.core.image_retriever.image_retriever._version_for_config") as mock_ver_config:
                                with patch("sagemaker.core.image_retriever.image_retriever._validate_arg"):
                                    with patch("sagemaker.core.image_retriever.image_retriever._validate_instance_deprecation"):
                                        mock_ver.return_value = "4.26"
                                        mock_py.return_value = "py310"
                                        mock_reg.return_value = "123456789"
                                        mock_proc.return_value = "gpu"
                                        mock_tag.return_value = "4.26-gpu-py310"
                                        mock_ver_config.return_value = "4.26"
                                        
                                        result = ImageRetriever.retrieve_hugging_face_uri(
                                            region="us-west-2",
                                            version="4.26",
                                            base_framework_version="pytorch2.0"
                                        )
                                        
                                        assert "123456789.dkr.ecr.us-west-2.amazonaws.com" in result

    @patch("sagemaker.core.image_retriever.image_retriever.SageMakerConfig")
    @patch("sagemaker.core.image_retriever.image_retriever._botocore_resolver")
    @patch("sagemaker.core.image_retriever.image_retriever._config_for_framework_and_scope")
    def test_retrieve_with_pytorch_framework(self, mock_config, mock_resolver, mock_sagemaker_config):
        """Test retrieve method with PyTorch framework."""
        mock_endpoint = {"hostname": "ecr.us-west-2.amazonaws.com"}
        mock_resolver.return_value.construct_endpoint.return_value = mock_endpoint
        mock_sagemaker_config.resolve_value_from_config.return_value = None
        
        with patch.object(ImageRetriever, "retrieve_pytorch_uri") as mock_pytorch:
            mock_pytorch.return_value = "pytorch-uri"
            
            result = ImageRetriever.retrieve(
                framework="pytorch",
                region="us-west-2",
                version="2.0"
            )
            
            assert result == "pytorch-uri"
            mock_pytorch.assert_called_once()

    @patch("sagemaker.core.image_retriever.image_retriever.SageMakerConfig")
    @patch("sagemaker.core.image_retriever.image_retriever._botocore_resolver")
    def test_retrieve_with_huggingface_framework(self, mock_resolver, mock_sagemaker_config):
        """Test retrieve method with HuggingFace framework."""
        mock_endpoint = {"hostname": "ecr.us-west-2.amazonaws.com"}
        mock_resolver.return_value.construct_endpoint.return_value = mock_endpoint
        mock_sagemaker_config.resolve_value_from_config.return_value = None
        
        with patch.object(ImageRetriever, "retrieve_hugging_face_uri") as mock_hf:
            mock_hf.return_value = "huggingface-uri"
            
            result = ImageRetriever.retrieve(
                framework="huggingface",
                region="us-west-2",
                version="4.26",
                base_framework_version="pytorch2.0"
            )
            
            assert result == "huggingface-uri"
            mock_hf.assert_called_once()

    @patch("sagemaker.core.image_retriever.image_retriever.SageMakerConfig")
    def test_retrieve_with_pipeline_variable_raises_error(self, mock_sagemaker_config):
        """Test that retrieve raises ValueError with pipeline variable."""
        from sagemaker.core.helper.pipeline_variable import PipelineVariable
        
        mock_sagemaker_config.resolve_value_from_config.return_value = None
        
        # Create a concrete implementation of PipelineVariable for testing
        class TestPipelineVariable(PipelineVariable):
            @property
            def expr(self):
                return {"test": "value"}
            
            @property
            def _referenced_steps(self):
                return []
        
        pipeline_var = TestPipelineVariable()
        
        with pytest.raises(ValueError, match="should not be a pipeline variable"):
            ImageRetriever.retrieve(
                framework="pytorch",
                region=pipeline_var,
                version="2.0"
            )

    def test_retrieve_jumpstart_uri_not_implemented(self):
        """Test that retrieve_jumpstart_uri is not yet implemented."""
        # This method currently has no implementation
        result = ImageRetriever.retrieve_jumpstart_uri()
        assert result is None
