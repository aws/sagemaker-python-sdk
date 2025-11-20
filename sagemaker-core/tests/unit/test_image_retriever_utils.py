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
"""Unit tests for sagemaker.core.image_retriever.image_retriever_utils module."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.core.image_retriever.image_retriever_utils import (
    _get_image_tag,
    _get_final_image_scope,
    _get_inference_tool,
    _get_latest_versions,
    _validate_arg,
    _validate_framework,
    _format_tag,
    _processor,
    _registry_from_region,
    _validate_py_version_and_set_if_needed,
    _validate_version_and_set_if_needed,
    _validate_instance_deprecation,
    _validate_for_suppported_frameworks_and_instance_type,
    _retrieve_pytorch_uri_inputs_are_all_default,
)


class TestGetImageTag:
    """Test _get_image_tag function."""

    def test_get_image_tag_basic(self):
        """Test _get_image_tag with basic parameters."""
        tag = _get_image_tag(
            container_version="cu118",
            distributed=False,
            final_image_scope="training",
            framework="pytorch",
            inference_tool=None,
            instance_type="ml.p3.2xlarge",
            processor="gpu",
            py_version="py310",
            tag_prefix="2.0",
            version="2.0"
        )
        assert "2.0" in tag
        assert "gpu" in tag
        assert "py310" in tag

    def test_get_image_tag_with_inference_tool(self):
        """Test _get_image_tag with inference tool."""
        tag = _get_image_tag(
            container_version="neuron",
            distributed=False,
            final_image_scope="inference",
            framework="pytorch",
            inference_tool="neuron",
            instance_type="ml.inf1.xlarge",
            processor="inf",
            py_version="py310",
            tag_prefix="2.0",
            version="2.0"
        )
        assert "neuron" in tag

    def test_get_image_tag_xgboost_graviton(self):
        """Test _get_image_tag for XGBoost on Graviton."""
        tag = _get_image_tag(
            container_version=None,
            distributed=False,
            final_image_scope="inference_graviton",
            framework="xgboost",
            inference_tool=None,
            instance_type="ml.c7g.xlarge",
            processor="cpu",
            py_version="py3",
            tag_prefix="1.5-1",
            version="1.5-1"
        )
        assert "1.5-1-arm64" in tag


class TestGetFinalImageScope:
    """Test _get_final_image_scope function."""

    def test_get_final_image_scope_graviton(self):
        """Test _get_final_image_scope for Graviton instances."""
        scope = _get_final_image_scope("xgboost", "ml.c7g.xlarge", "inference")
        assert scope == "inference_graviton"

    def test_get_final_image_scope_xgboost_default(self):
        """Test _get_final_image_scope for XGBoost without scope."""
        scope = _get_final_image_scope("xgboost", "ml.m5.xlarge", None)
        assert scope == "training"

    def test_get_final_image_scope_regular(self):
        """Test _get_final_image_scope for regular instances."""
        scope = _get_final_image_scope("pytorch", "ml.p3.2xlarge", "training")
        assert scope == "training"


class TestGetInferenceTool:
    """Test _get_inference_tool function."""

    def test_get_inference_tool_with_tool(self):
        """Test _get_inference_tool when tool is provided."""
        tool = _get_inference_tool("neuron", "ml.p3.2xlarge")
        assert tool == "neuron"

    def test_get_inference_tool_inf_instance(self):
        """Test _get_inference_tool for Inferentia instance."""
        tool = _get_inference_tool(None, "ml.inf1.xlarge")
        assert tool == "neuron"

    def test_get_inference_tool_trn_instance(self):
        """Test _get_inference_tool for Trainium instance."""
        tool = _get_inference_tool(None, "ml.trn1.2xlarge")
        assert tool == "neuron"

    def test_get_inference_tool_regular_instance(self):
        """Test _get_inference_tool for regular instance."""
        tool = _get_inference_tool(None, "ml.p3.2xlarge")
        assert tool is None


class TestGetLatestVersions:
    """Test _get_latest_versions function."""

    def test_get_latest_versions(self):
        """Test _get_latest_versions."""
        versions = ["1.0.0", "2.0.0", "1.5.0"]
        latest = _get_latest_versions(versions)
        assert latest == "2.0.0"

    def test_get_latest_versions_single(self):
        """Test _get_latest_versions with single version."""
        versions = ["1.0.0"]
        latest = _get_latest_versions(versions)
        assert latest == "1.0.0"


class TestValidateArg:
    """Test _validate_arg function."""

    def test_validate_arg_valid(self):
        """Test _validate_arg with valid argument."""
        _validate_arg("cpu", ["cpu", "gpu"], "processor")
        # Should not raise

    def test_validate_arg_invalid(self):
        """Test _validate_arg with invalid argument."""
        with pytest.raises(ValueError, match="Unsupported processor"):
            _validate_arg("tpu", ["cpu", "gpu"], "processor")


class TestValidateFramework:
    """Test _validate_framework function."""

    def test_validate_framework_valid(self):
        """Test _validate_framework with valid framework."""
        _validate_framework("pytorch", ["pytorch", "tensorflow"], "framework", "Trainium")
        # Should not raise

    def test_validate_framework_invalid(self):
        """Test _validate_framework with invalid framework."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            _validate_framework("xgboost", ["pytorch"], "framework", "Trainium")


class TestFormatTag:
    """Test _format_tag function."""

    def test_format_tag_basic(self):
        """Test _format_tag with basic parameters."""
        tag = _format_tag("2.0", "gpu", "py310", "cu118", None)
        assert tag == "2.0-gpu-py310-cu118"

    def test_format_tag_with_inference_tool(self):
        """Test _format_tag with inference tool."""
        tag = _format_tag("2.0", "inf", "py310", "neuron", "neuron")
        assert tag == "2.0-neuron-py310-neuron"

    def test_format_tag_with_none_values(self):
        """Test _format_tag with None values."""
        tag = _format_tag("2.0", "cpu", None, None, None)
        assert tag == "2.0-cpu"


class TestProcessor:
    """Test _processor function."""

    def test_processor_gpu_instance(self):
        """Test _processor for GPU instance."""
        proc = _processor("ml.p3.2xlarge", ["cpu", "gpu"])
        assert proc == "gpu"

    def test_processor_cpu_instance(self):
        """Test _processor for CPU instance."""
        proc = _processor("ml.m5.xlarge", ["cpu", "gpu"])
        assert proc == "cpu"

    def test_processor_local_cpu(self):
        """Test _processor for local CPU."""
        proc = _processor("local", ["cpu", "gpu"])
        assert proc == "cpu"

    def test_processor_local_gpu(self):
        """Test _processor for local GPU."""
        proc = _processor("local_gpu", ["cpu", "gpu"])
        assert proc == "gpu"

    def test_processor_inf_instance(self):
        """Test _processor for Inferentia instance."""
        proc = _processor("ml.inf1.xlarge", ["cpu", "gpu", "inf"])
        assert proc == "inf"

    def test_processor_trn_instance(self):
        """Test _processor for Trainium instance."""
        proc = _processor("ml.trn1.2xlarge", ["cpu", "gpu", "trn"])
        assert proc == "trn"

    def test_processor_serverless(self):
        """Test _processor with serverless config."""
        from sagemaker.core.serverless_inference_config import ServerlessInferenceConfig
        config = ServerlessInferenceConfig()
        proc = _processor(None, ["cpu", "gpu"], serverless_inference_config=config)
        assert proc == "cpu"

    def test_processor_no_instance_type_raises_error(self):
        """Test _processor without instance type raises error."""
        with pytest.raises(ValueError, match="Empty SageMaker instance type"):
            _processor(None, ["cpu", "gpu"])

    def test_processor_invalid_instance_type_raises_error(self):
        """Test _processor with invalid instance type raises error."""
        with pytest.raises(ValueError, match="Invalid SageMaker instance type"):
            _processor("invalid", ["cpu", "gpu"])


class TestRegistryFromRegion:
    """Test _registry_from_region function."""

    def test_registry_from_region_valid(self):
        """Test _registry_from_region with valid region."""
        registry_dict = {"us-west-2": "123456789", "us-east-1": "987654321"}
        registry = _registry_from_region("us-west-2", registry_dict)
        assert registry == "123456789"

    def test_registry_from_region_invalid(self):
        """Test _registry_from_region with invalid region."""
        registry_dict = {"us-west-2": "123456789"}
        with pytest.raises(ValueError, match="Unsupported region"):
            _registry_from_region("invalid-region", registry_dict)


class TestValidatePyVersionAndSetIfNeeded:
    """Test _validate_py_version_and_set_if_needed function."""

    def test_validate_py_version_valid(self):
        """Test _validate_py_version_and_set_if_needed with valid version."""
        version_config = {"repository": "pytorch", "py_versions": ["py38", "py310"]}
        result = _validate_py_version_and_set_if_needed("py310", version_config, "pytorch")
        assert result == "py310"

    @patch("sagemaker.core.image_retriever.image_retriever_utils.defaults", create=True)
    def test_validate_py_version_auto_select(self, mock_defaults):
        """Test _validate_py_version_and_set_if_needed with auto-selection."""
        mock_defaults.SPARK_NAME = "spark"
        version_config = {"repository": "pytorch", "py_versions": ["py310"]}
        result = _validate_py_version_and_set_if_needed(None, version_config, "pytorch")
        assert result == "py310"

    def test_validate_py_version_invalid(self):
        """Test _validate_py_version_and_set_if_needed with invalid version."""
        version_config = {"repository": "pytorch", "py_versions": ["py38", "py310"]}
        with pytest.raises(ValueError, match="Unsupported Python version"):
            _validate_py_version_and_set_if_needed("py36", version_config, "pytorch")


class TestValidateVersionAndSetIfNeeded:
    """Test _validate_version_and_set_if_needed function."""

    @patch("sagemaker.core.image_retriever.image_retriever_utils._get_latest_version")
    def test_validate_version_auto_select(self, mock_latest):
        """Test _validate_version_and_set_if_needed with auto-selection."""
        mock_latest.return_value = "2.0"
        config = {"versions": {"2.0": {}}}
        result = _validate_version_and_set_if_needed(None, config, "pytorch", "training")
        assert result == "2.0"

    def test_validate_version_single_version(self):
        """Test _validate_version_and_set_if_needed with single version."""
        config = {"versions": {"2.0": {}}}
        result = _validate_version_and_set_if_needed(None, config, "pytorch", "training")
        assert result == "2.0"


class TestValidateInstanceDeprecation:
    """Test _validate_instance_deprecation function."""

    def test_validate_instance_deprecation_p2_pytorch_new(self):
        """Test _validate_instance_deprecation for P2 with new PyTorch."""
        with pytest.raises(ValueError, match="P2 instances have been deprecated"):
            _validate_instance_deprecation("pytorch", "ml.p2.xlarge", "1.13")

    def test_validate_instance_deprecation_p2_tensorflow_new(self):
        """Test _validate_instance_deprecation for P2 with new TensorFlow."""
        with pytest.raises(ValueError, match="P2 instances have been deprecated"):
            _validate_instance_deprecation("tensorflow", "ml.p2.xlarge", "2.12")

    def test_validate_instance_deprecation_p2_pytorch_old(self):
        """Test _validate_instance_deprecation for P2 with old PyTorch."""
        _validate_instance_deprecation("pytorch", "ml.p2.xlarge", "1.12")
        # Should not raise

    def test_validate_instance_deprecation_p3(self):
        """Test _validate_instance_deprecation for P3."""
        _validate_instance_deprecation("pytorch", "ml.p3.2xlarge", "1.13")
        # Should not raise


class TestValidateForSupportedFrameworksAndInstanceType:
    """Test _validate_for_suppported_frameworks_and_instance_type function."""

    def test_validate_trainium_invalid_framework(self):
        """Test validation for Trainium with invalid framework."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            _validate_for_suppported_frameworks_and_instance_type("xgboost", "ml.trn1.2xlarge")

    def test_validate_trainium_valid_framework(self):
        """Test validation for Trainium with valid framework."""
        _validate_for_suppported_frameworks_and_instance_type("pytorch", "ml.trn1.2xlarge")
        # Should not raise

    def test_validate_graviton_invalid_framework(self):
        """Test validation for Graviton with invalid framework."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            _validate_for_suppported_frameworks_and_instance_type("mxnet", "ml.c7g.xlarge")

    def test_validate_graviton_valid_framework(self):
        """Test validation for Graviton with valid framework."""
        _validate_for_suppported_frameworks_and_instance_type("xgboost", "ml.c7g.xlarge")
        # Should not raise


class TestRetrievePytorchUriInputsAreAllDefault:
    """Test _retrieve_pytorch_uri_inputs_are_all_default function."""

    def test_all_defaults(self):
        """Test with all default values."""
        result = _retrieve_pytorch_uri_inputs_are_all_default()
        assert result is True

    def test_with_version(self):
        """Test with version specified."""
        result = _retrieve_pytorch_uri_inputs_are_all_default(version="2.0")
        assert result is False

    def test_with_instance_type(self):
        """Test with instance_type specified."""
        result = _retrieve_pytorch_uri_inputs_are_all_default(instance_type="ml.p3.2xlarge")
        assert result is False

    def test_with_distributed(self):
        """Test with distributed specified."""
        result = _retrieve_pytorch_uri_inputs_are_all_default(distributed=True)
        assert result is False
