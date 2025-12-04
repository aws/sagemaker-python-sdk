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

"""Unit tests for sagemaker.core.image_uris module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from packaging.version import Version

from sagemaker.core import image_uris
from sagemaker.core.jumpstart.enums import JumpStartModelType


class TestGetImageTag:
    """Test cases for _get_image_tag function"""

    def test_xgboost_graviton_instance(self):
        """Test XGBoost with Graviton instance"""
        tag = image_uris._get_image_tag(
            container_version="1",
            distribution=None,
            final_image_scope="inference_graviton",
            framework="xgboost",
            inference_tool=None,
            instance_type="ml.c7g.xlarge",
            processor="cpu",
            py_version="py3",
            tag_prefix="1.5-1",
            version="1.5-1",
        )
        assert tag == "1.5-1-arm64"

    def test_sklearn_graviton_instance(self):
        """Test SKLearn with Graviton instance"""
        tag = image_uris._get_image_tag(
            container_version="1",
            distribution=None,
            final_image_scope="inference_graviton",
            framework="sklearn",
            inference_tool=None,
            instance_type="ml.c7g.xlarge",
            processor="cpu",
            py_version="py3",
            tag_prefix="1.0-1",
            version="1.0-1",
        )
        assert tag == "1.0-1-arm64-cpu-py3"

    def test_format_tag_with_inference_tool(self):
        """Test tag formatting with inference tool"""
        tag = image_uris._get_image_tag(
            container_version="cu110",
            distribution=None,
            final_image_scope="inference",
            framework="pytorch",
            inference_tool="neuronx",
            instance_type="ml.inf2.xlarge",
            processor="inf",
            py_version="py39",
            tag_prefix="1.13.1",
            version="1.13.1",
        )
        assert "neuronx" in tag
        assert "py39" in tag

    def test_triton_gpu_tag(self):
        """Test Triton image tag for GPU (should not have -gpu suffix)"""
        tag = image_uris._get_image_tag(
            container_version="21.08",
            distribution=None,
            final_image_scope="inference",
            framework="sagemaker-tritonserver",
            inference_tool=None,
            instance_type="ml.g5.xlarge",
            processor="gpu",
            py_version="py38",
            tag_prefix="2.12",
            version="2.12",
        )
        assert not tag.endswith("-gpu")

    def test_triton_cpu_tag(self):
        """Test Triton image tag for CPU"""
        tag = image_uris._get_image_tag(
            container_version="21.08",
            distribution=None,
            final_image_scope="inference",
            framework="sagemaker-tritonserver",
            inference_tool=None,
            instance_type="ml.c5.xlarge",
            processor="cpu",
            py_version="py38",
            tag_prefix="2.12",
            version="2.12",
        )
        assert "-cpu" in tag

    def test_auto_select_container_version_p4d(self):
        """Test auto-selection of container version for p4d instances"""
        tag = image_uris._get_image_tag(
            container_version=None,
            distribution=None,
            final_image_scope="training",
            framework="tensorflow",
            inference_tool=None,
            instance_type="ml.p4d.24xlarge",
            processor="gpu",
            py_version="py37",
            tag_prefix="2.3",
            version="2.3",
        )
        # Should auto-select container version for p4d
        assert tag is not None


class TestConfigForFrameworkAndScope:
    """Test cases for _config_for_framework_and_scope function"""

    @patch("sagemaker.core.image_uris.config_for_framework")
    def test_with_accelerator_type(self, mock_config):
        """Test with accelerator type (EIA)"""
        mock_config.return_value = {
            "scope": ["training", "inference", "eia"],
            "eia": {"versions": {}},
        }

        result = image_uris._config_for_framework_and_scope(
            "tensorflow", "training", accelerator_type="ml.eia2.medium"
        )

        assert result == mock_config.return_value

    @patch("sagemaker.core.image_uris.config_for_framework")
    def test_single_scope_available(self, mock_config):
        """Test when only one scope is available"""
        mock_config.return_value = {"scope": ["training"], "training": {"versions": {}}}

        result = image_uris._config_for_framework_and_scope(
            "xgboost", "inference"  # Different from available
        )

        # Should default to the only available scope
        assert result == mock_config.return_value

    @patch("sagemaker.core.image_uris.config_for_framework")
    def test_training_inference_same_images(self, mock_config):
        """Test when training and inference use same images"""
        mock_config.return_value = {"scope": ["training", "inference"], "versions": {}}

        result = image_uris._config_for_framework_and_scope("sklearn", None)

        # Should return the config directly
        assert "versions" in result


class TestValidateInstanceDeprecation:
    """Test cases for _validate_instance_deprecation function"""

    def test_p2_with_pytorch_1_13(self):
        """Test P2 instance with PyTorch 1.13 (should raise)"""
        with pytest.raises(ValueError, match="P2 instances have been deprecated"):
            image_uris._validate_instance_deprecation("pytorch", "ml.p2.xlarge", "1.13")

    def test_p2_with_pytorch_1_12(self):
        """Test P2 instance with PyTorch 1.12 (should pass)"""
        # Should not raise
        image_uris._validate_instance_deprecation("pytorch", "ml.p2.xlarge", "1.12")

    def test_p2_with_tensorflow_2_12(self):
        """Test P2 instance with TensorFlow 2.12 (should raise)"""
        with pytest.raises(ValueError, match="P2 instances have been deprecated"):
            image_uris._validate_instance_deprecation("tensorflow", "ml.p2.xlarge", "2.12")

    def test_p2_with_tensorflow_2_11(self):
        """Test P2 instance with TensorFlow 2.11 (should pass)"""
        # Should not raise
        image_uris._validate_instance_deprecation("tensorflow", "ml.p2.xlarge", "2.11")

    def test_p3_instance(self):
        """Test P3 instance (should pass)"""
        # Should not raise
        image_uris._validate_instance_deprecation("pytorch", "ml.p3.2xlarge", "1.13")


class TestValidateForSupportedFrameworksAndInstanceType:
    """Test cases for _validate_for_suppported_frameworks_and_instance_type function"""

    def test_trainium_with_unsupported_framework(self):
        """Test Trainium instance with unsupported framework"""
        with pytest.raises(ValueError, match="framework"):
            image_uris._validate_for_suppported_frameworks_and_instance_type(
                "tensorflow", "ml.trn1.2xlarge"
            )

    def test_trainium_with_pytorch(self):
        """Test Trainium instance with PyTorch (should pass)"""
        # Should not raise
        image_uris._validate_for_suppported_frameworks_and_instance_type(
            "pytorch", "ml.trn1.2xlarge"
        )

    def test_graviton_with_unsupported_framework(self):
        """Test Graviton instance with unsupported framework"""
        with pytest.raises(ValueError, match="framework"):
            image_uris._validate_for_suppported_frameworks_and_instance_type(
                "mxnet", "ml.c7g.xlarge"
            )

    def test_graviton_with_xgboost(self):
        """Test Graviton instance with XGBoost (should pass)"""
        # Should not raise
        image_uris._validate_for_suppported_frameworks_and_instance_type("xgboost", "ml.c7g.xlarge")


class TestGetFinalImageScope:
    """Test cases for _get_final_image_scope function"""

    def test_graviton_instance_with_xgboost(self):
        """Test Graviton instance with XGBoost"""
        result = image_uris._get_final_image_scope("xgboost", "ml.c7g.xlarge", "inference")
        assert result == "inference_graviton"

    def test_graviton_instance_with_sklearn(self):
        """Test Graviton instance with SKLearn"""
        result = image_uris._get_final_image_scope("sklearn", "ml.c7g.xlarge", "training")
        assert result == "inference_graviton"

    def test_non_graviton_instance(self):
        """Test non-Graviton instance"""
        result = image_uris._get_final_image_scope("xgboost", "ml.m5.xlarge", "training")
        assert result == "training"

    def test_xgboost_with_none_scope(self):
        """Test XGBoost with None scope (should default to training)"""
        result = image_uris._get_final_image_scope("xgboost", "ml.m5.xlarge", None)
        assert result == "training"


class TestGetInferenceTool:
    """Test cases for _get_inference_tool function"""

    def test_inf_instance_without_tool(self):
        """Test Inferentia instance without explicit tool"""
        result = image_uris._get_inference_tool(None, "ml.inf1.xlarge")
        assert result == "neuron"

    def test_trn_instance_without_tool(self):
        """Test Trainium instance without explicit tool"""
        result = image_uris._get_inference_tool(None, "ml.trn1.2xlarge")
        assert result == "neuron"

    def test_with_explicit_tool(self):
        """Test with explicit inference tool"""
        result = image_uris._get_inference_tool("neuronx", "ml.inf2.xlarge")
        assert result == "neuronx"

    def test_regular_instance(self):
        """Test regular instance without tool"""
        result = image_uris._get_inference_tool(None, "ml.m5.xlarge")
        assert result is None


class TestGetLatestVersions:
    """Test cases for _get_latest_versions function"""

    def test_semantic_versions(self):
        """Test with semantic versions"""
        versions = ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]
        result = image_uris._get_latest_versions(versions)
        assert result == "2.0.0"

    def test_single_version(self):
        """Test with single version"""
        versions = ["1.0.0"]
        result = image_uris._get_latest_versions(versions)
        assert result == "1.0.0"

    def test_mixed_versions(self):
        """Test with mixed version formats"""
        versions = ["1.0", "1.1", "2.0"]
        result = image_uris._get_latest_versions(versions)
        assert result == "2.0"


class TestValidateAcceleratorType:
    """Test cases for _validate_accelerator_type function"""

    def test_valid_eia_accelerator(self):
        """Test valid EIA accelerator"""
        # Should not raise
        image_uris._validate_accelerator_type("ml.eia2.medium")

    def test_local_sagemaker_notebook(self):
        """Test local_sagemaker_notebook accelerator"""
        # Should not raise
        image_uris._validate_accelerator_type("local_sagemaker_notebook")

    def test_invalid_accelerator(self):
        """Test invalid accelerator type"""
        with pytest.raises(ValueError, match="Invalid SageMaker Elastic Inference"):
            image_uris._validate_accelerator_type("invalid-accelerator")


class TestValidateVersionAndSetIfNeeded:
    """Test cases for _validate_version_and_set_if_needed function"""

    @patch("sagemaker.core.image_uris._get_latest_version")
    def test_with_single_version(self, mock_get_latest):
        """Test when only one version is available"""
        config = {"versions": {"1.0": {}}}
        result = image_uris._validate_version_and_set_if_needed(None, config, "xgboost", "training")
        assert result == "1.0"

    @patch("sagemaker.core.image_uris._get_latest_version")
    def test_with_version_alias(self, mock_get_latest):
        """Test with version alias"""
        config = {"versions": {"1.0": {}, "2.0": {}}, "version_aliases": {"latest": "2.0"}}
        result = image_uris._validate_version_and_set_if_needed(
            "latest", config, "pytorch", "training"
        )
        assert result == "latest"

    def test_with_invalid_version(self):
        """Test with invalid version"""
        config = {"versions": {"1.0": {}, "1.5": {}}}
        with pytest.raises(ValueError, match="Unsupported"):
            image_uris._validate_version_and_set_if_needed("2.0", config, "xgboost", "training")


class TestVersionForConfig:
    """Test cases for _version_for_config function"""

    def test_with_version_alias(self):
        """Test with version alias"""
        config = {"version_aliases": {"latest": "2.0"}}
        result = image_uris._version_for_config("latest", config)
        assert result == "2.0"

    def test_without_alias(self):
        """Test without version alias"""
        config = {"versions": {"1.0": {}}}
        result = image_uris._version_for_config("1.0", config)
        assert result == "1.0"


class TestRegistryFromRegion:
    """Test cases for _registry_from_region function"""

    def test_valid_region(self):
        """Test with valid region"""
        registry_dict = {"us-west-2": "123456789012", "us-east-1": "987654321098"}
        result = image_uris._registry_from_region("us-west-2", registry_dict)
        assert result == "123456789012"

    def test_invalid_region(self):
        """Test with invalid region"""
        registry_dict = {"us-west-2": "123456789012"}
        with pytest.raises(ValueError, match="Unsupported"):
            image_uris._registry_from_region("invalid-region", registry_dict)


class TestProcessor:
    """Test cases for _processor function"""

    def test_local_cpu(self):
        """Test local CPU instance"""
        result = image_uris._processor("local", ["cpu", "gpu"])
        assert result == "cpu"

    def test_local_gpu(self):
        """Test local GPU instance"""
        result = image_uris._processor("local_gpu", ["cpu", "gpu"])
        assert result == "gpu"

    def test_neuron_instance(self):
        """Test neuron instance"""
        result = image_uris._processor("neuron", ["cpu", "neuron"])
        assert result == "neuron"

    def test_inf_instance(self):
        """Test Inferentia instance"""
        result = image_uris._processor("ml.inf1.xlarge", ["cpu", "inf"])
        assert result == "inf"

    def test_trn_instance(self):
        """Test Trainium instance"""
        result = image_uris._processor("ml.trn1.2xlarge", ["cpu", "trn"])
        assert result == "trn"

    def test_gpu_instance(self):
        """Test GPU instance"""
        result = image_uris._processor("ml.p3.2xlarge", ["cpu", "gpu"])
        assert result == "gpu"

    def test_cpu_instance(self):
        """Test CPU instance"""
        result = image_uris._processor("ml.m5.xlarge", ["cpu", "gpu"])
        assert result == "cpu"

    def test_specific_family(self):
        """Test specific instance family"""
        result = image_uris._processor("ml.c5.xlarge", ["cpu", "c5"])
        assert result == "c5"

    def test_serverless_inference(self):
        """Test serverless inference config"""
        serverless_config = Mock()
        result = image_uris._processor(None, ["cpu", "gpu"], serverless_config)
        assert result == "cpu"

    def test_single_processor_no_instance(self):
        """Test single processor without instance type"""
        result = image_uris._processor(None, ["cpu"])
        assert result == "cpu"

    def test_no_instance_type_multiple_processors(self):
        """Test no instance type with multiple processors"""
        with pytest.raises(ValueError, match="Empty SageMaker instance type"):
            image_uris._processor(None, ["cpu", "gpu"])

    def test_invalid_instance_type(self):
        """Test invalid instance type"""
        with pytest.raises(ValueError, match="Invalid SageMaker instance type"):
            image_uris._processor("invalid-instance", ["cpu", "gpu"])


class TestShouldAutoSelectContainerVersion:
    """Test cases for _should_auto_select_container_version function"""

    def test_p4d_instance(self):
        """Test P4D instance"""
        result = image_uris._should_auto_select_container_version("ml.p4d.24xlarge", None)
        assert result is True

    def test_smdistributed(self):
        """Test with smdistributed"""
        distribution = {"smdistributed": {"modelparallel": {"enabled": True}}}
        result = image_uris._should_auto_select_container_version("ml.p3.2xlarge", distribution)
        assert result is True

    def test_regular_instance(self):
        """Test regular instance"""
        result = image_uris._should_auto_select_container_version("ml.m5.xlarge", None)
        assert result is False


class TestValidatePyVersionAndSetIfNeeded:
    """Test cases for _validate_py_version_and_set_if_needed function"""

    def test_single_py_version(self):
        """Test with single Python version"""
        version_config = {"py3": {}}
        result = image_uris._validate_py_version_and_set_if_needed(None, version_config, "pytorch")
        assert result == "py3"

    def test_spark_framework(self):
        """Test with Spark framework"""
        version_config = {"py37": {}, "py38": {}}
        result = image_uris._validate_py_version_and_set_if_needed(None, version_config, "spark")
        assert result is None

    def test_no_py_versions(self):
        """Test with no Python versions"""
        version_config = {"repository": "test-repo"}
        result = image_uris._validate_py_version_and_set_if_needed("py3", version_config, "xgboost")
        assert result is None

    def test_invalid_py_version(self):
        """Test with invalid Python version"""
        version_config = {"py37": {}}
        with pytest.raises(ValueError, match="Unsupported"):
            image_uris._validate_py_version_and_set_if_needed("py39", version_config, "pytorch")


class TestValidateArg:
    """Test cases for _validate_arg function"""

    def test_valid_arg(self):
        """Test with valid argument"""
        # Should not raise
        image_uris._validate_arg("cpu", ["cpu", "gpu"], "processor")

    def test_invalid_arg(self):
        """Test with invalid argument"""
        with pytest.raises(ValueError, match="Unsupported processor"):
            image_uris._validate_arg("tpu", ["cpu", "gpu"], "processor")


class TestValidateFramework:
    """Test cases for _validate_framework function"""

    def test_valid_framework(self):
        """Test with valid framework"""
        # Should not raise
        image_uris._validate_framework(
            "pytorch", ["pytorch", "tensorflow"], "framework", "Trainium"
        )

    def test_invalid_framework(self):
        """Test with invalid framework"""
        with pytest.raises(ValueError, match="Unsupported framework"):
            image_uris._validate_framework("mxnet", ["pytorch"], "framework", "Trainium")


class TestFormatTag:
    """Test cases for _format_tag function"""

    def test_with_all_components(self):
        """Test with all components"""
        tag = image_uris._format_tag("1.13.1", "gpu", "py39", "cu118")
        assert tag == "1.13.1-gpu-py39-cu118"

    def test_with_inference_tool(self):
        """Test with inference tool"""
        tag = image_uris._format_tag("1.13.1", "inf", "py39", "cu118", "neuronx")
        assert tag == "1.13.1-neuronx-py39-cu118"

    def test_with_none_values(self):
        """Test with None values"""
        tag = image_uris._format_tag("1.0", None, None, None)
        assert tag == "1.0"


class TestGetBasePythonImageUri:
    """Test cases for get_base_python_image_uri function"""

    @patch("sagemaker.core.image_uris.config_for_framework")
    @patch("sagemaker.core.common_utils._botocore_resolver")
    def test_default_py_version(self, mock_resolver, mock_config):
        """Test with default Python version"""
        mock_endpoint = {"hostname": "ecr.us-west-2.amazonaws.com"}
        mock_resolver.return_value.construct_endpoint.return_value = mock_endpoint

        mock_config.return_value = {
            "versions": {
                "1.0": {
                    "repository": "sagemaker-base-python",
                    "registries": {"us-west-2": "123456789012"},
                }
            }
        }

        result = image_uris.get_base_python_image_uri("us-west-2")
        assert "sagemaker-base-python-310" in result
        assert "1.0" in result

    @patch("sagemaker.core.image_uris.config_for_framework")
    @patch("sagemaker.core.common_utils._botocore_resolver")
    def test_custom_py_version(self, mock_resolver, mock_config):
        """Test with custom Python version"""
        mock_endpoint = {"hostname": "ecr.us-west-2.amazonaws.com"}
        mock_resolver.return_value.construct_endpoint.return_value = mock_endpoint

        mock_config.return_value = {
            "versions": {
                "1.0": {
                    "repository": "sagemaker-base-python",
                    "registries": {"us-west-2": "123456789012"},
                }
            }
        }

        result = image_uris.get_base_python_image_uri("us-west-2", py_version="38")
        assert "sagemaker-base-python-38" in result


class TestFetchLatestVersionFromConfig:
    """Test cases for _fetch_latest_version_from_config function"""

    def test_with_version_aliases_in_scope(self):
        """Test with version aliases in image scope"""
        config = {"training": {"version_aliases": {"latest": "2.0"}}}
        result = image_uris._fetch_latest_version_from_config(config, "training")
        assert result == "2.0"

    def test_with_single_version(self):
        """Test with single version"""
        config = {"versions": {"1.0": {}}}
        result = image_uris._fetch_latest_version_from_config(config)
        assert result == "1.0"

    def test_with_x_versions(self):
        """Test with .x versions"""
        config = {"versions": {"1.x": {}, "2.x": {}}}
        result = image_uris._fetch_latest_version_from_config(config)
        assert result == "2.x"

    def test_with_semantic_versions(self):
        """Test with semantic versions"""
        config = {"versions": {"1.0.0": {}, "2.0.0": {}}}
        result = image_uris._fetch_latest_version_from_config(config)
        assert result == "2.0.0"  # Latest version

    def test_with_latest_keyword(self):
        """Test with 'latest' keyword"""
        config = {"versions": {"latest": {}, "1.0": {}}}
        result = image_uris._fetch_latest_version_from_config(config)
        assert result is None

    def test_with_processing_versions(self):
        """Test with processing versions"""
        config = {"processing": {"versions": {"1.0": {}, "2.0": {}}}}
        result = image_uris._fetch_latest_version_from_config(config)
        assert result in ["1.0", "2.0"]
