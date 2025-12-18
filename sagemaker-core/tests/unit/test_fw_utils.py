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
"""Unit tests for sagemaker.core.fw_utils module."""
from __future__ import absolute_import

import json
import os
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from packaging import version

from sagemaker.core.fw_utils import (
    validate_source_dir,
    validate_source_code_input_against_pipeline_variables,
    parse_mp_parameters,
    get_mp_parameters,
    validate_mp_config,
    tar_and_upload_dir,
    framework_name_from_image,
    framework_version_from_tag,
    model_code_key_prefix,
    warn_if_parameter_server_with_multi_gpu,
    profiler_config_deprecation_warning,
    validate_smdistributed,
    validate_distribution,
    validate_distribution_for_instance_type,
    validate_torch_distributed_distribution,
    validate_version_or_image_args,
    python_deprecation_warning,
    _region_supports_debugger,
    _region_supports_profiler,
    _instance_type_supports_profiler,
    _is_gpu_instance,
    _is_trainium_instance,
    UploadedCode,
)
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.instance_group import InstanceGroup


class TestValidateSourceDir:
    """Test validate_source_dir function."""

    def test_validate_source_dir_valid(self, tmp_path):
        """Test with valid script and directory."""
        script_file = tmp_path / "train.py"
        script_file.write_text("print('hello')")

        result = validate_source_dir("train.py", str(tmp_path))
        assert result is True

    def test_validate_source_dir_missing_script(self, tmp_path):
        """Test with missing script file."""
        with pytest.raises(ValueError, match="No file named"):
            validate_source_dir("missing.py", str(tmp_path))

    def test_validate_source_dir_no_directory(self):
        """Test with no directory specified."""
        result = validate_source_dir("train.py", None)
        assert result is True


class TestValidateSourceCodeInputAgainstPipelineVariables:
    """Test validate_source_code_input_against_pipeline_variables function."""

    def test_with_network_isolation_true_and_pipeline_variable_entry_point(self):
        """Test error when network isolation is True and entry_point is pipeline variable."""
        entry_point = ParameterString(name="EntryPoint")

        with pytest.raises(
            TypeError, match="entry_point, source_dir should not be pipeline variables"
        ):
            validate_source_code_input_against_pipeline_variables(
                entry_point=entry_point, enable_network_isolation=True
            )

    def test_with_git_config_and_pipeline_variable(self):
        """Test error when git_config is provided with pipeline variable."""
        source_dir = ParameterString(name="SourceDir")

        with pytest.raises(
            TypeError, match="entry_point, source_dir should not be pipeline variables"
        ):
            validate_source_code_input_against_pipeline_variables(
                source_dir=source_dir, git_config={"repo": "https://github.com/test/repo"}
            )

    def test_pipeline_variable_entry_point_without_source_dir(self):
        """Test error when entry_point is pipeline variable without source_dir."""
        entry_point = ParameterString(name="EntryPoint")

        with pytest.raises(TypeError, match="entry_point should not be a pipeline variable"):
            validate_source_code_input_against_pipeline_variables(entry_point=entry_point)

    def test_pipeline_variable_entry_point_with_local_source_dir(self):
        """Test error when entry_point is pipeline variable with local source_dir."""
        entry_point = ParameterString(name="EntryPoint")

        with pytest.raises(TypeError, match="entry_point should not be a pipeline variable"):
            validate_source_code_input_against_pipeline_variables(
                entry_point=entry_point, source_dir="/local/path"
            )

    def test_valid_pipeline_variable_entry_point_with_s3_source_dir(self):
        """Test valid case with pipeline variable entry_point and S3 source_dir."""
        entry_point = ParameterString(name="EntryPoint")
        # Should not raise
        validate_source_code_input_against_pipeline_variables(
            entry_point=entry_point, source_dir="s3://bucket/path"
        )


class TestParseMpParameters:
    """Test parse_mp_parameters function."""

    def test_parse_mp_parameters_dict(self):
        """Test parsing dict parameters."""
        params = {"partitions": 2, "microbatches": 4}
        result = parse_mp_parameters(params)
        assert result == params

    def test_parse_mp_parameters_file(self, tmp_path):
        """Test parsing parameters from file."""
        config_file = tmp_path / "config.json"
        params = {"partitions": 2, "microbatches": 4}
        config_file.write_text(json.dumps(params))

        result = parse_mp_parameters(str(config_file))
        assert result == params

    def test_parse_mp_parameters_invalid_json(self, tmp_path):
        """Test error with invalid JSON file."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("not json")

        with pytest.raises(ValueError, match="Cannot parse"):
            parse_mp_parameters(str(config_file))

    def test_parse_mp_parameters_nonexistent_file(self):
        """Test error with nonexistent file."""
        with pytest.raises(ValueError, match="Expected a string path"):
            parse_mp_parameters("/nonexistent/file.json")


class TestGetMpParameters:
    """Test get_mp_parameters function."""

    def test_get_mp_parameters_enabled(self):
        """Test getting parameters when modelparallel is enabled."""
        distribution = {
            "smdistributed": {"modelparallel": {"enabled": True, "parameters": {"partitions": 2}}}
        }
        result = get_mp_parameters(distribution)
        assert result == {"partitions": 2}

    def test_get_mp_parameters_disabled(self):
        """Test getting parameters when modelparallel is disabled."""
        distribution = {"smdistributed": {"modelparallel": {"enabled": False}}}
        result = get_mp_parameters(distribution)
        assert result is None

    def test_get_mp_parameters_not_present(self):
        """Test getting parameters when modelparallel is not present."""
        distribution = {"other": {}}
        result = get_mp_parameters(distribution)
        assert result is None


class TestValidateMpConfig:
    """Test validate_mp_config function."""

    def test_validate_mp_config_valid(self):
        """Test with valid config."""
        config = {
            "pipeline": "simple",
            "partitions": 2,
            "microbatches": 4,
            "placement_strategy": "spread",
        }
        # Should not raise
        validate_mp_config(config)

    def test_validate_mp_config_invalid_pipeline(self):
        """Test with invalid pipeline value."""
        config = {"pipeline": "invalid"}
        with pytest.raises(ValueError, match="pipeline must be a value in"):
            validate_mp_config(config)

    def test_validate_mp_config_invalid_partitions(self):
        """Test with invalid partitions value."""
        config = {"partitions": -1}
        with pytest.raises(ValueError, match="number of partitions must be a positive integer"):
            validate_mp_config(config)

    def test_validate_mp_config_missing_default_partition(self):
        """Test error when auto_partition is False without default_partition."""
        config = {"auto_partition": False, "partitions": 2}
        with pytest.raises(ValueError, match="default_partition must be supplied"):
            validate_mp_config(config)

    def test_validate_mp_config_default_partition_too_large(self):
        """Test error when default_partition >= partitions."""
        config = {"default_partition": 2, "partitions": 2}
        with pytest.raises(ValueError, match="default_partition must be less than"):
            validate_mp_config(config)

    def test_validate_mp_config_ddp_and_horovod(self):
        """Test error when both ddp and horovod are enabled."""
        config = {"ddp": True, "horovod": True}
        with pytest.raises(ValueError, match="ddp.*horovod.*cannot be simultaneously enabled"):
            validate_mp_config(config)


class TestTarAndUploadDir:
    """Test tar_and_upload_dir function."""

    @patch("sagemaker.core.common_utils.create_tar_file")
    def test_tar_and_upload_dir_s3_source(self, mock_create_tar):
        """Test with S3 source directory."""
        mock_session = Mock()

        result = tar_and_upload_dir(
            session=mock_session,
            bucket="test-bucket",
            s3_key_prefix="prefix",
            script="train.py",
            directory="s3://bucket/path",
        )

        assert result.s3_prefix == "s3://bucket/path"
        assert result.script_name == "train.py"
        mock_create_tar.assert_not_called()

    @patch("sagemaker.core.common_utils.create_tar_file")
    @patch("sagemaker.core.fw_utils.tempfile.mkdtemp")
    @patch("sagemaker.core.fw_utils.shutil.rmtree")
    def test_tar_and_upload_dir_local_file(
        self, mock_rmtree, mock_mkdtemp, mock_create_tar, tmp_path
    ):
        """Test with local file."""
        script_file = tmp_path / "train.py"
        script_file.write_text("print('hello')")

        mock_mkdtemp.return_value = str(tmp_path / "temp")
        mock_create_tar.return_value = str(tmp_path / "temp" / "source.tar.gz")

        mock_session = Mock()
        mock_s3_resource = Mock()
        mock_session.resource.return_value = mock_s3_resource
        mock_session.region_name = "us-west-2"

        result = tar_and_upload_dir(
            session=mock_session,
            bucket="test-bucket",
            s3_key_prefix="prefix",
            script=str(script_file),
        )

        assert result.s3_prefix == "s3://test-bucket/prefix/sourcedir.tar.gz"
        assert result.script_name == "train.py"


class TestFrameworkNameFromImage:
    """Test framework_name_from_image function."""

    def test_framework_name_from_image_tensorflow(self):
        """Test extracting TensorFlow framework info."""
        image = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:2.3-cpu-py37"
        fw, py, tag, scriptmode = framework_name_from_image(image)

        assert fw == "tensorflow"
        assert py == "py37"
        assert "2.3-cpu-py37" in tag

    def test_framework_name_from_image_pytorch(self):
        """Test extracting PyTorch framework info."""
        image = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-pytorch:1.8-gpu-py3"
        fw, py, tag, scriptmode = framework_name_from_image(image)

        assert fw == "pytorch"
        assert py == "py3"

    def test_framework_name_from_image_xgboost_short_tag(self):
        """Test extracting XGBoost with short tag."""
        image = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
        fw, py, tag, scriptmode = framework_name_from_image(image)

        assert fw == "xgboost"
        assert py == "py3"
        assert tag == "1.5-1"

    def test_framework_name_from_image_invalid(self):
        """Test with invalid image URI."""
        fw, py, tag, scriptmode = framework_name_from_image("invalid-image")

        assert fw is None
        assert py is None
        assert tag is None
        assert scriptmode is None


class TestFrameworkVersionFromTag:
    """Test framework_version_from_tag function."""

    def test_framework_version_from_tag_standard(self):
        """Test extracting version from standard tag."""
        version = framework_version_from_tag("2.3.1-cpu-py37")
        assert version == "2.3.1"

    def test_framework_version_from_tag_xgboost(self):
        """Test extracting version from XGBoost tag."""
        version = framework_version_from_tag("1.5-1")
        assert version == "1.5-1"

    def test_framework_version_from_tag_invalid(self):
        """Test with invalid tag."""
        version = framework_version_from_tag("invalid-tag")
        assert version is None


class TestModelCodeKeyPrefix:
    """Test model_code_key_prefix function."""

    def test_model_code_key_prefix_with_model_name(self):
        """Test with model name provided."""
        result = model_code_key_prefix("prefix", "my-model", "image:latest")
        assert "prefix" in result
        assert "my-model" in result

    def test_model_code_key_prefix_without_model_name(self):
        """Test without model name."""
        result = model_code_key_prefix("prefix", None, "sagemaker-tensorflow:2.3-cpu-py37")
        assert "prefix" in result
        assert "sagemaker-tensorflow" in result


class TestWarnIfParameterServerWithMultiGpu:
    """Test warn_if_parameter_server_with_multi_gpu function."""

    @patch("sagemaker.core.fw_utils.logger")
    def test_warn_with_multi_gpu_and_parameter_server(self, mock_logger):
        """Test warning with multi-GPU instance and parameter server."""
        distribution = {"parameter_server": {"enabled": True}}

        warn_if_parameter_server_with_multi_gpu("ml.p3.8xlarge", distribution)

        mock_logger.warning.assert_called_once()

    @patch("sagemaker.core.fw_utils.logger")
    def test_no_warn_with_single_gpu(self, mock_logger):
        """Test no warning with single GPU instance."""
        distribution = {"parameter_server": {"enabled": True}}

        warn_if_parameter_server_with_multi_gpu("ml.p2.xlarge", distribution)

        mock_logger.warning.assert_not_called()

    def test_no_warn_with_local(self):
        """Test no warning with local instance."""
        distribution = {"parameter_server": {"enabled": True}}
        # Should not raise
        warn_if_parameter_server_with_multi_gpu("local", distribution)


class TestValidateSmdistributed:
    """Test validate_smdistributed function."""

    def test_validate_smdistributed_dataparallel_valid(self):
        """Test valid dataparallel configuration."""
        distribution = {"smdistributed": {"dataparallel": {"enabled": True}}}
        # Should not raise
        validate_smdistributed(
            instance_type="ml.p3.16xlarge",
            framework_name="pytorch",
            framework_version="1.8.0",
            py_version="py38",
            distribution=distribution,
        )

    def test_validate_smdistributed_unsupported_framework_version(self):
        """Test error with unsupported framework version."""
        distribution = {"smdistributed": {"dataparallel": {"enabled": True}}}

        with pytest.raises(ValueError, match="framework_version.*not supported"):
            validate_smdistributed(
                instance_type="ml.p3.16xlarge",
                framework_name="pytorch",
                framework_version="1.0.0",
                py_version="py38",
                distribution=distribution,
            )

    def test_validate_smdistributed_multiple_strategies(self):
        """Test error with multiple strategies."""
        distribution = {
            "smdistributed": {"dataparallel": {"enabled": True}, "modelparallel": {"enabled": True}}
        }

        with pytest.raises(ValueError, match="Cannot use more than 1 smdistributed strategy"):
            validate_smdistributed(
                instance_type="ml.p3.16xlarge",
                framework_name="pytorch",
                framework_version="1.8.0",
                py_version="py38",
                distribution=distribution,
            )


class TestValidateDistributionForInstanceType:
    """Test validate_distribution_for_instance_type function."""

    def test_validate_distribution_for_trainium_valid(self):
        """Test valid distribution for Trainium instance."""
        distribution = {"torch_distributed": {"enabled": True}}
        # Should not raise
        validate_distribution_for_instance_type("ml.trn1.2xlarge", distribution)

    def test_validate_distribution_for_trainium_invalid(self):
        """Test invalid distribution for Trainium instance."""
        distribution = {"parameter_server": {"enabled": True}}

        with pytest.raises(ValueError, match="not supported for Trainium"):
            validate_distribution_for_instance_type("ml.trn1.2xlarge", distribution)

    def test_validate_distribution_for_trainium_multiple(self):
        """Test multiple distributions for Trainium instance."""
        distribution = {"torch_distributed": {"enabled": True}, "other": {"enabled": True}}

        with pytest.raises(ValueError, match="Multiple distribution strategies"):
            validate_distribution_for_instance_type("ml.trn1.2xlarge", distribution)


class TestValidateTorchDistributedDistribution:
    """Test validate_torch_distributed_distribution function."""

    def test_validate_torch_distributed_gpu_valid(self):
        """Test valid torch_distributed for GPU."""
        distribution = {"torch_distributed": {"enabled": True}}

        # Should not raise
        validate_torch_distributed_distribution(
            instance_type="ml.p3.2xlarge",
            distribution=distribution,
            framework_version="2.0.0",
            py_version="py38",
            image_uri=None,
            entry_point="train.py",
        )

    def test_validate_torch_distributed_unsupported_framework(self):
        """Test error with unsupported framework version."""
        distribution = {"torch_distributed": {"enabled": True}}

        with pytest.raises(ValueError, match="framework_version.*not supported"):
            validate_torch_distributed_distribution(
                instance_type="ml.p3.2xlarge",
                distribution=distribution,
                framework_version="1.0.0",
                py_version="py38",
                image_uri=None,
                entry_point="train.py",
            )

    def test_validate_torch_distributed_non_python_entry_point(self):
        """Test error with non-Python entry point."""
        distribution = {"torch_distributed": {"enabled": True}}

        with pytest.raises(ValueError, match="Unsupported entry point type"):
            validate_torch_distributed_distribution(
                instance_type="ml.p3.2xlarge",
                distribution=distribution,
                framework_version="2.0.0",
                py_version="py38",
                image_uri=None,
                entry_point="train.sh",
            )


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_gpu_instance_true(self):
        """Test _is_gpu_instance with GPU instance."""
        assert _is_gpu_instance("ml.p3.2xlarge") is True
        assert _is_gpu_instance("ml.g4dn.xlarge") is True
        assert _is_gpu_instance("local_gpu") is True
        assert _is_gpu_instance("ml.p6-b200.48xlarge") is True
        assert _is_gpu_instance("ml.g6e-12xlarge.xlarge") is True

    def test_is_gpu_instance_false(self):
        """Test _is_gpu_instance with non-GPU instance."""
        assert _is_gpu_instance("ml.m5.large") is False
        assert _is_gpu_instance("ml.c5.xlarge") is False

    def test_is_trainium_instance_true(self):
        """Test _is_trainium_instance with Trainium instance."""
        assert _is_trainium_instance("ml.trn1.2xlarge") is True
        assert _is_trainium_instance("ml.trn1.32xlarge") is True
        assert _is_trainium_instance("ml.trn1-n.2xlarge") is True

    def test_is_trainium_instance_false(self):
        """Test _is_trainium_instance with non-Trainium instance."""
        assert _is_trainium_instance("ml.p3.2xlarge") is False
        assert _is_trainium_instance("ml.m5.large") is False

    def test_region_supports_debugger(self):
        """Test _region_supports_debugger."""
        assert _region_supports_debugger("us-west-2") is True
        assert _region_supports_debugger("us-iso-east-1") is False

    def test_region_supports_profiler(self):
        """Test _region_supports_profiler."""
        assert _region_supports_profiler("us-west-2") is True
        assert _region_supports_profiler("us-iso-east-1") is False

    def test_instance_type_supports_profiler(self):
        """Test _instance_type_supports_profiler."""
        assert _instance_type_supports_profiler("ml.trn1-n.xlarge") is True
        assert _instance_type_supports_profiler("ml.trn1.2xlarge") is True
        assert _instance_type_supports_profiler("ml.p3.2xlarge") is False


class TestValidateVersionOrImageArgs:
    """Test validate_version_or_image_args function."""

    def test_validate_version_or_image_args_valid_versions(self):
        """Test with valid framework and py versions."""
        # Should not raise
        validate_version_or_image_args("2.3.0", "py38", None)

    def test_validate_version_or_image_args_valid_image(self):
        """Test with valid image URI."""
        # Should not raise
        validate_version_or_image_args(
            None, None, "123.dkr.ecr.us-west-2.amazonaws.com/image:latest"
        )

    def test_validate_version_or_image_args_missing_framework_version(self):
        """Test error when framework_version is None without image."""
        with pytest.raises(ValueError, match="framework_version or py_version was None"):
            validate_version_or_image_args(None, "py38", None)

    def test_validate_version_or_image_args_missing_py_version(self):
        """Test error when py_version is None without image."""
        with pytest.raises(ValueError, match="framework_version or py_version was None"):
            validate_version_or_image_args("2.3.0", None, None)


class TestPythonDeprecationWarning:
    """Test python_deprecation_warning function."""

    def test_python_deprecation_warning(self):
        """Test deprecation warning message."""
        result = python_deprecation_warning("tensorflow", "2.11")

        assert "2.11" in result
        assert "tensorflow" in result
        assert "Python 2" in result
        assert "py_version='py3'" in result
