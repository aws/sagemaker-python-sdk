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
"""Tests for PipelineVariable support in ModelTrainer (GH#5524).

Verifies that ModelTrainer fields accept PipelineVariable objects
(e.g., ParameterString) in addition to their concrete types, following
the existing V3 pattern established by SourceCode and OutputDataConfig.

See: https://github.com/aws/sagemaker-python-sdk/issues/5524
"""
from __future__ import absolute_import

import pytest
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.helper.pipeline_variable import PipelineVariable, StrPipeVar
from sagemaker.core.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.train.model_trainer import ModelTrainer, Mode
from sagemaker.train.configs import (
    Compute,
    StoppingCondition,
    OutputDataConfig,
)
from sagemaker.train.defaults import DEFAULT_INSTANCE_TYPE
from sagemaker.train.utils import _get_repo_name_from_image, safe_serialize


DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_BUCKET_PREFIX = "sample-prefix"
DEFAULT_REGION = "us-west-2"
DEFAULT_COMPUTE = Compute(instance_type=DEFAULT_INSTANCE_TYPE, instance_count=1)
DEFAULT_STOPPING = StoppingCondition(max_runtime_in_seconds=3600)
DEFAULT_OUTPUT = OutputDataConfig(
    s3_output_path=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BUCKET_PREFIX}/test-job",
)


@pytest.fixture(scope="module", autouse=True)
def modules_session():
    with patch("sagemaker.train.Session", spec=Session) as session_mock:
        session_instance = session_mock.return_value
        session_instance.default_bucket.return_value = DEFAULT_BUCKET
        session_instance.get_caller_identity_arn.return_value = DEFAULT_ROLE
        session_instance.default_bucket_prefix = DEFAULT_BUCKET_PREFIX
        session_instance.boto_session = MagicMock(spec="boto3.session.Session")
        session_instance.boto_region_name = DEFAULT_REGION
        yield session_instance


class TestModelTrainerPipelineVariableAcceptance:
    """Test that ModelTrainer fields accept PipelineVariable objects."""

    def test_training_image_accepts_parameter_string(self):
        """ModelTrainer.training_image should accept ParameterString (GH#5524)."""
        param = ParameterString(name="TrainingImage", default_value=DEFAULT_IMAGE)
        trainer = ModelTrainer(
            training_image=param,
            base_job_name="pipeline-test-job",  # Required: PipelineVariable can't generate job name
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_image is param

    def test_algorithm_name_accepts_parameter_string(self):
        """ModelTrainer.algorithm_name should accept ParameterString."""
        param = ParameterString(name="AlgorithmName", default_value="my-algo-arn")
        trainer = ModelTrainer(
            algorithm_name=param,
            base_job_name="pipeline-test-job",  # Required: PipelineVariable can't generate job name
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.algorithm_name is param

    def test_training_input_mode_accepts_parameter_string(self):
        """ModelTrainer.training_input_mode should accept ParameterString."""
        param = ParameterString(name="InputMode", default_value="File")
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            training_input_mode=param,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_input_mode is param

    def test_environment_values_accept_parameter_string(self):
        """ModelTrainer.environment dict values should accept ParameterString."""
        param = ParameterString(name="DatasetVersion", default_value="v1")
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            environment={"DATASET_VERSION": param, "STATIC_VAR": "hello"},
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.environment["DATASET_VERSION"] is param
        assert trainer.environment["STATIC_VAR"] == "hello"


class TestModelTrainerRealValuesStillWork:
    """Regression tests: verify that passing real values still works after the change."""

    def test_training_image_accepts_real_string(self):
        """ModelTrainer.training_image should still accept a plain string."""
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_image == DEFAULT_IMAGE

    def test_algorithm_name_accepts_real_string(self):
        """ModelTrainer.algorithm_name should still accept a plain string."""
        trainer = ModelTrainer(
            algorithm_name="arn:aws:sagemaker:us-west-2:000000000000:algorithm/my-algo",
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.algorithm_name == "arn:aws:sagemaker:us-west-2:000000000000:algorithm/my-algo"

    def test_training_input_mode_accepts_real_string(self):
        """ModelTrainer.training_input_mode should still accept a plain string."""
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            training_input_mode="Pipe",
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_input_mode == "Pipe"

    def test_environment_accepts_real_string_values(self):
        """ModelTrainer.environment should still accept plain string values."""
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            environment={"KEY1": "value1", "KEY2": "value2"},
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.environment == {"KEY1": "value1", "KEY2": "value2"}

    def test_training_image_rejects_invalid_type(self):
        """ModelTrainer.training_image should still reject invalid types (e.g., int)."""
        with pytest.raises(ValidationError):
            ModelTrainer(
                training_image=12345,
                role=DEFAULT_ROLE,
                compute=DEFAULT_COMPUTE,
                stopping_condition=DEFAULT_STOPPING,
                output_data_config=DEFAULT_OUTPUT,
            )


class TestValidateTrainingImageAndAlgorithmName:
    """Tests for _validate_training_image_and_algorithm_name with PipelineVariable."""

    def test_pipeline_variable_training_image_passes_validation(self):
        """PipelineVariable as training_image should pass validation."""
        param = ParameterString(name="TrainingImage", default_value=DEFAULT_IMAGE)
        trainer = ModelTrainer(
            training_image=param,
            base_job_name="pipeline-test-job",
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_image is param

    def test_pipeline_variable_algorithm_name_passes_validation(self):
        """PipelineVariable as algorithm_name should pass validation."""
        param = ParameterString(name="AlgoName", default_value="my-algo")
        trainer = ModelTrainer(
            algorithm_name=param,
            base_job_name="pipeline-test-job",
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.algorithm_name is param

    def test_both_pipeline_variables_raises_value_error(self):
        """Both training_image and algorithm_name as PipelineVariable should raise ValueError."""
        image_param = ParameterString(name="TrainingImage", default_value=DEFAULT_IMAGE)
        algo_param = ParameterString(name="AlgoName", default_value="my-algo")
        with pytest.raises(ValueError, match="Only one of"):
            ModelTrainer(
                training_image=image_param,
                algorithm_name=algo_param,
                base_job_name="pipeline-test-job",
                role=DEFAULT_ROLE,
                compute=DEFAULT_COMPUTE,
                stopping_condition=DEFAULT_STOPPING,
                output_data_config=DEFAULT_OUTPUT,
            )

    def test_neither_provided_raises_value_error(self):
        """Neither training_image nor algorithm_name should raise ValueError."""
        with pytest.raises(ValueError, match="Atleast one of"):
            ModelTrainer(
                training_image=None,
                algorithm_name=None,
                base_job_name="pipeline-test-job",
                role=DEFAULT_ROLE,
                compute=DEFAULT_COMPUTE,
                stopping_condition=DEFAULT_STOPPING,
                output_data_config=DEFAULT_OUTPUT,
            )


class TestGetRepoNameFromImage:
    """Tests for _get_repo_name_from_image with PipelineVariable."""

    def test_returns_none_for_pipeline_variable(self):
        """_get_repo_name_from_image should return None for PipelineVariable."""
        param = ParameterString(name="TrainingImage", default_value=DEFAULT_IMAGE)
        result = _get_repo_name_from_image(param)
        assert result is None

    def test_returns_repo_name_for_string(self):
        """_get_repo_name_from_image should return repo name for a normal string."""
        result = _get_repo_name_from_image(
            "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-repo:latest"
        )
        assert result == "my-repo"

    def test_returns_repo_name_without_tag(self):
        """_get_repo_name_from_image should handle image URIs without tags."""
        result = _get_repo_name_from_image(
            "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-repo"
        )
        assert result == "my-repo"


class TestSafeSerialize:
    """Tests for safe_serialize with PipelineVariable."""

    def test_safe_serialize_pipeline_variable_returns_variable(self):
        """safe_serialize should return the PipelineVariable object as-is."""
        param = ParameterInteger(name="MaxDepth", default_value=5)
        result = safe_serialize(param)
        assert result is param

    def test_safe_serialize_string_returns_string(self):
        """safe_serialize should return strings as-is."""
        result = safe_serialize("hello")
        assert result == "hello"

    def test_safe_serialize_int_returns_json(self):
        """safe_serialize should JSON-encode integers."""
        result = safe_serialize(5)
        assert result == "5"

    def test_safe_serialize_dict_returns_json(self):
        """safe_serialize should JSON-encode dicts."""
        result = safe_serialize({"key": "value"})
        assert result == '{"key": "value"}'

    def test_safe_serialize_parameter_string_returns_variable(self):
        """safe_serialize should return ParameterString as-is."""
        param = ParameterString(name="MyParam", default_value="val")
        result = safe_serialize(param)
        assert result is param
