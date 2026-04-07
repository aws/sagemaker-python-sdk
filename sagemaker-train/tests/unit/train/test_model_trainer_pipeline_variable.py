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


class TestModelTrainerPipelineVariableHyperparameters:
    """Test that PipelineVariable objects work correctly in ModelTrainer hyperparameters."""

    def test_hyperparameters_with_parameter_integer(self):
        """ParameterInteger in hyperparameters should be preserved through _create_training_job_args.

        This test documents the exact bug scenario from GH#5504: safe_serialize
        would fall back to str(data) for PipelineVariable objects, but
        PipelineVariable.__str__ intentionally raises TypeError.
        Before the fix, this call would have raised TypeError.
        """
        max_depth = ParameterInteger(name="MaxDepth", default_value=5)
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
            hyperparameters={"max_depth": max_depth},
        )
        # This call would have raised TypeError before the fix (GH#5504)
        args = trainer._create_training_job_args()
        # PipelineVariable should be preserved as-is, not stringified
        assert args["hyper_parameters"]["max_depth"] is max_depth

    def test_hyperparameters_with_parameter_string(self):
        """ParameterString in hyperparameters should be preserved through _create_training_job_args."""
        algo = ParameterString(name="Algorithm", default_value="xgboost")
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
            hyperparameters={"algorithm": algo},
        )
        args = trainer._create_training_job_args()
        assert args["hyper_parameters"]["algorithm"] is algo

    def test_hyperparameters_with_mixed_pipeline_and_static_values(self):
        """Mixed PipelineVariable and static values should both be handled correctly."""
        max_depth = ParameterInteger(name="MaxDepth", default_value=5)
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
            hyperparameters={
                "max_depth": max_depth,
                "eta": 0.1,
                "objective": "binary:logistic",
                "num_round": 100,
            },
        )
        args = trainer._create_training_job_args()
        hp = args["hyper_parameters"]
        # PipelineVariable preserved as-is
        assert hp["max_depth"] is max_depth
        # Static values serialized to strings
        assert hp["eta"] == "0.1"
        assert hp["objective"] == "binary:logistic"
        assert hp["num_round"] == "100"
