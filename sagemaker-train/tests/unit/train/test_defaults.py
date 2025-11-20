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
"""Tests for defaults module."""
from __future__ import absolute_import

import pytest
from unittest.mock import MagicMock, patch

from sagemaker.train.defaults import (
    TrainDefaults,
    JumpStartTrainDefaults,
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_INSTANCE_COUNT,
    DEFAULT_VOLUME_SIZE,
    DEFAULT_MAX_RUNTIME_IN_SECONDS,
)
from sagemaker.train.configs import Compute, StoppingCondition


class TestDefaultConstants:
    """Test default constant values."""

    def test_default_instance_type(self):
        """Test DEFAULT_INSTANCE_TYPE constant."""
        assert DEFAULT_INSTANCE_TYPE == "ml.m5.xlarge"

    def test_default_instance_count(self):
        """Test DEFAULT_INSTANCE_COUNT constant."""
        assert DEFAULT_INSTANCE_COUNT == 1

    def test_default_volume_size(self):
        """Test DEFAULT_VOLUME_SIZE constant."""
        assert DEFAULT_VOLUME_SIZE == 30

    def test_default_max_runtime(self):
        """Test DEFAULT_MAX_RUNTIME_IN_SECONDS constant."""
        assert DEFAULT_MAX_RUNTIME_IN_SECONDS == 3600


class TestTrainDefaultsGetSagemakerSession:
    """Test TrainDefaults.get_sagemaker_session method."""

    def test_returns_provided_session(self):
        """Test returns the provided session."""
        mock_session = MagicMock()
        result = TrainDefaults.get_sagemaker_session(sagemaker_session=mock_session)
        assert result == mock_session

    @patch("sagemaker.train.defaults.Session")
    def test_creates_default_session_when_none(self, mock_session_class):
        """Test creates default session when none provided."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        result = TrainDefaults.get_sagemaker_session(sagemaker_session=None)

        mock_session_class.assert_called_once()
        assert result == mock_session


class TestTrainDefaultsGetRole:
    """Test TrainDefaults.get_role method."""

    def test_returns_provided_role(self):
        """Test returns the provided role."""
        role = "arn:aws:iam::123456789012:role/MyRole"
        result = TrainDefaults.get_role(role=role)
        assert result == role

    @patch("sagemaker.train.defaults.get_execution_role")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    def test_gets_execution_role_when_none(self, mock_get_session, mock_get_role):
        """Test gets execution role when none provided."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        expected_role = "arn:aws:iam::123456789012:role/ExecutionRole"
        mock_get_role.return_value = expected_role

        result = TrainDefaults.get_role(role=None)

        mock_get_role.assert_called_once_with(mock_session)
        assert result == expected_role

    @patch("sagemaker.train.defaults.get_execution_role")
    def test_uses_provided_session_for_role(self, mock_get_role):
        """Test uses provided session when getting role."""
        mock_session = MagicMock()
        expected_role = "arn:aws:iam::123456789012:role/ExecutionRole"
        mock_get_role.return_value = expected_role

        result = TrainDefaults.get_role(role=None, sagemaker_session=mock_session)

        mock_get_role.assert_called_once_with(mock_session)
        assert result == expected_role


class TestTrainDefaultsGetBaseJobName:
    """Test TrainDefaults.get_base_job_name method."""

    def test_returns_provided_base_job_name(self):
        """Test returns the provided base job name."""
        base_name = "my-custom-job"
        result = TrainDefaults.get_base_job_name(base_job_name=base_name)
        assert result == base_name

    def test_generates_name_from_algorithm_name(self):
        """Test generates name from algorithm name."""
        algorithm_name = "xgboost"
        result = TrainDefaults.get_base_job_name(
            base_job_name=None, algorithm_name=algorithm_name
        )
        assert result == "xgboost-job"

    @patch("sagemaker.train.defaults._get_repo_name_from_image")
    def test_generates_name_from_training_image(self, mock_get_repo):
        """Test generates name from training image."""
        training_image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest"
        mock_get_repo.return_value = "my-image"

        result = TrainDefaults.get_base_job_name(
            base_job_name=None, training_image=training_image
        )

        mock_get_repo.assert_called_once_with(training_image)
        assert result == "my-image-job"

    def test_algorithm_name_takes_precedence_over_image(self):
        """Test algorithm name takes precedence over training image."""
        algorithm_name = "xgboost"
        training_image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest"

        result = TrainDefaults.get_base_job_name(
            base_job_name=None,
            algorithm_name=algorithm_name,
            training_image=training_image,
        )

        assert result == "xgboost-job"


class TestTrainDefaultsGetCompute:
    """Test TrainDefaults.get_compute method."""

    def test_returns_provided_compute(self):
        """Test returns the provided compute configuration."""
        compute = Compute(
            instance_type="ml.p3.2xlarge",
            instance_count=2,
            volume_size_in_gb=50,
        )
        result = TrainDefaults.get_compute(compute=compute)
        assert result == compute

    def test_creates_default_compute_when_none(self):
        """Test creates default compute when none provided."""
        result = TrainDefaults.get_compute(compute=None)

        assert result.instance_type == DEFAULT_INSTANCE_TYPE
        assert result.instance_count == DEFAULT_INSTANCE_COUNT
        assert result.volume_size_in_gb == DEFAULT_VOLUME_SIZE

    def test_default_compute_has_correct_values(self):
        """Test default compute has all expected values."""
        result = TrainDefaults.get_compute(compute=None)

        assert isinstance(result, Compute)
        assert result.instance_type == "ml.m5.xlarge"
        assert result.instance_count == 1
        assert result.volume_size_in_gb == 30


class TestTrainDefaultsGetStoppingCondition:
    """Test TrainDefaults.get_stopping_condition method."""

    def test_returns_provided_stopping_condition(self):
        """Test returns the provided stopping condition."""
        stopping_condition = StoppingCondition(
            max_runtime_in_seconds=7200,
            max_wait_time_in_seconds=10800,
        )
        result = TrainDefaults.get_stopping_condition(stopping_condition=stopping_condition)
        assert result == stopping_condition

    def test_creates_default_stopping_condition_when_none(self):
        """Test creates default stopping condition when none provided."""
        result = TrainDefaults.get_stopping_condition(stopping_condition=None)

        assert isinstance(result, StoppingCondition)
        assert result.max_runtime_in_seconds == DEFAULT_MAX_RUNTIME_IN_SECONDS

    def test_fills_missing_max_runtime(self):
        """Test fills missing max runtime in seconds."""
        stopping_condition = StoppingCondition(max_runtime_in_seconds=None)
        result = TrainDefaults.get_stopping_condition(stopping_condition=stopping_condition)

        assert result.max_runtime_in_seconds == DEFAULT_MAX_RUNTIME_IN_SECONDS

    def test_preserves_provided_max_runtime(self):
        """Test preserves provided max runtime."""
        max_runtime = 5400
        stopping_condition = StoppingCondition(max_runtime_in_seconds=max_runtime)
        result = TrainDefaults.get_stopping_condition(stopping_condition=stopping_condition)

        assert result.max_runtime_in_seconds == max_runtime


class TestTrainDefaultsGetOutputDataConfig:
    """Test TrainDefaults.get_output_data_config method."""

    @patch("sagemaker.train.defaults._default_s3_uri")
    def test_creates_output_data_config(self, mock_default_s3_uri):
        """Test creates output data config with default S3 URI."""
        base_job_name = "my-training-job"
        mock_session = MagicMock()
        mock_default_s3_uri.return_value = "s3://my-bucket/output"

        result = TrainDefaults.get_output_data_config(
            base_job_name=base_job_name,
            sagemaker_session=mock_session,
        )

        mock_default_s3_uri.assert_called_once_with(
            session=mock_session,
            additional_path=base_job_name,
        )
        assert result.s3_output_path == "s3://my-bucket/output"

    @patch("sagemaker.train.defaults.configs.OutputDataConfig")
    def test_uses_provided_output_data_config(self, mock_output_config_class):
        """Test uses provided output data config."""
        base_job_name = "my-training-job"
        mock_session = MagicMock()
        custom_config = MagicMock()
        custom_config.s3_output_path = "s3://custom-bucket/custom-path"

        result = TrainDefaults.get_output_data_config(
            base_job_name=base_job_name,
            output_data_config=custom_config,
            sagemaker_session=mock_session,
        )

        # Should not create new config when one is provided
        mock_output_config_class.assert_not_called()
        assert result == custom_config


class TestJumpStartTrainDefaultsGetTrainingComponentsModel:
    """Test JumpStartTrainDefaults._get_training_components_model method."""

    def test_returns_document_when_no_config_name(self):
        """Test returns document when no training config name."""
        mock_document = MagicMock()
        mock_config = MagicMock()
        mock_config.training_config_name = None

        result = JumpStartTrainDefaults._get_training_components_model(
            document=mock_document,
            jumpstart_config=mock_config,
        )

        assert result == mock_document

    def test_returns_training_config_component_when_config_name_provided(self):
        """Test returns training config component when config name provided."""
        mock_document = MagicMock()
        mock_config = MagicMock()
        mock_config.training_config_name = "config1"
        mock_document.TrainingConfigs = {"config1": {}, "config2": {}}
        mock_component = MagicMock()
        mock_document.TrainingConfigComponents = {mock_config: mock_component}

        result = JumpStartTrainDefaults._get_training_components_model(
            document=mock_document,
            jumpstart_config=mock_config,
        )

        assert result == mock_component

    def test_raises_error_when_config_not_found(self):
        """Test raises error when training config not found."""
        mock_document = MagicMock()
        mock_config = MagicMock()
        mock_config.training_config_name = "invalid_config"
        mock_config.model_id = "test-model"
        mock_document.TrainingConfigs = {"config1": {}, "config2": {}}

        with pytest.raises(ValueError, match="Training config invalid_config not found"):
            JumpStartTrainDefaults._get_training_components_model(
                document=mock_document,
                jumpstart_config=mock_config,
            )


class TestJumpStartTrainDefaultsGetTrainingVariant:
    """Test JumpStartTrainDefaults._get_training_variant method."""

    def test_returns_variant_by_instance_family(self):
        """Test returns variant by instance family."""
        mock_model = MagicMock()
        mock_variant = MagicMock()
        mock_model.TrainingInstanceTypeVariants.Variants = {
            "p3": mock_variant,
            "ml.p3.2xlarge": MagicMock(),
        }
        compute = Compute(instance_type="ml.p3.2xlarge")

        result = JumpStartTrainDefaults._get_training_variant(
            training_components_model=mock_model,
            compute=compute,
        )

        assert result == mock_variant

    def test_returns_variant_by_exact_instance_type(self):
        """Test returns variant by exact instance type when family not found."""
        mock_model = MagicMock()
        mock_variant = MagicMock()
        mock_model.TrainingInstanceTypeVariants.Variants = {
            "ml.p3.2xlarge": mock_variant,
        }
        compute = Compute(instance_type="ml.p3.2xlarge")

        result = JumpStartTrainDefaults._get_training_variant(
            training_components_model=mock_model,
            compute=compute,
        )

        assert result == mock_variant

    def test_returns_none_when_no_variant_found(self):
        """Test returns None when no variant found."""
        mock_model = MagicMock()
        mock_model.TrainingInstanceTypeVariants.Variants = {}
        compute = Compute(instance_type="ml.p3.2xlarge")

        result = JumpStartTrainDefaults._get_training_variant(
            training_components_model=mock_model,
            compute=compute,
        )

        assert result is None


class TestJumpStartTrainDefaultsGetCompute:
    """Test JumpStartTrainDefaults.get_compute method."""

    @patch("sagemaker.train.defaults.get_hub_content_and_document")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    def test_creates_default_compute_from_document(
        self, mock_get_session, mock_get_hub_content
    ):
        """Test creates default compute from JumpStart document."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_document = MagicMock()
        mock_document.DefaultTrainingInstanceType = "ml.p3.2xlarge"
        mock_document.TrainingVolumeSize = 100
        mock_get_hub_content.return_value = (None, mock_document)

        mock_config = MagicMock()
        mock_config.training_config_name = None

        result = JumpStartTrainDefaults.get_compute(
            jumpstart_config=mock_config,
            compute=None,
            sagemaker_session=mock_session,
        )

        assert result.instance_type == "ml.p3.2xlarge"
        assert result.instance_count == DEFAULT_INSTANCE_COUNT
        assert result.volume_size_in_gb == 100

    @patch("sagemaker.train.defaults.get_hub_content_and_document")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    def test_fills_missing_instance_type_from_document(
        self, mock_get_session, mock_get_hub_content
    ):
        """Test fills missing instance type from document."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_document = MagicMock()
        mock_document.DefaultTrainingInstanceType = "ml.p3.2xlarge"
        mock_document.TrainingVolumeSize = None
        mock_get_hub_content.return_value = (None, mock_document)

        mock_config = MagicMock()
        mock_config.training_config_name = None

        compute = Compute(instance_type=None, instance_count=2)

        result = JumpStartTrainDefaults.get_compute(
            jumpstart_config=mock_config,
            compute=compute,
            sagemaker_session=mock_session,
        )

        assert result.instance_type == "ml.p3.2xlarge"
        assert result.instance_count == 2

    @patch("sagemaker.train.defaults.get_hub_content_and_document")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    def test_uses_default_volume_size_when_not_in_document(
        self, mock_get_session, mock_get_hub_content
    ):
        """Test uses default volume size when not in document."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_document = MagicMock()
        mock_document.DefaultTrainingInstanceType = "ml.p3.2xlarge"
        mock_document.TrainingVolumeSize = None
        mock_get_hub_content.return_value = (None, mock_document)

        mock_config = MagicMock()
        mock_config.training_config_name = None

        compute = Compute(volume_size_in_gb=None)

        result = JumpStartTrainDefaults.get_compute(
            jumpstart_config=mock_config,
            compute=compute,
            sagemaker_session=mock_session,
        )

        assert result.volume_size_in_gb == DEFAULT_VOLUME_SIZE
