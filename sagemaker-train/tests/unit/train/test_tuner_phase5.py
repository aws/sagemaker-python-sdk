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
"""Phase 5: Additional HyperparameterTuner Tests for Coverage Boost."""
from __future__ import absolute_import

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from sagemaker.train.tuner import HyperparameterTuner
from sagemaker.core.parameter import ContinuousParameter, IntegerParameter


@pytest.fixture
def mock_model_trainer():
    """Create a mock ModelTrainer."""
    trainer = MagicMock()
    trainer.sagemaker_session = MagicMock()
    trainer.hyperparameters = {"learning_rate": 0.1}
    trainer.training_image = "123456789.dkr.ecr.us-west-2.amazonaws.com/sagemaker-training:latest"
    trainer.training_input_mode = "File"
    return trainer


@pytest.fixture
def hyperparameter_ranges():
    """Create sample hyperparameter ranges."""
    return {
        "learning_rate": ContinuousParameter(0.001, 0.1),
        "batch_size": IntegerParameter(32, 256),
    }


class TestHyperparameterTunerTune:
    """Test HyperparameterTuner.tune method."""

    def test_tune_with_wait_true(self, mock_model_trainer, hyperparameter_ranges):
        """Test tune method with wait=True."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
        )
        
        # Mock the _start_tuning_job method to avoid complex setup
        mock_tuning_job = MagicMock()
        mock_tuning_job.hyper_parameter_tuning_job_name = "test-tuning-job"
        tuner._start_tuning_job = MagicMock(return_value=mock_tuning_job)
        
        tuner.tune(wait=True)
        
        assert tuner.latest_tuning_job == mock_tuning_job
        mock_tuning_job.wait.assert_called_once()

    def test_tune_with_wait_false(self, mock_model_trainer, hyperparameter_ranges):
        """Test tune method with wait=False."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
        )
        
        # Mock the _start_tuning_job method to avoid complex setup
        mock_tuning_job = MagicMock()
        mock_tuning_job.hyper_parameter_tuning_job_name = "test-tuning-job"
        tuner._start_tuning_job = MagicMock(return_value=mock_tuning_job)
        
        tuner.tune(wait=False)
        
        assert tuner.latest_tuning_job == mock_tuning_job
        mock_tuning_job.wait.assert_not_called()

    def test_tune_with_inputs(self, mock_model_trainer, hyperparameter_ranges):
        """Test tune method with input data."""
        from sagemaker.train.configs import InputData
        
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
        )
        
        # Mock the _start_tuning_job method to avoid complex setup
        mock_tuning_job = MagicMock()
        mock_tuning_job.hyper_parameter_tuning_job_name = "test-tuning-job"
        tuner._start_tuning_job = MagicMock(return_value=mock_tuning_job)
        
        inputs = [
            InputData(channel_name="train", data_source="s3://bucket/train"),
            InputData(channel_name="validation", data_source="s3://bucket/val"),
        ]
        
        tuner.tune(inputs=inputs, wait=False)
        
        assert tuner.latest_tuning_job == mock_tuning_job
        tuner._start_tuning_job.assert_called_once_with(inputs)


class TestHyperparameterTunerCreate:
    """Test HyperparameterTuner.create class method."""

    @patch("sagemaker.train.tuner.HyperParameterTuningJob")
    def test_create_with_multiple_trainers(self, mock_tuning_job_class):
        """Test create method with multiple model trainers."""
        mock_trainer1 = MagicMock()
        mock_trainer1.sagemaker_session = MagicMock()
        mock_trainer1.hyperparameters = {}
        
        mock_trainer2 = MagicMock()
        mock_trainer2.sagemaker_session = MagicMock()
        mock_trainer2.hyperparameters = {}
        
        model_trainer_dict = {
            "trainer1": mock_trainer1,
            "trainer2": mock_trainer2,
        }
        
        objective_metric_name_dict = {
            "trainer1": "accuracy",
            "trainer2": "f1_score",
        }
        
        hyperparameter_ranges_dict = {
            "trainer1": {"lr": ContinuousParameter(0.001, 0.1)},
            "trainer2": {"lr": ContinuousParameter(0.01, 0.5)},
        }
        
        tuner = HyperparameterTuner.create(
            model_trainer_dict=model_trainer_dict,
            objective_metric_name_dict=objective_metric_name_dict,
            hyperparameter_ranges_dict=hyperparameter_ranges_dict,
        )
        
        assert tuner.model_trainer_dict == model_trainer_dict
        assert tuner.objective_metric_name_dict == objective_metric_name_dict

    def test_create_with_invalid_trainer_dict(self):
        """Test create raises error with None model_trainer_dict."""
        with pytest.raises(ValueError, match="At least one model_trainer should be provided"):
            HyperparameterTuner.create(
                model_trainer_dict=None,
                objective_metric_name_dict={"trainer1": "accuracy"},
                hyperparameter_ranges_dict={"trainer1": {}},
            )

    def test_create_with_mismatched_keys(self):
        """Test create raises error when dict keys don't match."""
        mock_trainer = MagicMock()
        mock_trainer.sagemaker_session = MagicMock()
        
        model_trainer_dict = {"trainer1": mock_trainer}
        objective_metric_name_dict = {"trainer2": "accuracy"}  # Different key
        hyperparameter_ranges_dict = {"trainer1": {}}
        
        with pytest.raises(ValueError):
            HyperparameterTuner.create(
                model_trainer_dict=model_trainer_dict,
                objective_metric_name_dict=objective_metric_name_dict,
                hyperparameter_ranges_dict=hyperparameter_ranges_dict,
            )


class TestHyperparameterTunerWarmStart:
    """Test HyperparameterTuner warm start functionality."""

    @patch("sagemaker.train.tuner.HyperParameterTuningJobWarmStartConfig")
    def test_transfer_learning_tuner(self, mock_warm_start_config, mock_model_trainer, hyperparameter_ranges):
        """Test transfer_learning_tuner method."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
        )
        tuner._current_job_name = "parent-tuning-job"
        
        # Mock latest_tuning_job to avoid "No tuning job available" error
        mock_tuning_job = MagicMock()
        mock_tuning_job.hyper_parameter_tuning_job_name = "parent-tuning-job"
        tuner.latest_tuning_job = mock_tuning_job
        
        # Mock the warm start config creation
        mock_config_instance = MagicMock()
        mock_warm_start_config.return_value = mock_config_instance
        
        new_tuner = tuner.transfer_learning_tuner()
        
        assert new_tuner is not None
        assert new_tuner.warm_start_config == mock_config_instance

    @patch("sagemaker.train.tuner.HyperParameterTuningJobWarmStartConfig")
    def test_transfer_learning_tuner_with_additional_parents(
        self, mock_warm_start_config, mock_model_trainer, hyperparameter_ranges
    ):
        """Test transfer_learning_tuner with additional parent jobs."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
        )
        tuner._current_job_name = "parent-tuning-job"
        
        # Mock latest_tuning_job to avoid "No tuning job available" error
        mock_tuning_job = MagicMock()
        mock_tuning_job.hyper_parameter_tuning_job_name = "parent-tuning-job"
        tuner.latest_tuning_job = mock_tuning_job
        
        # Mock the warm start config creation
        mock_config_instance = MagicMock()
        mock_warm_start_config.return_value = mock_config_instance
        
        additional_parents = ["other-parent-job-1", "other-parent-job-2"]
        new_tuner = tuner.transfer_learning_tuner(additional_parents=additional_parents)
        
        assert new_tuner is not None
        assert new_tuner.warm_start_config == mock_config_instance

    @patch("sagemaker.train.tuner.HyperParameterTuningJobWarmStartConfig")
    def test_transfer_learning_tuner_with_new_trainer(
        self, mock_warm_start_config, mock_model_trainer, hyperparameter_ranges
    ):
        """Test transfer_learning_tuner with a new model trainer."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
        )
        tuner._current_job_name = "parent-tuning-job"
        
        # Mock latest_tuning_job to avoid "No tuning job available" error
        mock_tuning_job = MagicMock()
        mock_tuning_job.hyper_parameter_tuning_job_name = "parent-tuning-job"
        tuner.latest_tuning_job = mock_tuning_job
        
        # Mock the warm start config creation
        mock_config_instance = MagicMock()
        mock_warm_start_config.return_value = mock_config_instance
        
        new_trainer = MagicMock()
        new_trainer.sagemaker_session = MagicMock()
        new_trainer.hyperparameters = {"learning_rate": 0.05}
        new_trainer.training_image = "123456789.dkr.ecr.us-west-2.amazonaws.com/sagemaker-training:latest"
        new_trainer.training_input_mode = "File"
        
        new_tuner = tuner.transfer_learning_tuner(model_trainer=new_trainer)
        
        assert new_tuner is not None
        assert new_tuner.model_trainer == new_trainer


class TestHyperparameterTunerPrepare:
    """Test HyperparameterTuner preparation methods."""

    def test_prepare_job_name_for_tuning_with_custom_name(
        self, mock_model_trainer, hyperparameter_ranges
    ):
        """Test _prepare_job_name_for_tuning with custom job name."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            base_tuning_job_name="custom-tuning",
        )
        
        tuner._prepare_job_name_for_tuning(job_name="my-specific-job")
        
        assert tuner._current_job_name == "my-specific-job"

    @patch("sagemaker.train.tuner.name_from_base")
    def test_prepare_job_name_for_tuning_auto_generated(
        self, mock_name_from_base, mock_model_trainer, hyperparameter_ranges
    ):
        """Test _prepare_job_name_for_tuning with auto-generated name."""
        mock_name_from_base.return_value = "auto-generated-job-name"
        
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            base_tuning_job_name="custom-tuning",
        )
        
        tuner._prepare_job_name_for_tuning()
        
        assert tuner._current_job_name == "auto-generated-job-name"
        mock_name_from_base.assert_called_once_with("custom-tuning", max_length=32, short=True)

    def test_prepare_static_hyperparameters_for_tuning(
        self, mock_model_trainer, hyperparameter_ranges
    ):
        """Test _prepare_static_hyperparameters_for_tuning method."""
        mock_model_trainer.hyperparameters = {
            "learning_rate": 0.1,
            "batch_size": 32,
            "epochs": 10,
        }
        
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
        )
        
        tuner._prepare_static_hyperparameters_for_tuning()
        
        # Static hyperparameters should exclude those in ranges
        assert tuner.static_hyperparameters is not None
        assert "epochs" in tuner.static_hyperparameters
        assert "learning_rate" not in tuner.static_hyperparameters

    def test_prepare_auto_parameters_for_tuning_disabled(
        self, mock_model_trainer, hyperparameter_ranges
    ):
        """Test _prepare_auto_parameters_for_tuning when autotune is disabled."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            autotune=False,
        )
        
        # Set static_hyperparameters before calling the method
        tuner.static_hyperparameters = {"epochs": 10}
        tuner._prepare_auto_parameters_for_tuning()
        
        # Should remain None when autotune is False
        assert tuner.auto_parameters is None

    def test_prepare_auto_parameters_for_tuning_enabled(
        self, mock_model_trainer
    ):
        """Test _prepare_auto_parameters_for_tuning when autotune is enabled."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges={},
            autotune=True,
        )
        
        tuner.static_hyperparameters = {"epochs": 10, "batch_size": 32}
        tuner._prepare_auto_parameters_for_tuning()
        
        # Auto parameters should be set when autotune is True
        assert tuner.auto_parameters is not None


class TestHyperparameterTunerOverrideResourceConfig:
    """Test HyperparameterTuner.override_resource_config method."""

    def test_override_resource_config_single_trainer(
        self, mock_model_trainer, hyperparameter_ranges
    ):
        """Test override_resource_config with single trainer."""
        from sagemaker.core.shapes import HyperParameterTuningInstanceConfig
        
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
        )
        
        instance_configs = [
            HyperParameterTuningInstanceConfig(
                instance_type="ml.p3.2xlarge",
                instance_count=2,
                volume_size_in_gb=50,
            )
        ]
        
        tuner.override_resource_config(instance_configs=instance_configs)
        
        assert tuner.instance_configs == instance_configs

    def test_override_resource_config_multiple_trainers(self):
        """Test override_resource_config with multiple trainers."""
        from sagemaker.core.shapes import HyperParameterTuningInstanceConfig
        
        mock_trainer1 = MagicMock()
        mock_trainer1.sagemaker_session = MagicMock()
        mock_trainer2 = MagicMock()
        mock_trainer2.sagemaker_session = MagicMock()
        
        tuner = HyperparameterTuner.create(
            model_trainer_dict={"trainer1": mock_trainer1, "trainer2": mock_trainer2},
            objective_metric_name_dict={"trainer1": "acc", "trainer2": "f1"},
            hyperparameter_ranges_dict={
                "trainer1": {"lr": ContinuousParameter(0.001, 0.1)},
                "trainer2": {"lr": ContinuousParameter(0.001, 0.1)},
            },
        )
        
        instance_configs_dict = {
            "trainer1": [
                HyperParameterTuningInstanceConfig(
                    instance_type="ml.p3.2xlarge",
                    instance_count=1,
                    volume_size_in_gb=30,
                )
            ],
            "trainer2": [
                HyperParameterTuningInstanceConfig(
                    instance_type="ml.g4dn.xlarge",
                    instance_count=1,
                    volume_size_in_gb=30,
                )
            ],
        }
        
        tuner.override_resource_config(instance_configs=instance_configs_dict)
        
        assert tuner.instance_configs_dict == instance_configs_dict


class TestHyperparameterTunerAddModelTrainer:
    """Test HyperparameterTuner._add_model_trainer method."""

    def test_add_model_trainer(self, mock_model_trainer, hyperparameter_ranges):
        """Test _add_model_trainer method."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            model_trainer_name="trainer1",
        )
        
        new_trainer = MagicMock()
        new_trainer.sagemaker_session = MagicMock()
        new_ranges = {"lr": ContinuousParameter(0.01, 0.5)}
        
        tuner._add_model_trainer(
            model_trainer_name="trainer2",
            model_trainer=new_trainer,
            objective_metric_name="f1_score",
            hyperparameter_ranges=new_ranges,
        )
        
        assert "trainer2" in tuner.model_trainer_dict
        assert tuner.model_trainer_dict["trainer2"] == new_trainer
        assert tuner.objective_metric_name_dict["trainer2"] == "f1_score"
        assert tuner._hyperparameter_ranges_dict["trainer2"] == new_ranges
