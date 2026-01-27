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
"""Tests for tuner module."""
from __future__ import absolute_import

import pytest
from unittest.mock import MagicMock, patch

from sagemaker.train.tuner import (
    HyperparameterTuner,
    WarmStartTypes,
    GRID_SEARCH,
)
from sagemaker.core.parameter import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
)
from sagemaker.core.shapes import (
    HyperParameterTuningJobWarmStartConfig,
    Channel,
    DataSource,
    S3DataSource,
)


# ---------------------------------------------------------------------------
# Factory functions for creating test objects (reduces fixture duplication)
# ---------------------------------------------------------------------------


def _create_mock_model_trainer(with_internal_channels=False):
    """Create a mock ModelTrainer with common attributes.

    Args:
        with_internal_channels: If True, adds internal channels (code, sm_drivers)
            to input_data_config for testing channel inclusion in tuning jobs.
    """
    trainer = MagicMock()
    trainer.sagemaker_session = MagicMock()
    trainer.hyperparameters = {"learning_rate": 0.1, "batch_size": 32, "optimizer": "adam"}
    trainer.training_image = "test-image:latest"
    trainer.training_input_mode = "File"
    trainer.role = "arn:aws:iam::123456789012:role/SageMakerRole"
    trainer.output_data_config = MagicMock()
    trainer.output_data_config.s3_output_path = "s3://bucket/output"
    trainer.compute = MagicMock()
    trainer.compute.instance_type = "ml.m5.xlarge"
    trainer.compute.instance_count = 1
    trainer.compute.volume_size_in_gb = 30
    trainer.stopping_condition = MagicMock()
    trainer.stopping_condition.max_runtime_in_seconds = 3600
    trainer.input_data_config = None

    if with_internal_channels:
        trainer.input_data_config = [
            _create_channel("code", "s3://bucket/code"),
            _create_channel("sm_drivers", "s3://bucket/drivers"),
        ]
    return trainer


def _create_hyperparameter_ranges():
    """Create sample hyperparameter ranges."""
    return {
        "learning_rate": ContinuousParameter(0.001, 0.1),
        "batch_size": IntegerParameter(32, 256),
        "optimizer": CategoricalParameter(["sgd", "adam"]),
    }


def _create_single_hp_range():
    """Create a single hyperparameter range for simple tests."""
    return {"learning_rate": ContinuousParameter(0.001, 0.1)}


def _create_channel(name: str, uri: str) -> Channel:
    """Create a Channel with S3 data source."""
    return Channel(
        channel_name=name,
        data_source=DataSource(
            s3_data_source=S3DataSource(
                s3_data_type="S3Prefix", s3_uri=uri, s3_data_distribution_type="FullyReplicated"
            )
        ),
    )


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestWarmStartTypes:
    """Test WarmStartTypes enum."""

    def test_identical_data_and_algorithm(self):
        """Test IDENTICAL_DATA_AND_ALGORITHM enum value."""
        assert WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM.value == "IdenticalDataAndAlgorithm"

    def test_transfer_learning(self):
        """Test TRANSFER_LEARNING enum value."""
        assert WarmStartTypes.TRANSFER_LEARNING.value == "TransferLearning"


class TestHyperparameterTunerInit:
    """Test HyperparameterTuner initialization."""

    @pytest.fixture
    def mock_model_trainer(self):
        """Create a mock ModelTrainer."""
        return _create_mock_model_trainer()

    @pytest.fixture
    def hyperparameter_ranges(self):
        """Create sample hyperparameter ranges."""
        return _create_hyperparameter_ranges()

    def test_init_with_basic_params(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with basic parameters."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
        )

        assert tuner.model_trainer == mock_model_trainer
        assert tuner.objective_metric_name == "accuracy"
        assert tuner._hyperparameter_ranges == hyperparameter_ranges
        assert tuner.strategy == "Bayesian"
        assert tuner.objective_type == "Maximize"
        assert tuner.max_jobs == 1
        assert tuner.max_parallel_jobs == 1

    def test_init_with_custom_strategy(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with custom strategy."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="loss",
            hyperparameter_ranges=hyperparameter_ranges,
            strategy="Random",
            objective_type="Minimize",
        )

        assert tuner.strategy == "Random"
        assert tuner.objective_type == "Minimize"

    def test_init_with_grid_search_strategy(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with Grid search strategy."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            strategy=GRID_SEARCH,
        )

        assert tuner.strategy == GRID_SEARCH
        assert tuner.max_jobs is None  # Grid search doesn't set default max_jobs

    def test_init_with_max_jobs(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with max_jobs specified."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=10,
            max_parallel_jobs=2,
        )

        assert tuner.max_jobs == 10
        assert tuner.max_parallel_jobs == 2

    def test_init_with_metric_definitions(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with metric definitions."""
        metric_definitions = [
            {"Name": "train:loss", "Regex": "loss: ([0-9\\.]+)"},
            {"Name": "validation:accuracy", "Regex": "accuracy: ([0-9\\.]+)"},
        ]

        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="validation:accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=metric_definitions,
        )

        assert tuner.metric_definitions == metric_definitions

    def test_init_with_tags(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with tags."""
        tags = [{"Key": "project", "Value": "ml-project"}]

        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            tags=tags,
        )

        assert tuner.tags == tags

    def test_init_with_base_tuning_job_name(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with base tuning job name."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            base_tuning_job_name="my-tuning-job",
        )

        assert tuner.base_tuning_job_name == "my-tuning-job"

    def test_init_with_warm_start_config(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with warm start config."""
        warm_start_config = MagicMock(spec=HyperParameterTuningJobWarmStartConfig)

        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            warm_start_config=warm_start_config,
        )

        assert tuner.warm_start_config == warm_start_config

    def test_init_with_early_stopping(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with early stopping."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            early_stopping_type="Auto",
        )

        assert tuner.early_stopping_type == "Auto"

    def test_init_with_random_seed(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with random seed."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            random_seed=42,
        )

        assert tuner.random_seed == 42

    def test_init_with_autotune(self, mock_model_trainer):
        """Test initialization with autotune enabled."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges={},
            autotune=True,
        )

        assert tuner.autotune is True

    def test_init_without_ranges_raises_error(self, mock_model_trainer):
        """Test initialization without hyperparameter ranges raises error."""
        with pytest.raises(ValueError, match="Need to specify hyperparameter ranges"):
            HyperparameterTuner(
                model_trainer=mock_model_trainer,
                objective_metric_name="accuracy",
                hyperparameter_ranges={},
            )

    def test_init_with_empty_ranges_raises_error(self, mock_model_trainer):
        """Test initialization with empty ranges raises error."""
        with pytest.raises(ValueError, match="Need to specify hyperparameter ranges"):
            HyperparameterTuner(
                model_trainer=mock_model_trainer,
                objective_metric_name="accuracy",
                hyperparameter_ranges=None,
            )

    def test_init_with_static_hyperparameters_without_autotune_raises_error(
        self, mock_model_trainer, hyperparameter_ranges
    ):
        """Test initialization with static hyperparameters without autotune raises error."""
        with pytest.raises(ValueError, match="hyperparameters_to_keep_static parameter is set"):
            HyperparameterTuner(
                model_trainer=mock_model_trainer,
                objective_metric_name="accuracy",
                hyperparameter_ranges=hyperparameter_ranges,
                hyperparameters_to_keep_static=["learning_rate"],
                autotune=False,
            )

    def test_init_with_duplicate_static_hyperparameters_raises_error(
        self, mock_model_trainer, hyperparameter_ranges
    ):
        """Test initialization with duplicate static hyperparameters raises error."""
        with pytest.raises(ValueError, match="Please remove duplicate names"):
            HyperparameterTuner(
                model_trainer=mock_model_trainer,
                objective_metric_name="accuracy",
                hyperparameter_ranges=hyperparameter_ranges,
                hyperparameters_to_keep_static=["learning_rate", "learning_rate"],
                autotune=True,
            )

    def test_init_with_model_trainer_name(self, mock_model_trainer, hyperparameter_ranges):
        """Test initialization with model_trainer_name."""
        tuner = HyperparameterTuner(
            model_trainer=mock_model_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            model_trainer_name="trainer1",
        )

        assert tuner.model_trainer is None
        assert tuner.model_trainer_dict == {"trainer1": mock_model_trainer}
        assert tuner.objective_metric_name_dict == {"trainer1": "accuracy"}
        assert tuner._hyperparameter_ranges_dict == {"trainer1": hyperparameter_ranges}


class TestHyperparameterTunerProperties:
    """Test HyperparameterTuner properties."""

    @pytest.fixture
    def tuner(self):
        """Create a basic tuner instance."""
        return HyperparameterTuner(
            model_trainer=_create_mock_model_trainer(),
            objective_metric_name="accuracy",
            hyperparameter_ranges=_create_single_hp_range(),
        )

    def test_sagemaker_session_property(self, tuner):
        """Test sagemaker_session property."""
        assert tuner.sagemaker_session == tuner.model_trainer.sagemaker_session

    def test_hyperparameter_ranges_property(self, tuner):
        """Test hyperparameter_ranges property."""
        ranges = tuner.hyperparameter_ranges()
        assert "ContinuousParameterRanges" in ranges
        assert len(ranges["ContinuousParameterRanges"]) == 1
        assert ranges["ContinuousParameterRanges"][0]["name"] == "learning_rate"

    def test_hyperparameter_ranges_dict_property_returns_none(self, tuner):
        """Test hyperparameter_ranges_dict property when dict is None."""
        assert tuner.hyperparameter_ranges_dict() is None

    def test_hyperparameter_ranges_dict_property_with_dict(self):
        """Test hyperparameter_ranges_dict property with model_trainer_dict."""
        tuner = HyperparameterTuner(
            model_trainer=_create_mock_model_trainer(),
            objective_metric_name="accuracy",
            hyperparameter_ranges=_create_single_hp_range(),
            model_trainer_name="trainer1",
        )

        ranges_dict = tuner.hyperparameter_ranges_dict()
        assert "trainer1" in ranges_dict
        assert "ContinuousParameterRanges" in ranges_dict["trainer1"]


class TestHyperparameterTunerMethods:
    """Test HyperparameterTuner methods."""

    @pytest.fixture
    def tuner_with_job(self):
        """Create a tuner with a latest_tuning_job."""
        tuner = HyperparameterTuner(
            model_trainer=_create_mock_model_trainer(),
            objective_metric_name="accuracy",
            hyperparameter_ranges=_create_single_hp_range(),
        )
        tuner.latest_tuning_job = MagicMock()
        tuner._current_job_name = "test-tuning-job"
        return tuner

    def test_ensure_last_tuning_job_raises_error_when_none(self):
        """Test _ensure_last_tuning_job raises error when no job exists."""
        tuner = HyperparameterTuner(
            model_trainer=_create_mock_model_trainer(),
            objective_metric_name="accuracy",
            hyperparameter_ranges=_create_single_hp_range(),
        )

        with pytest.raises(ValueError):
            tuner._ensure_last_tuning_job()

    def test_stop_tuning_job(self, tuner_with_job):
        """Test stop_tuning_job method."""
        tuner_with_job.stop_tuning_job()
        tuner_with_job.latest_tuning_job.stop.assert_called_once()

    def test_describe(self, tuner_with_job):
        """Test describe method."""
        tuner_with_job.describe()
        tuner_with_job.latest_tuning_job.refresh.assert_called_once()

    def test_wait(self, tuner_with_job):
        """Test wait method."""
        tuner_with_job.wait()
        tuner_with_job.latest_tuning_job.wait.assert_called_once()

    def test_best_training_job(self, tuner_with_job):
        """Test best_training_job method."""
        mock_best_job = MagicMock()
        mock_best_job.training_job_name = "best-job-123"
        mock_best_job.training_job_definition_name = "training-def"

        mock_tuning_job = MagicMock()
        mock_tuning_job.best_training_job = mock_best_job
        tuner_with_job.latest_tuning_job.refresh.return_value = mock_tuning_job

        best_job = tuner_with_job.best_training_job()
        assert best_job == "best-job-123"

    def test_analytics(self, tuner_with_job):
        """Test analytics method."""
        with patch("sagemaker.train.tuner.HyperparameterTuningJobAnalytics") as mock_analytics:
            tuner_with_job.analytics()
            # Analytics is called with positional args
            assert mock_analytics.called
            call_args = mock_analytics.call_args
            assert (
                call_args[0][0] == tuner_with_job.latest_tuning_job.hyper_parameter_tuning_job_name
            )


class TestHyperparameterTunerValidation:
    """Test HyperparameterTuner validation methods."""

    def test_validate_model_trainer_dict_with_none(self):
        """Test _validate_model_trainer_dict with None."""
        with pytest.raises(ValueError, match="At least one model_trainer should be provided"):
            HyperparameterTuner._validate_model_trainer_dict(None)

    def test_validate_model_trainer_dict_with_empty_dict(self):
        """Test _validate_model_trainer_dict with empty dict."""
        with pytest.raises(ValueError, match="At least one model_trainer should be provided"):
            HyperparameterTuner._validate_model_trainer_dict({})

    def test_validate_dict_argument_with_none(self):
        """Test _validate_dict_argument with None returns without error."""
        # None is allowed and returns without raising
        HyperparameterTuner._validate_dict_argument("test_arg", None, ["key1", "key2"])

    def test_validate_dict_argument_with_invalid_keys(self):
        """Test _validate_dict_argument with invalid keys."""
        with pytest.raises(ValueError):
            HyperparameterTuner._validate_dict_argument(
                "test_arg",
                {"key1": "value1", "invalid_key": "value2"},
                ["key1", "key2"],
            )

    def test_validate_dict_argument_with_require_same_keys(self):
        """Test _validate_dict_argument with require_same_keys."""
        with pytest.raises(ValueError):
            HyperparameterTuner._validate_dict_argument(
                "test_arg",
                {"key1": "value1"},
                ["key1", "key2"],
                require_same_keys=True,
            )


class TestHyperparameterTunerStaticMethods:
    """Test HyperparameterTuner static methods."""

    def test_prepare_static_hyperparameters(self):
        """Test _prepare_static_hyperparameters method."""
        mock_trainer = _create_mock_model_trainer()
        hyperparameter_ranges = _create_single_hp_range()

        static_hps = HyperparameterTuner._prepare_static_hyperparameters(
            mock_trainer, hyperparameter_ranges
        )

        assert "batch_size" in static_hps
        assert "optimizer" in static_hps
        assert "learning_rate" not in static_hps

    def test_prepare_parameter_ranges_from_job_description(self):
        """Test _prepare_parameter_ranges_from_job_description method."""
        parameter_ranges = {
            "ContinuousParameterRanges": [
                {"Name": "learning_rate", "MinValue": "0.001", "MaxValue": "0.1"}
            ],
            "IntegerParameterRanges": [{"Name": "batch_size", "MinValue": "32", "MaxValue": "256"}],
            "CategoricalParameterRanges": [
                {"Name": "optimizer", "Values": ["sgd", "adam", "rmsprop"]}
            ],
        }

        ranges = HyperparameterTuner._prepare_parameter_ranges_from_job_description(
            parameter_ranges
        )

        assert "learning_rate" in ranges
        assert isinstance(ranges["learning_rate"], ContinuousParameter)
        assert "batch_size" in ranges
        assert isinstance(ranges["batch_size"], IntegerParameter)
        assert "optimizer" in ranges
        assert isinstance(ranges["optimizer"], CategoricalParameter)

    def test_extract_hyperparameters_from_parameter_ranges(self):
        """Test _extract_hyperparameters_from_parameter_ranges method."""
        parameter_ranges = {
            "ContinuousParameterRanges": [
                {"Name": "learning_rate", "MinValue": "0.001", "MaxValue": "0.1"}
            ],
            "IntegerParameterRanges": [{"Name": "batch_size", "MinValue": "32", "MaxValue": "256"}],
            "CategoricalParameterRanges": [],
        }

        hyperparameters = HyperparameterTuner._extract_hyperparameters_from_parameter_ranges(
            parameter_ranges
        )

        assert "learning_rate" in hyperparameters
        assert "batch_size" in hyperparameters

    def test_prepare_parameter_ranges_for_tuning(self):
        """Test _prepare_parameter_ranges_for_tuning method."""
        parameter_ranges = _create_hyperparameter_ranges()

        processed_ranges = HyperparameterTuner._prepare_parameter_ranges_for_tuning(
            parameter_ranges
        )

        assert "ContinuousParameterRanges" in processed_ranges
        assert "IntegerParameterRanges" in processed_ranges
        assert "CategoricalParameterRanges" in processed_ranges
        assert len(processed_ranges["ContinuousParameterRanges"]) == 1
        assert len(processed_ranges["IntegerParameterRanges"]) == 1
        assert len(processed_ranges["CategoricalParameterRanges"]) == 1

    def test_build_training_job_definition_includes_internal_channels(self):
        """Test that _build_training_job_definition includes ModelTrainer's internal channels.

        This test verifies the fix for GitHub issue #5508 where tuning jobs were missing
        internal channels (code, sm_drivers) that ModelTrainer creates for custom training.
        """
        from sagemaker.core.training.configs import InputData

        # Create mock ModelTrainer with internal channels (code, sm_drivers)
        mock_trainer = _create_mock_model_trainer(with_internal_channels=True)

        # User-provided inputs
        user_inputs = [
            InputData(channel_name="train", data_source="s3://bucket/train"),
            InputData(channel_name="validation", data_source="s3://bucket/val"),
        ]

        tuner = HyperparameterTuner(
            model_trainer=mock_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_create_single_hp_range(),
        )

        # Build training job definition
        definition = tuner._build_training_job_definition(user_inputs)

        # Verify all channels are included
        channel_names = [ch.channel_name for ch in definition.input_data_config]
        assert "code" in channel_names, "Internal 'code' channel should be included"
        assert "sm_drivers" in channel_names, "Internal 'sm_drivers' channel should be included"
        assert "train" in channel_names, "User 'train' channel should be included"
        assert "validation" in channel_names, "User 'validation' channel should be included"
        assert len(channel_names) == 4, "Should have exactly 4 channels"
