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
from unittest.mock import MagicMock, patch, PropertyMock

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
from sagemaker.core.shapes import HyperParameterTuningJobWarmStartConfig


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
        trainer = MagicMock()
        trainer.sagemaker_session = MagicMock()
        trainer.hyperparameters = {"learning_rate": 0.1}
        return trainer

    @pytest.fixture
    def hyperparameter_ranges(self):
        """Create sample hyperparameter ranges."""
        return {
            "learning_rate": ContinuousParameter(0.001, 0.1),
            "batch_size": IntegerParameter(32, 256),
            "optimizer": CategoricalParameter(["sgd", "adam"]),
        }

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
        mock_trainer = MagicMock()
        mock_trainer.sagemaker_session = MagicMock()
        return HyperparameterTuner(
            model_trainer=mock_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges={
                "learning_rate": ContinuousParameter(0.001, 0.1),
            },
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
        mock_trainer = MagicMock()
        mock_trainer.sagemaker_session = MagicMock()
        
        tuner = HyperparameterTuner(
            model_trainer=mock_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges={
                "learning_rate": ContinuousParameter(0.001, 0.1),
            },
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
        mock_trainer = MagicMock()
        mock_trainer.sagemaker_session = MagicMock()
        tuner = HyperparameterTuner(
            model_trainer=mock_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges={
                "learning_rate": ContinuousParameter(0.001, 0.1),
            },
        )
        tuner.latest_tuning_job = MagicMock()
        tuner._current_job_name = "test-tuning-job"
        return tuner

    def test_ensure_last_tuning_job_raises_error_when_none(self):
        """Test _ensure_last_tuning_job raises error when no job exists."""
        mock_trainer = MagicMock()
        tuner = HyperparameterTuner(
            model_trainer=mock_trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges={
                "learning_rate": ContinuousParameter(0.001, 0.1),
            },
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
            assert call_args[0][0] == tuner_with_job.latest_tuning_job.hyper_parameter_tuning_job_name


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
        HyperparameterTuner._validate_dict_argument(
            "test_arg", None, ["key1", "key2"]
        )

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
        mock_trainer = MagicMock()
        mock_trainer.hyperparameters = {
            "learning_rate": 0.1,
            "batch_size": 32,
            "optimizer": "adam",
        }

        hyperparameter_ranges = {
            "learning_rate": ContinuousParameter(0.001, 0.1),
        }

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
            "IntegerParameterRanges": [
                {"Name": "batch_size", "MinValue": "32", "MaxValue": "256"}
            ],
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
            "IntegerParameterRanges": [
                {"Name": "batch_size", "MinValue": "32", "MaxValue": "256"}
            ],
            "CategoricalParameterRanges": [],
        }

        hyperparameters = HyperparameterTuner._extract_hyperparameters_from_parameter_ranges(
            parameter_ranges
        )

        assert "learning_rate" in hyperparameters
        assert "batch_size" in hyperparameters

    def test_prepare_parameter_ranges_for_tuning(self):
        """Test _prepare_parameter_ranges_for_tuning method."""
        parameter_ranges = {
            "learning_rate": ContinuousParameter(0.001, 0.1),
            "batch_size": IntegerParameter(32, 256),
            "optimizer": CategoricalParameter(["sgd", "adam"]),
        }

        processed_ranges = HyperparameterTuner._prepare_parameter_ranges_for_tuning(
            parameter_ranges
        )

        assert "ContinuousParameterRanges" in processed_ranges
        assert "IntegerParameterRanges" in processed_ranges
        assert "CategoricalParameterRanges" in processed_ranges
        assert len(processed_ranges["ContinuousParameterRanges"]) == 1
        assert len(processed_ranges["IntegerParameterRanges"]) == 1
        assert len(processed_ranges["CategoricalParameterRanges"]) == 1


class TestUploadSourceCodeIgnorePatterns:
    """Test _upload_source_code_and_configure_hyperparameters respects ignore_patterns."""

    @pytest.fixture
    def mock_model_trainer_with_source(self, tmp_path):
        """Create a mock ModelTrainer with source_code configured."""
        # Create test source directory with files that should be included/ignored
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create files that should be included
        (source_dir / "train.py").write_text("# training script")
        (source_dir / "utils.py").write_text("# utils")

        # Create files that should be ignored
        (source_dir / ".env").write_text("SECRET=123")
        (source_dir / "model.pt").write_text("fake model weights")

        # Create directories that should be ignored
        pycache_dir = source_dir / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "train.cpython-310.pyc").write_text("bytecode")

        git_dir = source_dir / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")

        # Create subdirectory with mixed files
        sub_dir = source_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "helper.py").write_text("# helper")
        (sub_dir / "cache.pyc").write_text("bytecode")

        # Create mock model trainer
        trainer = MagicMock()
        trainer.sagemaker_session = MagicMock()
        trainer.sagemaker_session.boto_session = MagicMock()
        trainer.sagemaker_session.boto_session.client.return_value = MagicMock()
        trainer.sagemaker_session.boto_region_name = "us-west-2"
        trainer.sagemaker_session.default_bucket.return_value = "test-bucket"
        trainer.base_job_name = "test-job"
        trainer.hyperparameters = {}

        # Create source_code mock
        source_code = MagicMock()
        source_code.source_dir = str(source_dir)
        source_code.entry_script = "train.py"
        source_code.ignore_patterns = [".env", ".git", "__pycache__", "*.pt", "*.pyc"]
        trainer.source_code = source_code

        return trainer, source_dir

    def test_upload_source_code_respects_ignore_patterns(self, mock_model_trainer_with_source):
        """Test that files matching ignore_patterns are excluded from tarball."""
        import tarfile
        import tempfile

        trainer, source_dir = mock_model_trainer_with_source

        # Call the method
        HyperparameterTuner._upload_source_code_and_configure_hyperparameters(trainer)

        # Get the uploaded tarball path from the mock call
        s3_client = trainer.sagemaker_session.boto_session.client.return_value
        upload_call = s3_client.upload_file.call_args
        assert upload_call is not None

        # The tarball was already deleted, so we need to test by recreating the scenario
        # Let's verify the logic by checking what would have been uploaded
        # We do this by examining the method behavior directly

        # For a more thorough test, let's capture what files were added
        # by patching tarfile.open
        with patch("tarfile.open") as mock_tarfile:
            mock_tar = MagicMock()
            mock_tarfile.return_value.__enter__.return_value = mock_tar

            # Reset the mock to test again
            trainer.hyperparameters = {}

            HyperparameterTuner._upload_source_code_and_configure_hyperparameters(trainer)

            # Check which files were added to the tarball
            added_files = [call[0][0] for call in mock_tar.add.call_args_list]
            added_arcnames = [call[1]["arcname"] for call in mock_tar.add.call_args_list]

            # Should include these files
            assert any("train.py" in f for f in added_files)
            assert any("utils.py" in f for f in added_files)
            assert any("helper.py" in f for f in added_files)

            # Should NOT include these files (ignored patterns)
            assert not any(".env" in f for f in added_files)
            assert not any("model.pt" in f for f in added_files)
            assert not any("__pycache__" in f for f in added_files)
            assert not any(".git" in f for f in added_files)
            assert not any(".pyc" in f for f in added_files)

    def test_upload_source_code_ignores_directories(self, mock_model_trainer_with_source):
        """Test that directories matching ignore_patterns are completely skipped."""
        trainer, source_dir = mock_model_trainer_with_source

        with patch("tarfile.open") as mock_tarfile:
            mock_tar = MagicMock()
            mock_tarfile.return_value.__enter__.return_value = mock_tar

            HyperparameterTuner._upload_source_code_and_configure_hyperparameters(trainer)

            added_files = [call[0][0] for call in mock_tar.add.call_args_list]

            # No files from __pycache__ directory should be added
            assert not any("__pycache__" in f for f in added_files)
            # No files from .git directory should be added
            assert not any(".git" in f for f in added_files)

    def test_upload_source_code_with_empty_ignore_patterns(self, tmp_path):
        """Test backward compatibility with None/empty ignore_patterns."""
        # Create simple source directory
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "train.py").write_text("# training script")
        (source_dir / ".env").write_text("SECRET=123")

        # Create mock trainer with no ignore_patterns
        trainer = MagicMock()
        trainer.sagemaker_session = MagicMock()
        trainer.sagemaker_session.boto_session = MagicMock()
        trainer.sagemaker_session.boto_session.client.return_value = MagicMock()
        trainer.sagemaker_session.boto_region_name = "us-west-2"
        trainer.sagemaker_session.default_bucket.return_value = "test-bucket"
        trainer.base_job_name = "test-job"
        trainer.hyperparameters = {}

        source_code = MagicMock()
        source_code.source_dir = str(source_dir)
        source_code.entry_script = "train.py"
        source_code.ignore_patterns = None  # No ignore patterns
        trainer.source_code = source_code

        with patch("tarfile.open") as mock_tarfile:
            mock_tar = MagicMock()
            mock_tarfile.return_value.__enter__.return_value = mock_tar

            HyperparameterTuner._upload_source_code_and_configure_hyperparameters(trainer)

            added_files = [call[0][0] for call in mock_tar.add.call_args_list]

            # Both files should be added when no ignore_patterns
            assert any("train.py" in f for f in added_files)
            assert any(".env" in f for f in added_files)

    def test_upload_source_code_with_default_ignore_patterns(self, tmp_path):
        """Test that default ignore patterns from SourceCode class work."""
        # Create source directory with default-ignored files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "train.py").write_text("# training script")
        (source_dir / ".DS_Store").write_text("mac file")

        cache_dir = source_dir / ".cache"
        cache_dir.mkdir()
        (cache_dir / "data.json").write_text("{}")

        # Create mock trainer with default ignore_patterns
        trainer = MagicMock()
        trainer.sagemaker_session = MagicMock()
        trainer.sagemaker_session.boto_session = MagicMock()
        trainer.sagemaker_session.boto_session.client.return_value = MagicMock()
        trainer.sagemaker_session.boto_region_name = "us-west-2"
        trainer.sagemaker_session.default_bucket.return_value = "test-bucket"
        trainer.base_job_name = "test-job"
        trainer.hyperparameters = {}

        source_code = MagicMock()
        source_code.source_dir = str(source_dir)
        source_code.entry_script = "train.py"
        # Default patterns as defined in SourceCode class
        source_code.ignore_patterns = [
            ".env", ".git", "__pycache__", ".DS_Store", ".cache", ".ipynb_checkpoints"
        ]
        trainer.source_code = source_code

        with patch("tarfile.open") as mock_tarfile:
            mock_tar = MagicMock()
            mock_tarfile.return_value.__enter__.return_value = mock_tar

            HyperparameterTuner._upload_source_code_and_configure_hyperparameters(trainer)

            added_files = [call[0][0] for call in mock_tar.add.call_args_list]

            # train.py should be included
            assert any("train.py" in f for f in added_files)
            # .DS_Store should be ignored
            assert not any(".DS_Store" in f for f in added_files)
            # .cache directory should be ignored
            assert not any(".cache" in f for f in added_files)
