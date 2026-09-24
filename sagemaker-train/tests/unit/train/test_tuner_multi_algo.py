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
"""Tests for multi-algorithm HPO tuning and compression_type passthrough."""
from __future__ import absolute_import

import pytest
from unittest.mock import MagicMock, patch

from sagemaker.train.tuner import HyperparameterTuner
from sagemaker.core.parameter import ContinuousParameter, IntegerParameter
from sagemaker.core.training.configs import InputData
from sagemaker.core.shapes import (
    Channel,
    DataSource,
    S3DataSource,
    OutputDataConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_model_trainer(**overrides):
    """Create a mock ModelTrainer with realistic attributes."""
    trainer = MagicMock()
    trainer.sagemaker_session = MagicMock()
    trainer.hyperparameters = {"learning_rate": "0.1", "epochs": "10"}
    trainer.training_image = "123456789.dkr.ecr.us-west-2.amazonaws.com/training:latest"
    trainer.training_input_mode = "File"
    trainer.role = "arn:aws:iam::123456789012:role/SageMakerRole"
    trainer.output_data_config = MagicMock()
    trainer.output_data_config.s3_output_path = "s3://bucket/output"
    trainer.output_data_config.compression_type = None
    trainer.compute = MagicMock()
    trainer.compute.instance_type = "ml.m5.xlarge"
    trainer.compute.instance_count = 1
    trainer.compute.volume_size_in_gb = 30
    trainer.compute.enable_managed_spot_training = None
    trainer.stopping_condition = MagicMock()
    trainer.stopping_condition.max_runtime_in_seconds = 3600
    trainer.stopping_condition.max_wait_time_in_seconds = None
    trainer.environment = {"ENV_VAR": "value"}
    trainer.vpc_config = None
    trainer.input_data_config = None
    trainer._tuner_channels = None

    for key, value in overrides.items():
        setattr(trainer, key, value)
    return trainer


def _hp_ranges():
    return {"learning_rate": ContinuousParameter(0.001, 0.1)}


def _create_channel(name, uri):
    return Channel(
        channel_name=name,
        data_source=DataSource(
            s3_data_source=S3DataSource(
                s3_data_type="S3Prefix",
                s3_uri=uri,
                s3_data_distribution_type="FullyReplicated",
            )
        ),
    )


def _create_multi_algo_tuner(trainer1=None, trainer2=None, **tuner_kwargs):
    """Create a multi-algo tuner via HyperparameterTuner.create().

    Calls _prepare_static_hyperparameters_for_tuning() so that
    static_hyperparameters_dict is set (as it would be during tune()).
    """
    t1 = trainer1 or _mock_model_trainer()
    t2 = trainer2 or _mock_model_trainer(
        training_image="987654321.dkr.ecr.us-west-2.amazonaws.com/other:latest"
    )

    defaults = dict(
        model_trainer_dict={"xgboost": t1, "lightgbm": t2},
        objective_metric_name_dict={"xgboost": "auc", "lightgbm": "auc"},
        hyperparameter_ranges_dict={
            "xgboost": {"learning_rate": ContinuousParameter(0.001, 0.1)},
            "lightgbm": {"learning_rate": ContinuousParameter(0.01, 0.5)},
        },
        objective_type="Maximize",
        strategy="Bayesian",
        max_jobs=10,
        max_parallel_jobs=2,
    )
    defaults.update(tuner_kwargs)
    tuner = HyperparameterTuner.create(**defaults)
    tuner._prepare_static_hyperparameters_for_tuning()
    return tuner


# ---------------------------------------------------------------------------
# Tests: _start_tuning_job branching (single vs multi-algo)
# ---------------------------------------------------------------------------


class TestStartTuningJobBranching:
    """Test that _start_tuning_job routes to the correct builder."""

    @patch("sagemaker.train.tuner.HyperParameterTuningJob")
    def test_single_algo_calls_build_training_job_definition(self, mock_tuning_job_class):
        """Single-algo tuner should call _build_training_job_definition (singular)."""
        mock_tuning_job_class.create.return_value = MagicMock()
        trainer = _mock_model_trainer()

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
            max_jobs=5,
            max_parallel_jobs=2,
        )
        tuner._current_job_name = "test-single-algo"

        with (
            patch.object(tuner, "_build_training_job_definition") as mock_single,
            patch.object(tuner, "_build_training_job_definitions") as mock_multi,
        ):
            mock_single.return_value = MagicMock()
            tuner._start_tuning_job(inputs=None)

            mock_single.assert_called_once_with(None)
            mock_multi.assert_not_called()

    @patch("sagemaker.train.tuner.HyperParameterTuningJob")
    def test_multi_algo_calls_build_training_job_definitions(self, mock_tuning_job_class):
        """Multi-algo tuner should call _build_training_job_definitions (plural)."""
        mock_tuning_job_class.create.return_value = MagicMock()
        tuner = _create_multi_algo_tuner()
        tuner._current_job_name = "test-multi-algo"

        with (
            patch.object(tuner, "_build_training_job_definition") as mock_single,
            patch.object(tuner, "_build_training_job_definitions") as mock_multi,
        ):
            mock_multi.return_value = [MagicMock()]
            inputs = {"xgboost": "s3://bucket/xgb", "lightgbm": "s3://bucket/lgbm"}
            tuner._start_tuning_job(inputs=inputs)

            mock_multi.assert_called_once_with(inputs)
            mock_single.assert_not_called()

    @patch("sagemaker.train.tuner.HyperParameterTuningJob")
    def test_multi_algo_passes_training_job_definitions_in_request(self, mock_tuning_job_class):
        """Multi-algo should pass training_job_definitions (plural) to create()."""
        mock_tuning_job_class.create.return_value = MagicMock()
        tuner = _create_multi_algo_tuner()
        tuner._current_job_name = "test-multi-request"

        mock_definitions = [MagicMock(), MagicMock()]
        with patch.object(tuner, "_build_training_job_definitions", return_value=mock_definitions):
            tuner._start_tuning_job(inputs=None)

        call_kwargs = mock_tuning_job_class.create.call_args[1]
        assert call_kwargs["training_job_definition"] is None
        assert call_kwargs["training_job_definitions"] == mock_definitions

    @patch("sagemaker.train.tuner.HyperParameterTuningJob")
    def test_single_algo_passes_training_job_definition_in_request(self, mock_tuning_job_class):
        """Single-algo should pass training_job_definition (singular) to create()."""
        mock_tuning_job_class.create.return_value = MagicMock()
        trainer = _mock_model_trainer()

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
            max_jobs=5,
            max_parallel_jobs=2,
        )
        tuner._current_job_name = "test-single-request"

        mock_definition = MagicMock()
        with patch.object(tuner, "_build_training_job_definition", return_value=mock_definition):
            tuner._start_tuning_job(inputs=None)

        call_kwargs = mock_tuning_job_class.create.call_args[1]
        assert call_kwargs["training_job_definition"] == mock_definition
        assert call_kwargs["training_job_definitions"] is None


# ---------------------------------------------------------------------------
# Tests: _build_training_job_definitions (multi-algo)
# ---------------------------------------------------------------------------


class TestBuildTrainingJobDefinitions:
    """Test the new _build_training_job_definitions method for multi-algo tuning."""

    def test_returns_one_definition_per_trainer(self):
        """Should return a list with one definition per trainer in model_trainer_dict."""
        tuner = _create_multi_algo_tuner()
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)

        assert len(definitions) == 2

    def test_definition_names_match_trainer_keys(self):
        """Each definition should have definition_name matching its dict key."""
        tuner = _create_multi_algo_tuner()
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        definition_names = {d.definition_name for d in definitions}

        assert definition_names == {"xgboost", "lightgbm"}

    def test_training_images_from_each_trainer(self):
        """Each definition should use the training_image from its respective trainer."""
        t1 = _mock_model_trainer(training_image="image-xgb:latest")
        t2 = _mock_model_trainer(training_image="image-lgbm:latest")
        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        images = {d.definition_name: d.algorithm_specification.training_image for d in definitions}

        assert images["xgboost"] == "image-xgb:latest"
        assert images["lightgbm"] == "image-lgbm:latest"

    def test_per_trainer_objective(self):
        """Each definition should have its own tuning objective from the dict."""
        tuner = _create_multi_algo_tuner()
        tuner.objective_metric_name_dict = {
            "xgboost": "auc-roc",
            "lightgbm": "f1-score",
        }
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        objectives = {d.definition_name: d.tuning_objective.metric_name for d in definitions}

        assert objectives["xgboost"] == "auc-roc"
        assert objectives["lightgbm"] == "f1-score"

    def test_per_trainer_hyperparameter_ranges(self):
        """Each definition should have its own hyperparameter_ranges."""
        tuner = _create_multi_algo_tuner()
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)

        for defn in definitions:
            assert defn.hyper_parameter_ranges is not None

    def test_static_hyperparameters_included(self):
        """Static hyperparameters should be passed through for each trainer."""
        t1 = _mock_model_trainer(hyperparameters={"lr": "0.1", "epochs": "50"})
        t2 = _mock_model_trainer(hyperparameters={"lr": "0.2", "num_leaves": "31"})
        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner.static_hyperparameters_dict = {
            "xgboost": {"epochs": "50"},
            "lightgbm": {"num_leaves": "31"},
        }
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        statics = {d.definition_name: d.static_hyper_parameters for d in definitions}

        assert statics["xgboost"]["epochs"] == "50"
        assert statics["lightgbm"]["num_leaves"] == "31"

    def test_resource_config_from_each_trainer(self):
        """Resource config should be derived from each trainer's compute settings."""
        t1 = _mock_model_trainer()
        t1.compute.instance_type = "ml.p3.2xlarge"
        t1.compute.instance_count = 2
        t1.compute.volume_size_in_gb = 50

        t2 = _mock_model_trainer()
        t2.compute.instance_type = "ml.m5.xlarge"
        t2.compute.instance_count = 1
        t2.compute.volume_size_in_gb = 30

        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        resources = {d.definition_name: d.resource_config for d in definitions}

        assert resources["xgboost"].instance_type == "ml.p3.2xlarge"
        assert resources["xgboost"].instance_count == 2
        assert resources["lightgbm"].instance_type == "ml.m5.xlarge"
        assert resources["lightgbm"].instance_count == 1

    def test_stopping_condition_from_each_trainer(self):
        """Stopping condition should come from each trainer."""
        t1 = _mock_model_trainer()
        t1.stopping_condition.max_runtime_in_seconds = 7200

        t2 = _mock_model_trainer()
        t2.stopping_condition.max_runtime_in_seconds = 3600

        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        stopping = {d.definition_name: d.stopping_condition for d in definitions}

        assert stopping["xgboost"].max_runtime_in_seconds == 7200
        assert stopping["lightgbm"].max_runtime_in_seconds == 3600

    def test_role_from_each_trainer(self):
        """Role ARN should come from each trainer."""
        t1 = _mock_model_trainer(role="arn:aws:iam::111:role/RoleA")
        t2 = _mock_model_trainer(role="arn:aws:iam::222:role/RoleB")

        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        roles = {d.definition_name: d.role_arn for d in definitions}

        assert roles["xgboost"] == "arn:aws:iam::111:role/RoleA"
        assert roles["lightgbm"] == "arn:aws:iam::222:role/RoleB"

    def test_input_data_config_with_string_inputs(self):
        """When inputs are string S3 URIs, should create Channel objects."""
        tuner = _create_multi_algo_tuner()
        tuner._current_job_name = "test-job"

        inputs = {
            "xgboost": "s3://bucket/xgb-data",
            "lightgbm": "s3://bucket/lgbm-data",
        }

        definitions = tuner._build_training_job_definitions(inputs=inputs)

        for defn in definitions:
            assert len(defn.input_data_config) >= 1
            channel_names = [c.channel_name for c in defn.input_data_config]
            assert "training" in channel_names

    def test_input_data_config_with_dict_inputs(self):
        """When inputs are dicts mapping channel names to URIs, should create Channels."""
        tuner = _create_multi_algo_tuner()
        tuner._current_job_name = "test-job"

        inputs = {
            "xgboost": {"train": "s3://bucket/train", "val": "s3://bucket/val"},
            "lightgbm": {"train": "s3://bucket/train", "val": "s3://bucket/val"},
        }

        definitions = tuner._build_training_job_definitions(inputs=inputs)

        for defn in definitions:
            channel_names = [c.channel_name for c in defn.input_data_config]
            assert "train" in channel_names
            assert "val" in channel_names

    def test_input_data_config_with_input_data_list(self):
        """When inputs are lists of InputData, should convert to Channel objects."""
        tuner = _create_multi_algo_tuner()
        tuner._current_job_name = "test-job"

        input_list = [
            InputData(channel_name="training", data_source="s3://bucket/train"),
            InputData(channel_name="validation", data_source="s3://bucket/val"),
        ]
        inputs = {"xgboost": input_list, "lightgbm": input_list}

        definitions = tuner._build_training_job_definitions(inputs=inputs)

        for defn in definitions:
            channel_names = [c.channel_name for c in defn.input_data_config]
            assert "training" in channel_names
            assert "validation" in channel_names

    def test_input_data_config_with_channel_list(self):
        """When inputs are lists of Channel objects, should pass them through."""
        tuner = _create_multi_algo_tuner()
        tuner._current_job_name = "test-job"

        channel_list = [
            _create_channel("training", "s3://bucket/train"),
            _create_channel("validation", "s3://bucket/val"),
        ]
        inputs = {"xgboost": channel_list, "lightgbm": channel_list}

        definitions = tuner._build_training_job_definitions(inputs=inputs)

        for defn in definitions:
            channel_names = [c.channel_name for c in defn.input_data_config]
            assert "training" in channel_names
            assert "validation" in channel_names

    def test_internal_channels_included(self):
        """Internal channels from model_trainer (code, sm_drivers) should be included."""
        internal_channels = [
            _create_channel("code", "s3://bucket/code"),
            _create_channel("sm_drivers", "s3://bucket/drivers"),
        ]
        t1 = _mock_model_trainer(input_data_config=internal_channels)
        t2 = _mock_model_trainer(input_data_config=internal_channels)

        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)

        for defn in definitions:
            channel_names = [c.channel_name for c in defn.input_data_config]
            assert "code" in channel_names
            assert "sm_drivers" in channel_names

    def test_tuner_channels_included(self):
        """Channels from _tuner_channels (set by _prepare_model_trainer_for_tuning) should be included."""
        tuner_channels = [
            _create_channel("sm_drivers", "s3://bucket/tuner-drivers"),
        ]
        t1 = _mock_model_trainer(_tuner_channels=tuner_channels)
        t2 = _mock_model_trainer(_tuner_channels=tuner_channels)

        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)

        for defn in definitions:
            channel_names = [c.channel_name for c in defn.input_data_config]
            assert "sm_drivers" in channel_names

    def test_no_duplicate_channels(self):
        """Internal channels should not duplicate user-provided channels."""
        user_code_channel = _create_channel("code", "s3://bucket/user-code")
        internal_channels = [_create_channel("code", "s3://bucket/internal-code")]

        t1 = _mock_model_trainer(input_data_config=internal_channels)
        tuner = _create_multi_algo_tuner(trainer1=t1)
        tuner._current_job_name = "test-job"

        inputs = {"xgboost": [user_code_channel], "lightgbm": None}
        definitions = tuner._build_training_job_definitions(inputs=inputs)

        xgb_def = next(d for d in definitions if d.definition_name == "xgboost")
        code_channels = [c for c in xgb_def.input_data_config if c.channel_name == "code"]
        assert len(code_channels) == 1, "Should not have duplicate 'code' channels"

    def test_metric_definitions_included(self):
        """Metric definitions should be set on the algorithm specification."""
        tuner = _create_multi_algo_tuner(
            metric_definitions_dict={
                "xgboost": [{"Name": "auc", "Regex": r"auc: ([0-9\.]+)"}],
                "lightgbm": [{"Name": "f1", "Regex": r"f1: ([0-9\.]+)"}],
            }
        )
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)

        xgb_def = next(d for d in definitions if d.definition_name == "xgboost")
        lgbm_def = next(d for d in definitions if d.definition_name == "lightgbm")

        assert len(xgb_def.algorithm_specification.metric_definitions) == 1
        assert len(lgbm_def.algorithm_specification.metric_definitions) == 1

    def test_vpc_config_from_trainer(self):
        """VPC config should be passed through when set on the trainer."""
        from sagemaker.core.shapes import VpcConfig

        mock_vpc = VpcConfig(
            security_group_ids=["sg-123"],
            subnets=["subnet-abc"],
        )
        networking = MagicMock()
        networking._to_vpc_config.return_value = mock_vpc

        t1 = _mock_model_trainer(networking=networking)
        tuner = _create_multi_algo_tuner(trainer1=t1)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        xgb_def = next(d for d in definitions if d.definition_name == "xgboost")

        assert xgb_def.vpc_config is not None
        assert xgb_def.vpc_config.security_group_ids == ["sg-123"]

    def test_environment_from_trainer(self):
        """Environment variables should be passed through from each trainer."""
        t1 = _mock_model_trainer(environment={"KEY1": "val1"})
        t2 = _mock_model_trainer(environment={"KEY2": "val2"})

        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        envs = {d.definition_name: d.environment for d in definitions}

        assert envs["xgboost"] == {"KEY1": "val1"}
        assert envs["lightgbm"] == {"KEY2": "val2"}

    def test_none_inputs_produces_none_input_config(self):
        """When inputs is None and no internal channels, input_data_config should be None."""
        t1 = _mock_model_trainer(input_data_config=None, _tuner_channels=None)
        tuner = _create_multi_algo_tuner(trainer1=t1)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)

        xgb_def = next(d for d in definitions if d.definition_name == "xgboost")
        assert xgb_def.input_data_config is None


# ---------------------------------------------------------------------------
# Tests: compression_type passthrough in OutputDataConfig
# ---------------------------------------------------------------------------


class TestCompressionTypePassthrough:
    """Test that compression_type is correctly passed through in OutputDataConfig."""

    def test_compression_type_none_single_algo(self):
        """compression_type='NONE' should appear in single-algo training job definition."""
        trainer = _mock_model_trainer()
        trainer.output_data_config = OutputDataConfig(
            s3_output_path="s3://bucket/output",
            compression_type="NONE",
        )

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)

        assert definition.output_data_config.compression_type == "NONE"

    def test_compression_type_gzip_single_algo(self):
        """compression_type='GZIP' should be passed through."""
        trainer = _mock_model_trainer()
        trainer.output_data_config = OutputDataConfig(
            s3_output_path="s3://bucket/output",
            compression_type="GZIP",
        )

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)

        assert definition.output_data_config.compression_type == "GZIP"

    def test_compression_type_omitted_when_not_set(self):
        """When compression_type is not set, it should not be passed."""
        trainer = _mock_model_trainer()
        trainer.output_data_config = MagicMock()
        trainer.output_data_config.s3_output_path = "s3://bucket/output"
        trainer.output_data_config.compression_type = MagicMock()

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)

        # MagicMock is not a string, so isinstance check should skip it
        # and compression_type should be None (not a MagicMock)
        assert not isinstance(definition.output_data_config.compression_type, MagicMock)

    def test_compression_type_none_multi_algo(self):
        """compression_type='NONE' should appear in multi-algo definitions."""
        t1 = _mock_model_trainer()
        t1.output_data_config = OutputDataConfig(
            s3_output_path="s3://bucket/xgb-output",
            compression_type="NONE",
        )
        t2 = _mock_model_trainer()
        t2.output_data_config = OutputDataConfig(
            s3_output_path="s3://bucket/lgbm-output",
            compression_type="NONE",
        )

        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)

        for defn in definitions:
            assert (
                defn.output_data_config.compression_type == "NONE"
            ), f"{defn.definition_name} should have compression_type='NONE'"

    def test_compression_type_mixed_multi_algo(self):
        """Different trainers can have different compression_type values."""
        t1 = _mock_model_trainer()
        t1.output_data_config = OutputDataConfig(
            s3_output_path="s3://bucket/xgb-output",
            compression_type="NONE",
        )
        t2 = _mock_model_trainer()
        t2.output_data_config = OutputDataConfig(
            s3_output_path="s3://bucket/lgbm-output",
            compression_type="GZIP",
        )

        tuner = _create_multi_algo_tuner(trainer1=t1, trainer2=t2)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        compression = {
            d.definition_name: d.output_data_config.compression_type for d in definitions
        }

        assert compression["xgboost"] == "NONE"
        assert compression["lightgbm"] == "GZIP"

    def test_compression_type_mock_not_leaked_multi_algo(self):
        """MagicMock compression_type should not leak into multi-algo definitions."""
        t1 = _mock_model_trainer()
        t1.output_data_config = MagicMock()
        t1.output_data_config.s3_output_path = "s3://bucket/output"
        t1.output_data_config.compression_type = MagicMock()

        tuner = _create_multi_algo_tuner(trainer1=t1)
        tuner._current_job_name = "test-job"

        definitions = tuner._build_training_job_definitions(inputs=None)
        xgb_def = next(d for d in definitions if d.definition_name == "xgboost")

        assert not isinstance(xgb_def.output_data_config.compression_type, MagicMock)
