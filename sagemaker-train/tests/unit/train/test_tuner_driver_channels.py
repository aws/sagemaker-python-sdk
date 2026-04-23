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
"""Unit tests for HyperparameterTuner driver/code channel building (PR #5634).

Tests cover:
- _prepare_model_trainer_for_tuning guard logic
- _build_driver_and_code_channels sm_drivers channel creation
- _build_training_job_definition picking up _tuner_channels
- Environment and VPC config passthrough in _build_training_job_definition
- sourcedir.tar.gz upload and sagemaker_submit_directory hyperparameter
- getattr fallback for static_hyperparameters
"""
from __future__ import absolute_import

import json
import os
import shutil
from tempfile import TemporaryDirectory

import pytest
from unittest.mock import MagicMock, patch

from sagemaker.train.tuner import HyperparameterTuner
from sagemaker.train.constants import SM_DRIVERS_LOCAL_PATH
from sagemaker.core.parameter import ContinuousParameter
from sagemaker.core.shapes import (
    Channel,
    DataSource,
    S3DataSource,
    VpcConfig,
    OutputDataConfig,
)
from sagemaker.core.utils.utils import Unassigned


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_channel(name, uri="s3://bucket/data"):
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


def _is_unassigned(val):
    """Check if a value is the Unassigned sentinel used by Pydantic shapes."""
    return isinstance(val, Unassigned)


def _mock_model_trainer(**overrides):
    """Create a mock ModelTrainer with sensible defaults."""
    trainer = MagicMock()
    trainer.sagemaker_session = MagicMock()
    trainer.sagemaker_session.default_bucket.return_value = "my-bucket"
    trainer.sagemaker_session.default_bucket_prefix = None
    trainer.sagemaker_session.boto_session.client.return_value = MagicMock()
    trainer.sagemaker_session.boto_region_name = "us-west-2"
    trainer.hyperparameters = {"learning_rate": 0.1}
    trainer.training_image = "test-image:latest"
    trainer.training_input_mode = "File"
    trainer.role = "arn:aws:iam::123456789012:role/SageMakerRole"
    trainer.output_data_config = OutputDataConfig(s3_output_path="s3://bucket/output")
    trainer.compute = MagicMock()
    trainer.compute.instance_type = "ml.m5.xlarge"
    trainer.compute.instance_count = 1
    trainer.compute.volume_size_in_gb = 30
    trainer.stopping_condition = MagicMock()
    trainer.stopping_condition.max_runtime_in_seconds = 3600
    trainer.input_data_config = None
    trainer.base_job_name = "test-tuning"
    trainer.distributed = None
    trainer.environment = None
    trainer.tags = []
    trainer.source_code = None
    trainer.configure_mock(**{"networking": None})
    for k, v in overrides.items():
        setattr(trainer, k, v)
    return trainer


def _make_source_code(entry_script="train.py", source_dir=None):
    sc = MagicMock()
    sc.entry_script = entry_script
    sc.source_dir = source_dir
    sc.model_dump.return_value = {"entry_script": entry_script, "source_dir": source_dir}
    sc.ignore_patterns = [".git"]
    return sc


def _hp_ranges():
    return {"learning_rate": ContinuousParameter(0.001, 0.1)}


# ---------------------------------------------------------------------------
# _prepare_model_trainer_for_tuning – guard logic
# ---------------------------------------------------------------------------

class TestPrepareModelTrainerForTuning:
    """Tests for the guard clauses in _prepare_model_trainer_for_tuning."""

    def test_skips_when_source_code_is_none(self):
        """Should return early when model_trainer.source_code is None."""
        trainer = _mock_model_trainer(source_code=None)
        HyperparameterTuner._prepare_model_trainer_for_tuning(trainer)
        trainer.create_input_data_channel.assert_not_called()

    def test_skips_when_entry_script_is_none(self):
        """Should return early when source_code exists but entry_script is None."""
        source_code = MagicMock()
        source_code.entry_script = None
        trainer = _mock_model_trainer(source_code=source_code)
        HyperparameterTuner._prepare_model_trainer_for_tuning(trainer)
        trainer.create_input_data_channel.assert_not_called()

    def test_skips_when_entry_script_is_mock(self):
        """Should return early when entry_script is a MagicMock (not a str).

        This guards against calling _build_driver_and_code_channels on
        MagicMock model trainers in multi-trainer tuning scenarios.
        """
        source_code = MagicMock()
        source_code.entry_script = MagicMock()  # not a string
        trainer = _mock_model_trainer(source_code=source_code)
        HyperparameterTuner._prepare_model_trainer_for_tuning(trainer)
        trainer.create_input_data_channel.assert_not_called()

    @patch.object(HyperparameterTuner, "_build_driver_and_code_channels")
    def test_calls_build_when_entry_script_is_string(self, mock_build):
        """Should call _build_driver_and_code_channels when entry_script is a real string."""
        source_code = MagicMock()
        source_code.entry_script = "train.py"
        trainer = _mock_model_trainer(source_code=source_code)
        HyperparameterTuner._prepare_model_trainer_for_tuning(trainer)
        mock_build.assert_called_once_with(trainer)


# ---------------------------------------------------------------------------
# _build_driver_and_code_channels
#
# The method does real file I/O (writes JSON, copies driver files, creates
# tarballs) so these tests use real temp directories via tmp_path.
# ---------------------------------------------------------------------------

class TestBuildDriverAndCodeChannels:
    """Tests for _build_driver_and_code_channels."""

    def _make_trainer(self, tmp_path, source_dir=None, distributed=None):
        """Build a mock trainer whose TemporaryDirectory points at a real path."""
        source_code = _make_source_code(source_dir=source_dir)
        trainer = _mock_model_trainer(
            source_code=source_code,
            distributed=distributed,
        )
        trainer.create_input_data_channel.return_value = _create_channel(
            "sm_drivers", "s3://bucket/drivers"
        )
        trainer.hyperparameters = {"learning_rate": 0.1}
        return trainer

    def test_creates_sm_drivers_channel_and_sets_hp(self, tmp_path):
        """Should create sm_drivers channel and set sagemaker_program HP."""
        trainer = self._make_trainer(tmp_path)

        HyperparameterTuner._build_driver_and_code_channels(trainer)

        # sm_drivers channel stored on trainer
        assert len(trainer._tuner_channels) == 1
        assert trainer._tuner_channels[0].channel_name == "sm_drivers"
        # sagemaker_program HP set
        assert trainer.hyperparameters["sagemaker_program"] == "train.py"
        # create_input_data_channel called with sm_drivers
        trainer.create_input_data_channel.assert_called_once()
        args, kwargs = trainer.create_input_data_channel.call_args
        assert kwargs.get("channel_name") == "sm_drivers" or (args and args[0] == "sm_drivers")
        # _prepare_train_script called
        trainer._prepare_train_script.assert_called_once()

    def test_copies_distributed_drivers(self, tmp_path):
        """Should copy distributed drivers when model_trainer.distributed is set."""
        # Create a real driver_dir with a file so copytree has something to copy
        driver_dir = str(tmp_path / "torchrun_drivers")
        os.makedirs(driver_dir)
        with open(os.path.join(driver_dir, "torchrun_driver.py"), "w") as f:
            f.write("# driver")

        distributed = MagicMock()
        distributed.driver_dir = driver_dir
        distributed.model_dump.return_value = {"type": "Torchrun"}

        trainer = self._make_trainer(tmp_path, distributed=distributed)
        trainer.hyperparameters = {}

        HyperparameterTuner._build_driver_and_code_channels(trainer)

        # _prepare_train_script should have received the distributed config
        call_kwargs = trainer._prepare_train_script.call_args[1]
        assert call_kwargs["distributed"] is distributed

    def test_writes_sourcecode_and_distributed_json(self, tmp_path):
        """Should write sourcecode.json and distributed.json to the temp dir."""
        trainer = self._make_trainer(tmp_path)
        trainer.hyperparameters = {}

        HyperparameterTuner._build_driver_and_code_channels(trainer)

        # The temp dir is stored on the trainer
        temp_dir_path = trainer._tuner_temp_dir.name
        sc_path = os.path.join(temp_dir_path, "sourcecode.json")
        dist_path = os.path.join(temp_dir_path, "distributed.json")
        assert os.path.exists(sc_path)
        assert os.path.exists(dist_path)

        with open(sc_path) as f:
            sc_data = json.load(f)
        assert sc_data["entry_script"] == "train.py"

        with open(dist_path) as f:
            dist_data = json.load(f)
        # distributed is None → empty dict
        assert dist_data == {}

    def test_initializes_hyperparameters_when_none(self, tmp_path):
        """Should initialize hyperparameters dict when it's None."""
        trainer = self._make_trainer(tmp_path)
        trainer.hyperparameters = None

        HyperparameterTuner._build_driver_and_code_channels(trainer)

        assert trainer.hyperparameters is not None
        assert trainer.hyperparameters["sagemaker_program"] == "train.py"

    def test_uploads_sourcedir_tar_gz(self, tmp_path):
        """Should create and upload sourcedir.tar.gz when source_dir is a local path."""
        src_dir = str(tmp_path / "src")
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, "train.py"), "w") as f:
            f.write("print('hello')")

        mock_s3_client = MagicMock()
        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = None
        mock_session.boto_session.client.return_value = mock_s3_client
        mock_session.boto_region_name = "us-west-2"

        source_code = _make_source_code(source_dir=src_dir)
        trainer = _mock_model_trainer(source_code=source_code)
        trainer.sagemaker_session = mock_session
        trainer.create_input_data_channel.return_value = _create_channel("sm_drivers")
        trainer.hyperparameters = {}

        HyperparameterTuner._build_driver_and_code_channels(trainer)

        # S3 upload called
        mock_s3_client.upload_file.assert_called_once()
        upload_args = mock_s3_client.upload_file.call_args[0]
        assert upload_args[1] == "my-bucket"
        assert "sourcedir.tar.gz" in upload_args[2]
        # sagemaker_submit_directory set
        assert trainer.hyperparameters["sagemaker_submit_directory"].startswith("s3://my-bucket/")

    def test_sets_submit_directory_for_s3_source_dir(self, tmp_path):
        """Should set sagemaker_submit_directory directly when source_dir is an S3 URI."""
        source_code = _make_source_code(source_dir="s3://my-bucket/code/sourcedir.tar.gz")
        trainer = _mock_model_trainer(source_code=source_code)
        trainer.create_input_data_channel.return_value = _create_channel("sm_drivers")
        trainer.hyperparameters = {}

        HyperparameterTuner._build_driver_and_code_channels(trainer)

        assert trainer.hyperparameters["sagemaker_submit_directory"] == (
            "s3://my-bucket/code/sourcedir.tar.gz"
        )

    def test_stores_temp_dir_reference(self, tmp_path):
        """Should store temp dir reference on model_trainer to prevent premature cleanup."""
        trainer = self._make_trainer(tmp_path)
        trainer.hyperparameters = {}

        HyperparameterTuner._build_driver_and_code_channels(trainer)

        assert hasattr(trainer, "_tuner_temp_dir")
        assert trainer._tuner_temp_dir is not None
        # Should be a TemporaryDirectory instance
        assert hasattr(trainer._tuner_temp_dir, "name")


# ---------------------------------------------------------------------------
# _build_training_job_definition – _tuner_channels inclusion
# ---------------------------------------------------------------------------

class TestBuildTrainingJobDefinitionTunerChannels:
    """Tests for _tuner_channels being picked up by _build_training_job_definition."""

    def test_includes_tuner_channels(self):
        """_tuner_channels should appear in the definition's input_data_config."""
        trainer = _mock_model_trainer()
        trainer._tuner_channels = [
            _create_channel("sm_drivers", "s3://bucket/drivers"),
        ]

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        channel_names = [ch.channel_name for ch in definition.input_data_config]
        assert "sm_drivers" in channel_names

    def test_tuner_channels_no_duplicates(self):
        """Should not duplicate a channel already present from input_data_config."""
        trainer = _mock_model_trainer()
        trainer.input_data_config = [
            _create_channel("sm_drivers", "s3://bucket/existing"),
        ]
        trainer._tuner_channels = [
            _create_channel("sm_drivers", "s3://bucket/new"),
        ]

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        sm_channels = [c for c in definition.input_data_config if c.channel_name == "sm_drivers"]
        assert len(sm_channels) == 1

    def test_no_tuner_channels_still_works(self):
        """Definition should build fine when _tuner_channels is not set."""
        trainer = _mock_model_trainer()

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        assert definition is not None

    def test_tuner_channels_with_user_inputs(self):
        """_tuner_channels should coexist with user-provided input channels."""
        from sagemaker.core.training.configs import InputData

        trainer = _mock_model_trainer()
        trainer._tuner_channels = [
            _create_channel("sm_drivers", "s3://bucket/drivers"),
        ]

        user_inputs = [
            InputData(channel_name="train", data_source="s3://bucket/train"),
        ]

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(user_inputs)
        channel_names = [ch.channel_name for ch in definition.input_data_config]
        assert "train" in channel_names
        assert "sm_drivers" in channel_names


# ---------------------------------------------------------------------------
# Environment and VPC passthrough in _build_training_job_definition
# ---------------------------------------------------------------------------

class TestBuildTrainingJobDefinitionPassthrough:
    """Tests for environment and VPC config passthrough."""

    def test_passes_environment_variables(self):
        """Should set definition.environment from model_trainer.environment."""
        trainer = _mock_model_trainer(environment={"MY_VAR": "value", "OTHER": "123"})

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        assert definition.environment == {"MY_VAR": "value", "OTHER": "123"}

    def test_passes_empty_environment(self):
        """Should pass through empty dict environment as-is.

        An empty dict is valid for the SageMaker API, so we pass it through
        rather than silently converting it to None/Unassigned.
        """
        trainer = _mock_model_trainer(environment={})

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        assert definition.environment == {}, (
            "Empty dict environment should be passed through as-is"
        )

    def test_skips_environment_when_none(self):
        """Should not set environment when model_trainer.environment is None.

        When environment is None, it is not passed to the Pydantic constructor,
        so the field stays as Unassigned (excluded from serialization).
        """
        trainer = _mock_model_trainer(environment=None)

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        assert _is_unassigned(definition.environment), (
            "Environment should be Unassigned when model_trainer.environment is None"
        )

    def test_skips_environment_when_not_dict(self):
        """Should not set environment when it's not a dict (e.g. MagicMock).

        Non-dict values are not passed to the Pydantic constructor to avoid
        validation errors. The field stays as Unassigned.
        """
        trainer = _mock_model_trainer(environment=MagicMock())

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        assert _is_unassigned(definition.environment), (
            "Environment should be Unassigned when model_trainer.environment is not a dict"
        )

    def test_passes_vpc_config(self):
        """Should set definition.vpc_config from model_trainer.networking._to_vpc_config()."""
        real_vpc = VpcConfig(
            security_group_ids=["sg-123"],
            subnets=["subnet-abc"],
        )
        networking = MagicMock()
        networking._to_vpc_config.return_value = real_vpc

        trainer = _mock_model_trainer()
        trainer.networking = networking

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        assert definition.vpc_config == real_vpc
        assert definition.vpc_config.security_group_ids == ["sg-123"]
        assert definition.vpc_config.subnets == ["subnet-abc"]

    def test_skips_vpc_when_networking_none(self):
        """Should not set vpc_config when networking is None."""
        trainer = _mock_model_trainer()
        trainer.networking = None

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        assert _is_unassigned(definition.vpc_config)

    def test_vpc_config_exception_swallowed(self):
        """Should not raise when _to_vpc_config() throws an exception."""
        networking = MagicMock()
        networking._to_vpc_config.side_effect = RuntimeError("mock error")

        trainer = _mock_model_trainer()
        trainer.networking = networking

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        # Should not raise
        definition = tuner._build_training_job_definition(inputs=None)
        assert definition is not None

    def test_skips_vpc_when_to_vpc_config_returns_none(self):
        """Should not set vpc_config when _to_vpc_config() returns None."""
        networking = MagicMock()
        networking._to_vpc_config.return_value = None

        trainer = _mock_model_trainer()
        trainer.networking = networking

        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )

        definition = tuner._build_training_job_definition(inputs=None)
        assert _is_unassigned(definition.vpc_config)


# ---------------------------------------------------------------------------
# static_hyperparameters getattr fallback
# ---------------------------------------------------------------------------

class TestStaticHyperparametersGetattr:
    """Test that _build_training_job_definition uses getattr for static_hyperparameters."""

    def test_uses_static_hyperparameters_when_set(self):
        trainer = _mock_model_trainer()
        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )
        tuner.static_hyperparameters = {"batch_size": "32"}

        definition = tuner._build_training_job_definition(inputs=None)
        assert definition.static_hyper_parameters == {"batch_size": "32"}

    def test_falls_back_to_empty_dict_when_none(self):
        trainer = _mock_model_trainer()
        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )
        tuner.static_hyperparameters = None

        definition = tuner._build_training_job_definition(inputs=None)
        assert definition.static_hyper_parameters == {}

    def test_falls_back_when_attribute_deleted(self):
        trainer = _mock_model_trainer()
        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name="accuracy",
            hyperparameter_ranges=_hp_ranges(),
        )
        if hasattr(tuner, "static_hyperparameters"):
            delattr(tuner, "static_hyperparameters")

        definition = tuner._build_training_job_definition(inputs=None)
        assert definition.static_hyper_parameters == {}
