# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License.
"""Unit tests for BaseTrainer._train_serverful_smtj dataset channel injection.

These cover the fix that injects the resolved dataset channel mount paths into
the recipe override spec (so the rendered recipe's ``train_files`` /
``val_files`` are non-empty) and forwards the trainer ``environment`` to
``ModelTrainer.from_recipe``.
"""
from unittest.mock import patch, MagicMock

import pytest

from sagemaker.train.base_trainer import BaseTrainer


class _ConcreteTrainer(BaseTrainer):
    """Minimal concrete BaseTrainer for exercising _train_serverful_smtj."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = "nova-lite"
        self._customization_technique = "sft"
        self.training_type = "lora"
        self.training_dataset = None
        self.validation_dataset = None
        self.compute = MagicMock(
            instance_type="ml.p4d.24xlarge",
            instance_count=1,
            volume_size_in_gb=100,
            keep_alive_period_in_seconds=None,
        )
        self.networking = None
        self.stopping_condition = None
        self.training_image = "123.dkr.ecr.us-east-1.amazonaws.com/nova:latest"
        self.base_job_name = "nova-lite-sft"

    def train(self, *args, **kwargs):  # pragma: no cover - abstract impl
        return self._train_serverful_smtj(*args, **kwargs)


def _run_serverful(trainer):
    """Invoke _train_serverful_smtj with all external boundaries mocked.

    Returns a tuple of (override_spec captured at render time, from_recipe kwargs).
    """
    captured = {}

    def _capture_render(recipe_content, override_spec):
        captured["override_spec"] = override_spec
        return recipe_content

    mock_model_trainer = MagicMock()
    mock_model_trainer._latest_training_job = MagicMock()

    mock_session = MagicMock()
    # s3 download_file is a no-op; the temp recipe file stays empty.
    mock_session.boto_session.client.return_value.download_file.return_value = None

    with patch(
        "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session",
        return_value=mock_session,
    ), patch(
        "sagemaker.train.defaults.TrainDefaults.get_role", return_value="arn:aws:iam::1:role/x"
    ), patch(
        "sagemaker.train.common_utils.finetune_utils.get_recipe_s3_uri",
        return_value="s3://bucket/recipe.yaml",
    ), patch(
        "sagemaker.train.common_utils.finetune_utils.get_training_image",
        return_value="image:latest",
    ), patch(
        "sagemaker.train.common_utils.finetune_utils._validate_hyperparameter_values"
    ), patch(
        "sagemaker.train.common_utils.finetune_utils._get_smtj_override_spec",
        return_value={},
    ), patch(
        "sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders",
        side_effect=_capture_render,
    ), patch(
        "sagemaker.train.model_trainer.ModelTrainer.from_recipe",
        return_value=mock_model_trainer,
    ) as mock_from_recipe:
        trainer.hyperparameters = MagicMock()
        trainer.hyperparameters.to_dict.return_value = {}
        trainer.train(wait=False)

    return captured.get("override_spec"), mock_from_recipe.call_args.kwargs


class TestChannelMountInjection:
    """The resolved dataset paths must be injected into the recipe override spec."""

    def test_s3_prefix_maps_to_channel_directory(self):
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train/"
        trainer.validation_dataset = "s3://my-bucket/data/val/"

        override_spec, _ = _run_serverful(trainer)

        assert override_spec["data_path"]["default"] == "/opt/ml/input/data/train"
        assert (
            override_spec["validation_data_path"]["default"]
            == "/opt/ml/input/data/validation"
        )

    def test_s3_object_key_maps_to_mounted_file(self):
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train.jsonl"
        trainer.validation_dataset = "s3://my-bucket/data/val.jsonl"

        override_spec, _ = _run_serverful(trainer)

        assert (
            override_spec["data_path"]["default"]
            == "/opt/ml/input/data/train/train.jsonl"
        )
        assert (
            override_spec["validation_data_path"]["default"]
            == "/opt/ml/input/data/validation/val.jsonl"
        )

    def test_no_validation_dataset_leaves_validation_path_unset(self):
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train/"

        override_spec, _ = _run_serverful(trainer)

        assert override_spec["data_path"]["default"] == "/opt/ml/input/data/train"
        assert "validation_data_path" not in override_spec

    def test_existing_spec_entry_is_updated_in_place(self):
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train/"

        def _capture_render(recipe_content, override_spec):
            _capture_render.spec = override_spec
            return recipe_content

        # Pre-seed the override spec with an existing data_path entry so we can
        # assert the fix mutates it (preserving type) rather than replacing it.
        with patch(
            "sagemaker.train.common_utils.finetune_utils._get_smtj_override_spec",
            return_value={"data_path": {"default": "", "type": "string"}},
        ), patch(
            "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session",
            return_value=MagicMock(),
        ), patch(
            "sagemaker.train.defaults.TrainDefaults.get_role", return_value="role"
        ), patch(
            "sagemaker.train.common_utils.finetune_utils.get_recipe_s3_uri",
            return_value="s3://bucket/recipe.yaml",
        ), patch(
            "sagemaker.train.common_utils.finetune_utils.get_training_image",
            return_value="image:latest",
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._validate_hyperparameter_values"
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders",
            side_effect=_capture_render,
        ), patch(
            "sagemaker.train.model_trainer.ModelTrainer.from_recipe",
            return_value=MagicMock(),
        ):
            trainer.hyperparameters = MagicMock()
            trainer.hyperparameters.to_dict.return_value = {}
            trainer.train(wait=False)

        entry = _capture_render.spec["data_path"]
        assert entry == {"default": "/opt/ml/input/data/train", "type": "string"}


class TestEnvironmentForwarding:
    """The trainer environment must be forwarded to ModelTrainer.from_recipe."""

    def test_environment_passed_through(self):
        trainer = _ConcreteTrainer(environment={"FI_PROVIDER": "efa"})
        trainer.training_dataset = "s3://my-bucket/data/train/"

        _, from_recipe_kwargs = _run_serverful(trainer)

        assert from_recipe_kwargs["environment"] == {"FI_PROVIDER": "efa"}

    def test_empty_environment_forwarded_as_none(self):
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train/"

        _, from_recipe_kwargs = _run_serverful(trainer)

        # self.environment defaults to {} -> ``or None`` collapses to None.
        assert from_recipe_kwargs["environment"] is None
