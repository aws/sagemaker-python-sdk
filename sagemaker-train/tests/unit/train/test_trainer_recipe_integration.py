# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License.
"""Integration tests for get_resolved_recipe() on all trainer types."""
import os
import tempfile

import pytest
import yaml
from unittest.mock import patch, MagicMock, Mock


# --- Fixtures ---


@pytest.fixture
def recipe_file(tmp_path):
    """Create a temporary recipe YAML file for testing."""
    recipe_content = {
        "training_config": {
            "learning_rate": 2e-5,
            "batch_size": 8,
        }
    }
    recipe_path = tmp_path / "test_recipe.yaml"
    recipe_path.write_text(yaml.dump(recipe_content))
    return str(recipe_path)


@pytest.fixture
def mock_hyperparams():
    """Create a mock hyperparameters object with _specs and _get_recipe_template."""
    mock_hp = MagicMock()
    mock_hp._specs = {
        "learning_rate": {"default": 1e-5, "type": "float", "min": 1e-7, "max": 1.0},
        "num_epochs": {"default": 3, "type": "integer", "min": 1, "max": 100},
        "batch_size": {"default": 1, "type": "integer", "min": 1, "max": 64},
    }
    mock_hp._get_recipe_template = MagicMock(return_value=None)
    mock_hp.to_dict = MagicMock(return_value={"learning_rate": "1e-5", "num_epochs": "3"})
    return mock_hp


# --- SFTTrainer Tests ---


class TestSFTTrainerRecipeIntegration:
    """Tests for SFTTrainer.get_resolved_recipe()."""

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_sft_with_recipe_and_overrides(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        recipe_file, mock_hyperparams
    ):
        """SFTTrainer with recipe + overrides returns merged result."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            recipe=recipe_file,
            overrides={"training_config": {"num_epochs": 5}},
        )

        resolved = trainer.get_resolved_recipe()

        # Overrides take highest precedence
        assert resolved["training_config"]["num_epochs"] == 5
        # Recipe file values take precedence over defaults
        assert resolved["training_config"]["learning_rate"] == 2e-5
        assert resolved["training_config"]["batch_size"] == 8

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_sft_no_recipe_no_overrides_raises(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams
    ):
        """SFTTrainer with no recipe/overrides raises ValueError."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
        )

        with pytest.raises(ValueError, match="get_resolved_recipe\\(\\) requires"):
            trainer.get_resolved_recipe()

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_sft_direct_hyperparameter_assignment_resolves(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
    ):
        """SFTTrainer with direct hyperparameter assignment resolves recipe."""
        from sagemaker.train.common import FineTuningOptions

        hp = FineTuningOptions({
            "learning_rate": {"default": 1e-5, "type": "float", "min": 1e-7, "max": 1.0},
            "num_epochs": {"default": 3, "type": "integer", "min": 1, "max": 100},
            "batch_size": {"default": 1, "type": "integer", "min": 1, "max": 64},
        })
        mock_get_options.return_value = (hp, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
        )

        # Simulate direct assignment (the common user pattern)
        trainer.hyperparameters.learning_rate = 2e-5
        trainer.hyperparameters.num_epochs = 5

        resolved = trainer.get_resolved_recipe()

        assert resolved["training_config"]["learning_rate"] == 2e-5
        assert resolved["training_config"]["num_epochs"] == 5
        # Unset params keep their defaults
        assert resolved["training_config"]["batch_size"] == 1

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_sft_overrides_plus_direct_hyperparameter_assignment(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
    ):
        """SFTTrainer with overrides AND direct hyperparameter assignment merges both."""
        from sagemaker.train.common import FineTuningOptions

        hp = FineTuningOptions({
            "learning_rate": {"default": 1e-5, "type": "float", "min": 1e-7, "max": 1.0},
            "num_epochs": {"default": 3, "type": "integer", "min": 1, "max": 100},
            "max_steps": {"default": 100, "type": "integer", "min": 1, "max": 10000},
            "save_steps": {"default": 50, "type": "integer", "min": 1, "max": 10000},
        })
        mock_get_options.return_value = (hp, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            overrides={"training_config": {"learning_rate": 5e-6, "num_epochs": 2}},
        )

        # Direct assignment on top of overrides
        trainer.hyperparameters.max_steps = 5
        trainer.hyperparameters.save_steps = 5

        resolved = trainer.get_resolved_recipe()

        # Overrides should be present
        assert resolved["training_config"]["learning_rate"] == 5e-6
        assert resolved["training_config"]["num_epochs"] == 2
        # Direct hyperparameter assignments should also be present (layered on top)
        assert resolved["training_config"]["max_steps"] == 5
        assert resolved["training_config"]["save_steps"] == 5


# --- RLVRTrainer Tests ---


class TestRLVRTrainerRecipeIntegration:
    """Tests for RLVRTrainer.get_resolved_recipe()."""

    @patch("sagemaker.train.rlvr_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.rlvr_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.rlvr_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.rlvr_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_rlvr_with_recipe_and_overrides(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        recipe_file, mock_hyperparams
    ):
        """RLVRTrainer with recipe + overrides returns merged result."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.rlvr_trainer import RLVRTrainer

        trainer = RLVRTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            recipe=recipe_file,
            overrides={"training_config": {"num_epochs": 7}},
        )

        resolved = trainer.get_resolved_recipe()

        # Overrides take highest precedence
        assert resolved["training_config"]["num_epochs"] == 7
        # Recipe file values take precedence over defaults
        assert resolved["training_config"]["learning_rate"] == 2e-5
        assert resolved["training_config"]["batch_size"] == 8

    @patch("sagemaker.train.rlvr_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.rlvr_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.rlvr_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.rlvr_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_rlvr_no_recipe_no_overrides_raises(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams
    ):
        """RLVRTrainer with no recipe/overrides raises ValueError."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.rlvr_trainer import RLVRTrainer

        trainer = RLVRTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
        )

        with pytest.raises(ValueError, match="get_resolved_recipe\\(\\) requires"):
            trainer.get_resolved_recipe()


# --- DPOTrainer Tests ---


class TestDPOTrainerRecipeIntegration:
    """Tests for DPOTrainer.get_resolved_recipe()."""

    @patch("sagemaker.train.dpo_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.dpo_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_dpo_with_recipe_and_overrides(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        recipe_file, mock_hyperparams
    ):
        """DPOTrainer with recipe + overrides returns merged result."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.dpo_trainer import DPOTrainer

        trainer = DPOTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            recipe=recipe_file,
            overrides={"training_config": {"num_epochs": 10}},
        )

        resolved = trainer.get_resolved_recipe()

        # Overrides take highest precedence
        assert resolved["training_config"]["num_epochs"] == 10
        # Recipe file values take precedence over defaults
        assert resolved["training_config"]["learning_rate"] == 2e-5
        assert resolved["training_config"]["batch_size"] == 8

    @patch("sagemaker.train.dpo_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.dpo_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_dpo_no_recipe_no_overrides_raises(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams
    ):
        """DPOTrainer with no recipe/overrides raises ValueError."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.dpo_trainer import DPOTrainer

        trainer = DPOTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
        )

        with pytest.raises(ValueError, match="get_resolved_recipe\\(\\) requires"):
            trainer.get_resolved_recipe()


# --- BenchMarkEvaluator Tests ---


class TestBenchMarkEvaluatorRecipeIntegration:
    """Tests for BenchMarkEvaluator.get_resolved_recipe()."""

    def test_benchmark_evaluator_with_recipe_and_overrides(self, tmp_path):
        """BenchMarkEvaluator with recipe + overrides returns merged result."""
        # Create a recipe file with inference section
        recipe_content = {
            "inference": {
                "max_new_tokens": 2048,
                "temperature": 0.7,
            }
        }
        recipe_path = tmp_path / "eval_recipe.yaml"
        recipe_path.write_text(yaml.dump(recipe_content))

        # Mock session
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-east-1"

        # Mock the model resolution that happens in the validator
        mock_model_info = MagicMock()
        mock_model_info.base_model_name = "nova-pro-v2"
        mock_model_info.base_model_arn = "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/nova-pro-v2/1.0"
        mock_model_info.source_model_package_arn = None

        with patch(
            "sagemaker.train.common_utils.model_resolution._resolve_base_model",
            return_value=mock_model_info,
        ), patch(
            "sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn",
            return_value=None,
        ):
            from sagemaker.train.evaluate.benchmark_evaluator import BenchMarkEvaluator, _Benchmark

            evaluator = BenchMarkEvaluator(
                benchmark=_Benchmark.MMLU,
                model="nova-pro-v2",
                s3_output_path="s3://bucket/output",
                sagemaker_session=mock_session,
                recipe=str(recipe_path),
                overrides={"inference": {"max_new_tokens": 4096}},
            )

            # Manually set _hyperparameters to simulate what the property would return
            mock_hp = MagicMock()
            mock_hp._specs = {
                "max_new_tokens": {"default": 1024, "type": "integer", "min": 1, "max": 8192},
                "temperature": {"default": 1.0, "type": "float", "min": 0.0, "max": 2.0},
            }
            object.__setattr__(evaluator, '_hyperparameters', mock_hp)

            resolved = evaluator.get_resolved_recipe()

            # Overrides take highest precedence
            assert resolved["inference"]["max_new_tokens"] == 4096
            # Recipe file values take precedence over defaults
            assert resolved["inference"]["temperature"] == 0.7

    def test_benchmark_evaluator_no_recipe_no_overrides_raises(self):
        """BenchMarkEvaluator with no recipe/overrides raises ValueError."""
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-east-1"

        mock_model_info = MagicMock()
        mock_model_info.base_model_name = "nova-pro-v2"
        mock_model_info.base_model_arn = "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/nova-pro-v2/1.0"
        mock_model_info.source_model_package_arn = None

        with patch(
            "sagemaker.train.common_utils.model_resolution._resolve_base_model",
            return_value=mock_model_info,
        ), patch(
            "sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn",
            return_value=None,
        ):
            from sagemaker.train.evaluate.benchmark_evaluator import BenchMarkEvaluator, _Benchmark

            evaluator = BenchMarkEvaluator(
                benchmark=_Benchmark.MMLU,
                model="nova-pro-v2",
                s3_output_path="s3://bucket/output",
                sagemaker_session=mock_session,
            )

            with pytest.raises(ValueError, match="get_resolved_recipe\\(\\) requires"):
                evaluator.get_resolved_recipe()


# --- ModelTrainer (from_recipe) Tests ---


class TestModelTrainerRecipeIntegration:
    """Tests for ModelTrainer.get_resolved_recipe() via from_recipe."""

    def test_model_trainer_from_recipe_get_resolved(self, tmp_path):
        """ModelTrainer.from_recipe with recipe_overrides returns resolved result."""
        from omegaconf import OmegaConf
        from sagemaker.train.model_trainer import ModelTrainer
        from sagemaker.train.configs import Compute, SourceCode
        from sagemaker.core.helper.session_helper import Session

        # Create a temp recipe file
        recipe_content = {
            "trainer": {"num_nodes": 1},
            "model": {"name": "test-model", "hidden_size": 768},
            "run": {"results_dir": "/opt/ml/model", "name": "test-run"},
        }
        recipe_path = tmp_path / "training_recipe.yaml"
        recipe_path.write_text(yaml.dump(recipe_content))

        # Mock session - use spec=Session so Pydantic validates it
        mock_session = MagicMock(spec=Session)
        mock_session.boto_region_name = "us-east-1"
        mock_session.default_bucket_prefix = None
        mock_session.default_bucket.return_value = "sagemaker-us-east-1-123456789012"

        # Mock _get_args_from_recipe to return minimal args with a temp dir
        recipe_tmp_dir = tempfile.TemporaryDirectory(prefix="test_recipe_")
        source_dir = recipe_tmp_dir.name

        # The base recipe loaded during from_recipe
        base_recipe_cfg = OmegaConf.create(recipe_content)

        # The merged recipe returned when get_resolved_recipe calls _load_base_recipe
        merged_recipe_cfg = OmegaConf.create({
            "trainer": {"num_nodes": 1},
            "model": {"name": "test-model", "hidden_size": 768},
            "run": {"results_dir": "/opt/ml/output", "name": "test-run"},
        })

        recipe_overrides = {"run": {"results_dir": "/opt/ml/output"}}
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)

        with patch("sagemaker.train.model_trainer._determine_device_type", return_value="gpu"), \
             patch("sagemaker.train.model_trainer._load_base_recipe", return_value=base_recipe_cfg), \
             patch("sagemaker.train.model_trainer._is_nova_recipe", return_value=False), \
             patch("sagemaker.train.model_trainer._is_llmft_recipe", return_value=False), \
             patch("sagemaker.train.model_trainer._get_args_from_recipe", return_value=(
                 {
                     "source_code": SourceCode(source_dir=source_dir),
                     "training_image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/test:latest",
                     "compute": Compute(instance_type="ml.p5.48xlarge", instance_count=1),
                     "hyperparameters": {"config-path": ".", "config-name": "recipe.yaml"},
                 },
                 recipe_tmp_dir,
             )), \
             patch("sagemaker.train.model_trainer.TrainDefaults.get_sagemaker_session", return_value=mock_session), \
             patch("sagemaker.train.model_trainer.TrainDefaults.get_role", return_value="arn:aws:iam::123456789012:role/SageMakerRole"):

            model_trainer = ModelTrainer.from_recipe(
                training_recipe=str(recipe_path),
                compute=compute,
                recipe_overrides=recipe_overrides,
            )

        # Now call get_resolved_recipe - it imports _load_base_recipe locally
        with patch(
            "sagemaker.train.sm_recipes.utils._load_base_recipe",
            return_value=merged_recipe_cfg,
        ):
            resolved = model_trainer.get_resolved_recipe()

        assert resolved["run"]["results_dir"] == "/opt/ml/output"
        assert resolved["model"]["name"] == "test-model"
        assert resolved["trainer"]["num_nodes"] == 1

        # Clean up
        recipe_tmp_dir.cleanup()

    def test_model_trainer_no_from_recipe_raises(self):
        """ModelTrainer not created via from_recipe raises AttributeError."""
        from sagemaker.train.model_trainer import ModelTrainer
        from sagemaker.train.configs import Compute

        # Construction resolves an execution role via STS; mock it so this unit
        # test (which only exercises get_resolved_recipe) runs without AWS creds.
        with patch(
            "sagemaker.train.defaults.TrainDefaults.get_role",
            return_value="arn:aws:iam::123456789012:role/test-role",
        ):
            trainer = ModelTrainer(
                training_image="123456789012.dkr.ecr.us-east-1.amazonaws.com/test:latest",
                compute=Compute(instance_type="ml.p5.48xlarge", instance_count=1),
            )

        with pytest.raises(AttributeError, match="get_resolved_recipe\\(\\) is only available"):
            trainer.get_resolved_recipe()


# --- Full Recipe Template Tests ---


class TestFullRecipeTemplateResolution:
    """Tests that the full recipe template enables overriding non-spec keys."""

    @pytest.fixture
    def full_recipe_template(self):
        """A full recipe template simulating what SmtjRecipeTemplateS3Uri returns."""
        return {
            "training_config": {
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "batch_size": 32,
                "sequence_length": 4096,
                "warmup_ratio": 0.1,
                "gradient_accumulation_steps": 4,
                "weight_decay": 0.01,
                "lr_scheduler": {
                    "warmup_steps": 15,
                    "min_lr": 1e-6,
                },
            }
        }

    @pytest.fixture
    def mock_hyperparams_with_full_template(self, full_recipe_template):
        """Mock hyperparameters with _specs (subset) and _full_recipe_template."""
        mock_hp = MagicMock()
        mock_hp._specs = {
            "learning_rate": {"default": 1e-4, "type": "float", "min": 1e-7, "max": 1.0},
            "num_epochs": {"default": 10, "type": "integer", "min": 1, "max": 100},
            "batch_size": {"default": 32, "type": "integer", "min": 1, "max": 64},
        }
        mock_hp._full_recipe_template = full_recipe_template
        mock_hp.to_dict = MagicMock(return_value={
            "learning_rate": "0.0001", "num_epochs": "10", "batch_size": "32"
        })
        return mock_hp

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_override_non_spec_keys_with_full_template(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams_with_full_template
    ):
        """Overriding keys like sequence_length that are in full template but not in spec."""
        mock_get_options.return_value = (mock_hyperparams_with_full_template, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            overrides={
                "training_config": {
                    "sequence_length": 8192,
                    "warmup_ratio": 0.05,
                }
            },
        )

        resolved = trainer.get_resolved_recipe()

        # Non-spec keys from full template are present with overridden values
        assert resolved["training_config"]["sequence_length"] == 8192
        assert resolved["training_config"]["warmup_ratio"] == 0.05
        # Non-spec keys retain their full template defaults where not overridden
        assert resolved["training_config"]["gradient_accumulation_steps"] == 4
        assert resolved["training_config"]["weight_decay"] == 0.01
        # Spec keys retain their defaults
        assert resolved["training_config"]["learning_rate"] == 1e-4
        assert resolved["training_config"]["num_epochs"] == 10

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_full_template_with_recipe_file_and_overrides(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams_with_full_template, tmp_path
    ):
        """3-level merge with full template: full_template < recipe file < overrides."""
        mock_get_options.return_value = (mock_hyperparams_with_full_template, "model-arn", False)

        recipe_content = {
            "training_config": {
                "sequence_length": 2048,
                "learning_rate": 5e-5,
            }
        }
        recipe_path = tmp_path / "recipe.yaml"
        recipe_path.write_text(yaml.dump(recipe_content))

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            recipe=str(recipe_path),
            overrides={"training_config": {"learning_rate": 2e-5}},
        )

        resolved = trainer.get_resolved_recipe()

        # Overrides win
        assert resolved["training_config"]["learning_rate"] == 2e-5
        # Recipe file wins over full template defaults
        assert resolved["training_config"]["sequence_length"] == 2048
        # Full template defaults where nothing overrides
        assert resolved["training_config"]["warmup_ratio"] == 0.1
        assert resolved["training_config"]["gradient_accumulation_steps"] == 4

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_non_spec_keys_flow_into_train_hyperparameters(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams_with_full_template
    ):
        """Non-spec keys from full template are included in final training hyperparameters."""
        mock_get_options.return_value = (mock_hyperparams_with_full_template, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            training_dataset="s3://bucket/train.jsonl",
            overrides={"training_config": {"sequence_length": 8192}},
        )

        with patch("sagemaker.train.sft_trainer.TrainingJob") as mock_tj, \
             patch("sagemaker.train.sft_trainer.TrainDefaults") as mock_defaults, \
             patch("sagemaker.train.sft_trainer._create_input_data_config") as mock_input, \
             patch("sagemaker.train.sft_trainer._convert_input_data_to_channels", return_value=[]), \
             patch("sagemaker.train.sft_trainer._create_output_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_serverless_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_mlflow_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._create_model_package_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._validate_hyperparameter_values"), \
             patch("sagemaker.train.sft_trainer._get_studio_tags", return_value=[]):

            mock_session = MagicMock()
            mock_session.boto_session.region_name = "us-west-2"
            mock_defaults.get_sagemaker_session.return_value = mock_session
            mock_defaults.get_role.return_value = "arn:aws:iam::123:role/Role"
            mock_tj.create.return_value = MagicMock()

            trainer.train(wait=False)

            call_kwargs = mock_tj.create.call_args[1]
            final_hp = call_kwargs["hyper_parameters"]

            # Non-spec key is now in final hyperparameters
            assert "sequence_length" in final_hp
            assert final_hp["sequence_length"] == "8192"
            # Other full template keys also present
            assert "warmup_ratio" in final_hp
            assert final_hp["warmup_ratio"] == "0.1"

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_nested_keys_flow_into_train_hyperparameters(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams_with_full_template
    ):
        """Nested recipe keys (lr_scheduler.warmup_steps) are flattened into final hyperparameters."""
        mock_get_options.return_value = (mock_hyperparams_with_full_template, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            training_dataset="s3://bucket/train.jsonl",
            overrides={"training_config": {"lr_scheduler": {"warmup_steps": 30}}},
        )

        with patch("sagemaker.train.sft_trainer.TrainingJob") as mock_tj, \
             patch("sagemaker.train.sft_trainer.TrainDefaults") as mock_defaults, \
             patch("sagemaker.train.sft_trainer._create_input_data_config") as mock_input, \
             patch("sagemaker.train.sft_trainer._convert_input_data_to_channels", return_value=[]), \
             patch("sagemaker.train.sft_trainer._create_output_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_serverless_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_mlflow_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._create_model_package_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._validate_hyperparameter_values"), \
             patch("sagemaker.train.sft_trainer._get_studio_tags", return_value=[]):

            mock_session = MagicMock()
            mock_session.boto_session.region_name = "us-west-2"
            mock_defaults.get_sagemaker_session.return_value = mock_session
            mock_defaults.get_role.return_value = "arn:aws:iam::123:role/Role"
            mock_tj.create.return_value = MagicMock()

            trainer.train(wait=False)

            call_kwargs = mock_tj.create.call_args[1]
            final_hp = call_kwargs["hyper_parameters"]

            # Nested key overridden and flattened to leaf name
            assert "warmup_steps" in final_hp
            assert final_hp["warmup_steps"] == "30"
            # Other nested leaf retains default from full template
            assert "min_lr" in final_hp
            assert final_hp["min_lr"] == "1e-06"

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_deeply_nested_peft_keys_flow_into_hyperparameters(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
    ):
        """Deeply nested keys (peft.lora_tuning.alpha) flatten into hyperparameters."""
        mock_hp = MagicMock()
        mock_hp._specs = {
            "learning_rate": {"default": 1e-4, "type": "float", "min": 1e-7, "max": 1.0},
        }
        mock_hp._full_recipe_template = {
            "training_config": {
                "learning_rate": 1e-4,
                "peft": {
                    "peft_scheme": "lora",
                    "lora_tuning": {
                        "alpha": 64,
                        "rank": 16,
                    }
                },
            }
        }
        mock_hp.to_dict = MagicMock(return_value={"learning_rate": "0.0001"})
        mock_get_options.return_value = (mock_hp, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            training_dataset="s3://bucket/train.jsonl",
            overrides={"training_config": {"peft": {"lora_tuning": {"alpha": 128}}}},
        )

        with patch("sagemaker.train.sft_trainer.TrainingJob") as mock_tj, \
             patch("sagemaker.train.sft_trainer.TrainDefaults") as mock_defaults, \
             patch("sagemaker.train.sft_trainer._create_input_data_config"), \
             patch("sagemaker.train.sft_trainer._convert_input_data_to_channels", return_value=[]), \
             patch("sagemaker.train.sft_trainer._create_output_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_serverless_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_mlflow_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._create_model_package_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._validate_hyperparameter_values"), \
             patch("sagemaker.train.sft_trainer._get_studio_tags", return_value=[]):

            mock_session = MagicMock()
            mock_session.boto_session.region_name = "us-west-2"
            mock_defaults.get_sagemaker_session.return_value = mock_session
            mock_defaults.get_role.return_value = "arn:aws:iam::123:role/Role"
            mock_tj.create.return_value = MagicMock()

            trainer.train(wait=False)

            call_kwargs = mock_tj.create.call_args[1]
            final_hp = call_kwargs["hyper_parameters"]

            # Deeply nested override flattened
            assert final_hp["alpha"] == "128"
            # Unchanged deeply nested default preserved
            assert final_hp["rank"] == "16"
            # String-only leaf that's deeply nested
            assert final_hp["peft_scheme"] == "lora"

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_no_dicts_or_lists_in_final_hyperparameters(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams_with_full_template
    ):
        """Final hyperparameters contain only string values — no dicts or lists leak through."""
        mock_get_options.return_value = (mock_hyperparams_with_full_template, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            training_dataset="s3://bucket/train.jsonl",
            overrides={"training_config": {"learning_rate": 5e-6}},
        )

        with patch("sagemaker.train.sft_trainer.TrainingJob") as mock_tj, \
             patch("sagemaker.train.sft_trainer.TrainDefaults") as mock_defaults, \
             patch("sagemaker.train.sft_trainer._create_input_data_config"), \
             patch("sagemaker.train.sft_trainer._convert_input_data_to_channels", return_value=[]), \
             patch("sagemaker.train.sft_trainer._create_output_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_serverless_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_mlflow_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._create_model_package_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._validate_hyperparameter_values"), \
             patch("sagemaker.train.sft_trainer._get_studio_tags", return_value=[]):

            mock_session = MagicMock()
            mock_session.boto_session.region_name = "us-west-2"
            mock_defaults.get_sagemaker_session.return_value = mock_session
            mock_defaults.get_role.return_value = "arn:aws:iam::123:role/Role"
            mock_tj.create.return_value = MagicMock()

            trainer.train(wait=False)

            call_kwargs = mock_tj.create.call_args[1]
            final_hp = call_kwargs["hyper_parameters"]

            # Every value must be a string
            for k, v in final_hp.items():
                assert isinstance(v, str), (
                    f"Hyperparameter '{k}' has type {type(v).__name__}, expected str. Value: {v}"
                )

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_spec_validation_still_applies_with_full_template(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams_with_full_template
    ):
        """Spec validation still rejects out-of-range values for spec keys."""
        mock_get_options.return_value = (mock_hyperparams_with_full_template, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            overrides={"training_config": {"num_epochs": 999}},
        )

        with pytest.raises(ValueError, match="above maximum"):
            trainer.get_resolved_recipe()


# --- Tests that recipe/overrides flow into train() ---


class TestSFTTrainerRecipeFlowsIntoTrain:
    """Tests that recipe/overrides values are applied to hyperparameters in train()."""

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_sft_train_applies_recipe_overrides_to_hyperparameters(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        recipe_file, mock_hyperparams
    ):
        """SFTTrainer.train() applies resolved recipe values to final_hyperparameters."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            training_dataset="s3://bucket/train.jsonl",
            recipe=recipe_file,
            overrides={"training_config": {"num_epochs": 7}},
        )

        # Mock TrainingJob.create to capture what hyperparameters are sent
        with patch("sagemaker.train.sft_trainer.TrainingJob") as mock_tj, \
             patch("sagemaker.train.sft_trainer.TrainDefaults") as mock_defaults, \
             patch("sagemaker.train.sft_trainer._create_input_data_config") as mock_input, \
             patch("sagemaker.train.sft_trainer._convert_input_data_to_channels", return_value=[]), \
             patch("sagemaker.train.sft_trainer._create_output_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_serverless_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_mlflow_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._create_model_package_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._validate_hyperparameter_values"), \
             patch("sagemaker.train.sft_trainer._get_studio_tags", return_value=[]):

            mock_session = MagicMock()
            mock_session.boto_session.region_name = "us-west-2"
            mock_defaults.get_sagemaker_session.return_value = mock_session
            mock_defaults.get_role.return_value = "arn:aws:iam::123:role/Role"

            mock_tj.create.return_value = MagicMock()

            trainer.train(wait=False)

            # Check the hyper_parameters passed to TrainingJob.create
            call_kwargs = mock_tj.create.call_args[1]
            final_hp = call_kwargs["hyper_parameters"]

            # Recipe file had learning_rate: 2e-5, overrides had num_epochs: 7
            # Hub default was learning_rate: "1e-5", num_epochs: "3"
            assert final_hp["learning_rate"] == "2e-05", f"Expected recipe value, got {final_hp['learning_rate']}"
            assert final_hp["num_epochs"] == "7", f"Expected override value, got {final_hp['num_epochs']}"

    @patch("sagemaker.train.sft_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.sft_trainer._validate_and_resolve_model_package_group", return_value="my-group")
    @patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    def test_sft_train_without_recipe_uses_hyperparameters_unchanged(
        self, mock_resolve, mock_validate_group, mock_get_options, mock_eula,
        mock_hyperparams
    ):
        """SFTTrainer.train() without recipe/overrides uses hyperparameters.to_dict() as-is."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model="nova-lite-v2",
            model_package_group="my-group",
            training_dataset="s3://bucket/train.jsonl",
        )

        with patch("sagemaker.train.sft_trainer.TrainingJob") as mock_tj, \
             patch("sagemaker.train.sft_trainer.TrainDefaults") as mock_defaults, \
             patch("sagemaker.train.sft_trainer._create_input_data_config") as mock_input, \
             patch("sagemaker.train.sft_trainer._convert_input_data_to_channels", return_value=[]), \
             patch("sagemaker.train.sft_trainer._create_output_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_serverless_config", return_value=MagicMock()), \
             patch("sagemaker.train.sft_trainer._create_mlflow_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._create_model_package_config", return_value=None), \
             patch("sagemaker.train.sft_trainer._validate_hyperparameter_values"), \
             patch("sagemaker.train.sft_trainer._get_studio_tags", return_value=[]):

            mock_session = MagicMock()
            mock_session.boto_session.region_name = "us-west-2"
            mock_defaults.get_sagemaker_session.return_value = mock_session
            mock_defaults.get_role.return_value = "arn:aws:iam::123:role/Role"

            mock_tj.create.return_value = MagicMock()

            trainer.train(wait=False)

            call_kwargs = mock_tj.create.call_args[1]
            final_hp = call_kwargs["hyper_parameters"]

            # Should match hyperparameters.to_dict() exactly (no recipe applied)
            assert final_hp == {"learning_rate": "1e-5", "num_epochs": "3"}


# --- Tests that recipe/overrides flow into evaluate() ---


class TestBenchMarkEvaluatorRecipeFlowsIntoEvaluate:
    """Tests that recipe/overrides are used by _get_effective_hyperparameters in evaluate()."""

    def test_effective_hyperparameters_with_recipe(self, tmp_path):
        """_get_effective_hyperparameters returns recipe values when recipe is provided."""
        recipe_content = {
            "inference": {
                "max_new_tokens": 2048,
                "temperature": 0,
            }
        }
        recipe_path = tmp_path / "eval_recipe.yaml"
        recipe_path.write_text(yaml.dump(recipe_content))

        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-east-1"

        mock_model_info = MagicMock()
        mock_model_info.base_model_name = "nova-pro-v2"
        mock_model_info.base_model_arn = "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/nova-pro-v2/1.0"
        mock_model_info.source_model_package_arn = None

        with patch(
            "sagemaker.train.common_utils.model_resolution._resolve_base_model",
            return_value=mock_model_info,
        ), patch(
            "sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn",
            return_value=None,
        ):
            from sagemaker.train.evaluate.benchmark_evaluator import BenchMarkEvaluator, _Benchmark

            evaluator = BenchMarkEvaluator(
                benchmark=_Benchmark.MMLU,
                model="nova-pro-v2",
                s3_output_path="s3://bucket/output",
                sagemaker_session=mock_session,
                recipe=str(recipe_path),
                overrides={"inference": {"max_new_tokens": 4096}},
            )

            # Set _hyperparameters with specs for template building
            mock_hp = MagicMock()
            mock_hp._specs = {
                "max_new_tokens": {"default": 1024, "type": "integer", "min": 1, "max": 8192},
                "temperature": {"default": 1, "type": "integer", "min": 0, "max": 2},
            }
            mock_hp.to_dict.return_value = {"max_new_tokens": "1024", "temperature": "1"}
            object.__setattr__(evaluator, '_hyperparameters', mock_hp)

            # _get_effective_hyperparameters should return resolved recipe values
            effective = evaluator._get_effective_hyperparameters()

            assert effective["max_new_tokens"] == 4096, f"Override should win, got {effective['max_new_tokens']}"
            assert effective["temperature"] == 0, f"Recipe value should be used, got {effective['temperature']}"

    def test_effective_hyperparameters_without_recipe_uses_to_dict(self, tmp_path):
        """_get_effective_hyperparameters falls back to hyperparameters.to_dict() without recipe."""
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-east-1"

        mock_model_info = MagicMock()
        mock_model_info.base_model_name = "nova-pro-v2"
        mock_model_info.base_model_arn = "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/nova-pro-v2/1.0"
        mock_model_info.source_model_package_arn = None

        with patch(
            "sagemaker.train.common_utils.model_resolution._resolve_base_model",
            return_value=mock_model_info,
        ), patch(
            "sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn",
            return_value=None,
        ):
            from sagemaker.train.evaluate.benchmark_evaluator import BenchMarkEvaluator, _Benchmark

            evaluator = BenchMarkEvaluator(
                benchmark=_Benchmark.MMLU,
                model="nova-pro-v2",
                s3_output_path="s3://bucket/output",
                sagemaker_session=mock_session,
            )

            mock_hp = MagicMock()
            mock_hp._specs = {}
            mock_hp.to_dict.return_value = {"max_new_tokens": "1024", "temperature": "1"}
            object.__setattr__(evaluator, '_hyperparameters', mock_hp)

            effective = evaluator._get_effective_hyperparameters()

            # Should be the raw to_dict() output
            assert effective == {"max_new_tokens": "1024", "temperature": "1"}


# --- MultiTurnRLTrainer Tests ---


class TestMultiTurnRLTrainerRecipeIntegration:
    """Tests for MultiTurnRLTrainer recipe/overrides support."""

    @patch("sagemaker.train.multi_turn_rl_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.multi_turn_rl_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.multi_turn_rl_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    @patch("sagemaker.train.multi_turn_rl_trainer._validate_s3_path_exists")
    @patch("sagemaker.train.multi_turn_rl_trainer._get_default_s3_output_path", return_value="s3://bucket/output/")
    def test_mtrl_trainer_with_recipe_and_overrides(
        self, mock_s3_default, mock_s3_validate, mock_resolve, mock_get_options, mock_eula,
        recipe_file, mock_hyperparams
    ):
        """MultiTurnRLTrainer with recipe + overrides returns merged result via get_resolved_recipe()."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

        with patch.object(MultiTurnRLTrainer, '_validate_agent_config'), \
             patch.object(MultiTurnRLTrainer, '_validate_networking'), \
             patch.object(MultiTurnRLTrainer, '_resolve_model_package_group', return_value="my-group"), \
             patch.object(MultiTurnRLTrainer, '_resolve_intermediate_checkpoint_mpg', return_value=None):

            trainer = MultiTurnRLTrainer(
                model="nova-lite-v2",
                agent_env="arn:aws:lambda:us-west-2:123456789012:function:my-agent",
                training_dataset="s3://bucket/train.jsonl",
                recipe=recipe_file,
                overrides={"training_config": {"num_epochs": 10}},
            )

            resolved = trainer.get_resolved_recipe()

            # Overrides take highest precedence
            assert resolved["training_config"]["num_epochs"] == 10
            # Recipe file values take precedence over defaults
            assert resolved["training_config"]["learning_rate"] == 2e-5
            assert resolved["training_config"]["batch_size"] == 8

    @patch("sagemaker.train.multi_turn_rl_trainer._validate_eula_for_gated_model", return_value=False)
    @patch("sagemaker.train.multi_turn_rl_trainer._get_fine_tuning_options_and_model_arn")
    @patch("sagemaker.train.multi_turn_rl_trainer._resolve_model_and_name", return_value=("model_obj", "nova-lite-v2"))
    @patch("sagemaker.train.multi_turn_rl_trainer._validate_s3_path_exists")
    @patch("sagemaker.train.multi_turn_rl_trainer._get_default_s3_output_path", return_value="s3://bucket/output/")
    def test_mtrl_trainer_no_recipe_no_overrides_raises(
        self, mock_s3_default, mock_s3_validate, mock_resolve, mock_get_options, mock_eula,
        mock_hyperparams
    ):
        """MultiTurnRLTrainer with no recipe/overrides raises ValueError."""
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)

        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

        with patch.object(MultiTurnRLTrainer, '_validate_agent_config'), \
             patch.object(MultiTurnRLTrainer, '_validate_networking'), \
             patch.object(MultiTurnRLTrainer, '_resolve_model_package_group', return_value="my-group"), \
             patch.object(MultiTurnRLTrainer, '_resolve_intermediate_checkpoint_mpg', return_value=None):

            trainer = MultiTurnRLTrainer(
                model="nova-lite-v2",
                agent_env="arn:aws:lambda:us-west-2:123456789012:function:my-agent",
                training_dataset="s3://bucket/train.jsonl",
            )

            with pytest.raises(ValueError, match="get_resolved_recipe\\(\\) requires"):
                trainer.get_resolved_recipe()


# --- MultiTurnRLEvaluator Tests ---


class TestMultiTurnRLEvaluatorRecipeIntegration:
    """Tests for MultiTurnRLEvaluator recipe/overrides support."""

    def test_mtrl_evaluator_with_recipe_and_overrides(self, tmp_path):
        """MultiTurnRLEvaluator with recipe + overrides returns merged result."""
        recipe_content = {
            "inference": {
                "max_tokens": 2048,
                "sampling_temperature": 1,
            }
        }
        recipe_path = tmp_path / "mtrl_eval_recipe.yaml"
        recipe_path.write_text(yaml.dump(recipe_content))

        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-east-1"

        mock_model_info = MagicMock()
        mock_model_info.base_model_name = "nova-lite-v2"
        mock_model_info.base_model_arn = "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/nova-lite-v2/1.0"
        mock_model_info.source_model_package_arn = None
        mock_model_info.model_type = MagicMock()

        with patch(
            "sagemaker.train.common_utils.model_resolution._resolve_base_model",
            return_value=mock_model_info,
        ), patch(
            "sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn",
            return_value=None,
        ):
            from sagemaker.train.evaluate.multi_turn_rl_evaluator import MultiTurnRLEvaluator

            evaluator = MultiTurnRLEvaluator(
                model="nova-lite-v2",
                agent_config="arn:aws:lambda:us-west-2:123456789012:function:my-agent",
                dataset="s3://bucket/eval-data.jsonl",
                s3_output_path="s3://bucket/eval-output/",
                sagemaker_session=mock_session,
                recipe=str(recipe_path),
                overrides={"inference": {"max_tokens": 4096}},
            )

            mock_hp = MagicMock()
            mock_hp._specs = {
                "max_tokens": {"default": 1024, "type": "integer", "min": 1, "max": 8192},
                "sampling_temperature": {"default": 1, "type": "integer", "min": 0, "max": 2},
            }
            mock_hp.to_dict.return_value = {"max_tokens": "1024", "sampling_temperature": "1"}
            object.__setattr__(evaluator, '_hyperparameters', mock_hp)

            resolved = evaluator.get_resolved_recipe()

            assert resolved["inference"]["max_tokens"] == 4096
            assert resolved["inference"]["sampling_temperature"] == 1

    def test_mtrl_evaluator_no_recipe_raises(self):
        """MultiTurnRLEvaluator without recipe/overrides raises ValueError."""
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-east-1"

        mock_model_info = MagicMock()
        mock_model_info.base_model_name = "nova-lite-v2"
        mock_model_info.base_model_arn = "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/nova-lite-v2/1.0"
        mock_model_info.source_model_package_arn = None
        mock_model_info.model_type = MagicMock()

        with patch(
            "sagemaker.train.common_utils.model_resolution._resolve_base_model",
            return_value=mock_model_info,
        ), patch(
            "sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn",
            return_value=None,
        ):
            from sagemaker.train.evaluate.multi_turn_rl_evaluator import MultiTurnRLEvaluator

            evaluator = MultiTurnRLEvaluator(
                model="nova-lite-v2",
                agent_config="arn:aws:lambda:us-west-2:123456789012:function:my-agent",
                dataset="s3://bucket/eval-data.jsonl",
                s3_output_path="s3://bucket/eval-output/",
                sagemaker_session=mock_session,
            )

            with pytest.raises(ValueError, match="get_resolved_recipe\\(\\) requires"):
                evaluator.get_resolved_recipe()
