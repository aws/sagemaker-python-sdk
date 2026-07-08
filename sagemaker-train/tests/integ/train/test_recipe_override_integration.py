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
"""Integration tests for recipe override feature (get_resolved_recipe)."""
from __future__ import absolute_import

import os
import tempfile
import time

import pytest
import yaml

from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType


# Ensure bundled service model is available for botocore
@pytest.fixture(autouse=True)
def setup_aws_data_path():
    sample_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "sagemaker-core", "sample"
    )
    # Resolve relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    sample_path = os.path.join(repo_root, "sagemaker-core", "sample")
    if os.path.isdir(sample_path):
        os.environ["AWS_DATA_PATH"] = sample_path
    yield
    os.environ.pop("AWS_DATA_PATH", None)


class TestSFTTrainerRecipeOverrideInteg:
    """Integration tests for SFTTrainer with recipe override."""

    def test_sft_get_resolved_recipe_with_local_yaml(self):
        """Test that SFTTrainer.get_resolved_recipe() returns merged config from a local YAML."""
        # Create a recipe file
        recipe_content = {
            "training_config": {
                "learning_rate": 1e-5,
                "num_epochs": 3,
                "batch_size": 8,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            sft_trainer = SFTTrainer(
                model="meta-textgeneration-llama-3-2-1b-instruct",
                training_type=TrainingType.LORA,
                model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
                training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
                accept_eula=True,
                recipe=recipe_path,
                overrides={
                    "training_config": {
                        "learning_rate": 2e-5,
                        "num_epochs": 5,
                    }
                },
            )

            resolved = sft_trainer.get_resolved_recipe()

            # Overrides win
            assert resolved["training_config"]["learning_rate"] == 2e-5
            assert resolved["training_config"]["num_epochs"] == 5
            # Recipe file value preserved where no override
            assert resolved["training_config"]["batch_size"] == 8

        finally:
            os.unlink(recipe_path)

    def test_sft_get_resolved_recipe_overrides_only(self):
        """Test get_resolved_recipe() with only overrides (no recipe file)."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "learning_rate": 3e-5,
                }
            },
        )

        resolved = sft_trainer.get_resolved_recipe()

        # Override applied on top of Hub defaults
        assert resolved["training_config"]["learning_rate"] == 3e-5

    def test_sft_get_resolved_recipe_no_recipe_raises(self, sagemaker_session):
        """Test that get_resolved_recipe() raises when no recipe or overrides provided."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            sagemaker_session=sagemaker_session,
        )

        with pytest.raises(ValueError, match=r"requires a 'recipe', 'overrides'"):
            sft_trainer.get_resolved_recipe()

    @pytest.mark.skip(reason="Skipping GPU resource intensive test - submits actual training job")
    def test_sft_train_with_recipe_e2e(self):
        """End-to-end test: SFTTrainer with recipe, inspect, then train."""
        recipe_content = {
            "training_config": {
                "learning_rate": 1e-5,
                "num_epochs": 1,
                "batch_size": 4,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            sft_trainer = SFTTrainer(
                model="meta-textgeneration-llama-3-2-1b-instruct",
                training_type=TrainingType.LORA,
                model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
                training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
                s3_output_path="s3://mc-flows-sdk-testing/output/",
                accept_eula=True,
                recipe=recipe_path,
                overrides={"training_config": {"learning_rate": 2e-5}},
            )

            # Verify resolved recipe before training
            resolved = sft_trainer.get_resolved_recipe()
            assert resolved["training_config"]["learning_rate"] == 2e-5

            # Submit training job
            training_job = sft_trainer.train(wait=False)

            # Wait for completion
            max_wait_time = 3600
            poll_interval = 30
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                training_job.refresh()
                status = training_job.training_job_status
                if status in ["Completed", "Failed", "Stopped"]:
                    break
                time.sleep(poll_interval)

            assert training_job.training_job_status == "Completed"

        finally:
            os.unlink(recipe_path)


class TestSFTTrainerFullRecipeOverrideInteg:
    """Integration tests for SFTTrainer overriding non-spec keys via full recipe template."""

    def test_sft_override_non_spec_keys(self):
        """Test that non-spec keys (max_length, save_top_k) can be overridden."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "max_length": 16384,
                    "save_top_k": 3,
                }
            },
        )

        resolved = sft_trainer.get_resolved_recipe()

        # Non-spec keys should be overridable when full recipe template is available
        assert resolved["training_config"]["max_length"] == 16384
        assert resolved["training_config"]["save_top_k"] == 3

    def test_sft_override_nested_non_spec_keys(self):
        """Test that nested non-spec keys (training_args.max_len) can be overridden."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "training_args": {"max_len": 8192},
                }
            },
        )

        resolved = sft_trainer.get_resolved_recipe()

        # Nested non-spec key overridden (merged into resolved recipe as new key)
        assert resolved["training_config"]["training_args"]["max_len"] == 8192
        # Spec-level default preserved at training_config level (seed is a spec key)
        assert resolved["training_config"]["seed"] == 42

    def test_sft_full_recipe_defaults_preserved(self):
        """Test that full recipe defaults are present for keys not overridden."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "learning_rate": 3e-5,
                }
            },
        )

        resolved = sft_trainer.get_resolved_recipe()

        # Override applied
        assert resolved["training_config"]["learning_rate"] == 3e-5
        # Full recipe template keys are present (not just spec keys)
        training_config = resolved.get("training_config", {})
        assert len(training_config) > 3, (
            f"Expected more keys from full recipe template, got only: {list(training_config.keys())}"
        )

    def test_sft_full_recipe_with_recipe_file_and_overrides(self):
        """Test 3-level merge: full_template < recipe file < overrides with non-spec keys."""
        recipe_content = {
            "training_config": {
                "max_length": 8192,
                "learning_rate": 1e-5,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            sft_trainer = SFTTrainer(
                model="meta-textgeneration-llama-3-2-1b-instruct",
                training_type=TrainingType.LORA,
                model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
                training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
                accept_eula=True,
                recipe=recipe_path,
                overrides={
                    "training_config": {
                        "learning_rate": 2e-5,
                    }
                },
            )

            resolved = sft_trainer.get_resolved_recipe()

            # Overrides win
            assert resolved["training_config"]["learning_rate"] == 2e-5
            # Recipe file wins over full template default
            assert resolved["training_config"]["max_length"] == 8192

        finally:
            os.unlink(recipe_path)


class TestSFTTrainerNestedRecipeOverrideInteg:
    """Integration tests for nested recipe key overrides and deep flattening."""

    def test_sft_nested_override_flows_to_hyperparameters(self):
        """Test that nested overrides (lr_scheduler.warmup_steps) flatten into hyperparameters."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "training_args": {"max_len": 8192, "lr_warmup_ratio": 0.05},
                }
            },
        )

        # Simulate what train() does: to_dict() then _apply_recipe_to_hyperparameters()
        final_hp = sft_trainer.hyperparameters.to_dict()
        final_hp = sft_trainer._apply_recipe_to_hyperparameters(final_hp)

        # Nested keys are flattened to leaf names
        assert "max_len" in final_hp, f"max_len missing from: {list(final_hp.keys())}"
        assert final_hp["max_len"] == "8192"
        assert "lr_warmup_ratio" in final_hp
        assert final_hp["lr_warmup_ratio"] == "0.05"
        # All values are strings (SageMaker API requirement)
        for k, v in final_hp.items():
            assert isinstance(v, str), f"Key '{k}' has non-string value: {type(v).__name__}"

    def test_sft_nested_defaults_preserved_in_hyperparameters(self):
        """Test that unchanged nested defaults also flatten into hyperparameters."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "learning_rate": 5e-6,
                }
            },
        )

        final_hp = sft_trainer.hyperparameters.to_dict()
        final_hp = sft_trainer._apply_recipe_to_hyperparameters(final_hp)

        # Nested leaf defaults from full recipe are present
        assert "seed" in final_hp, f"Nested default 'seed' missing from: {list(final_hp.keys())}"
        assert "gradient_clipping" in final_hp
        # Override applied
        assert final_hp["learning_rate"] == "5e-06"

    def test_sft_recipe_file_overrides_nested_keys(self):
        """Test that a recipe file can override nested keys in the full template."""
        recipe_content = {
            "training_config": {
                "training_args": {
                    "max_len": 2048,
                    "max_epochs": 2,
                }
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            sft_trainer = SFTTrainer(
                model="meta-textgeneration-llama-3-2-1b-instruct",
                training_type=TrainingType.LORA,
                model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
                training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
                accept_eula=True,
                recipe=recipe_path,
                overrides={
                    "training_config": {
                        "training_args": {"max_len": 4096},
                    }
                },
            )

            final_hp = sft_trainer.hyperparameters.to_dict()
            final_hp = sft_trainer._apply_recipe_to_hyperparameters(final_hp)

            # Overrides > recipe file > full template
            assert final_hp["max_len"] == "4096"
            # Recipe file value preserved (no override for max_epochs)
            assert final_hp["max_epochs"] == "2"

        finally:
            os.unlink(recipe_path)


class TestBenchMarkEvaluatorRecipeOverrideInteg:
    """Integration tests for BenchMarkEvaluator with recipe override."""

    def test_evaluator_get_resolved_recipe_with_local_yaml(self, monkeypatch):
        """Test BenchMarkEvaluator.get_resolved_recipe() with recipe + overrides."""
        # The base model lives in the public hub, not the private "sdktest"
        # recipe hub that the session-scoped use_private_hub fixture pins
        # SAGEMAKER_HUB_NAME to. The evaluator's JumpStart model resolution does
        # not fall back to the public hub, so resolve the base model against
        # SageMakerPublicHub explicitly.
        monkeypatch.setenv("SAGEMAKER_HUB_NAME", "SageMakerPublicHub")

        from sagemaker.train.evaluate import BenchMarkEvaluator, get_benchmarks

        Benchmark = get_benchmarks()

        recipe_content = {
            "inference": {
                "max_new_tokens": 256,
                "temperature": 1,
                "top_p": 0.9,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            evaluator = BenchMarkEvaluator(
                benchmark=Benchmark.MMLU,
                subtasks=["abstract_algebra"],
                model="meta-textgeneration-llama-3-2-1b-instruct",
                s3_output_path="s3://mc-flows-sdk-testing/eval-output/",
                recipe=recipe_path,
                overrides={
                    "inference": {
                        "max_new_tokens": 512,
                        "temperature": 0,
                    }
                },
            )

            resolved = evaluator.get_resolved_recipe()

            # Overrides win
            assert resolved["inference"]["max_new_tokens"] == 512
            assert resolved["inference"]["temperature"] == 0
            # Recipe value preserved
            assert resolved["inference"]["top_p"] == 0.9

        finally:
            os.unlink(recipe_path)

    def test_evaluator_get_resolved_recipe_no_recipe_raises(self, monkeypatch):
        """Test that get_resolved_recipe() raises without recipe/overrides."""
        # See test_evaluator_get_resolved_recipe_with_local_yaml: resolve the
        # base model against the public hub since the evaluator does not fall
        # back to it from the pinned private "sdktest" hub.
        monkeypatch.setenv("SAGEMAKER_HUB_NAME", "SageMakerPublicHub")

        from sagemaker.train.evaluate import BenchMarkEvaluator, get_benchmarks

        Benchmark = get_benchmarks()

        evaluator = BenchMarkEvaluator(
            benchmark=Benchmark.MMLU,
            subtasks=["abstract_algebra"],
            model="meta-textgeneration-llama-3-2-1b-instruct",
            s3_output_path="s3://mc-flows-sdk-testing/eval-output/",
        )

        with pytest.raises(ValueError, match=r"requires a 'recipe', 'overrides'"):
            evaluator.get_resolved_recipe()


class TestSFTTrainerValidationFailuresInteg:
    """Integration tests for validation failures triggered via SFTTrainer overrides."""

    def test_sft_rejects_save_steps_greater_than_max_steps(self):
        """Test that save_steps > max_steps raises ValueError via SFTTrainer."""
        sft_trainer = SFTTrainer(
            model="nova-textgeneration-lite-v2",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "max_steps": 50,
                    "save_steps": 200,
                }
            },
        )

        with pytest.raises(ValueError, match="save_steps.*must be less than or equal to.*max_steps"):
            sft_trainer.get_resolved_recipe()

    def test_sft_rejects_learning_rate_above_maximum(self):
        """Test that learning_rate > 1 (spec max) raises ValueError."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "learning_rate": 5.0,
                }
            },
        )

        with pytest.raises(ValueError, match="above maximum"):
            sft_trainer.get_resolved_recipe()

    def test_sft_rejects_invalid_type_for_learning_rate(self):
        """Test that a string learning_rate raises type validation error."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "learning_rate": "not_a_number",
                }
            },
        )

        with pytest.raises(ValueError, match="Invalid type for"):
            sft_trainer.get_resolved_recipe()

    def test_sft_rejects_invalid_enum_value_for_seed(self):
        """Test that an invalid enum value (e.g., batch_size not in allowed set) raises ValueError."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "lora_alpha": 99,
                }
            },
        )

        with pytest.raises(ValueError, match="not in allowed values"):
            sft_trainer.get_resolved_recipe()

    @pytest.mark.us_east_1
    def test_sft_rejects_max_steps_below_minimum(self, sagemaker_session_us_east_1):
        """Test that max_steps below spec minimum raises ValueError."""
        sft_trainer = SFTTrainer(
            model="nova-textgeneration-lite-v2",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            sagemaker_session=sagemaker_session_us_east_1,
            overrides={
                "training_config": {
                    "max_steps": 1,
                }
            },
        )

        with pytest.raises(ValueError, match="below minimum"):
            sft_trainer.get_resolved_recipe()

    def test_sft_rejects_invalid_instance_type_with_compute(self):
        """Test that an unsupported instance_type in HyperPodCompute raises ValueError."""
        from sagemaker.core.training.configs import HyperPodCompute

        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.t3.medium",
        )

        sft_trainer = SFTTrainer(
            model="nova-textgeneration-lite-v2",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            compute=compute,
            overrides={
                "training_config": {
                    "learning_rate": 1e-5,
                }
            },
        )

        with pytest.raises(ValueError, match="not supported"):
            sft_trainer.get_resolved_recipe()

    def test_sft_rejects_invalid_instance_type_with_hyperpod_compute(self):
        """Test that an unsupported instance_type in HyperPodCompute raises ValueError."""
        from sagemaker.core.training.configs import HyperPodCompute

        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.t3.medium",
        )

        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            compute=compute,
            overrides={
                "training_config": {
                    "learning_rate": 1e-5,
                }
            },
        )

        with pytest.raises(ValueError, match="not supported"):
            sft_trainer.get_resolved_recipe()

    def test_sft_rejects_invalid_node_count_with_hyperpod_compute(self):
        """Test that an unsupported node_count in HyperPodCompute raises ValueError."""
        from sagemaker.core.training.configs import HyperPodCompute

        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
            node_count=7,
        )

        sft_trainer = SFTTrainer(
            model="nova-textgeneration-lite-v2",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            compute=compute,
            overrides={
                "training_config": {
                    "learning_rate": 1e-5,
                }
            },
        )

        with pytest.raises(ValueError, match="not supported"):
            sft_trainer.get_resolved_recipe()

    def test_sft_valid_instance_type_passes_with_compute(self):
        """Test that a valid instance_type in Compute passes validation."""
        from sagemaker.core.training.configs import Compute

        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)

        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            compute=compute,
            overrides={
                "training_config": {
                    "learning_rate": 1e-5,
                }
            },
        )

        # Should not raise
        resolved = sft_trainer.get_resolved_recipe()
        assert resolved["training_config"]["learning_rate"] == 1e-5

    def test_sft_serverless_skips_instance_type_validation(self):
        """Test that serverless mode (no compute) skips instance_type validation."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "learning_rate": 1e-5,
                }
            },
        )

        # No compute = serverless, instance_type validation skipped
        resolved = sft_trainer.get_resolved_recipe()
        assert resolved["training_config"]["learning_rate"] == 1e-5

    def test_sft_save_steps_equal_to_max_steps_passes(self):
        """Test that save_steps == max_steps is valid (boundary condition)."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "max_steps": 100,
                    "save_steps": 100,
                }
            },
        )

        # Boundary condition: save_steps == max_steps should pass
        resolved = sft_trainer.get_resolved_recipe()
        assert resolved["training_config"]["max_steps"] == 100
        assert resolved["training_config"]["save_steps"] == 100

    def test_sft_recipe_file_with_invalid_value_raises(self):
        """Test that validation catches errors from recipe file values."""
        recipe_content = {
            "training_config": {
                "learning_rate": 999.0,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            sft_trainer = SFTTrainer(
                model="meta-textgeneration-llama-3-2-1b-instruct",
                training_type=TrainingType.LORA,
                model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
                training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
                accept_eula=True,
                recipe=recipe_path,
            )

            with pytest.raises(ValueError, match="above maximum"):
                sft_trainer.get_resolved_recipe()
        finally:
            os.unlink(recipe_path)

    def test_sft_override_corrects_invalid_recipe_value(self):
        """Test that a programmatic override can fix an invalid recipe file value."""
        recipe_content = {
            "training_config": {
                "learning_rate": 999.0,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            sft_trainer = SFTTrainer(
                model="meta-textgeneration-llama-3-2-1b-instruct",
                training_type=TrainingType.LORA,
                model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
                training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
                accept_eula=True,
                recipe=recipe_path,
                overrides={
                    "training_config": {
                        "learning_rate": 1e-5,
                    }
                },
            )

            # Override wins, fixing the bad recipe file value
            resolved = sft_trainer.get_resolved_recipe()
            assert resolved["training_config"]["learning_rate"] == 1e-5
        finally:
            os.unlink(recipe_path)

    def test_sft_nonexistent_recipe_file_raises(self):
        """Test that a non-existent recipe file path raises ValueError."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            recipe="/tmp/nonexistent_recipe_file_abc123.yaml",
        )

        with pytest.raises(ValueError, match="Recipe file not found"):
            sft_trainer.get_resolved_recipe()

    def test_sft_http_recipe_url_rejected(self):
        """Test that HTTP/HTTPS recipe URLs are rejected for security."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            recipe="https://evil.example.com/recipe.yaml",
        )

        with pytest.raises(ValueError, match="HTTP/HTTPS recipe URLs are not supported"):
            sft_trainer.get_resolved_recipe()

    def test_sft_resolved_recipe_is_idempotent(self):
        """Test that calling get_resolved_recipe() twice returns the same result."""
        sft_trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
            training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
            accept_eula=True,
            overrides={
                "training_config": {
                    "learning_rate": 2e-5,
                }
            },
        )

        result1 = sft_trainer.get_resolved_recipe()
        result2 = sft_trainer.get_resolved_recipe()

        assert result1 == result2
        assert result1["training_config"]["learning_rate"] == 2e-5

        # Mutating the returned dict doesn't affect cached result
        result1["training_config"]["learning_rate"] = 999
        result3 = sft_trainer.get_resolved_recipe()
        assert result3["training_config"]["learning_rate"] == 2e-5

    def test_sft_invalid_yaml_content_raises(self):
        """Test that a YAML file with non-dict content raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("- just\n- a\n- list\n")
            recipe_path = f.name

        try:
            sft_trainer = SFTTrainer(
                model="meta-textgeneration-llama-3-2-1b-instruct",
                training_type=TrainingType.LORA,
                model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
                training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
                accept_eula=True,
                recipe=recipe_path,
            )

            with pytest.raises(ValueError, match="did not parse as a YAML mapping"):
                sft_trainer.get_resolved_recipe()
        finally:
            os.unlink(recipe_path)


class TestModelTrainerRecipeOverrideInteg:
    """Integration tests for ModelTrainer.from_recipe with get_resolved_recipe()."""

    def test_model_trainer_get_resolved_recipe_with_local_yaml(self):
        """Test ModelTrainer.from_recipe + get_resolved_recipe() with a local YAML."""
        from sagemaker.train import ModelTrainer
        from sagemaker.train.configs import Compute

        recipe_content = {
            "run": {
                "name": "integ-test-experiment",
                "model_type": "amazon.nova-lite",
                "model_name_or_path": "amazon-nova-lite-v2",
                "replicas": 1,
            },
            "training_config": {
                "learning_rate": 1e-5,
                "num_epochs": 3,
                "batch_size": 4,
                "sequence_length": 4096,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)

            model_trainer = ModelTrainer.from_recipe(
                training_recipe=recipe_path,
                compute=compute,
                training_image="123456789012.dkr.ecr.us-west-2.amazonaws.com/nova:latest",
                recipe_overrides={
                    "run": {"name": "overridden-experiment"},
                    "training_config": {
                        "learning_rate": 2e-5,
                        "num_epochs": 5,
                    },
                },
            )

            resolved = model_trainer.get_resolved_recipe()

            # Overrides win
            assert resolved["run"]["name"] == "overridden-experiment"
            assert resolved["training_config"]["learning_rate"] == 2e-5
            assert resolved["training_config"]["num_epochs"] == 5

            # Recipe values preserved where no override
            assert resolved["training_config"]["batch_size"] == 4
            assert resolved["training_config"]["sequence_length"] == 4096
            assert resolved["run"]["model_type"] == "amazon.nova-lite"
            assert resolved["run"]["replicas"] == 1

        finally:
            os.unlink(recipe_path)

    def test_model_trainer_get_resolved_recipe_overrides_only(self):
        """Test ModelTrainer.from_recipe with only recipe_overrides (no external changes)."""
        from sagemaker.train import ModelTrainer
        from sagemaker.train.configs import Compute

        recipe_content = {
            "run": {
                "name": "base-experiment",
                "model_type": "amazon.nova-lite",
                "model_name_or_path": "amazon-nova-lite-v2",
                "replicas": 2,
            },
            "training_config": {
                "learning_rate": 1e-5,
                "num_epochs": 3,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            compute = Compute(instance_type="ml.p5.48xlarge", instance_count=2)

            model_trainer = ModelTrainer.from_recipe(
                training_recipe=recipe_path,
                compute=compute,
                training_image="123456789012.dkr.ecr.us-west-2.amazonaws.com/nova:latest",
            )

            resolved = model_trainer.get_resolved_recipe()

            # All values come from the recipe file (no overrides)
            assert resolved["run"]["name"] == "base-experiment"
            assert resolved["run"]["replicas"] == 2
            assert resolved["training_config"]["learning_rate"] == 1e-5
            assert resolved["training_config"]["num_epochs"] == 3

        finally:
            os.unlink(recipe_path)

    def test_model_trainer_get_resolved_recipe_is_idempotent(self):
        """Test that get_resolved_recipe() returns same result on repeated calls."""
        from sagemaker.train import ModelTrainer
        from sagemaker.train.configs import Compute

        recipe_content = {
            "run": {
                "name": "idempotent-test",
                "model_type": "amazon.nova-lite",
                "model_name_or_path": "amazon-nova-lite-v2",
                "replicas": 1,
            },
            "training_config": {"learning_rate": 1e-5},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(recipe_content, f)
            recipe_path = f.name

        try:
            compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)

            model_trainer = ModelTrainer.from_recipe(
                training_recipe=recipe_path,
                compute=compute,
                training_image="123456789012.dkr.ecr.us-west-2.amazonaws.com/nova:latest",
                recipe_overrides={"training_config": {"learning_rate": 3e-5}},
            )

            result1 = model_trainer.get_resolved_recipe()
            result2 = model_trainer.get_resolved_recipe()

            assert result1 == result2
            assert result1["training_config"]["learning_rate"] == 3e-5

            # Mutating the returned dict doesn't affect future calls
            result1["training_config"]["learning_rate"] = 999
            result3 = model_trainer.get_resolved_recipe()
            assert result3["training_config"]["learning_rate"] == 3e-5

        finally:
            os.unlink(recipe_path)
