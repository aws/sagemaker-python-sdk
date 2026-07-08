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
"""Unit tests for CPTTrainer data mixing integration."""

import pytest
from unittest.mock import Mock, patch, call

from sagemaker.train.cpt_trainer import CPTTrainer
from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.core.training.configs import HyperPodCompute, Compute


# Patch paths for CPTTrainer constructor dependencies
_PATCH_RESOLVE_MODEL = "sagemaker.train.cpt_trainer._resolve_model_and_name"
_PATCH_VALIDATE_GROUP = "sagemaker.train.cpt_trainer._validate_and_resolve_model_package_group"
_PATCH_VALIDATE_EULA = "sagemaker.train.cpt_trainer._validate_eula_for_gated_model"
_PATCH_RESOLVE_HP_CONTEXT = "sagemaker.train.cpt_trainer.resolve_hyperpod_datamix_context"
_PATCH_VALIDATE_CATEGORIES = "sagemaker.train.cpt_trainer.validate_data_mixing_categories"
_PATCH_BUILD_HP_FROM_CONTEXT = "sagemaker.train.cpt_trainer.build_hyperpod_datamix_recipe_from_context"
_PATCH_VALIDATE_DM_MODEL = "sagemaker.train.cpt_trainer.validate_data_mixing_model"
_PATCH_TRAIN_HYPERPOD = "sagemaker.train.cpt_trainer.CPTTrainer._train_hyperpod"


class TestCPTTrainerDataMixingConstruction:
    """Tests for CPTTrainer constructor data_mixing_config handling."""

    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_accepts_data_mixing_config(self, mock_resolve, mock_validate_group, mock_eula):
        """Test CPTTrainer accepts a DataMixingConfig instance."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )
        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            data_mixing_config=config,
        )
        assert trainer.data_mixing_config is config

    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_accepts_none_data_mixing_config(self, mock_resolve, mock_validate_group, mock_eula):
        """Test CPTTrainer constructor accepts None for data_mixing_config (no data mixing)."""
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )
        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            data_mixing_config=None,
        )
        assert trainer.data_mixing_config is None


class TestCPTTrainerDataMixingTrain:
    """Tests for CPTTrainer.train() data mixing integration."""

    @patch(_PATCH_TRAIN_HYPERPOD, return_value="job-name")
    @patch(_PATCH_BUILD_HP_FROM_CONTEXT, return_value=("fine-tuning/nova/nova_lite_2_0_datamix-abc", "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-CPT-latest"))
    @patch(_PATCH_VALIDATE_CATEGORIES)
    @patch(_PATCH_RESOLVE_HP_CONTEXT)
    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_train_includes_serialized_config_in_overrides(
        self, mock_resolve, mock_validate_group, mock_eula,
        mock_resolve_context, mock_validate_cats, mock_build_from_context, mock_train_hp
    ):
        """Test train() generates a datamix recipe and overrides compute recipe path."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )

        # Setup mock context
        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve_context.return_value = mock_context
        mock_validate_cats.return_value = config

        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
        )
        trainer.train(wait=False)

        # Verify 3-step pipeline was called
        mock_resolve_context.assert_called_once()
        # Verify CPT-specific parameters
        call_kwargs = mock_resolve_context.call_args[1]
        assert call_kwargs["customization_technique"] == "CPT"
        assert call_kwargs["training_type"] == "FULL"

        mock_validate_cats.assert_called_once_with(config, mock_context.categories)
        mock_build_from_context.assert_called_once_with(mock_context, config)

        # Verify _recipe_path was set with the generated recipe
        assert trainer._recipe_path == "fine-tuning/nova/nova_lite_2_0_datamix-abc"

        mock_train_hp.assert_called_once()

    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_train_raises_valueerror_for_plain_compute(
        self, mock_resolve, mock_validate_group, mock_eula
    ):
        """Test train() raises ValueError when compute is plain Compute instance with data_mixing_config.

        Note: CPT always requires HyperPod, so the error comes from the general CPT compute check.
        """
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        # CPTTrainer rejects non-HyperPodCompute at constructor level
        # So we create with HyperPod then swap compute to test train() path
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )
        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
        )
        # Replace compute with plain Compute to test train() validation
        trainer.compute = Compute(instance_type="ml.p5.48xlarge", instance_count=4)

        with pytest.raises(ValueError, match="CPT requires HyperPod compute"):
            trainer.train(wait=False)

    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_train_raises_valueerror_for_none_compute(
        self, mock_resolve, mock_validate_group, mock_eula
    ):
        """Test train() raises ValueError when compute is None with data_mixing_config.

        Note: CPT always requires HyperPod, so the error comes from the general CPT compute check.
        """
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )
        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
        )
        # Replace compute with None to test train() validation
        trainer.compute = None

        with pytest.raises(ValueError, match="CPT requires HyperPod compute"):
            trainer.train(wait=False)

    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("meta-llama-3", "meta-llama-3"))
    def test_train_raises_valueerror_for_non_nova_model(
        self, mock_resolve, mock_validate_group, mock_eula
    ):
        """Test train() raises ValueError for non-Nova model with data_mixing_config."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )
        trainer = CPTTrainer(
            model="meta-llama-3",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
        )

        with pytest.raises(ValueError, match="Data mixing is only supported for Nova models"):
            trainer.train(wait=False)

    @patch(_PATCH_TRAIN_HYPERPOD, return_value="job-name")
    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_train_without_data_mixing_config_omits_overrides(
        self, mock_resolve, mock_validate_group, mock_eula, mock_train_hp
    ):
        """Test train() without data_mixing_config omits data mixing override parameters."""
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )
        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
        )
        trainer.train(wait=False)

        # _recipe_path should NOT have been set by data mixing since no config provided
        assert trainer._recipe_path is None
        mock_train_hp.assert_called_once()

    @patch(_PATCH_TRAIN_HYPERPOD, return_value="job-name")
    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("meta-llama-3", "meta-llama-3"))
    def test_train_skips_validation_when_no_config(
        self, mock_resolve, mock_validate_group, mock_eula, mock_train_hp
    ):
        """Test train() skips model/platform validation when no data_mixing_config provided."""
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )
        # Non-Nova model without data_mixing_config should succeed
        trainer = CPTTrainer(
            model="meta-llama-3",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
        )
        # Should NOT raise ValueError even though model is non-Nova
        trainer.train(wait=False)
        mock_train_hp.assert_called_once()


class TestCPTTrainerDataMixingOrchestration:
    @patch(_PATCH_TRAIN_HYPERPOD, return_value="job-name")
    @patch(_PATCH_BUILD_HP_FROM_CONTEXT, return_value=("fine-tuning/nova/nova_lite_2_0_datamix-abc", "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-CPT-latest"))
    @patch(_PATCH_VALIDATE_CATEGORIES)
    @patch(_PATCH_RESOLVE_HP_CONTEXT)
    @patch(_PATCH_VALIDATE_DM_MODEL)
    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_orchestration_order_validate_model_resolve_validate_cats_build(
        self, mock_resolve_model, mock_validate_group, mock_eula,
        mock_validate_dm_model, mock_resolve_context, mock_validate_cats,
        mock_build_from_context, mock_train_hp
    ):
        """Test full orchestration: validate_data_mixing_model → resolve → validate_categories → build in order."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve_context.return_value = mock_context
        mock_validate_cats.return_value = config

        # Use a shared call tracker to verify ordering
        call_order = []
        mock_validate_dm_model.side_effect = lambda *a, **kw: call_order.append("validate_model")
        mock_resolve_context.side_effect = lambda *a, **kw: (call_order.append("resolve"), mock_context)[1]
        mock_validate_cats.side_effect = lambda *a, **kw: (call_order.append("validate_categories"), config)[1]
        mock_build_from_context.side_effect = lambda *a, **kw: (
            call_order.append("build"),
            ("fine-tuning/nova/nova_lite_2_0_datamix-abc", "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-CPT-latest"),
        )[1]

        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
        )
        trainer.train(wait=False)

        assert call_order == ["validate_model", "resolve", "validate_categories", "build"]

    @patch(_PATCH_TRAIN_HYPERPOD, return_value="job-name")
    @patch(_PATCH_BUILD_HP_FROM_CONTEXT, return_value=("fine-tuning/nova/nova_lite_2_0_datamix-abc", "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-CPT-latest"))
    @patch(_PATCH_VALIDATE_CATEGORIES)
    @patch(_PATCH_RESOLVE_HP_CONTEXT)
    @patch(_PATCH_VALIDATE_DM_MODEL)
    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_customization_technique_cpt_and_training_type_full_passed_to_resolve(
        self, mock_resolve_model, mock_validate_group, mock_eula,
        mock_validate_dm_model, mock_resolve_context, mock_validate_cats,
        mock_build_from_context, mock_train_hp
    ):
        """Test customization_technique='CPT' and training_type='FULL' are passed to resolve."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve_context.return_value = mock_context
        mock_validate_cats.return_value = config

        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
        )
        trainer.train(wait=False)

        mock_resolve_context.assert_called_once()
        call_kwargs = mock_resolve_context.call_args[1]
        assert call_kwargs["customization_technique"] == "CPT"
        assert call_kwargs["training_type"] == "FULL"

    @patch(_PATCH_TRAIN_HYPERPOD, return_value="job-name")
    @patch(_PATCH_BUILD_HP_FROM_CONTEXT, return_value=("fine-tuning/nova/nova_lite_2_0_datamix-abc", "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-CPT-latest"))
    @patch(_PATCH_VALIDATE_CATEGORIES)
    @patch(_PATCH_RESOLVE_HP_CONTEXT)
    @patch(_PATCH_VALIDATE_DM_MODEL)
    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_recipe_path_set_to_returned_relative_recipe_path(
        self, mock_resolve_model, mock_validate_group, mock_eula,
        mock_validate_dm_model, mock_resolve_context, mock_validate_cats,
        mock_build_from_context, mock_train_hp
    ):
        """Test self._recipe_path is set to the relative_recipe_path returned by build."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve_context.return_value = mock_context
        mock_validate_cats.return_value = config

        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
        )
        trainer.train(wait=False)

        assert trainer._recipe_path == "fine-tuning/nova/nova_lite_2_0_datamix-abc"

    @patch(_PATCH_TRAIN_HYPERPOD, return_value="job-name")
    @patch(_PATCH_BUILD_HP_FROM_CONTEXT, return_value=("fine-tuning/nova/nova_lite_2_0_datamix-abc", "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-CPT-latest"))
    @patch(_PATCH_VALIDATE_CATEGORIES)
    @patch(_PATCH_RESOLVE_HP_CONTEXT)
    @patch(_PATCH_VALIDATE_DM_MODEL)
    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_training_image_set_from_image_uri_when_not_already_set(
        self, mock_resolve_model, mock_validate_group, mock_eula,
        mock_validate_dm_model, mock_resolve_context, mock_validate_cats,
        mock_build_from_context, mock_train_hp
    ):
        """Test self.training_image set from image_uri when not already set and not None."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve_context.return_value = mock_context
        mock_validate_cats.return_value = config

        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
            training_image=None,  # Not already set
        )
        trainer.train(wait=False)

        assert trainer.training_image == "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-CPT-latest"

    @patch(_PATCH_TRAIN_HYPERPOD, return_value="job-name")
    @patch(_PATCH_BUILD_HP_FROM_CONTEXT, return_value=("fine-tuning/nova/nova_lite_2_0_datamix-abc", "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-CPT-latest"))
    @patch(_PATCH_VALIDATE_CATEGORIES)
    @patch(_PATCH_RESOLVE_HP_CONTEXT)
    @patch(_PATCH_VALIDATE_DM_MODEL)
    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_training_image_not_overwritten_when_already_set(
        self, mock_resolve_model, mock_validate_group, mock_eula,
        mock_validate_dm_model, mock_resolve_context, mock_validate_cats,
        mock_build_from_context, mock_train_hp
    ):
        """Test self.training_image is NOT overwritten when already set by the user."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve_context.return_value = mock_context
        mock_validate_cats.return_value = config

        user_custom_image = "123456789.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:v1"
        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
            training_image=user_custom_image,  # User-provided image
        )
        trainer.train(wait=False)

        # training_image should remain the user-provided value, not overwritten by build result
        assert trainer.training_image == user_custom_image

    @patch(_PATCH_TRAIN_HYPERPOD, return_value="job-name")
    @patch(_PATCH_BUILD_HP_FROM_CONTEXT, return_value=("fine-tuning/nova/nova_lite_2_0_datamix-abc", None))
    @patch(_PATCH_VALIDATE_CATEGORIES)
    @patch(_PATCH_RESOLVE_HP_CONTEXT)
    @patch(_PATCH_VALIDATE_DM_MODEL)
    @patch(_PATCH_VALIDATE_EULA, return_value=False)
    @patch(_PATCH_VALIDATE_GROUP, return_value="test-group")
    @patch(_PATCH_RESOLVE_MODEL, return_value=("nova-textgeneration-lite-v2", "nova-textgeneration-lite-v2"))
    def test_training_image_not_set_when_image_uri_is_none(
        self, mock_resolve_model, mock_validate_group, mock_eula,
        mock_validate_dm_model, mock_resolve_context, mock_validate_cats,
        mock_build_from_context, mock_train_hp
    ):
        """Test self.training_image is NOT set when image_uri returned by build is None."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
        )

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve_context.return_value = mock_context
        mock_validate_cats.return_value = config

        trainer = CPTTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="test-group",
            compute=compute,
            training_dataset="s3://bucket/corpus.jsonl",
            data_mixing_config=config,
            training_image=None,
        )
        trainer.train(wait=False)

        # training_image should remain None since build returned None for image_uri
        assert trainer.training_image is None
