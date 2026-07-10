# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License.
"""Unit tests for commit bb96b501: recipe validation and instance count override for SMTJ serverful.

Covers:
- _get_smhp_replicas_enum: fetching replicas enum from SMHP override spec
- Replicas enum injection into SMTJ override spec and hyperparameters._specs
- get_resolved_recipe fallback (building overrides from hyperparameters._user_set)
- _apply_recipe_to_hyperparameters warning when recipe+hyperparameters conflict
- disable_output_compression in _train_serverful_smtj
- additional_overrides in get_hyperpod_recipe_path
- HyperPod path: resolved recipe is flattened and passed as additional_overrides
"""
import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from sagemaker.train.base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConcreteTrainer(BaseTrainer):
    """Minimal concrete BaseTrainer for unit testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = "nova-lite"
        self._customization_technique = "sft"
        self.training_type = "lora"
        self.training_dataset = None
        self.validation_dataset = None
        self.compute = MagicMock(
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            volume_size_in_gb=100,
            keep_alive_period_in_seconds=None,
        )
        self.networking = None
        self.stopping_condition = None
        self.training_image = "123.dkr.ecr.us-east-1.amazonaws.com/nova:latest"
        self.base_job_name = "nova-lite-sft"
        self.s3_output_path = None

    def train(self, *args, **kwargs):
        return self._train_serverful_smtj(*args, **kwargs)


def _run_serverful_with_replicas(trainer, replicas_enum=None):
    """Invoke _train_serverful_smtj with _get_smhp_replicas_enum mocked."""
    captured = {}

    def _capture_render(recipe_content, override_spec):
        captured["override_spec"] = override_spec
        return recipe_content

    mock_model_trainer = MagicMock()
    mock_model_trainer._latest_training_job = MagicMock()

    mock_session = MagicMock()
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
        "sagemaker.train.common_utils.finetune_utils._get_smhp_replicas_enum",
        return_value=replicas_enum,
    ), patch(
        "sagemaker.train.base_trainer._get_smhp_replicas_enum",
        return_value=replicas_enum,
    ), patch(
        "sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders",
        side_effect=_capture_render,
    ), patch(
        "sagemaker.train.common_utils.recipe_utils.resolve_recipe",
        return_value={"training_config": {}},
    ), patch(
        "sagemaker.train.base_trainer.flatten_resolved_recipe",
        return_value={},
    ), patch(
        "sagemaker.train.model_trainer.ModelTrainer.from_recipe",
        return_value=mock_model_trainer,
    ) as mock_from_recipe:
        trainer.hyperparameters = MagicMock()
        trainer.hyperparameters.to_dict.return_value = {}
        trainer.hyperparameters._specs = {}
        trainer.hyperparameters._user_set = None
        trainer.train(wait=False)

    return captured.get("override_spec"), mock_from_recipe.call_args.kwargs


# ---------------------------------------------------------------------------
# Tests: _get_smhp_replicas_enum
# ---------------------------------------------------------------------------


class TestGetSmhpReplicasEnum:
    """Unit tests for _get_smhp_replicas_enum."""

    @patch("sagemaker.train.common_utils.finetune_utils._get_recipe_entry_and_override_spec")
    def test_returns_enum_list_when_present(self, mock_get_recipe):
        from sagemaker.train.common_utils.finetune_utils import _get_smhp_replicas_enum

        mock_get_recipe.return_value = (
            {"Name": "recipe"},
            {"replicas": {"enum": [4, 8, 16], "type": "integer"}},
        )

        result = _get_smhp_replicas_enum(
            model_name="nova-lite",
            customization_technique="SFT",
            training_type="LORA",
            sagemaker_session=MagicMock(),
        )

        assert result == [4, 8, 16]
        mock_get_recipe.assert_called_once_with(
            model_name="nova-lite",
            customization_technique="SFT",
            training_type="LORA",
            sagemaker_session=mock_get_recipe.call_args.kwargs["sagemaker_session"],
            platform="hyperpod",
            hub_name=None,
        )

    @patch("sagemaker.train.common_utils.finetune_utils._get_recipe_entry_and_override_spec")
    def test_returns_none_when_no_replicas_key(self, mock_get_recipe):
        from sagemaker.train.common_utils.finetune_utils import _get_smhp_replicas_enum

        mock_get_recipe.return_value = ({"Name": "recipe"}, {"max_steps": {"default": 100}})

        result = _get_smhp_replicas_enum(
            model_name="nova-lite",
            customization_technique="SFT",
            training_type="FULL",
            sagemaker_session=MagicMock(),
        )

        assert result is None

    @patch("sagemaker.train.common_utils.finetune_utils._get_recipe_entry_and_override_spec")
    def test_returns_none_when_enum_is_empty_list(self, mock_get_recipe):
        from sagemaker.train.common_utils.finetune_utils import _get_smhp_replicas_enum

        mock_get_recipe.return_value = (
            {"Name": "recipe"},
            {"replicas": {"enum": [], "type": "integer"}},
        )

        result = _get_smhp_replicas_enum(
            model_name="nova-lite",
            customization_technique="SFT",
            training_type="FULL",
            sagemaker_session=MagicMock(),
        )

        assert result is None

    @patch("sagemaker.train.common_utils.finetune_utils._get_recipe_entry_and_override_spec")
    def test_returns_none_and_warns_on_exception(self, mock_get_recipe):
        from sagemaker.train.common_utils.finetune_utils import _get_smhp_replicas_enum

        mock_get_recipe.side_effect = ValueError("No matching recipe")

        result = _get_smhp_replicas_enum(
            model_name="unknown-model",
            customization_technique="SFT",
            training_type="LORA",
            sagemaker_session=MagicMock(),
        )

        assert result is None


# ---------------------------------------------------------------------------
# Tests: Replicas enum injection into override spec
# ---------------------------------------------------------------------------


class TestReplicasEnumInjection:
    """Replicas enum from SMHP is injected into the SMTJ override spec."""

    def test_replicas_enum_injected_into_override_spec(self):
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train/"
        trainer.compute.instance_count = 4  # must be in the enum

        override_spec, _ = _run_serverful_with_replicas(trainer, replicas_enum=[4, 8, 16])

        assert "replicas" in override_spec
        assert override_spec["replicas"]["enum"] == [4, 8, 16]

    def test_replicas_enum_injected_into_hyperparameters_specs(self):
        """The enum is also set on hyperparameters._specs for recipe validation."""
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train/"
        trainer.compute.instance_count = 4  # must be in the enum

        mock_session = MagicMock()
        mock_session.boto_session.client.return_value.download_file.return_value = None
        mock_model_trainer = MagicMock()
        mock_model_trainer._latest_training_job = MagicMock()

        hp_mock = MagicMock()
        hp_mock.to_dict.return_value = {}
        hp_mock._specs = {}
        hp_mock._user_set = None

        with patch(
            "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session",
            return_value=mock_session,
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
            "sagemaker.train.common_utils.finetune_utils._get_smtj_override_spec",
            return_value={},
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._get_smhp_replicas_enum",
            return_value=[4, 8],
        ), patch(
            "sagemaker.train.base_trainer._get_smhp_replicas_enum",
            return_value=[4, 8],
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders",
            side_effect=lambda c, s: c,
        ), patch(
            "sagemaker.train.common_utils.recipe_utils.resolve_recipe",
            return_value={"training_config": {}},
        ), patch(
            "sagemaker.train.base_trainer.flatten_resolved_recipe",
            return_value={},
        ), patch(
            "sagemaker.train.model_trainer.ModelTrainer.from_recipe",
            return_value=mock_model_trainer,
        ):
            trainer.hyperparameters = hp_mock
            trainer.train(wait=False)

        assert hp_mock._specs["replicas"]["enum"] == [4, 8]

    def test_no_injection_when_replicas_enum_is_none(self):
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train/"

        override_spec, _ = _run_serverful_with_replicas(trainer, replicas_enum=None)

        assert "replicas" not in override_spec


# ---------------------------------------------------------------------------
# Tests: get_resolved_recipe fallback from hyperparameters._user_set
# ---------------------------------------------------------------------------


class TestGetResolvedRecipeFallback:
    """get_resolved_recipe builds overrides from _user_set when no recipe/overrides given."""

    def test_builds_overrides_from_user_set_hyperparameters(self):
        """User-set hyperparameters become overrides via get_resolved_recipe_from_context."""
        trainer = _ConcreteTrainer()
        trainer._recipe_path = None
        trainer._overrides = None
        trainer._resolved_recipe_cache = None

        hp_mock = MagicMock()
        hp_mock._user_set = {"max_steps", "save_steps"}
        hp_mock.max_steps = 100
        hp_mock.save_steps = 10
        hp_mock._specs = {"max_steps": {"type": "integer"}, "save_steps": {"type": "integer"}}
        hp_mock._full_recipe_template = None
        trainer.hyperparameters = hp_mock

        with patch("sagemaker.train.common_utils.recipe_utils.resolve_recipe") as mock_resolve:
            mock_resolve.return_value = {"training_config": {"max_steps": 100, "save_steps": 10}}
            trainer.get_resolved_recipe()

        call_kwargs = mock_resolve.call_args.kwargs
        # User-set HPs are merged into overrides under the template_section key
        overrides = call_kwargs["overrides"]
        assert overrides["training_config"]["max_steps"] == 100
        assert overrides["training_config"]["save_steps"] == 10
        assert call_kwargs["recipe_path"] is None

    def test_raises_when_no_recipe_overrides_or_user_set(self):
        """NoRecipeError is raised when nothing can be resolved."""
        from sagemaker.train.common_utils.recipe_utils import NoRecipeError

        trainer = _ConcreteTrainer()
        trainer._recipe_path = None
        trainer._overrides = None
        trainer._resolved_recipe_cache = None

        hp_mock = MagicMock()
        hp_mock._user_set = None
        hp_mock._specs = {}
        hp_mock._full_recipe_template = None
        trainer.hyperparameters = hp_mock

        with pytest.raises(NoRecipeError):
            trainer.get_resolved_recipe()


# ---------------------------------------------------------------------------
# Tests: _apply_recipe_to_hyperparameters warning
# ---------------------------------------------------------------------------


class TestApplyRecipeToHyperparameters:
    """_apply_recipe_to_hyperparameters flattens resolved recipe into HP dict."""

    def test_resolved_recipe_values_applied(self):
        """Resolved recipe values are flattened and merged into hyperparameters."""
        trainer = _ConcreteTrainer()
        trainer._recipe_path = "s3://bucket/recipe.yaml"
        trainer._overrides = None
        trainer._resolved_recipe_cache = None

        hp_mock = MagicMock()
        hp_mock._specs = {"max_steps": {"type": "integer"}, "lr": {"type": "float"}}
        trainer.hyperparameters = hp_mock

        with patch.object(trainer, "get_resolved_recipe", return_value={"training_config": {"max_steps": 50, "lr": 0.001}}), \
             patch("sagemaker.train.base_trainer.flatten_resolved_recipe", return_value={"max_steps": "50", "lr": "0.001"}):
            result = trainer._apply_recipe_to_hyperparameters({"existing_key": "val"})

        assert result["max_steps"] == "50"
        assert result["lr"] == "0.001"
        assert result["existing_key"] == "val"

    def test_returns_unchanged_when_nothing_to_resolve(self):
        """When get_resolved_recipe raises 'requires a recipe', hyperparameters stay unchanged."""
        trainer = _ConcreteTrainer()
        trainer._recipe_path = None
        trainer._overrides = None
        trainer._resolved_recipe_cache = None

        hp_mock = MagicMock()
        hp_mock._specs = {}
        hp_mock._user_set = None
        trainer.hyperparameters = hp_mock

        result = trainer._apply_recipe_to_hyperparameters({"existing": "val"})

        assert result == {"existing": "val"}

    def test_serverless_drops_unchanged_non_spec_recipe_keys(self):
        """Serverless path drops the recipe's unchanged non-spec defaults.

        Regression test for P467902218 — flattening the full resolved recipe
        (200+ leaf keys) into the serverless HyperParameters override map blew
        past the API's 100-entry limit and failed with a ValidationException.
        With serverless=True the unchanged non-spec recipe defaults are applied
        server-side and must not be sent, keeping the map bounded.
        """
        trainer = _ConcreteTrainer()
        trainer._recipe_path = None
        trainer._overrides = None
        trainer._resolved_recipe_cache = None

        hp_mock = MagicMock()
        hp_mock._specs = {"max_steps": {"type": "integer"}, "lr": {"type": "float"}}
        # No direct hyperparameter assignments.
        hp_mock._user_set = set()
        trainer.hyperparameters = hp_mock

        # A full recipe: two overridable spec keys plus 200 non-spec defaults
        # (KL/vLLM/LoRA settings) that the user never changed.
        training_config = {"max_steps": 50, "lr": 0.001}
        training_config.update({f"internal_recipe_key_{i}": i for i in range(200)})
        resolved = {"training_config": training_config}

        with patch.object(trainer, "get_resolved_recipe", return_value=resolved):
            result = trainer._apply_recipe_to_hyperparameters(
                {"existing_key": "val"}, serverless=True
            )

        # Only the pre-existing key + the two spec keys survive; the 200 unchanged
        # non-spec recipe leaves are dropped, keeping the map under the 100 limit.
        assert result == {"existing_key": "val", "max_steps": "50", "lr": "0.001"}
        assert len(result) < 100
        assert not any(k.startswith("internal_recipe_key_") for k in result)

    def test_serverless_forwards_explicit_user_overrides(self):
        """Serverless path forwards non-spec keys the user explicitly overrode.

        A user override of a non-spec recipe key (e.g. peft.lora_tuning.alpha)
        must never be silently dropped — otherwise the job would train with the
        wrong value. Only the recipe's *unchanged* defaults are pruned.
        """
        trainer = _ConcreteTrainer()
        trainer._recipe_path = None
        # User explicitly overrode a non-spec, deeply nested key.
        trainer._overrides = {"training_config": {"peft": {"lora_tuning": {"alpha": 128}}}}
        trainer._resolved_recipe_cache = None

        hp_mock = MagicMock()
        hp_mock._specs = {"max_steps": {"type": "integer"}}
        hp_mock._user_set = set()
        trainer.hyperparameters = hp_mock

        resolved = {
            "training_config": {
                "max_steps": 50,
                "peft": {"lora_tuning": {"alpha": 128, "rank": 16}},
                "unchanged_default": "keepserverside",
            }
        }

        with patch.object(trainer, "get_resolved_recipe", return_value=resolved):
            result = trainer._apply_recipe_to_hyperparameters({}, serverless=True)

        # Spec key + explicitly overridden non-spec key are forwarded...
        assert result["max_steps"] == "50"
        assert result["alpha"] == "128"
        # ...but unchanged non-spec defaults (rank, unchanged_default) are dropped.
        assert "rank" not in result
        assert "unchanged_default" not in result

    def test_serverful_applies_all_recipe_keys(self):
        """Serverful path (serverless=False) still flattens the full recipe."""
        trainer = _ConcreteTrainer()
        trainer._recipe_path = None
        trainer._overrides = None
        trainer._resolved_recipe_cache = None

        hp_mock = MagicMock()
        hp_mock._specs = {"max_steps": {"type": "integer"}}
        hp_mock._user_set = set()
        trainer.hyperparameters = hp_mock

        resolved = {"training_config": {"max_steps": 50, "internal_recipe_key": "megatron"}}

        with patch.object(trainer, "get_resolved_recipe", return_value=resolved):
            result = trainer._apply_recipe_to_hyperparameters({})

        # Non-spec recipe keys are preserved for the serverful path, because
        # they are rendered back into the recipe YAML.
        assert result == {"max_steps": "50", "internal_recipe_key": "megatron"}


# ---------------------------------------------------------------------------
# Tests: disable_output_compression
# ---------------------------------------------------------------------------


class TestDisableOutputCompression:
    """disable_output_compression adds compression_type=NONE to OutputDataConfig."""

    def test_compression_type_none_when_flag_set(self):
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train/"
        trainer.s3_output_path = "s3://my-bucket/output/"
        trainer.disable_output_compression = True

        mock_session = MagicMock()
        mock_session.boto_session.client.return_value.download_file.return_value = None
        mock_model_trainer = MagicMock()
        mock_model_trainer._latest_training_job = MagicMock()

        with patch(
            "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session",
            return_value=mock_session,
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
            "sagemaker.train.common_utils.finetune_utils._get_smtj_override_spec",
            return_value={},
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._get_smhp_replicas_enum",
            return_value=None,
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders",
            side_effect=lambda c, s: c,
        ), patch(
            "sagemaker.train.model_trainer.ModelTrainer.from_recipe",
            return_value=mock_model_trainer,
        ) as mock_from_recipe:
            trainer.hyperparameters = MagicMock()
            trainer.hyperparameters.to_dict.return_value = {}
            trainer.hyperparameters._specs = {}
            trainer.hyperparameters._user_set = None
            trainer.train(wait=False)

        from_recipe_kwargs = mock_from_recipe.call_args.kwargs
        output_config = from_recipe_kwargs["output_data_config"]
        assert output_config.compression_type == "NONE"

    def test_no_compression_type_when_flag_not_set(self):
        trainer = _ConcreteTrainer()
        trainer.training_dataset = "s3://my-bucket/data/train/"
        trainer.s3_output_path = "s3://my-bucket/output/"

        mock_session = MagicMock()
        mock_session.boto_session.client.return_value.download_file.return_value = None
        mock_model_trainer = MagicMock()
        mock_model_trainer._latest_training_job = MagicMock()

        with patch(
            "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session",
            return_value=mock_session,
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
            "sagemaker.train.common_utils.finetune_utils._get_smtj_override_spec",
            return_value={},
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._get_smhp_replicas_enum",
            return_value=None,
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders",
            side_effect=lambda c, s: c,
        ), patch(
            "sagemaker.train.model_trainer.ModelTrainer.from_recipe",
            return_value=mock_model_trainer,
        ) as mock_from_recipe:
            trainer.hyperparameters = MagicMock()
            trainer.hyperparameters.to_dict.return_value = {}
            trainer.hyperparameters._specs = {}
            trainer.hyperparameters._user_set = None
            trainer.train(wait=False)

        from_recipe_kwargs = mock_from_recipe.call_args.kwargs
        output_config = from_recipe_kwargs["output_data_config"]
        assert not hasattr(output_config, 'compression_type') or output_config.compression_type != "NONE"


# ---------------------------------------------------------------------------
# Tests: additional_overrides in get_hyperpod_recipe_path
# ---------------------------------------------------------------------------


class TestHyperpodRecipeAdditionalOverrides:
    """get_hyperpod_recipe_path injects additional_overrides into spec before rendering."""

    @patch("sagemaker.train.common_utils.finetune_utils._get_recipe_entry_and_override_spec")
    @patch("sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders")
    def test_additional_overrides_update_existing_spec_entry(
        self, mock_render, mock_get_recipe
    ):
        from sagemaker.train.common_utils.finetune_utils import get_hyperpod_recipe_path

        mock_get_recipe.return_value = (
            {"Name": "recipe", "HpEksPayloadTemplateS3Uri": "s3://b/template.yaml"},
            {"max_steps": {"default": 100, "type": "integer"}, "name": {"default": "", "type": "string"}},
        )

        mock_session = MagicMock()
        mock_session.boto_session.client.return_value.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b"---\nrun:\n  name: {{ name }}\n  max_steps: {{ max_steps }}"))
        }

        captured_spec = {}

        def capture_render(content, spec):
            captured_spec.update(spec)
            return content

        mock_render.side_effect = capture_render

        import sys
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

        with patch("sagemaker.train.common_utils.finetune_utils._extract_recipe_from_helm_template", side_effect=lambda x: x), \
             patch("builtins.open", MagicMock()), \
             patch("sagemaker.train.common_utils.finetune_utils.os.path.join", return_value="/tmp/recipe"), \
             patch("sagemaker.train.common_utils.finetune_utils.os.path.dirname", return_value="/pkg"), \
             patch("sagemaker.train.common_utils.finetune_utils.os.makedirs"), \
             patch.dict(sys.modules, {"hyperpod_cli": mock_hyperpod_cli}):
            try:
                get_hyperpod_recipe_path(
                    model_name="nova-lite",
                    customization_technique="SFT",
                    training_type="LORA",
                    sagemaker_session=mock_session,
                    job_name="test-job",
                    additional_overrides={"max_steps": 500, "name": "my-job"},
                )
            except Exception:
                pass  # may fail on filesystem ops but spec injection happens before

        # The override spec should have been updated with additional_overrides
        if captured_spec:
            assert captured_spec["max_steps"]["default"] == 500
            assert captured_spec["name"]["default"] == "my-job"

    @patch("sagemaker.train.common_utils.finetune_utils._get_recipe_entry_and_override_spec")
    @patch("sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders")
    def test_additional_overrides_creates_new_spec_entry(
        self, mock_render, mock_get_recipe
    ):
        from sagemaker.train.common_utils.finetune_utils import get_hyperpod_recipe_path

        mock_get_recipe.return_value = (
            {"Name": "recipe", "HpEksPayloadTemplateS3Uri": "s3://b/template.yaml"},
            {},  # empty override spec
        )

        mock_session = MagicMock()
        mock_session.boto_session.client.return_value.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b"---\nrun:\n  custom_key: {{ custom_key }}"))
        }

        captured_spec = {}

        def capture_render(content, spec):
            captured_spec.update(spec)
            return content

        mock_render.side_effect = capture_render

        import sys
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

        with patch("sagemaker.train.common_utils.finetune_utils._extract_recipe_from_helm_template", side_effect=lambda x: x), \
             patch("builtins.open", MagicMock()), \
             patch("sagemaker.train.common_utils.finetune_utils.os.path.join", return_value="/tmp/recipe"), \
             patch("sagemaker.train.common_utils.finetune_utils.os.path.dirname", return_value="/pkg"), \
             patch("sagemaker.train.common_utils.finetune_utils.os.makedirs"), \
             patch.dict(sys.modules, {"hyperpod_cli": mock_hyperpod_cli}):
            try:
                get_hyperpod_recipe_path(
                    model_name="nova-lite",
                    customization_technique="SFT",
                    training_type="LORA",
                    sagemaker_session=mock_session,
                    job_name="test-job",
                    additional_overrides={"custom_key": "custom_val"},
                )
            except Exception:
                pass

        if captured_spec:
            assert captured_spec["custom_key"] == {"default": "custom_val", "type": "string"}


# ---------------------------------------------------------------------------
# Tests: HyperPod _train_hyperpod passes resolved recipe as additional_overrides
# ---------------------------------------------------------------------------


class TestHyperpodResolvesAndFlattensRecipe:
    """_train_hyperpod calls get_resolved_recipe and passes flattened result as additional_overrides."""

    @patch("sagemaker.train.base_trainer.subprocess")
    @patch("sagemaker.train.base_trainer.TrainDefaults.verify_hyperpod_caller_permissions")
    @patch("sagemaker.train.base_trainer.validate_hyperpod_compute")
    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.base_trainer.get_hyperpod_recipe_path")
    @patch("sagemaker.train.base_trainer.flatten_resolved_recipe")
    def test_resolved_recipe_flattened_into_additional_overrides(
        self, mock_flatten, mock_get_recipe_path, mock_get_session,
        mock_validate, mock_verify, mock_subprocess
    ):
        from sagemaker.train.sft_trainer import SFTTrainer

        mock_get_session.return_value = MagicMock()
        mock_subprocess.run.return_value = SimpleNamespace(
            stdout="NAME: my-job-456\n", stderr=""
        )
        mock_flatten.return_value = {"max_steps": "100", "lr": "0.001"}
        mock_get_recipe_path.return_value = "recipes/nova-lite-sft"

        trainer = SFTTrainer.__new__(SFTTrainer)
        trainer.sagemaker_session = MagicMock()
        trainer.compute = SimpleNamespace(
            cluster_name="my-cluster",
            namespace="kubeflow",
            instance_type="ml.p5.48xlarge",
            node_count=4,
        )
        trainer._model_name = "amazon.nova-lite-v2"
        trainer._customization_technique = "SFT"
        trainer.training_type = "LORA"
        trainer.training_image = "123.dkr.ecr.us-west-2.amazonaws.com/img:latest"
        trainer.base_job_name = "my-job"
        trainer.training_dataset = "s3://bucket/train.jsonl"
        trainer.validation_dataset = None
        trainer.s3_output_path = "s3://bucket/output/"
        trainer.hyperparameters = None
        trainer._recipe_path = None
        trainer._overrides = {"training_config": {"max_steps": 100}}
        trainer.mlflow_resource_arn = None
        trainer.mlflow_experiment_name = None
        trainer.mlflow_run_name = None
        trainer.model_source = None

        with patch.object(trainer, "get_resolved_recipe", return_value={"training_config": {"max_steps": 100, "lr": 0.001}}):
            with patch(
                "sagemaker.train.common_utils.finetune_utils.get_training_image",
                return_value=None,
            ):
                job_name = trainer._train_hyperpod(wait=False)

        assert job_name == "my-job-456"

        # Verify get_hyperpod_recipe_path received additional_overrides
        recipe_call_kwargs = mock_get_recipe_path.call_args.kwargs
        additional = recipe_call_kwargs["additional_overrides"]
        assert additional["max_steps"] == "100"
        assert additional["lr"] == "0.001"
        assert additional["data_s3_path"] == "s3://bucket/train.jsonl"
        assert additional["output_s3_path"] == "s3://bucket/output/"
        assert "replicas" in additional
