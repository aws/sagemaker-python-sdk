# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License.
"""Unit tests for recipe_resolver module."""
import os
import tempfile

import pytest
import yaml

from sagemaker.train.recipe_resolver import (
    RecipeResolver,
    render_template,
    flatten_resolved_recipe,
    _load_user_recipe,
    _validate_value,
    _get_nested_value,
)


# --- render_template tests ---


class TestRenderTemplate:
    def test_basic_placeholder_rendering(self):
        template = {
            "training_config": {
                "learning_rate": "'{{learning_rate}}'",
                "num_epochs": "'{{num_epochs}}'",
            },
            "run": {
                "name": "'{{name}}'",
            },
        }
        spec = {
            "learning_rate": {"default": 1e-5, "type": "float"},
            "num_epochs": {"default": 3, "type": "integer"},
            "name": {"default": "experiment-1", "type": "string"},
        }

        rendered, key_path_map = render_template(template, spec)

        assert rendered["training_config"]["learning_rate"] == 1e-5
        assert rendered["training_config"]["num_epochs"] == 3
        assert rendered["run"]["name"] == "experiment-1"
        assert key_path_map["learning_rate"] == "training_config.learning_rate"
        assert key_path_map["num_epochs"] == "training_config.num_epochs"
        assert key_path_map["name"] == "run.name"

    def test_non_placeholder_values_preserved(self):
        template = {
            "optimizer": {
                "name": "distributed_fused_adam",
                "lr": "'{{learning_rate}}'",
            }
        }
        spec = {"learning_rate": {"default": 1e-4}}

        rendered, key_path_map = render_template(template, spec)

        assert rendered["optimizer"]["name"] == "distributed_fused_adam"
        assert rendered["optimizer"]["lr"] == 1e-4
        assert "learning_rate" in key_path_map

    def test_missing_spec_key_returns_none(self):
        template = {"config": {"unknown": "'{{missing_key}}'"}}
        spec = {}

        rendered, key_path_map = render_template(template, spec)

        assert rendered["config"]["unknown"] is None
        assert key_path_map["missing_key"] == "config.unknown"

    def test_nested_dicts(self):
        template = {
            "level1": {
                "level2": {
                    "param": "'{{deep_param}}'",
                }
            }
        }
        spec = {"deep_param": {"default": 42, "type": "integer"}}

        rendered, key_path_map = render_template(template, spec)

        assert rendered["level1"]["level2"]["param"] == 42
        assert key_path_map["deep_param"] == "level1.level2.param"


# --- _load_user_recipe tests ---


class TestLoadUserRecipe:
    def test_load_local_yaml(self, tmp_path):
        recipe_content = {"training_config": {"learning_rate": 2e-5, "batch_size": 4}}
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text(yaml.dump(recipe_content))

        result = _load_user_recipe(str(recipe_file))

        assert result == recipe_content

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(ValueError, match="Recipe file not found"):
            _load_user_recipe("/nonexistent/path/recipe.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        recipe_file = tmp_path / "bad.yaml"
        recipe_file.write_text("not a mapping: [")

        with pytest.raises((ValueError, yaml.YAMLError)):
            _load_user_recipe(str(recipe_file))

    def test_load_non_dict_yaml_raises(self, tmp_path):
        recipe_file = tmp_path / "list.yaml"
        recipe_file.write_text("- item1\n- item2\n")

        with pytest.raises(ValueError, match="did not parse as a YAML mapping"):
            _load_user_recipe(str(recipe_file))


# --- _validate_value tests ---


class TestValidateValue:
    def test_valid_float(self):
        _validate_value("lr", 0.001, {"type": "float", "min": 1e-7, "max": 1.0}, "test")

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="expected float"):
            _validate_value("lr", "not_a_float", {"type": "float"}, "test")

    def test_below_min(self):
        with pytest.raises(ValueError, match="below minimum"):
            _validate_value("lr", 1e-8, {"type": "float", "min": 1e-7, "max": 1.0}, "test")

    def test_above_max(self):
        with pytest.raises(ValueError, match="above maximum"):
            _validate_value("epochs", 200, {"type": "integer", "min": 1, "max": 100}, "test")

    def test_invalid_enum(self):
        with pytest.raises(ValueError, match="not in allowed values"):
            _validate_value("mode", "invalid", {"type": "string", "enum": ["full", "lora"]}, "test")

    def test_valid_enum(self):
        _validate_value("mode", "full", {"type": "string", "enum": ["full", "lora"]}, "test")

    def test_none_value_passes(self):
        _validate_value("param", None, {"type": "float", "min": 0, "max": 1}, "test")

    def test_int_accepted_for_float_type(self):
        _validate_value("lr", 1, {"type": "float", "min": 0, "max": 10}, "test")


# --- RecipeResolver tests ---


class TestRecipeResolver:
    def _make_template(self):
        return {
            "training_config": {
                "learning_rate": "'{{learning_rate}}'",
                "num_epochs": "'{{num_epochs}}'",
                "batch_size": "'{{batch_size}}'",
            },
            "run": {
                "model_type": "'{{model_type}}'",
                "name": "'{{name}}'",
            },
        }

    def _make_spec(self):
        return {
            "learning_rate": {"default": 1e-5, "type": "float", "min": 1e-7, "max": 1.0},
            "num_epochs": {"default": 3, "type": "integer", "min": 1, "max": 100},
            "batch_size": {"default": 1, "type": "integer", "min": 1, "max": 64},
            "model_type": {"default": "amazon.nova-pro", "type": "string"},
            "name": {"default": "default-job", "type": "string"},
        }

    def test_resolve_base_only(self):
        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
        )

        result = resolver.resolve()

        assert result["training_config"]["learning_rate"] == 1e-5
        assert result["training_config"]["num_epochs"] == 3
        assert result["run"]["model_type"] == "amazon.nova-pro"

    def test_resolve_with_overrides(self):
        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
            overrides={"training_config": {"learning_rate": 2e-5, "num_epochs": 5}},
        )

        result = resolver.resolve()

        assert result["training_config"]["learning_rate"] == 2e-5
        assert result["training_config"]["num_epochs"] == 5
        assert result["training_config"]["batch_size"] == 1  # default preserved

    def test_resolve_with_user_recipe(self, tmp_path):
        user_recipe = {
            "training_config": {"learning_rate": 3e-5, "batch_size": 8},
            "run": {"name": "my-experiment"},
        }
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text(yaml.dump(user_recipe))

        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
            user_recipe_path=str(recipe_file),
        )

        result = resolver.resolve()

        assert result["training_config"]["learning_rate"] == 3e-5
        assert result["training_config"]["batch_size"] == 8
        assert result["run"]["name"] == "my-experiment"
        # Defaults for unset params
        assert result["training_config"]["num_epochs"] == 3

    def test_overrides_beat_user_recipe(self, tmp_path):
        user_recipe = {"training_config": {"learning_rate": 3e-5}}
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text(yaml.dump(user_recipe))

        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
            user_recipe_path=str(recipe_file),
            overrides={"training_config": {"learning_rate": 5e-5}},
        )

        result = resolver.resolve()

        # Overrides win over user recipe
        assert result["training_config"]["learning_rate"] == 5e-5

    def test_protected_keys_stripped(self, tmp_path):
        user_recipe = {
            "training_config": {"learning_rate": 2e-5},
            "run": {"model_type": "hacked-model"},
        }
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text(yaml.dump(user_recipe))

        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
            user_recipe_path=str(recipe_file),
            protected_keys={"model_type"},
        )

        result = resolver.resolve()

        # model_type should remain the default, not "hacked-model"
        assert result["run"]["model_type"] == "amazon.nova-pro"
        # Non-protected key still gets user recipe value
        assert result["training_config"]["learning_rate"] == 2e-5

    def test_validation_fails_for_out_of_range(self):
        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
            overrides={"training_config": {"learning_rate": 5.0}},
        )

        with pytest.raises(ValueError, match="above maximum"):
            resolver.resolve()

    def test_validation_fails_for_wrong_type(self):
        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
            overrides={"training_config": {"num_epochs": "not_an_int"}},
        )

        with pytest.raises(ValueError, match="expected integer"):
            resolver.resolve()

    def test_resolve_is_idempotent(self):
        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
        )

        result1 = resolver.resolve()
        result2 = resolver.resolve()

        assert result1 == result2

    def test_resolve_returns_deep_copy(self):
        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
        )

        result1 = resolver.resolve()
        result1["training_config"]["learning_rate"] = 999

        result2 = resolver.resolve()
        assert result2["training_config"]["learning_rate"] == 1e-5

    def test_inputs_deep_copied_at_construction(self):
        template = self._make_template()
        spec = self._make_spec()
        overrides = {"training_config": {"learning_rate": 2e-5}}

        resolver = RecipeResolver(
            recipe_template=template,
            override_spec=spec,
            overrides=overrides,
        )

        # Mutate originals after construction
        overrides["training_config"]["learning_rate"] = 9e-1
        spec["learning_rate"]["max"] = 0.001

        result = resolver.resolve()

        # Should use the values from construction time
        assert result["training_config"]["learning_rate"] == 2e-5

    def test_get_resolved_recipe_same_as_resolve(self):
        resolver = RecipeResolver(
            recipe_template=self._make_template(),
            override_spec=self._make_spec(),
        )

        assert resolver.get_resolved_recipe() == resolver.resolve()


# --- Full Recipe Template tests ---


class TestFullRecipeTemplate:
    """Tests for RecipeResolver with full_recipe_template."""

    def _make_full_template(self):
        return {
            "training_config": {
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "batch_size": 32,
                "sequence_length": 4096,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
            }
        }

    def _make_spec(self):
        return {
            "learning_rate": {"default": 1e-4, "type": "float", "min": 1e-7, "max": 1.0},
            "num_epochs": {"default": 10, "type": "integer", "min": 1, "max": 100},
            "batch_size": {"default": 32, "type": "integer", "min": 1, "max": 64},
        }

    def test_full_template_as_base_layer(self):
        """Full recipe template provides all keys as base defaults."""
        resolver = RecipeResolver(
            recipe_template={"training_config": {}},
            override_spec=self._make_spec(),
            full_recipe_template=self._make_full_template(),
            overrides={"training_config": {"sequence_length": 8192}},
        )

        result = resolver.resolve()

        # Override applied
        assert result["training_config"]["sequence_length"] == 8192
        # Non-spec keys retain full template defaults
        assert result["training_config"]["warmup_ratio"] == 0.1
        assert result["training_config"]["weight_decay"] == 0.01
        # Spec keys retain defaults
        assert result["training_config"]["learning_rate"] == 1e-4

    def test_full_template_with_user_recipe(self, tmp_path):
        """User recipe overrides full template base values."""
        recipe_content = {"training_config": {"warmup_ratio": 0.2, "learning_rate": 5e-5}}
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text(yaml.dump(recipe_content))

        resolver = RecipeResolver(
            recipe_template={"training_config": {}},
            override_spec=self._make_spec(),
            full_recipe_template=self._make_full_template(),
            user_recipe_path=str(recipe_file),
        )

        result = resolver.resolve()

        # User recipe overrides
        assert result["training_config"]["warmup_ratio"] == 0.2
        assert result["training_config"]["learning_rate"] == 5e-5
        # Full template defaults where no user recipe value
        assert result["training_config"]["sequence_length"] == 4096
        assert result["training_config"]["weight_decay"] == 0.01

    def test_full_template_three_level_merge(self, tmp_path):
        """Precedence: overrides > user recipe > full template."""
        recipe_content = {"training_config": {"warmup_ratio": 0.2, "learning_rate": 5e-5}}
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text(yaml.dump(recipe_content))

        resolver = RecipeResolver(
            recipe_template={"training_config": {}},
            override_spec=self._make_spec(),
            full_recipe_template=self._make_full_template(),
            user_recipe_path=str(recipe_file),
            overrides={"training_config": {"learning_rate": 2e-5}},
        )

        result = resolver.resolve()

        # Overrides win
        assert result["training_config"]["learning_rate"] == 2e-5
        # User recipe wins over template
        assert result["training_config"]["warmup_ratio"] == 0.2
        # Template default
        assert result["training_config"]["sequence_length"] == 4096

    def test_spec_validation_with_full_template(self):
        """Spec validation still applies for spec keys."""
        resolver = RecipeResolver(
            recipe_template={"training_config": {}},
            override_spec=self._make_spec(),
            full_recipe_template=self._make_full_template(),
            overrides={"training_config": {"num_epochs": 200}},
        )

        with pytest.raises(ValueError, match="above maximum"):
            resolver.resolve()

    def test_non_spec_keys_not_validated(self):
        """Non-spec keys are not subject to spec validation."""
        resolver = RecipeResolver(
            recipe_template={"training_config": {}},
            override_spec=self._make_spec(),
            full_recipe_template=self._make_full_template(),
            overrides={"training_config": {"warmup_ratio": 999.0}},
        )

        # Should not raise — warmup_ratio is not in the spec
        result = resolver.resolve()
        assert result["training_config"]["warmup_ratio"] == 999.0

    def test_protected_keys_still_enforced(self):
        """Protected keys cannot be overridden even with full template."""
        full_template = self._make_full_template()
        full_template["training_config"]["model_type"] = "original"

        resolver = RecipeResolver(
            recipe_template={"training_config": {}},
            override_spec=self._make_spec(),
            full_recipe_template=full_template,
            overrides={"training_config": {"model_type": "hacked"}},
            protected_keys={"model_type"},
        )

        result = resolver.resolve()
        assert result["training_config"]["model_type"] == "original"


class TestBuildKeyPathMap:
    """Tests for _build_key_path_map helper."""

    def test_finds_spec_keys_in_nested_dict(self):
        from sagemaker.train.recipe_resolver import _build_key_path_map

        recipe = {
            "training_config": {
                "learning_rate": 1e-4,
                "nested": {"batch_size": 32},
            }
        }
        spec_keys = {"learning_rate", "batch_size"}

        result = _build_key_path_map(recipe, spec_keys)

        assert result["learning_rate"] == "training_config.learning_rate"
        assert result["batch_size"] == "training_config.nested.batch_size"

    def test_missing_spec_keys_not_in_map(self):
        from sagemaker.train.recipe_resolver import _build_key_path_map

        recipe = {"training_config": {"learning_rate": 1e-4}}
        spec_keys = {"learning_rate", "nonexistent_key"}

        result = _build_key_path_map(recipe, spec_keys)

        assert "learning_rate" in result
        assert "nonexistent_key" not in result


# --- _get_nested_value tests ---


class TestFlattenResolvedRecipe:
    """Tests for flatten_resolved_recipe deep flattening."""

    def test_flat_structure(self):
        """Single-level section is flattened to leaf keys."""
        resolved = {
            "training_config": {
                "learning_rate": 1e-4,
                "num_epochs": 10,
            }
        }
        flat = flatten_resolved_recipe(resolved)
        assert flat == {"learning_rate": 1e-4, "num_epochs": 10}

    def test_nested_structure_flattened_to_leaves(self):
        """Nested dicts are recursively walked to scalar leaves."""
        resolved = {
            "training_config": {
                "max_steps": 100,
                "lr_scheduler": {
                    "warmup_steps": 15,
                    "min_lr": 1e-6,
                },
                "optim_config": {
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                    "adam_beta1": 0.9,
                },
            }
        }
        flat = flatten_resolved_recipe(resolved)
        assert flat["max_steps"] == 100
        assert flat["warmup_steps"] == 15
        assert flat["min_lr"] == 1e-6
        assert flat["lr"] == 1e-4
        assert flat["weight_decay"] == 0.01
        assert flat["adam_beta1"] == 0.9

    def test_multiple_sections(self):
        """Multiple top-level sections are all walked."""
        resolved = {
            "run": {"name": "experiment-1", "replicas": 4},
            "training_config": {"max_steps": 100},
        }
        flat = flatten_resolved_recipe(resolved)
        assert flat["name"] == "experiment-1"
        assert flat["replicas"] == 4
        assert flat["max_steps"] == 100

    def test_deeply_nested(self):
        """Three levels of nesting still produces leaf values."""
        resolved = {
            "training_config": {
                "peft": {
                    "lora_tuning": {
                        "alpha": 64,
                        "rank": 16,
                    }
                }
            }
        }
        flat = flatten_resolved_recipe(resolved)
        assert flat["alpha"] == 64
        assert flat["rank"] == 16

    def test_none_values_preserved(self):
        """None values are included in the flat dict."""
        resolved = {"training_config": {"data_path": None, "lr": 1e-4}}
        flat = flatten_resolved_recipe(resolved)
        assert "data_path" in flat
        assert flat["data_path"] is None
        assert flat["lr"] == 1e-4

    def test_list_values_skipped(self):
        """List values at leaf positions are skipped."""
        resolved = {
            "training_config": {
                "max_steps": 100,
                "instance_types": ["ml.p4d.24xlarge", "ml.p5.48xlarge"],
            }
        }
        flat = flatten_resolved_recipe(resolved)
        assert flat["max_steps"] == 100
        assert "instance_types" not in flat

    def test_duplicate_leaf_keys_last_wins(self):
        """When same key name appears at multiple levels, last encountered wins."""
        resolved = {
            "training_config": {
                "learning_rate": 1e-4,
                "nested": {
                    "learning_rate": 5e-5,
                },
            }
        }
        flat = flatten_resolved_recipe(resolved)
        # depth-first: training_config.learning_rate is visited first,
        # then training_config.nested.learning_rate overwrites it
        assert flat["learning_rate"] == 5e-5


# --- _get_nested_value tests ---


class TestGetNestedValue:
    def test_simple_path(self):
        d = {"a": {"b": {"c": 42}}}
        assert _get_nested_value(d, "a.b.c") == 42

    def test_missing_path(self):
        d = {"a": {"b": 1}}
        assert _get_nested_value(d, "a.c.d") is None

    def test_top_level(self):
        d = {"key": "value"}
        assert _get_nested_value(d, "key") == "value"
