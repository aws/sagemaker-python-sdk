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
"""Recipe resolution with 3-level override precedence for Nova model training."""
from __future__ import absolute_import

import copy
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Set, Tuple, Union

import yaml
from omegaconf import OmegaConf

from sagemaker.core.training.configs import HyperPodCompute, TrainingJobCompute
from sagemaker.train.sm_recipes.utils import _register_custom_resolvers

logger = logging.getLogger(__name__)


def render_template(
    template: Dict[str, Any],
    override_spec: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Render a Hub recipe template by filling {{placeholder}} values.

    Args:
        template: Hub recipe template dict containing '{{key}}' placeholders.
        override_spec: Flat dict mapping spec keys to their metadata
            (including 'default', 'type', 'min', 'max', 'enum').

    Returns:
        Tuple of (rendered_dict, key_path_map) where key_path_map maps flat spec keys
        to their dotpath location in the recipe structure.
        e.g. {"learning_rate": "training_config.learning_rate"}
    """
    key_path_map = {}

    def _walk(obj, path_parts):
        if isinstance(obj, dict):
            return {k: _walk(v, path_parts + [k]) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_walk(item, path_parts + [str(i)]) for i, item in enumerate(obj)]
        elif isinstance(obj, str) and "{{" in obj and "}}" in obj:
            spec_key = obj.removeprefix("'").removesuffix("'")
            spec_key = spec_key.removeprefix('"').removesuffix('"')
            spec_key = spec_key.removeprefix("{{").removesuffix("}}")
            spec_key = spec_key.strip()
            key_path_map[spec_key] = ".".join(path_parts)
            spec_entry = override_spec.get(spec_key, {})
            return spec_entry.get("default")
        else:
            return obj

    rendered = _walk(template, [])
    return rendered, key_path_map


def _load_user_recipe(recipe_path: str) -> Dict[str, Any]:
    """Load a user recipe from a local path or S3 URI.

    Args:
        recipe_path: Local file path or S3 URI to a YAML recipe file.

    Returns:
        Parsed recipe as a dict.

    Raises:
        ValueError: If the file cannot be loaded or parsed.
    """
    if recipe_path.startswith("s3://"):
        try:
            import boto3
            parts = recipe_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
            s3.download_file(bucket, key, tmp.name)
            tmp.close()
            with open(tmp.name, "r") as f:
                content = yaml.safe_load(f)
            os.unlink(tmp.name)
            if not isinstance(content, dict):
                raise ValueError(
                    f"Recipe file at {recipe_path} did not parse as a YAML mapping."
                )
            return content
        except ImportError:
            raise ValueError(
                "boto3 is required to load recipes from S3. Install it with: pip install boto3"
            )
        except Exception as e:
            raise ValueError(f"Could not load recipe from {recipe_path}: {e}")
    elif recipe_path.startswith("http://") or recipe_path.startswith("https://"):
        raise ValueError(
            f"HTTP/HTTPS recipe URLs are not supported for security reasons. "
            f"Use a local file path or S3 URI instead: {recipe_path}"
        )
    else:
        if not os.path.isfile(recipe_path):
            raise ValueError(f"Recipe file not found: {recipe_path}")
        with open(recipe_path, "r") as f:
            content = yaml.safe_load(f)
        if not isinstance(content, dict):
            raise ValueError(
                f"Recipe file at {recipe_path} did not parse as a YAML mapping."
            )
        return content


def _validate_value(
    key: str,
    value: Any,
    spec: Dict[str, Any],
    source: str,
    resolved_recipe: Optional[Dict[str, Any]] = None,
    dotpath: Optional[str] = None,
) -> None:
    """Validate a single value against its spec entry.

    Performs type checking, range validation, enum membership, and required
    field presence checks.

    Args:
        key: The parameter name.
        value: The value to validate.
        spec: The spec entry dict with 'type', 'min', 'max', 'enum', 'required' fields.
        source: Human-readable source label (e.g., "overrides dict", "user recipe").
        resolved_recipe: Optional resolved recipe dict (used for required check context).
        dotpath: Optional dotpath of the key in the recipe (used for required check context).

    Raises:
        ValueError: If validation fails.
    """
    # --- Required field presence check ---
    if spec.get("required", False):
        if not dotpath:
            raise ValueError(
                f"'{key}' is required but was not found in the resolved recipe."
            )
        if value is None:
            raise ValueError(
                f"'{key}' is required but was not found in the resolved recipe."
            )

    if value is None:
        return

    expected_type = spec.get("type")
    if expected_type:
        type_map = {
            "float": (int, float),
            "integer": (int,),
            "int": (int,),
            "string": (str,),
            "boolean": (bool,),
            "bool": (bool,),
        }
        allowed_types = type_map.get(expected_type.lower())
        if allowed_types and not isinstance(value, allowed_types):
            raise ValueError(
                f"Invalid type for '{key}': expected {expected_type}, "
                f"got {type(value).__name__} (value: {value}). Source: {source}."
            )

    min_val = spec.get("min")
    if min_val is not None and isinstance(value, (int, float)):
        if value < min_val:
            raise ValueError(
                f"Invalid value for '{key}': {value} is below minimum {min_val}. "
                f"Allowed range: [{min_val}, {spec.get('max', '...')}]. Source: {source}."
            )

    max_val = spec.get("max")
    if max_val is not None and isinstance(value, (int, float)):
        if value > max_val:
            raise ValueError(
                f"Invalid value for '{key}': {value} is above maximum {max_val}. "
                f"Allowed range: [{spec.get('min', '...')}, {max_val}]. Source: {source}."
            )

    enum_values = spec.get("enum")
    if enum_values is not None:
        if value == "" or value == spec.get("default"):
            pass
        elif value not in enum_values:
            raise ValueError(
                f"Invalid value for '{key}': {value} is not in allowed values {enum_values}. "
                f"Source: {source}."
            )


def _validate_step_constraints(
    resolved_recipe: Dict[str, Any],
    key_path_map: Dict[str, str],
) -> None:
    """Perform cross-field validation on a resolved recipe.

    Validates constraints that involve relationships between multiple
    recipe parameters (e.g., save_steps must be <= max_steps).

    Args:
        resolved_recipe: The fully resolved recipe dict (nested structure).
        key_path_map: Mapping of flat spec key names to their dotpath
            location in the resolved recipe.

    Raises:
        ValueError: Immediately on the first validation failure.
    """
    # --- Cross-field validation: save_steps must be <= max_steps ---
    save_steps_path = key_path_map.get("save_steps")
    max_steps_path = key_path_map.get("max_steps")

    if save_steps_path and max_steps_path:
        save_steps = _get_nested_value(resolved_recipe, save_steps_path)
        max_steps = _get_nested_value(resolved_recipe, max_steps_path)

        if (
            isinstance(save_steps, (int, float))
            and isinstance(max_steps, (int, float))
            and save_steps > max_steps
        ):
            raise ValueError(
                f"'save_steps' ({save_steps}) must be less than or equal to "
                f"'max_steps' ({max_steps})."
            )


def flatten_resolved_recipe(resolved: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a resolved recipe dict into a single-level key-value map.

    Recursively walks all nested dicts and extracts scalar leaf values
    keyed by their leaf key name. Used by trainers and evaluators to
    apply resolved recipe values as flat hyperparameters to the
    SageMaker training API.

    For nested structures like:
        training_config:
          lr_scheduler:
            warmup_steps: 15
            min_lr: 1e-6

    This produces: {"warmup_steps": 15, "min_lr": 1e-6}

    If duplicate leaf keys exist at different nesting levels, the last
    one encountered wins (depth-first traversal).

    Args:
        resolved: The resolved recipe dict (nested by section).

    Returns:
        Flat dict of all scalar leaf key-value pairs across all sections.
    """
    flat = {}

    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    _walk(v)
                elif not isinstance(v, list):
                    flat[k] = v
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    _walk(item)

    _walk(resolved)
    return flat


def _get_nested_value(d: Dict[str, Any], dotpath: str) -> Any:
    """Get a value from a nested dict using a dot-separated path."""
    parts = dotpath.split(".")
    current = d
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _set_nested_value(d: Dict[str, Any], dotpath: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated path."""
    parts = dotpath.split(".")
    current = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _build_key_path_map(recipe_dict: Dict[str, Any], spec_keys: set) -> Dict[str, str]:
    """Build a key_path_map by finding spec keys in a recipe dict.

    Walks the recipe dict and maps spec key names to their dotpath locations.
    Used when a full recipe template is provided instead of a synthetic one.

    Args:
        recipe_dict: The full recipe dict (nested).
        spec_keys: Set of flat spec key names to locate.

    Returns:
        Dict mapping spec key names to their dotpath in the recipe.
    """
    key_path_map = {}

    def _walk(obj, path_parts):
        if isinstance(obj, dict):
            for k, v in obj.items():
                current_path = path_parts + [k]
                if k in spec_keys and k not in key_path_map:
                    key_path_map[k] = ".".join(current_path)
                _walk(v, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, path_parts + [str(i)])

    _walk(recipe_dict, [])
    return key_path_map


class RecipeResolver:
    """Resolves a 3-level recipe configuration for Nova model training.

    Precedence (highest wins):
        1. Programmatic overrides (dict)
        2. User recipe (YAML file)
        3. Base defaults (rendered from Hub template + override-params spec)

    Immutable after construction — all inputs are deep-copied.
    resolve() is idempotent: second call returns cached result.
    """

    def __init__(
        self,
        recipe_template: Dict[str, Any],
        override_spec: Dict[str, Any],
        user_recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        protected_keys: Optional[Set[str]] = None,
        full_recipe_template: Optional[Dict[str, Any]] = None,
        compute: Optional[Union["Compute", "HyperPodCompute"]] = None,
    ):
        """Initialize the resolver.

        Args:
            recipe_template: Hub recipe template dict (with {{placeholder}} syntax).
            override_spec: Flat dict of parameter specs from Hub (type/min/max/enum/default).
            user_recipe_path: Optional path to user's recipe YAML file (local or S3).
            overrides: Optional programmatic overrides dict (nested structure).
            protected_keys: Optional set of keys that cannot be overridden by user
                recipe or overrides (e.g., 'model_type', 'task').
            full_recipe_template: Optional full recipe dict fetched from Hub
                (SmtjRecipeTemplateS3Uri). When provided, used as the base layer
                instead of the synthetic template — enables overriding any key in
                the full recipe, not just the spec-exposed subset.
            compute: Optional compute configuration. Union of
                ``sagemaker.core.training.configs.Compute`` (TrainingJobCompute) or
                ``sagemaker.core.training.configs.HyperPodCompute``.
                None indicates SMTJ Serverless.
        """
        self._recipe_template = copy.deepcopy(recipe_template)
        self._override_spec = copy.deepcopy(override_spec)
        self._user_recipe_path = user_recipe_path
        self._overrides = copy.deepcopy(overrides) if overrides else {}
        self._protected_keys = protected_keys or set()
        self._full_recipe_template = copy.deepcopy(full_recipe_template) if full_recipe_template else None
        self._compute = compute
        self._resolved: Optional[Dict[str, Any]] = None

    def resolve(self) -> Dict[str, Any]:
        """Perform template render, 3-level merge, and validation.

        Returns:
            The fully resolved recipe as a plain dict.

        Raises:
            ValueError: If validation fails for any parameter.
        """
        if self._resolved is not None:
            return copy.deepcopy(self._resolved)

        # Phase 1: Determine base layer and key_path_map
        if self._full_recipe_template:
            # Use the full recipe from Hub as the base layer.
            # render_template resolves any {{placeholder}} values to spec defaults
            # and maps those placeholder keys to their dotpaths.
            base_dict, key_path_map = render_template(
                self._full_recipe_template, self._override_spec
            )
            # For keys that appear as plain values (not placeholders) in the
            # full template, locate them by name so validation and protected-key
            # stripping still work.
            extra_keys = (set(self._override_spec.keys()) | self._protected_keys) - set(key_path_map.keys())
            if extra_keys:
                extra_paths = _build_key_path_map(base_dict, extra_keys)
                key_path_map.update(extra_paths)
        else:
            # Synthetic template built from spec keys only (legacy path)
            base_dict, key_path_map = render_template(
                self._recipe_template, self._override_spec
            )

        # Phase 2: Load user recipe if provided
        user_dict = {}
        if self._user_recipe_path:
            user_dict = _load_user_recipe(self._user_recipe_path)
            if not self._full_recipe_template:
                self._warn_unknown_keys(user_dict, base_dict)

        # Warn about unknown keys in overrides (only in legacy mode)
        if self._overrides and not self._full_recipe_template:
            self._warn_unknown_keys(self._overrides, base_dict)

        # Phase 3: Strip protected keys from copies (don't mutate loaded inputs)
        user_dict_for_merge = copy.deepcopy(user_dict)
        overrides_for_merge = copy.deepcopy(self._overrides)
        self._strip_protected_keys(user_dict_for_merge, key_path_map)
        self._strip_protected_keys(overrides_for_merge, key_path_map)

        # Phase 4: OmegaConf 3-way merge (base < user < overrides)
        _register_custom_resolvers()
        base_cfg = OmegaConf.create(base_dict)
        user_cfg = OmegaConf.create(user_dict_for_merge)
        overrides_cfg = OmegaConf.create(overrides_for_merge)

        merged = OmegaConf.merge(base_cfg, user_cfg, overrides_cfg)

        # Phase 5: Resolve interpolations
        try:
            OmegaConf.resolve(merged)
        except Exception as e:
            raise ValueError(
                f"Failed to resolve recipe interpolations: {e}. "
                f"Ensure all referenced keys exist in the base template or user recipe."
            )

        resolved_dict = OmegaConf.to_container(merged, resolve=True)

        # Phase 6: Validate against override spec using key_path_map
        self._validate(resolved_dict, key_path_map, compute=self._compute)

        self._resolved = resolved_dict
        return copy.deepcopy(self._resolved)

    def get_resolved_recipe(self) -> Dict[str, Any]:
        """Return the resolved recipe as a read-only deep copy.

        Callable before or after train()/evaluate(). Triggers resolution
        on first call if not already resolved.

        Returns:
            Deep copy of the fully resolved recipe dict.
        """
        return self.resolve()

    def _warn_unknown_keys(
        self, user_dict: Dict[str, Any], base_dict: Dict[str, Any]
    ) -> None:
        """Log warnings for keys in user recipe that don't exist in base."""

        def _collect_keys(d, prefix=""):
            keys = set()
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                keys.add(full_key)
                if isinstance(v, dict):
                    keys.update(_collect_keys(v, full_key))
            return keys

        base_keys = _collect_keys(base_dict)
        user_keys = _collect_keys(user_dict)
        unknown = user_keys - base_keys
        for key in sorted(unknown):
            logger.warning(
                f"Unknown key '{key}' in user recipe will be included but is not "
                f"part of the base template. It will not be validated."
            )

    def _strip_protected_keys(
        self, d: Dict[str, Any], key_path_map: Dict[str, str]
    ) -> None:
        """Remove protected keys from a dict and log warnings."""
        for spec_key in self._protected_keys:
            dotpath = key_path_map.get(spec_key)
            if dotpath:
                parts = dotpath.split(".")
                current = d
                for part in parts[:-1]:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        current = None
                        break
                if isinstance(current, dict) and parts[-1] in current:
                    logger.warning(
                        f"Protected key '{spec_key}' (at {dotpath}) cannot be overridden. "
                        f"Ignoring user-provided value."
                    )
                    del current[parts[-1]]

    def _validate(
        self,
        resolved: Dict[str, Any],
        key_path_map: Dict[str, str],
        compute: Optional[Union["TrainingJobCompute", "HyperPodCompute"]] = None,
    ) -> None:
        """Validate resolved values against the override spec.

        Performs per-value validation (type, range, enum, required) and
        compute-level checks (instance_type), then runs
        steps validation.

        Args:
            resolved: The fully resolved recipe dict.
            key_path_map: Mapping of spec keys to their dotpath in the recipe.
            compute: Optional compute configuration. Union of
                Compute (TrainingJobCompute) or HyperPodCompute.
                None indicates SMTJ Serverless.
        """

        is_serverless = compute is None

        for spec_key, spec_entry in self._override_spec.items():
            # --- Instance type: validated from compute, not from recipe ---
            if spec_key == "instance_type":
                if is_serverless:
                    continue
                instance_type = getattr(compute, "instance_type", None)
                allowed_values = spec_entry.get("enum")
                if instance_type is None:
                    raise ValueError(
                        "instance_type must be specified in Compute parameter. "
                        f"Allowed types: {sorted(allowed_values) if allowed_values else 'unknown'}."
                    )
                if allowed_values and instance_type not in allowed_values:
                    raise ValueError(
                        f"Instance type '{instance_type}' is not supported. "
                        f"Allowed types: {sorted(allowed_values)}."
                    )
                continue

            # --- Replicas: validated from compute, not from recipe ---
            if spec_key == "replicas":
                if is_serverless:
                    continue
                node_count = getattr(compute, "node_count", None) or getattr(
                    compute, "instance_count", None
                )

                allowed_values = spec_entry.get("enum")
                if node_count is None:
                    raise ValueError(
                        "node_count (or instance_count) must be specified in Compute parameter. "
                        f"Allowed values: {sorted(allowed_values) if allowed_values else 'unknown'}."
                    )
                if allowed_values and node_count not in allowed_values:
                    raise ValueError(
                        f"Node/Instance count '{node_count}' is not supported. "
                        f"Allowed values: {sorted(allowed_values)}."
                    )
                continue

            # Skip keys not mapped into the recipe structure
            dotpath = key_path_map.get(spec_key)
            if not dotpath:
                # Still check required even when not in key_path_map
                if spec_entry.get("required", False):
                    raise ValueError(
                        f"'{spec_key}' is required but was not found in the "
                        f"resolved recipe."
                    )
                continue

            value = _get_nested_value(resolved, dotpath)

            # Determine source for error messages
            if self._overrides:
                override_value = _get_nested_value(self._overrides, dotpath)
                if override_value is not None:
                    source = "overrides dict"
                elif self._user_recipe_path:
                    source = f"user recipe ({self._user_recipe_path})"
                else:
                    source = "base defaults"
            else:
                source = "resolved recipe"

            _validate_value(spec_key, value, spec_entry, source, resolved, dotpath)

        _validate_step_constraints(resolved, key_path_map)
