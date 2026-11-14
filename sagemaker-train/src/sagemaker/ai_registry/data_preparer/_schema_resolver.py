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
"""Hub-backed JobConfigSchema resolver with process-local caching.

PRIVATE — not exported from sagemaker.ai_registry.__init__. Only
DatasetJobConfig._resolve_and_validate and DatasetJobConfig.schema / print_schema
call into this module.
"""

from __future__ import absolute_import

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, create_model

from sagemaker.ai_registry.data_preparer.constants import (
    JobCategory,
    SCHEMA_VERSION_CEILING,
)
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerSchemaNotFoundError,
)
from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger(__name__)


_SCHEMA_CACHE: Dict[Tuple[JobCategory, str], Dict[str, Any]] = {}
_MODEL_CACHE: Dict[Tuple[JobCategory, str], Type[BaseModel]] = {}

# Regex for converting PascalCase to snake_case
_PASCAL_TO_SNAKE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


_PYDANTIC_RESERVED = {
    "model_config",
    "model_fields",
    "model_computed_fields",
    "model_extra",
    "model_fields_set",
}


def _pascal_to_snake(name: str) -> str:
    """Convert PascalCase to snake_case, replacing non-identifier chars with underscores."""
    result = _PASCAL_TO_SNAKE_RE.sub("_", name).lower()
    # Replace any non-identifier characters with underscore
    result = re.sub(r"[^a-z0-9_]", "_", result)
    # Avoid Pydantic reserved attribute names
    if result in _PYDANTIC_RESERVED:
        result = result + "_field"
    return result


def _major(version: str) -> int:
    """Extract major version number."""
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError):
        return 0


def _semver_key(version: str) -> Tuple:
    """Parse a semver string into a comparable tuple.

    Pre-release versions (containing '-') sort lower than their release counterpart.
    Normalized to a fixed-length tuple so that versions with different part counts
    (e.g. "1.0" vs "1.0.0") compare consistently: "1.0" is treated as "1.0.0".
    """
    # Split off pre-release suffix
    base, _, prerelease = version.partition("-")
    parts = base.split(".")
    numeric_parts: list = []
    for part in parts:
        try:
            numeric_parts.append(int(part))
        except ValueError:
            # Non-numeric parts sort lower than any numeric part
            numeric_parts.append(-1)
    # Pad to 3 parts so length-different versions compare consistently
    while len(numeric_parts) < 3:
        numeric_parts.append(0)
    # Pre-release sorts lower: release gets (1,) suffix, pre-release gets (0, prerelease)
    if prerelease:
        numeric_parts.append(0)
        # Include the prerelease string so different prereleases order deterministically
        numeric_parts.append(prerelease)
    else:
        numeric_parts.append(1)
    return tuple(numeric_parts)


class _DynamicBase(BaseModel):
    """Base class for dynamically-generated schema models."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class _SchemaResolver:
    """Resolves service-published JobConfigSchema and builds dynamic Pydantic models.

    Backend: SageMaker's ``ListJobSchemaVersions`` /
    ``DescribeJobSchemaVersion`` APIs (part of the public service model).
    ``ListJobSchemaVersions`` returns the list of available versions (with
    version string per item); the schema body is fetched via
    ``DescribeJobSchemaVersion``. Results are process-locally cached keyed
    by ``(category, version)``.
    """

    @staticmethod
    def _client(sagemaker_session: Optional[Session] = None):
        """Return a boto SageMaker client bound to the resolver's session.

        Constructs a default :class:`Session` when none is supplied so the
        resolver remains usable from discoverability entrypoints
        (``DatasetJobConfig.schema``, ``DatasetJobConfig.print_schema``) that do not
        have an explicit session in scope.
        """
        if sagemaker_session is None:
            sagemaker_session = Session()
        return sagemaker_session.sagemaker_client

    @staticmethod
    def list_versions(
        category: JobCategory,
        sagemaker_session: Optional[Session] = None,
    ) -> List[str]:
        """List available schema versions for a category (newest first per semver)."""
        try:
            client = _SchemaResolver._client(sagemaker_session)
            versions: List[str] = []
            next_token: Optional[str] = None
            while True:
                kwargs: Dict[str, Any] = {"JobCategory": category.value}
                if next_token is not None:
                    kwargs["NextToken"] = next_token
                resp = client.list_job_schema_versions(**kwargs)
                for item in resp.get("JobConfigSchemas", []):
                    v = item.get("JobConfigSchemaVersion")
                    if v:
                        versions.append(v)
                next_token = resp.get("NextToken")
                if not next_token:
                    break
            return versions
        except Exception:
            return []

    @staticmethod
    def get_schema(
        category: JobCategory,
        version: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Resolve (version, schema). If version is None, returns the latest.

        Uses ``ListJobSchemaVersions`` (returns version + full schema in one
        call) for the latest-version case, and ``DescribeJobSchemaVersion``
        for explicit-version pinning. Results are cached keyed by
        ``(category, version)``.

        Returns:
            Tuple of (resolved_version, schema_dict).

        Raises:
            DataPreparerSchemaNotFoundError: If no schema is published for the
                category, or if the requested version is not found.
        """
        client = _SchemaResolver._client(sagemaker_session)

        # Latest-version path: consult ListJobSchemaVersions to enumerate
        # available versions; DescribeJobSchemaVersion is used to fetch the
        # schema body. We speculatively read any inline JobConfigSchema field
        # from list items as a cache warm-up in case the service ever returns
        # the schema inline — harmless today since the field is absent.
        if version is None:
            try:
                resp = client.list_job_schema_versions(JobCategory=category.value)
            except Exception as e:  # network / auth / service error
                raise DataPreparerSchemaNotFoundError(
                    category=category,
                    schema_version="latest",
                    available_versions=[],
                    transient=True,
                ) from e

            items = resp.get("JobConfigSchemas", []) or []
            if not items:
                raise DataPreparerSchemaNotFoundError(
                    category=category,
                    schema_version="latest",
                    available_versions=[],
                )

            # Pick the max-semver item, speculatively cache any (version, schema)
            # pairs where the schema body was returned inline (defensive; the service
            # currently returns only the version string per item, not the schema body,
            # so this branch is effectively a no-op today but kept for compatibility).
            available: List[str] = []
            for item in items:
                v = item.get("JobConfigSchemaVersion")
                if not v:
                    continue
                available.append(v)
                s = item.get("JobConfigSchema")
                if s:
                    parsed = json.loads(s) if isinstance(s, str) else s
                    _SCHEMA_CACHE[(category, v)] = parsed

            if not available:
                raise DataPreparerSchemaNotFoundError(
                    category=category,
                    schema_version="latest",
                    available_versions=[],
                )

            # Filter to versions within major version ceiling.
            ceiling = SCHEMA_VERSION_CEILING.get(category)
            if ceiling is not None:
                filtered = [v for v in available if _major(v) <= ceiling]
                if filtered:
                    available = filtered
                else:
                    logger.warning(
                        "No schema versions found within major version %d for %s. "
                        "Falling back to latest available: %s",
                        ceiling,
                        category.value,
                        max(available, key=_semver_key),
                    )

            resolved_version = max(available, key=_semver_key)

            # If the schema wasn't inline in the list response, fetch via Describe.
            if (category, resolved_version) not in _SCHEMA_CACHE:
                desc_resp = client.describe_job_schema_version(
                    JobCategory=category.value,
                    JobConfigSchemaVersion=resolved_version,
                )
                s = desc_resp.get("JobConfigSchema")
                if not s:
                    raise DataPreparerSchemaNotFoundError(
                        category=category,
                        schema_version=resolved_version,
                        available_versions=available,
                    )
                parsed = json.loads(s) if isinstance(s, str) else s
                _SCHEMA_CACHE[(category, resolved_version)] = parsed

            schema = _SCHEMA_CACHE[(category, resolved_version)]
            logger.info(
                "Using JobConfigSchema v%s for %s (latest of %d)",
                resolved_version,
                category.value,
                len(available),
            )
            return resolved_version, schema

        # Explicit-version path: consult cache first, else DescribeJobSchemaVersion.
        cache_key = (category, version)
        if cache_key in _SCHEMA_CACHE:
            logger.debug("Schema cache hit for (%s, %s)", category.value, version)
            return version, _SCHEMA_CACHE[cache_key]

        try:
            response = client.describe_job_schema_version(
                JobCategory=category.value,
                JobConfigSchemaVersion=version,
            )
        except Exception as e:
            exc_name = type(e).__name__
            if (
                "ResourceNotFound" in exc_name
                or "ResourceNotFoundException" in exc_name
                or "ValidationException" in exc_name
                or "ClientError" in exc_name
            ):
                available = _SchemaResolver.list_versions(category, sagemaker_session)
                raise DataPreparerSchemaNotFoundError(
                    category=category,
                    schema_version=version,
                    available_versions=available,
                ) from e
            raise

        document = response.get("JobConfigSchema", "{}")
        schema = json.loads(document) if isinstance(document, str) else document
        _SCHEMA_CACHE[cache_key] = schema
        logger.info("Using JobConfigSchema v%s for %s (pinned)", version, category.value)
        return version, schema

    @staticmethod
    def build_dynamic_model(
        category: JobCategory,
        version: str,
        schema: Dict[str, Any],
    ) -> Type[BaseModel]:
        """Recursively build a tree of Pydantic models mirroring the schema.

        Returns the root model. Uses populate_by_name=True + field aliases so
        callers can pass snake_case or PascalCase; serialization emits PascalCase.
        """
        cache_key = (category, version)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        definitions = schema.get("$defs", schema.get("definitions", {}))
        ref_cache: Dict[str, Type[BaseModel]] = {}

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        model = _SchemaResolver._build_object_model(
            name=f"{category.value}Config",
            props=properties,
            required=required,
            definitions=definitions,
            ref_cache=ref_cache,
        )
        _MODEL_CACHE[cache_key] = model
        return model

    @staticmethod
    def _build_object_model(
        name: str,
        props: Dict[str, Any],
        required: set,
        definitions: Dict[str, Any],
        ref_cache: Dict[str, Type[BaseModel]],
    ) -> Type[BaseModel]:
        """Build a Pydantic BaseModel for a single object schema."""
        field_definitions: Dict[str, Any] = {}

        for prop_name, prop_schema in props.items():
            snake_name = _pascal_to_snake(prop_name)
            python_type = _SchemaResolver._resolve_type(prop_schema, definitions, ref_cache)

            default = prop_schema.get("default")
            # For optional fields, ignore schema-declared defaults. The service
            # resolves defaults server-side. Client-side defaults from the schema
            # can be stale or wrong (e.g., "AIRegistry" for HubName which should
            # be a per-account computed value). Treat all optional fields as
            # default=None so they're omitted from the payload unless explicitly set.
            if prop_name not in required:
                default = None
            if prop_name in required:
                if default is not None:
                    field_definitions[snake_name] = (
                        python_type,
                        Field(default=default, alias=prop_name),
                    )
                else:
                    field_definitions[snake_name] = (
                        python_type,
                        Field(..., alias=prop_name),
                    )
            else:
                if default is not None:
                    field_definitions[snake_name] = (
                        Optional[python_type],
                        Field(default=default, alias=prop_name),
                    )
                else:
                    field_definitions[snake_name] = (
                        Optional[python_type],
                        Field(default=None, alias=prop_name),
                    )

        model = create_model(name, __base__=_DynamicBase, **field_definitions)
        return model

    @staticmethod
    def _resolve_type(
        prop_schema: Dict[str, Any],
        definitions: Dict[str, Any],
        ref_cache: Dict[str, Type[BaseModel]],
    ) -> type:
        """Resolve a JSON schema property to a Python type."""
        # Handle $ref
        if "$ref" in prop_schema:
            ref_path = prop_schema["$ref"]
            if ref_path in ref_cache:
                return ref_cache[ref_path]

            # Resolve the reference
            ref_name = ref_path.split("/")[-1]
            ref_schema = definitions.get(ref_name, {})

            # Build the referenced model
            ref_props = ref_schema.get("properties", {})
            ref_required = set(ref_schema.get("required", []))
            model = _SchemaResolver._build_object_model(
                name=ref_name,
                props=ref_props,
                required=ref_required,
                definitions=definitions,
                ref_cache=ref_cache,
            )
            ref_cache[ref_path] = model
            return model

        # Handle oneOf / anyOf unions — map to Any
        if "oneOf" in prop_schema or "anyOf" in prop_schema:
            return Any

        # No "type" places no constraint on the value, so the service validates it.
        if "type" not in prop_schema:
            return Any

        schema_type = prop_schema["type"]

        if schema_type == "object":
            nested_props = prop_schema.get("properties", {})
            if nested_props:
                nested_required = set(prop_schema.get("required", []))
                title = prop_schema.get("title", "NestedModel")
                return _SchemaResolver._build_object_model(
                    name=title,
                    props=nested_props,
                    required=nested_required,
                    definitions=definitions,
                    ref_cache=ref_cache,
                )
            return dict

        if schema_type == "array":
            items_schema = prop_schema.get("items", {})
            item_type = _SchemaResolver._resolve_type(items_schema, definitions, ref_cache)
            return List[item_type]

        # Primitive type mapping
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }
        return type_map.get(schema_type, str)
