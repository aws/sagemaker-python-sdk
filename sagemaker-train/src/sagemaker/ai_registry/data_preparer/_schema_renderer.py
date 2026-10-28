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
"""Schema tree renderer — walks raw JSON schema and prints every field.

PRIVATE — used by DatasetJobConfig.print_schema only.
"""
from __future__ import absolute_import

import re
from typing import Any, Dict, List, Optional, Set

_PASCAL_TO_SNAKE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_CONSTRAINT_KEYS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "pattern", "format",
    "minItems", "maxItems", "multipleOf", "uniqueItems",
)


def _pascal_to_snake(name: str) -> str:
    result = _PASCAL_TO_SNAKE_RE.sub("_", name).lower()
    return re.sub(r"[^a-z0-9_]", "_", result)


def render_schema_tree(schema: Dict[str, Any], title: str = "DatasetJobConfig") -> str:
    """Render a full JSON schema as a tree string. Returns the rendered text."""
    lines: List[str] = []
    definitions = schema.get("$defs", schema.get("definitions", {}))
    lines.append(title)
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    _render_properties(properties, required, definitions, lines, prefix="", visited=set())
    return "\n".join(lines)


def _render_properties(
    properties: Dict[str, Any],
    required: Set[str],
    definitions: Dict[str, Any],
    lines: List[str],
    prefix: str,
    visited: Set[str],
) -> None:
    items = list(properties.items())
    for i, (prop_name, prop_schema) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")
        _render_field(prop_name, prop_schema, prop_name in required, definitions,
                      lines, prefix, connector, child_prefix, visited)


def _render_field(
    prop_name: str,
    prop_schema: Dict[str, Any],
    is_required: bool,
    definitions: Dict[str, Any],
    lines: List[str],
    prefix: str,
    connector: str,
    child_prefix: str,
    visited: Set[str],
) -> None:
    # Resolve $ref (track for cycle detection)
    ref = _ref_name(prop_schema)
    if ref and ref in visited:
        snake_name = _pascal_to_snake(prop_name)
        req_str = " [required]" if is_required else " (optional)"
        lines.append(f"{prefix}{connector}{snake_name}: <circular ref>{req_str}")
        return
    resolved = _resolve_ref(prop_schema, definitions)

    # Add ref to visited set for this branch
    new_visited = visited | {ref} if ref else visited

    snake_name = _pascal_to_snake(prop_name)
    alias_str = ""
    if snake_name != prop_name:
        alias_str = f" (alias: {prop_name})"

    type_str = _type_string(resolved, definitions)
    req_str = " [required]" if is_required else " (optional)"
    default = resolved.get("default")
    default_str = f" = {_format_default(default)}" if default is not None else ""

    lines.append(f"{prefix}{connector}{snake_name}: {type_str}{req_str}{default_str}{alias_str}")

    # Description
    desc = resolved.get("description")
    if desc:
        lines.append(f"{child_prefix}{desc}")

    # Constraints
    constraints = _gather_constraints(resolved)
    if constraints:
        lines.append(f"{child_prefix}constraints: {constraints}")

    # Examples
    examples = resolved.get("examples")
    if examples:
        lines.append(f"{child_prefix}examples: {examples}")

    # Enum values (from JSON schema enum keyword)
    enum_vals = resolved.get("enum")
    if enum_vals:
        lines.append(f"{child_prefix}values: {' | '.join(repr(v) for v in enum_vals)}")

    # additionalProperties
    add_props = resolved.get("additionalProperties")
    if add_props is True:
        lines.append(f"{child_prefix}(allows additional properties)")
    elif isinstance(add_props, dict):
        val_type = _type_string(add_props, definitions)
        lines.append(f"{child_prefix}(additional properties: {val_type})")

    # Recurse into nested structures
    _recurse_into(resolved, definitions, lines, child_prefix, new_visited)


def _recurse_into(
    schema: Dict[str, Any],
    definitions: Dict[str, Any],
    lines: List[str],
    prefix: str,
    visited: Set[str],
) -> None:
    """Recurse into object properties, array items, oneOf/anyOf branches."""
    schema = _resolve_ref(schema, definitions)

    # Object with properties
    props = schema.get("properties")
    if props:
        req = set(schema.get("required", []))
        _render_properties(props, req, definitions, lines, prefix, visited)
        return

    # Array items
    if schema.get("type") == "array":
        items_schema = schema.get("items", {})
        ref = _ref_name(items_schema)
        if ref and ref in visited:
            lines.append(f"{prefix}<circular ref>")
            return
        resolved_items = _resolve_ref(items_schema, definitions)
        new_visited = visited | {ref} if ref else visited
        if resolved_items.get("properties"):
            lines.append(f"{prefix}[items]:")
            req = set(resolved_items.get("required", []))
            _render_properties(resolved_items["properties"], req, definitions,
                               lines, prefix + "    ", new_visited)
        elif "oneOf" in resolved_items or "anyOf" in resolved_items:
            _render_union_branches(resolved_items, definitions, lines, prefix, new_visited)
        return

    # oneOf / anyOf at field level
    if "oneOf" in schema or "anyOf" in schema:
        _render_union_branches(schema, definitions, lines, prefix, visited)


def _render_union_branches(
    schema: Dict[str, Any],
    definitions: Dict[str, Any],
    lines: List[str],
    prefix: str,
    visited: Set[str],
) -> None:
    branches = schema.get("oneOf") or schema.get("anyOf") or []
    keyword = "oneOf" if "oneOf" in schema else "anyOf"
    for idx, branch in enumerate(branches):
        resolved_branch = _resolve_ref(branch, definitions)
        branch_type = _type_string(resolved_branch, definitions)
        is_last = idx == len(branches) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")
        lines.append(f"{prefix}{connector}[{keyword} branch {idx + 1}]: {branch_type}")
        desc = resolved_branch.get("description")
        if desc:
            lines.append(f"{child_prefix}{desc}")
        if resolved_branch.get("properties"):
            req = set(resolved_branch.get("required", []))
            _render_properties(resolved_branch["properties"], req, definitions,
                               lines, child_prefix, visited)


def _type_string(schema: Dict[str, Any], definitions: Dict[str, Any]) -> str:
    """Produce a human-readable type string from a JSON schema node."""
    schema = _resolve_ref(schema, definitions)

    # Literal / enum
    enum_vals = schema.get("enum")
    if enum_vals:
        return " | ".join(repr(v) for v in enum_vals)

    # const
    if "const" in schema:
        return repr(schema["const"])

    # oneOf / anyOf
    if "oneOf" in schema or "anyOf" in schema:
        branches = schema.get("oneOf") or schema.get("anyOf") or []
        # Check if it's Optional (union with null)
        non_null = [b for b in branches if _resolve_ref(b, definitions).get("type") != "null"]
        has_null = len(non_null) < len(branches)
        if has_null and len(non_null) == 1:
            inner = _type_string(_resolve_ref(non_null[0], definitions), definitions)
            return f"Optional[{inner}]"
        parts = [_type_string(_resolve_ref(b, definitions), definitions) for b in branches
                 if _resolve_ref(b, definitions).get("type") != "null"]
        result = " | ".join(parts)
        if has_null:
            result = f"Optional[{result}]"
        return result

    schema_type = schema.get("type")

    if schema_type == "object":
        props = schema.get("properties")
        if props:
            title = schema.get("title", "object")
            return title
        add_props = schema.get("additionalProperties")
        if isinstance(add_props, dict):
            val_type = _type_string(add_props, definitions)
            return f"Dict[str, {val_type}]"
        return "Dict[str, Any]" if add_props else "object"

    if schema_type == "array":
        items_schema = schema.get("items", {})
        item_type = _type_string(_resolve_ref(items_schema, definitions), definitions)
        return f"List[{item_type}]"

    if isinstance(schema_type, list):
        # JSON schema allows type: ["string", "null"]
        non_null = [t for t in schema_type if t != "null"]
        has_null = "null" in schema_type
        type_str = " | ".join(_primitive_name(t) for t in non_null)
        if has_null:
            return f"Optional[{type_str}]"
        return type_str

    if schema_type:
        return _primitive_name(schema_type)

    return "Any"


def _primitive_name(t: str) -> str:
    return {"string": "str", "integer": "int", "number": "float",
            "boolean": "bool", "null": "None"}.get(t, t)


def _resolve_ref(schema: Dict[str, Any], definitions: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a $ref to its definition."""
    if "$ref" not in schema:
        return schema
    ref_path = schema["$ref"]
    ref_name = ref_path.split("/")[-1]
    return definitions.get(ref_name, schema)


def _ref_name(schema: Dict[str, Any]) -> Optional[str]:
    """Extract the ref name from a schema node, or None if not a $ref."""
    ref_path = schema.get("$ref")
    if ref_path:
        return ref_path.split("/")[-1]
    return None


def _gather_constraints(schema: Dict[str, Any]) -> str:
    parts = []
    for key in _CONSTRAINT_KEYS:
        if key in schema:
            parts.append(f"{key}={schema[key]}")
    return ", ".join(parts)


def _format_default(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)
