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
"""Case conversion helpers for DatasetJobConfig payload normalization.

PRIVATE — not exported from sagemaker.ai_registry.__init__.
"""
from __future__ import absolute_import

from typing import Any


def _snake_to_pascal(key: str) -> str:
    """Convert a snake_case key to PascalCase.

    Rules:
    - Split on underscores, capitalize first letter of each segment, join.
    - Idempotent on already-PascalCase: if the key contains any uppercase
      letter AND no underscore, it is returned unchanged.
    - If the key contains underscores (even mixed like ``Foo_Bar``), it is
      treated as snake-ish: split on ``_``, capitalize first char of each
      piece, preserve the rest of each piece as-is.
    - Non-string keys (int, None) are returned unchanged.
    - Empty string returns empty string.
    - Leading/trailing/consecutive underscores are stripped (e.g. "_foo_" → "Foo").

    Examples:
        >>> _snake_to_pascal("foo_bar")
        'FooBar'
        >>> _snake_to_pascal("s3_uri")
        'S3Uri'
        >>> _snake_to_pascal("FooBar")
        'FooBar'
    """
    if not isinstance(key, str):
        return key
    if not key:
        return key
    # Already PascalCase: has uppercase and no underscores → return as-is
    if "_" not in key and any(c.isupper() for c in key):
        return key
    # Split on underscores, capitalize first letter of each part
    parts = key.split("_")
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


# An operation's ``Config`` holds the operator's own settings, defined by the
# service's operator catalog rather than the job schema; their keys (``languages``,
# ``Ratios`` split names, ``ActionMap`` entity types) must reach the service as written.
_VERBATIM_KEYS = frozenset({"Config"})


def _normalize_keys_to_pascal(obj: Any) -> Any:
    """Recursively convert all dict keys in *obj* to PascalCase.

    - dict → new dict with converted keys, recursed values; the value of a
      ``Config`` key is kept as written.
    - list/tuple → same container type with recursed elements.
    - Anything else → returned as-is.

    Does not mutate input.
    """
    if isinstance(obj, dict):
        converted = {}
        for key, value in obj.items():
            key = _snake_to_pascal(key)
            converted[key] = value if key in _VERBATIM_KEYS else _normalize_keys_to_pascal(value)
        return converted
    if isinstance(obj, list):
        return [_normalize_keys_to_pascal(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_keys_to_pascal(item) for item in obj)
    return obj
