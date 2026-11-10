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
"""One SHA-256 and one canonical JSON form for every digest an auto-labeling run records.

Digests ride in run-log entries, manifests and re-attach checks, so the bytes hashed
must not drift between modules: a request is re-attached only when its digest matches
the one recorded. ``ascii`` mirrors ``json.dumps``'s ``ensure_ascii`` because the two
forms differ on non-ASCII text, and each recorded digest keeps the form it was first
written with.
"""

from __future__ import absolute_import

import hashlib
import json
from typing import Any, Union

from sagemaker.core.workflow.utilities import hash_file


def sha256_hex(data: Union[str, bytes]) -> str:
    """The SHA-256 hex digest of ``data`` (text is UTF-8 encoded)."""
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    """The SHA-256 hex digest of a file's bytes: sagemaker-core's :func:`hash_file`."""
    return hash_file(str(path))


def canonical_json(value: Any, ascii: bool = False) -> str:  # pylint: disable=redefined-builtin
    """``value`` as JSON with sorted keys, one spelling per value."""
    return json.dumps(value, sort_keys=True, ensure_ascii=ascii)


def sha256_json(value: Any, ascii: bool = False) -> str:  # pylint: disable=redefined-builtin
    """The SHA-256 hex digest of :func:`canonical_json`."""
    return sha256_hex(canonical_json(value, ascii=ascii))


def canonical(value: Any) -> Any:
    """``value`` with every list in a fixed order.

    Spans or answers given in another order compare equal; dict keys are kept.
    """
    if isinstance(value, dict):
        return {k: canonical(x) for k, x in value.items()}
    if isinstance(value, (list, tuple)):
        return sorted((canonical(x) for x in value), key=canonical_json)
    return value
