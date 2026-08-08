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
"""Client-side validation for pipeline step ``arguments`` blocks.

Validates the **top-level keys** of a step's ``arguments`` dict against
the corresponding public AWS API input shape from botocore, and rejects
fields that SageMaker Pipelines is known not to support. This fails fast
at step construction with a clear error, instead of a server-side parse
failure at ``CreatePipeline`` time.

Values are intentionally not validated: they may be pipeline variables
(parameter references, step property references, ``Join``/``JsonGet``
expressions) that only resolve at pipeline compile or execution time.

If the installed botocore release does not know the target operation
(for example, a very old botocore without newer Bedrock APIs), shape
validation is skipped and the service remains the authority.
"""

from __future__ import absolute_import

import logging
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

import botocore.session
from botocore.exceptions import UnknownServiceError
from botocore.model import OperationNotFoundError

logger = logging.getLogger(__name__)

# Cache of (service, operation, pascal_case) -> allowed top-level keys.
# ``None`` means botocore does not know the operation; skip shape checks.
_SHAPE_CACHE: Dict[Tuple[str, str, bool], Optional[FrozenSet[str]]] = {}


def _allowed_top_level_keys(
    service_name: str, operation_name: str, pascal_case: bool
) -> Optional[FrozenSet[str]]:
    """Return the allowed top-level keys for an operation input shape.

    Args:
        service_name (str): botocore service name (e.g. ``sagemaker``).
        operation_name (str): operation name (e.g. ``CreateEndpointConfig``).
        pascal_case (bool): If True, convert member names to PascalCase
            (used for Bedrock, whose JSON API members are camelCase but
            whose pipeline ``Arguments`` fields are PascalCase).

    Returns:
        The allowed key set, or ``None`` if the installed botocore does
        not know the operation (validation should then be skipped).
    """
    cache_key = (service_name, operation_name, pascal_case)
    if cache_key not in _SHAPE_CACHE:
        try:
            session = botocore.session.get_session()
            service_model = session.get_service_model(service_name)
            operation_model = service_model.operation_model(operation_name)
            members = operation_model.input_shape.members.keys()
            if pascal_case:
                members = [m[0].upper() + m[1:] for m in members]
            _SHAPE_CACHE[cache_key] = frozenset(members)
        except (UnknownServiceError, OperationNotFoundError):
            logger.warning(
                "Installed botocore does not know %s.%s; skipping "
                "client-side argument shape validation for this step.",
                service_name,
                operation_name,
            )
            _SHAPE_CACHE[cache_key] = None
    return _SHAPE_CACHE[cache_key]


def validate_step_arguments(
    step_class_name: str,
    arguments: Dict[str, Any],
    service_name: str,
    operation_name: str,
    unsupported_fields: Sequence[str] = (),
    pascal_case: bool = False,
) -> None:
    """Validate the top-level keys of a step ``arguments`` dict.

    Args:
        step_class_name (str): Step class name, used in error messages.
        arguments (Dict[str, Any]): The user-provided ``arguments`` dict.
        service_name (str): botocore service name of the wrapped API.
        operation_name (str): Operation whose input shape defines the
            allowed top-level fields.
        unsupported_fields (Sequence[str]): Fields that exist in the
            public API shape but are rejected by SageMaker Pipelines.
        pascal_case (bool): Convert botocore member names to PascalCase
            before comparison (Bedrock APIs).

    Raises:
        ValueError: If ``arguments`` is not a non-empty dict with string
            keys, contains an unsupported field, or contains a key that
            is not part of the operation's input shape.
    """
    if arguments is None:
        raise ValueError(f"arguments is required for {step_class_name}.")
    if not isinstance(arguments, dict) or not arguments:
        raise ValueError(f"{step_class_name}: arguments must be a non-empty dict.")
    non_string_keys = [key for key in arguments if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError(
            f"{step_class_name}: argument keys must be strings; got {non_string_keys!r}."
        )
    rejected = sorted(field for field in unsupported_fields if field in arguments)
    if rejected:
        raise ValueError(
            f"{step_class_name}: field(s) {rejected} are not supported by "
            "SageMaker Pipelines and would be rejected at pipeline creation "
            "time. Remove them from arguments."
        )
    allowed = _allowed_top_level_keys(service_name, operation_name, pascal_case)
    if allowed is None:
        return
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ValueError(
            f"{step_class_name}: unknown argument field(s) {unknown}. "
            f"Allowed top-level fields (from {service_name}.{operation_name}): "
            f"{sorted(allowed)}."
        )
