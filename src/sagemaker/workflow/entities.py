# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Defines the base entities used in workflow."""
from __future__ import absolute_import

import abc

from enum import EnumMeta
from typing import Any, Dict, List, Union

PrimitiveType = Union[str, int, bool, float, None]
RequestType = Union[Dict[str, Any], List[Dict[str, Any]]]


class Entity(abc.ABC):
    """Base object for workflow entities.

    Entities must implement the to_request method.
    """

    @abc.abstractmethod
    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""


class DefaultEnumMeta(EnumMeta):
    """An EnumMeta which defaults to the first value in the Enum list."""

    default = object()

    def __call__(cls, *args, value=default, **kwargs):
        """Defaults to the first value in the Enum list."""
        if value is DefaultEnumMeta.default:
            return next(iter(cls))
        return super().__call__(value, *args, **kwargs)

    factory = __call__


class Expression(abc.ABC):
    """Base object for expressions.

    Expressions must implement the expr property.
    """

    @property
    @abc.abstractmethod
    def expr(self) -> RequestType:
        """Get the expression structure for workflow service calls."""
