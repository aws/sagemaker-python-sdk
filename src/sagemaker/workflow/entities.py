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
"""Defines the base entities used in workflow."""
from __future__ import absolute_import

import abc

from enum import EnumMeta
from typing import Any, Dict, List, Union, Optional

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


class PipelineVariable(Expression):
    """Base object for pipeline variables

    PipelineVariables must implement the expr property.
    """

    def __add__(self, other: Union[Expression, PrimitiveType]):
        """Add function for PipelineVariable

        Args:
            other (Union[Expression, PrimitiveType]): The other object to be concatenated.

        Always raise an error since pipeline variables do not support concatenation
        """

        raise TypeError("Pipeline variables do not support concatenation.")

    def __str__(self):
        """Override built-in String function for PipelineVariable"""
        raise TypeError("Pipeline variables do not support __str__ operation.")

    def __int__(self):
        """Override built-in Integer function for PipelineVariable"""
        raise TypeError("Pipeline variables do not support __int__ operation.")

    def __float__(self):
        """Override built-in Float function for PipelineVariable"""
        raise TypeError("Pipeline variables do not support __float__ operation.")

    def to_string(self):
        """Prompt the pipeline to convert the pipeline variable to String in runtime"""
        from sagemaker.workflow.functions import Join

        return Join(on="", values=[self])

    @property
    @abc.abstractmethod
    def expr(self) -> RequestType:
        """Get the expression structure for workflow service calls."""

    def startswith(
        self,
        prefix: Union[str, tuple],  # pylint: disable=unused-argument
        start: Optional[int] = None,  # pylint: disable=unused-argument
        end: Optional[int] = None,  # pylint: disable=unused-argument
    ) -> bool:
        """Simulate the Python string's built-in method: startswith

        Args:
            prefix (str, tuple): The (tuple of) string to be checked.
            start (int): To set the start index of the matching boundary (default: None).
            end (int): To set the end index of the matching boundary (default: None).

        Return:
            bool: Always return False as Pipeline variables are parsed during execution runtime
        """
        return False

    def endswith(
        self,
        suffix: Union[str, tuple],  # pylint: disable=unused-argument
        start: Optional[int] = None,  # pylint: disable=unused-argument
        end: Optional[int] = None,  # pylint: disable=unused-argument
    ) -> bool:
        """Simulate the Python string's built-in method: endswith

        Args:
            suffix (str, tuple): The (tuple of) string to be checked.
            start (int): To set the start index of the matching boundary (default: None).
            end (int): To set the end index of the matching boundary (default: None).

        Return:
            bool: Always return False as Pipeline variables are parsed during execution runtime
        """
        return False
