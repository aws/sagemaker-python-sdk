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
"""Conditions for condition steps.

Ideally, some of these comparison conditions would be implemented as "partial classes",
but use of functools.partial doesn't set correct metadata/type information.
"""
from __future__ import absolute_import

from enum import Enum
from typing import Dict, List, Union

import attr

from sagemaker.workflow.entities import (
    DefaultEnumMeta,
    Entity,
    Expression,
    PrimitiveType,
    RequestType,
)
from sagemaker.workflow.execution_variables import ExecutionVariable
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.properties import Properties

# TODO: consider base class for those with an expr method, rather than defining a type here
ConditionValueType = Union[ExecutionVariable, Parameter, Properties]


class ConditionTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """Condition type enum."""

    EQ = "Equals"
    GT = "GreaterThan"
    GTE = "GreaterThanOrEqualTo"
    IN = "In"
    LT = "LessThan"
    LTE = "LessThanOrEqualTo"
    NOT = "Not"
    OR = "Or"


@attr.s
class Condition(Entity):
    """Abstract Condition entity.

    Attributes:
        condition_type (ConditionTypeEnum): The type of condition.
    """

    condition_type: ConditionTypeEnum = attr.ib(factory=ConditionTypeEnum.factory)


@attr.s
class ConditionComparison(Condition):
    """Generic comparison condition that can be used to derive specific condition comparisons.

    Attributes:
        condition_type (ConditionTypeEnum): The type of condition.
        left (ConditionValueType): The execution variable, parameter, or
            property to use in comparison.
        right (Union[ConditionValueType, PrimitiveType]): The execution variable,
            parameter, property, or python primitive value to compare to.
    """

    left: ConditionValueType = attr.ib(default=None)
    right: Union[ConditionValueType, PrimitiveType] = attr.ib(default=None)

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        return {
            "Type": self.condition_type.value,
            "LeftValue": self.left.expr,
            "RightValue": primitive_or_expr(self.right),
        }


class ConditionEquals(ConditionComparison):
    """A condition for equality comparisons.

    Attributes:
        condition_type (ConditionTypeEnum): A `ConditionTypeEnum.EQ` type of condition.
        left (ConditionValueType): The execution variable, parameter, or
            property to use in comparison.
        right (Union[ConditionValueType, PrimitiveType]): The execution variable,
            parameter, property, or python primitive value to compare to.
    """

    def __init__(self, left: ConditionValueType, right: Union[ConditionValueType, PrimitiveType]):
        """Constructs an instance of ConditionEquals.

        Args:
            left (ConditionValueType): The execution variable, parameter,
                or property to use in comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or python primitive value to compare to.
        """

        super(ConditionEquals, self).__init__(ConditionTypeEnum.EQ, left, right)


class ConditionGreaterThan(ConditionComparison):
    """A condition for greater than comparisons.

    Attributes:
        condition_type (ConditionTypeEnum): A `ConditionTypeEnum.GT` type of condition.
        left (ConditionValueType): The execution variable, parameter, or
            property to use in comparison.
        right (Union[ConditionValueType, PrimitiveType]): The execution variable,
            parameter, property, or python primitive value to compare to.
    """

    def __init__(self, left: ConditionValueType, right: Union[ConditionValueType, PrimitiveType]):
        """Constructs an instance of ConditionGreaterThan.

        Args:
            left (ConditionValueType): The execution variable, parameter,
                or property to use in comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or python primitive value to compare to.
        """

        super(ConditionGreaterThan, self).__init__(ConditionTypeEnum.GT, left, right)


class ConditionGreaterThanOrEqualTo(ConditionComparison):
    """A condition for greater than or equal to comparisons.

    Attributes:
        condition_type (ConditionTypeEnum): A `ConditionTypeEnum.GTE` type of condition.
        left (ConditionValueType): The execution variable, parameter, or
            property to use in comparison.
        right (Union[ConditionValueType, PrimitiveType]): The execution variable,
            parameter, property, or python primitive value to compare to.
    """

    def __init__(self, left: ConditionValueType, right: Union[ConditionValueType, PrimitiveType]):
        """Constructs an instance of ConditionGreaterThanOrEqualTo.

        Args:
            left (ConditionValueType): The execution variable, parameter,
                or property to use in comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or python primitive value to compare to.
        """

        super(ConditionGreaterThanOrEqualTo, self).__init__(ConditionTypeEnum.GTE, left, right)


class ConditionLessThan(ConditionComparison):
    """A condition for less than comparisons.

    Attributes:
        condition_type (ConditionTypeEnum): A `ConditionTypeEnum.LT` type of condition.
        left (ConditionValueType): The execution variable, parameter, or
            property to use in comparison.
        right (Union[ConditionValueType, PrimitiveType]): The execution variable,
            parameter, property, or python primitive value to compare to.
    """

    def __init__(self, left: ConditionValueType, right: Union[ConditionValueType, PrimitiveType]):
        """Constructs an instance of ConditionLessThan.

        Args:
            left (ConditionValueType): The execution variable, parameter,
                or property to use in comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or python primitive value to compare to.
        """

        super(ConditionLessThan, self).__init__(ConditionTypeEnum.LT, left, right)


class ConditionLessThanOrEqualTo(ConditionComparison):
    """A condition for less than or equal to comparisons.

    Attributes:
        condition_type (ConditionTypeEnum): A `ConditionTypeEnum.LTE` type of condition.
        left (ConditionValueType): The execution variable, parameter, or
            property to use in comparison.
        right (Union[ConditionValueType, PrimitiveType]): The execution variable,
            parameter, property, or python primitive value to compare to.
    """

    def __init__(self, left: ConditionValueType, right: Union[ConditionValueType, PrimitiveType]):
        """Constructs an instance of ConditionLessThanOrEqualTo.

        Args:
            left (ConditionValueType): The execution variable, parameter,
                or property to use in comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or python primitive value to compare to.
        """

        super(ConditionLessThanOrEqualTo, self).__init__(ConditionTypeEnum.LTE, left, right)


class ConditionIn(Condition):
    """An In condition for negating another `Condition`.

    Attributes:
        condition_type (ConditionTypeEnum): A `ConditionTypeEnum.IN` type of condition.
        value (ConditionValueType): The value to check membership of.
        in_values (List[Union[ConditionValueType, PrimitiveType]]): A list of values to check
            membership in.
    """

    def __init__(
        self, value: ConditionValueType, in_values: List[Union[ConditionValueType, PrimitiveType]]
    ):
        """Constructs a `ConditionIn` condition.

        Args:
            value (ConditionValueType): The execution variable,
                parameter, or property to use for the in comparison.
            in_values (List[Union[ConditionValueType, PrimitiveType]]): The list
                of values to check the in comparison of.
        """
        super(ConditionIn, self).__init__(ConditionTypeEnum.IN)
        self.value = value
        self.in_values = in_values

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        return {
            "Type": self.condition_type.value,
            "Value": self.value.expr,
            "In": [primitive_or_expr(in_value) for in_value in self.in_values],
        }


class ConditionNot(Condition):
    """A Not condition for negating another `Condition`.

    Attributes:
        condition_type (ConditionTypeEnum): A `ConditionTypeEnum.NOT` type of condition.
        expression (Condition): A `Condition` to take the negation of.
    """

    def __init__(self, expression: Condition):
        """Constructs a `ConditionNot` condition.

        Attributes:
            expression (Condition): A `Condition` to take the negation of.
        """
        super(ConditionNot, self).__init__(ConditionTypeEnum.NOT)
        self.expression = expression

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        return {"Type": self.condition_type.value, "Expression": self.expression.to_request()}


class ConditionOr(Condition):
    """An Or condition for logical OR of a list of `Condition` instances.

    Attributes:
        condition_type (ConditionTypeEnum): A `ConditionTypeEnum.OR` type of condition.
        conditions (List[Condition]): A list of `Condition` to logically OR.
    """

    def __init__(self, conditions: List[Condition] = None):
        """Constructs a `ConditionOr` condition.

        Attributes:
            conditions (List[Condition]): A list of `Condition` instances.
        """
        super(ConditionOr, self).__init__(ConditionTypeEnum.OR)
        self.conditions = conditions or []

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        return {
            "Type": self.condition_type.value,
            "Conditions": [condition.to_request() for condition in self.conditions],
        }


def primitive_or_expr(
    value: Union[ExecutionVariable, Expression, PrimitiveType, Parameter, Properties]
) -> Union[Dict[str, str], PrimitiveType]:
    """Provides the expression of the value or leaves the primitive value alone.

    Args:
        value (Union[ConditionValueType, PrimitiveType]): If it is a
            PrimitiveType, leave it be. Otherwise, get the expr of the instance.

    Returns:
        Either the expression of the value or the primitive value.
    """
    if isinstance(value, (ExecutionVariable, Expression, Parameter, Properties)):
        return value.expr
    return value
