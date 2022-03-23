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
"""This module stores filters related to SageMaker JumpStart."""
from __future__ import absolute_import
from ast import literal_eval
from enum import Enum
from typing import List, Union, Any

from sagemaker.jumpstart.types import JumpStartDataHolderType


class BooleanValues(str, Enum):
    """Enum class for boolean values."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    UNEVALUATED = "unevaluated"


class FilterOperators(str, Enum):
    """Enum class for filter operators for JumpStart models."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"


FILTER_OPERATOR_STRING_MAPPINGS = {
    FilterOperators.EQUALS: ["===", "==", "equals", "is"],
    FilterOperators.NOT_EQUALS: ["!==", "!=", "not equals", "is not"],
    FilterOperators.IN: ["in"],
    FilterOperators.NOT_IN: ["not in"],
}


class Operand:
    """Operand class for filtering JumpStart content."""

    def __init__(
        self, unresolved_value: Any, resolved_value: BooleanValues = BooleanValues.UNEVALUATED
    ):
        self.unresolved_value = unresolved_value
        self.resolved_value = resolved_value

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield self

    def eval(self) -> None:
        """Evaluates operand."""
        return

    @staticmethod
    def validate_operand(operand: Any) -> Any:
        """Validate operand and return ``Operand`` object."""
        if isinstance(operand, str):
            if operand.lower() == BooleanValues.TRUE.lower():
                operand = Operand(operand, resolved_value=BooleanValues.TRUE)
            elif operand.lower() == BooleanValues.FALSE.lower():
                operand = Operand(operand, resolved_value=BooleanValues.FALSE)
            elif operand.lower() == BooleanValues.UNKNOWN.lower():
                operand = Operand(operand, resolved_value=BooleanValues.UNKNOWN)
            else:
                operand = Operand(parse_filter_string(operand))
        elif not issubclass(type(operand), Operand):
            raise RuntimeError()
        return operand


class Operator(Operand):
    """Operator class for filtering JumpStart content."""

    def __init__(
        self,
        resolved_value: BooleanValues = BooleanValues.UNEVALUATED,
        unresolved_value: Any = None,
    ):
        """Initializes ``Operator`` instance.

        Args:
            resolved_value (BooleanValues): Optional. The resolved value of the operator.
                (Default: BooleanValues.UNEVALUATED).
            unresolved_value (Any): Optional. The unresolved value of the operator.
                (Default: None).
        """
        super().__init__(unresolved_value=unresolved_value, resolved_value=resolved_value)

    def eval(self) -> None:
        """Evaluates operator."""
        return

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield self


class And(Operator):
    """And operator class for filtering JumpStart content."""

    def __init__(
        self,
        *operands: Union[Operand, str],
    ) -> None:
        """Instantiates And object.

        Args:
            operand (Operand): Operand for And-ing.
        """
        self.operands: List[Operand] = list(operands)  # type: ignore
        for i in range(len(self.operands)):
            self.operands[i] = Operand.validate_operand(self.operands[i])
        super().__init__()

    def eval(self) -> None:
        """Evaluates operator."""
        incomplete_expression = False
        for operand in self.operands:
            if not issubclass(type(operand), Operand):
                raise RuntimeError()
            if operand.resolved_value == BooleanValues.UNEVALUATED:
                operand.eval()
            if operand.resolved_value == BooleanValues.UNEVALUATED:
                raise RuntimeError()
            if not isinstance(operand.resolved_value, BooleanValues):
                raise RuntimeError()
            if operand.resolved_value == BooleanValues.FALSE:
                self.resolved_value = BooleanValues.FALSE
                return
            if operand.resolved_value == BooleanValues.UNKNOWN:
                incomplete_expression = True
        if not incomplete_expression:
            self.resolved_value = BooleanValues.TRUE
        else:
            self.resolved_value = BooleanValues.UNKNOWN

    def __iter__(self) -> Any:
        """Returns an iterator."""
        for operand in self.operands:
            yield from operand
        yield self


class Constant(Operator):
    """Constant operator class for filtering JumpStart content."""

    def __init__(
        self,
        constant: BooleanValues,
    ):
        """Instantiates Constant operator object.

        Args:
            constant (BooleanValues]): Value of constant.
        """
        super().__init__(constant)

    def eval(self) -> None:
        """Evaluates constant"""
        return

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield self


class Identity(Operator):
    """Identity operator class for filtering JumpStart content."""

    def __init__(
        self,
        operand: Union[Operand, str],
    ):
        """Instantiates Identity object.

        Args:
            operand (Union[Operand, str]): Operand for identity operation.
        """
        super().__init__()
        self.operand = Operand.validate_operand(operand)

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield self
        yield from self.operand

    def eval(self) -> Any:
        """Evaluates operator."""
        if not issubclass(type(self.operand), Operand):
            raise RuntimeError()
        if self.operand.resolved_value == BooleanValues.UNEVALUATED:
            self.operand.eval()
        if self.operand.resolved_value == BooleanValues.UNEVALUATED:
            raise RuntimeError()
        if not isinstance(self.operand.resolved_value, BooleanValues):
            raise RuntimeError(self.operand.resolved_value)
        self.resolved_value = self.operand.resolved_value


class Or(Operator):
    """Or operator class for filtering JumpStart content."""

    def __init__(
        self,
        *operands: Union[Operand, str],
    ) -> None:
        """Instantiates Or object.

        Args:
            operands (Operand): Operand for Or-ing.
        """
        self.operands: List[Operand] = list(operands)  # type: ignore
        for i in range(len(self.operands)):
            self.operands[i] = Operand.validate_operand(self.operands[i])
        super().__init__()

    def eval(self) -> None:
        """Evaluates operator."""
        incomplete_expression = False
        for operand in self.operands:
            if not issubclass(type(operand), Operand):
                raise RuntimeError()
            if operand.resolved_value == BooleanValues.UNEVALUATED:
                operand.eval()
            if operand.resolved_value == BooleanValues.UNEVALUATED:
                raise RuntimeError()
            if not isinstance(operand.resolved_value, BooleanValues):
                raise RuntimeError()
            if operand.resolved_value == BooleanValues.TRUE:
                self.resolved_value = BooleanValues.TRUE
                return
            if operand.resolved_value == BooleanValues.UNKNOWN:
                incomplete_expression = True
        if not incomplete_expression:
            self.resolved_value = BooleanValues.FALSE
        else:
            self.resolved_value = BooleanValues.UNKNOWN

    def __iter__(self) -> Any:
        """Returns an iterator."""
        for operand in self.operands:
            yield from operand
        yield self


class Not(Operator):
    """Not operator class for filtering JumpStart content."""

    def __init__(
        self,
        operand: Union[Operand, str],
    ) -> None:
        """Instantiates Not object.

        Args:
            operand (Operand): Operand for Not-ing.
        """
        self.operand: Operand = Operand.validate_operand(operand)
        super().__init__()

    def eval(self) -> None:
        """Evaluates operator."""

        if not issubclass(type(self.operand), Operand):
            raise RuntimeError()
        if self.operand.resolved_value == BooleanValues.UNEVALUATED:
            self.operand.eval()
        if self.operand.resolved_value == BooleanValues.UNEVALUATED:
            raise RuntimeError()
        if not isinstance(self.operand.resolved_value, BooleanValues):
            raise RuntimeError()
        if self.operand.resolved_value == BooleanValues.TRUE:
            self.resolved_value = BooleanValues.FALSE
            return
        if self.operand.resolved_value == BooleanValues.FALSE:
            self.resolved_value = BooleanValues.TRUE
            return
        self.resolved_value = BooleanValues.UNKNOWN

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield from self.operand
        yield self


class ModelFilter(JumpStartDataHolderType):
    """Data holder class to store model filters.

    For a given filter string "task == ic", the key corresponds to
    "task" and the value corresponds to "ic", with the operation being
    "==".
    """

    __slots__ = ["key", "value", "operator"]

    def __init__(self, key: str, value: str, operator: str):
        """Instantiates ``ModelFilter`` object.

        Args:
            key (str): The key in metadata for the model filter.
            value (str): The value of the metadata for the model filter.
            operator (str): The operator used in the model filter.
        """
        self.key = key
        self.value = value
        self.operator = operator


def parse_filter_string(filter_string: str) -> ModelFilter:
    """Parse filter string and return a serialized ``ModelFilter`` object.

    Args:
        filter_string (str): The filter string to be serialized to an object.
    """

    pad_alphabetic_operator = (
        lambda operator: " " + operator + " "
        if any(character.isalpha() for character in operator)
        else operator
    )

    acceptable_operators_in_parse_order = (
        list(
            map(
                pad_alphabetic_operator, FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_EQUALS]
            )
        )
        + list(
            map(pad_alphabetic_operator, FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_IN])
        )
        + list(
            map(pad_alphabetic_operator, FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.EQUALS])
        )
        + list(map(pad_alphabetic_operator, FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.IN]))
    )
    for operator in acceptable_operators_in_parse_order:
        split_filter_string = filter_string.split(operator)
        if len(split_filter_string) == 2:
            return ModelFilter(
                split_filter_string[0].strip(), split_filter_string[1].strip(), operator.strip()
            )
    raise RuntimeError(f"Cannot parse filter string: {filter_string}")


def evaluate_filter_expression(  # pylint: disable=too-many-return-statements
    model_filter: ModelFilter, cached_model_value: Any
) -> BooleanValues:
    """Evaluates model filter with cached model spec value, returns boolean.

    Args:
        model_filter (ModelFilter): The model filter for evaluation.
        cached_model_value (Any): The value in the model manifest/spec that should be used to
            evaluate the filter.
    """
    if model_filter.operator in FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.EQUALS]:
        model_filter_value = model_filter.value
        if isinstance(cached_model_value, bool):
            cached_model_value = str(cached_model_value).lower()
            model_filter_value = model_filter.value.lower()
        if str(model_filter_value) == str(cached_model_value):
            return BooleanValues.TRUE
        return BooleanValues.FALSE
    if model_filter.operator in FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_EQUALS]:
        if isinstance(cached_model_value, bool):
            cached_model_value = str(cached_model_value).lower()
            model_filter.value = model_filter.value.lower()
        if str(model_filter.value) == str(cached_model_value):
            return BooleanValues.FALSE
        return BooleanValues.TRUE
    if model_filter.operator in FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.IN]:
        if cached_model_value in literal_eval(model_filter.value):
            return BooleanValues.TRUE
        return BooleanValues.FALSE
    if model_filter.operator in FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_IN]:
        if cached_model_value in literal_eval(model_filter.value):
            return BooleanValues.FALSE
        return BooleanValues.TRUE
    raise RuntimeError(f"Bad operator: {model_filter.operator}")
