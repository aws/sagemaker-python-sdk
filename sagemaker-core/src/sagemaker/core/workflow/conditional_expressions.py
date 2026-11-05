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
"""Conditional argument expressions for SageMaker Pipelines."""

from __future__ import absolute_import

from typing import Any, Callable, List, Sequence, Union, TYPE_CHECKING

import attr

from sagemaker.core.helper.pipeline_variable import PipelineVariable, RequestType
from sagemaker.core.workflow.conditions import Condition

if TYPE_CHECKING:
    from sagemaker.mlops.workflow.steps import Step


def _serialize_expression_value(value: Any) -> Any:
    """Recursively translate pipeline variables into service expression dictionaries."""
    if isinstance(value, PipelineVariable):
        return value.expr
    if isinstance(value, dict):
        return value.__class__(
            (key, _serialize_expression_value(child)) for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return [_serialize_expression_value(child) for child in value]
    return value


def _branch_to_string(value: Any) -> Any:
    """Stringify one branch value, leaving structures as they are."""
    if isinstance(value, PipelineVariable):
        return value.to_string()
    if isinstance(value, (bool, int, float)):
        return str(value)
    return value


def _referenced_steps(value: Any) -> List[Union["Step", str]]:
    """Collect upstream steps from a nested expression value."""
    if isinstance(value, PipelineVariable):
        return list(value._referenced_steps)
    if isinstance(value, dict):
        result = []
        for child in value.values():
            result.extend(_referenced_steps(child))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for child in value:
            result.extend(_referenced_steps(child))
        return result
    return []


@attr.s(frozen=True, slots=True)
class Omit(PipelineVariable):
    """Explicitly omit the containing API member, map entry, or list element."""

    @property
    def expr(self) -> RequestType:
        """Return the canonical ``Std:Omit`` service expression."""
        return {"Std:Omit": True}

    def to_string(self) -> "Omit":
        """Return self; wrapping in ``Std:Join`` would stop the member from being omitted."""
        return self

    @property
    def _referenced_steps(self) -> List[Union["Step", str]]:
        """Return no references because omission has no operands."""
        return []


@attr.s(frozen=True, slots=True)
class ElseIfBranch:
    """One ordered ``ElseIf`` branch inside an :class:`IfExpression`.

    Each branch carries its own :class:`Condition` and target-shaped value. Branches are
    evaluated in list order by the SageMaker service; the first branch whose condition holds
    supplies its ``then_value`` as the selected result.

    Args:
        condition: Existing SageMaker Pipelines condition expression.
        then_value: Value selected when this branch's condition evaluates to true.
    """

    condition: Condition = attr.ib(validator=attr.validators.instance_of(Condition))
    then_value: Any = attr.ib()

    def to_request(self) -> RequestType:
        """Return the canonical serialized form for this ``ElseIf`` entry."""
        return {
            "Condition": _serialize_expression_value(self.condition.to_request()),
            "Then": _serialize_expression_value(self.then_value),
        }


@attr.s(frozen=True, slots=True)
class IfExpression(PipelineVariable):
    """Select one target-shaped argument value using a pipeline condition.

    An optional ordered list of ``ElseIfBranch`` entries is evaluated between the outer
    ``then_value`` and the outer ``else_value``. The first branch whose condition holds
    supplies the selected value. If no branch matches, ``else_value`` is used.

    ``ElseIfBranch`` entries and further nested ``IfExpression`` instances can be mixed
    freely; the wire format emits the ``ElseIf`` key only when the list is non-empty so
    existing recursive-``Else`` pipelines remain byte-identical.

    Args:
        condition: Existing SageMaker Pipelines condition expression.
        then_value: Value selected when ``condition`` evaluates to true.
        else_value: Value selected when no branch matches.
        else_if_branches: Optional ordered list of :class:`ElseIfBranch` entries.
    """

    condition: Condition = attr.ib(validator=attr.validators.instance_of(Condition))
    then_value: Any = attr.ib()
    else_value: Any = attr.ib()
    else_if_branches: List[ElseIfBranch] = attr.ib(
        factory=list,
        converter=lambda values: [] if values is None else list(values),
    )

    @else_if_branches.validator
    def _validate_else_if_branches(
        self,
        _attribute: attr.Attribute,
        value: Sequence[ElseIfBranch],
    ) -> None:
        """Ensure every ``ElseIfBranch`` supplied uses a valid :class:`Condition`."""
        for index, branch in enumerate(value):
            if not isinstance(branch, ElseIfBranch):
                raise TypeError(f"else_if_branches[{index}] must be an ElseIfBranch instance.")

    @property
    def expr(self) -> RequestType:
        """Return the canonical recursive ``Std:If`` service expression."""
        body = {
            "Condition": _serialize_expression_value(self.condition.to_request()),
            "Then": _serialize_expression_value(self.then_value),
        }
        if self.else_if_branches:
            body["ElseIf"] = [branch.to_request() for branch in self.else_if_branches]
        body["Else"] = _serialize_expression_value(self.else_value)
        return {"Std:If": body}

    def to_string(self) -> "IfExpression":
        """Stringify each branch so ``Std:If`` stays outermost and ``Std:Omit`` still omits."""
        return self._map_branches(_branch_to_string)

    def _map_branches(self, fn: Callable[[Any], Any]) -> "IfExpression":
        def apply(value):
            if isinstance(value, IfExpression):
                return value._map_branches(fn)
            if isinstance(value, Omit):
                return value
            return fn(value)

        return IfExpression(
            condition=self.condition,
            then_value=apply(self.then_value),
            else_value=apply(self.else_value),
            else_if_branches=[
                ElseIfBranch(condition=b.condition, then_value=apply(b.then_value))
                for b in self.else_if_branches
            ],
        )

    @property
    def _referenced_steps(self) -> List[Union["Step", str]]:
        """Collect references conservatively from the condition and every branch."""
        references: List[Union["Step", str]] = list(self.condition._referenced_steps)
        references.extend(_referenced_steps(self.then_value))
        for branch in self.else_if_branches:
            references.extend(branch.condition._referenced_steps)
            references.extend(_referenced_steps(branch.then_value))
        references.extend(_referenced_steps(self.else_value))
        return references


__all__ = ["ElseIfBranch", "IfExpression", "Omit"]
