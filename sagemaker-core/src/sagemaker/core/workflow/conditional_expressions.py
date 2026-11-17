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

# The service rejects Std:If expressions nested deeper than this.
MAX_IF_DEPTH = 2


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


def _serialize_branch(value: Any) -> Any:
    """Serialize a Then or Else branch, the only place ``Omit`` is allowed."""
    if isinstance(value, Omit):
        return {"Std:Omit": True}
    return _serialize_expression_value(value)


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


def _if_depth(value: Any) -> int:
    """Return how deeply ``If`` expressions are nested in a branch value."""
    if isinstance(value, If):
        branches = [value.then, value.else_] + [branch.then for branch in value.else_if]
        return 1 + max(_if_depth(branch) for branch in branches)
    if isinstance(value, dict):
        return max((_if_depth(child) for child in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return max((_if_depth(child) for child in value), default=0)
    return 0


@attr.s(frozen=True, slots=True)
class Omit(PipelineVariable):
    """Omit the containing API member, map entry, or list element.

    Only valid as a complete ``then`` or ``else_`` branch of :class:`If` or :class:`ElseIf`.
    Anywhere else the definition is rejected when it is built.
    """

    @property
    def expr(self) -> RequestType:
        """Reject an ``Omit`` used outside a branch, which the service would reject."""
        raise ValueError(
            "Omit() is supported only as a complete then or else_ branch of an If expression."
        )

    def to_string(self) -> "Omit":
        """Return self; wrapping in ``Std:Join`` would stop the member from being omitted."""
        return self

    @property
    def _referenced_steps(self) -> List[Union["Step", str]]:
        """Return no references because omission has no operands."""
        return []


@attr.s(frozen=True, slots=True)
class ElseIf:
    """One ordered ``ElseIf`` branch inside an :class:`If`.

    Branches are evaluated in list order by the SageMaker service; the first branch whose
    condition holds supplies its ``then`` value.

    Args:
        condition: Existing SageMaker Pipelines condition expression.
        then: Value selected when this branch's condition evaluates to true.
    """

    condition: Condition = attr.ib(validator=attr.validators.instance_of(Condition))
    then: Any = attr.ib()

    def to_request(self) -> RequestType:
        """Return the canonical serialized form for this ``ElseIf`` entry."""
        return {
            "Condition": _serialize_expression_value(self.condition.to_request()),
            "Then": _serialize_branch(self.then),
        }


@attr.s(frozen=True, slots=True)
class If(PipelineVariable):
    """Select one target-shaped argument value using a pipeline condition.

    Compiles to ``Std:If``. Optional ordered ``else_if`` branches are evaluated after
    ``condition``; the first that holds supplies the value, otherwise ``else_`` is used. The
    ``ElseIf`` key is emitted only when ``else_if`` is non-empty. A branch may hold one
    nested ``If``; the service rejects deeper nesting, so it is rejected here too.

    Args:
        condition: Existing SageMaker Pipelines condition expression.
        then: Value selected when ``condition`` evaluates to true.
        else_: Value selected when no branch matches.
        else_if: Optional ordered list of :class:`ElseIf` branches.
    """

    condition: Condition = attr.ib(validator=attr.validators.instance_of(Condition))
    then: Any = attr.ib()
    else_: Any = attr.ib()
    else_if: List[ElseIf] = attr.ib(
        factory=list,
        converter=lambda values: [] if values is None else list(values),
    )

    @else_if.validator
    def _validate_else_if(
        self,
        _attribute: attr.Attribute,
        value: Sequence[ElseIf],
    ) -> None:
        """Ensure every ``else_if`` entry is an :class:`ElseIf`."""
        for index, branch in enumerate(value):
            if not isinstance(branch, ElseIf):
                raise TypeError(f"else_if[{index}] must be an ElseIf instance.")

    def __attrs_post_init__(self) -> None:
        """Reject nesting deeper than the service accepts."""
        if _if_depth(self) > MAX_IF_DEPTH:
            raise ValueError(f"If expression depth cannot exceed {MAX_IF_DEPTH}.")

    @property
    def expr(self) -> RequestType:
        """Return the canonical recursive ``Std:If`` service expression."""
        body = {
            "Condition": _serialize_expression_value(self.condition.to_request()),
            "Then": _serialize_branch(self.then),
        }
        if self.else_if:
            body["ElseIf"] = [branch.to_request() for branch in self.else_if]
        body["Else"] = _serialize_branch(self.else_)
        return {"Std:If": body}

    def to_string(self) -> "If":
        """Stringify each branch so ``Std:If`` stays outermost and ``Std:Omit`` still omits."""
        return self._map_branches(_branch_to_string)

    def _map_branches(self, fn: Callable[[Any], Any]) -> "If":
        def apply(value):
            if isinstance(value, If):
                return value._map_branches(fn)
            if isinstance(value, Omit):
                return value
            return fn(value)

        return If(
            condition=self.condition,
            then=apply(self.then),
            else_=apply(self.else_),
            else_if=[ElseIf(condition=b.condition, then=apply(b.then)) for b in self.else_if],
        )

    @property
    def _referenced_steps(self) -> List[Union["Step", str]]:
        """Collect references conservatively from the condition and every branch."""
        references: List[Union["Step", str]] = list(self.condition._referenced_steps)
        references.extend(_referenced_steps(self.then))
        for branch in self.else_if:
            references.extend(branch.condition._referenced_steps)
            references.extend(_referenced_steps(branch.then))
        references.extend(_referenced_steps(self.else_))
        return references


__all__ = ["ElseIf", "If", "Omit"]
