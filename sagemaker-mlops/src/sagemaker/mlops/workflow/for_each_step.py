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
"""The `ForEach` step definitions for SageMaker Pipelines Workflows."""

from __future__ import absolute_import

import re
from enum import Enum
from typing import Any, List, Optional, Union

from sagemaker.core.helper.pipeline_variable import PipelineVariable, RequestType
from sagemaker.core.workflow.properties import Properties
from sagemaker.core.workflow.utilities import list_to_request
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum

MAX_CONCURRENCY_MIN = 1
MAX_CONCURRENCY_MAX = 50

_CHILD_OUTPUT_NAMESPACE = "ChildOutput"
_CURRENT_ITEM_PATH = "ForEach.CurrentItem"

# Step types that may not appear in a ForEachBody.
_DISALLOWED_BODY_STEP_TYPES = frozenset(
    {StepTypeEnum.CONDITION, StepTypeEnum.FAIL, StepTypeEnum.FOR_EACH}
)


class _ForEachPropertyReference(PipelineVariable):
    """A drillable ``{"Get": <path>}`` reference used by the ForEach step.

    Supports attribute access (``ref.Member``) and indexing (``ref[0]``,
    ``ref["key"]``) to build up a property path, mirroring the drill-down
    behavior of :class:`~sagemaker.core.workflow.properties.Properties`.
    """

    def __init__(self, path: str, referenced_steps: Optional[List[Any]] = None):
        """Initialize a _ForEachPropertyReference.

        Args:
            path (str): The full ``Get`` expression path.
            referenced_steps (List): Steps this reference depends on.
        """
        self._path = path
        self._refs = list(referenced_steps) if referenced_steps else []

    def __getattr__(self, name: str) -> "_ForEachPropertyReference":
        """Drill into a member of the referenced structure."""
        if name.startswith("_"):
            raise AttributeError(name)
        return _ForEachPropertyReference(f"{self._path}.{name}", self._refs)

    def __getitem__(self, item: Union[int, str]) -> "_ForEachPropertyReference":
        """Drill into a list index or map key of the referenced structure."""
        if isinstance(item, int):
            return _ForEachPropertyReference(f"{self._path}[{item}]", self._refs)
        return _ForEachPropertyReference(f"{self._path}['{item}']", self._refs)

    @property
    def expr(self) -> RequestType:
        """The 'Get' expression dict for this reference."""
        return {"Get": self._path}

    @property
    def _referenced_steps(self) -> List[Any]:
        """List of steps that this reference depends on."""
        return self._refs


class _TerminalPropertyReference(_ForEachPropertyReference):
    """A property reference that rejects further drilling.

    Used for paths that are at their maximum allowed depth (e.g. a single-level
    CurrentItem field like ``ForEach.CurrentItem.name``).
    """

    def __getattr__(self, name: str) -> "_ForEachPropertyReference":
        """Reject further attribute access."""
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"Cannot drill deeper into '{self._path}'. The service only "
            f"supports one level of field access on CurrentItem "
            f"(e.g. CurrentItem().name, not CurrentItem().a.b)."
        )

    def __getitem__(self, item: Union[int, str]) -> "_ForEachPropertyReference":
        """Reject indexing on a terminal reference."""
        raise TypeError(
            f"Cannot drill deeper into '{self._path}'. The service only "
            f"supports one level of field access on CurrentItem."
        )


class CurrentItem(_ForEachPropertyReference):
    """Reference to the current iteration item inside a ForEach body step.

    Valid ONLY within the steps of a ``ForEachStep``'s ``for_each_body``.
    The bare reference resolves to the whole item (for scalar items). For
    JSON-object items, drill one level into a top-level field::

        item = CurrentItem()
        item.expr             # {"Get": "ForEach.CurrentItem"}
        item.name.expr        # {"Get": "ForEach.CurrentItem.name"}

    Only one level of field access is supported. Deeper paths
    (``item.a.b``) and bracket indexing (``item["score"]``) are rejected.
    """

    def __init__(self):
        """Initialize a CurrentItem reference."""
        super().__init__(_CURRENT_ITEM_PATH)

    def __getattr__(self, name: str) -> "_ForEachPropertyReference":
        """Allow one level of field access, returning a terminal reference."""
        if name.startswith("_"):
            raise AttributeError(name)
        return _TerminalPropertyReference(f"{self._path}.{name}", self._refs)

    def __getitem__(self, item: Union[int, str]) -> "_ForEachPropertyReference":
        """Reject indexing; the CurrentItem grammar only supports dot paths."""
        raise TypeError(
            f"CurrentItem does not support indexing ('[{item!r}]'). Use attribute "
            f"access instead, e.g. CurrentItem().{item} for field access."
        )


class ChildOutput(_ForEachPropertyReference):
    """Reference to a ForEach body step's output, for use in ``ForEachSelector.field``.

    Valid ONLY as the ``field`` of a :class:`ForEachSelector`, and must target a
    body step of the same ForEach step, drilling at least one member deep::

        ChildOutput(process_step).OutputParameters["InstanceCount"].expr
        # {"Get": "ChildOutput.ProcessItem.OutputParameters['InstanceCount']"}

    Args:
        body_step (Union[str, Step]): The body step (or its name) whose output
            the selector compares across children.
    """

    def __init__(self, body_step: Union[str, Step]):
        """Initialize a ChildOutput reference rooted at the given body step."""
        step_name = body_step.name if isinstance(body_step, Step) else body_step
        super().__init__(f"{_CHILD_OUTPUT_NAMESPACE}.{step_name}")


class _ChildOutputsAccessor:
    """Indexed accessor for a ForEach step's per-child outputs.

    ``ChildOutputs`` MUST be indexed (``properties.ChildOutputs[0].<member>``);
    a bare ``ChildOutputs.<member>`` reference is invalid, so this accessor is
    intentionally not a ``PipelineVariable`` itself.
    """

    def __init__(self, step: Step):
        """Initialize a _ChildOutputsAccessor for the given ForEach step."""
        self._step = step

    def __getitem__(self, index: int) -> _ForEachPropertyReference:
        """Get a drillable reference to the output of the child at ``index``."""
        if not isinstance(index, int):
            raise TypeError(
                f"ChildOutputs of ForEach step '{self._step.name}' must be indexed "
                f"with an integer, e.g. properties.ChildOutputs[0]."
            )
        return _ForEachPropertyReference(
            f"Steps.{self._step.name}.ChildOutputs[{index}]", [self._step]
        )


class ForEachSelectorPickType(Enum):
    """Enum of pick strategies for a :class:`ForEachSelector`."""

    MAX_BY = "MaxBy"
    MIN_BY = "MinBy"


class ForEachSelector:
    """Selects a single child of a ForEach step by comparing a field across children.

    The selected child's output is exposed to downstream steps via
    ``Steps.<forEachName>.Selected.<member>``.

    Args:
        pick (Union[str, ForEachSelectorPickType]): The pick strategy,
            ``"MaxBy"`` or ``"MinBy"``.
        field (ChildOutput): A ``ChildOutput`` reference targeting a body step
            of the same ForEach step, drilled at least one member deep.
    """

    def __init__(
        self,
        pick: Union[str, ForEachSelectorPickType],
        field: PipelineVariable,
    ):
        """Initialize a ForEachSelector."""
        if pick is None:
            raise ValueError("ForEachSelector requires 'pick' (MaxBy or MinBy).")
        if isinstance(pick, str):
            try:
                pick = ForEachSelectorPickType(pick)
            except ValueError:
                raise ValueError(
                    f"Invalid ForEachSelector pick type: '{pick}'. "
                    f"Valid values: "
                    f"{[member.value for member in ForEachSelectorPickType]}."
                )
        if not isinstance(pick, ForEachSelectorPickType):
            raise ValueError(
                f"ForEachSelector 'pick' must be a str or ForEachSelectorPickType, "
                f"got: {type(pick)}."
            )
        if field is None:
            raise ValueError("ForEachSelector requires 'field' (a ChildOutput reference).")
        if not isinstance(field, PipelineVariable):
            raise ValueError(
                "ForEachSelector 'field' must be a ChildOutput reference, e.g. "
                'ChildOutput(body_step).OutputParameters["MyOutput"], '
                f"got: {type(field)}."
            )
        self.pick = pick
        self.field = field

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        return {
            "Pick": self.pick.value,
            "Field": self.field,
        }


class ForEachStep(Step):
    """ForEach step for dynamic fan-out in SageMaker Pipelines Workflows.

    A ForEach step takes a list of items (resolved at runtime), runs its body
    step once per item as child steps, enforces a concurrency cap, and exposes
    per-child outputs plus aggregate results to downstream steps.

    Child steps get IDs of the form ``<bodyStepName>-<index>`` (e.g.
    ``ProcessItem-0``). Because of this, body step names must be unique across
    all ForEach bodies in the pipeline, and no top-level step name may look
    like ``<bodyStepName>-<digits>`` — the service rejects such collisions.

    Downstream steps can reference the parent step's properties:

    - ``properties.ChildOutputs[i].<member>`` — per-child output (delegates to
      the body step's property schema; must be indexed)
    - ``properties.Selected.<member>`` — output of the selector-picked child
    - ``properties.Status`` (String: Succeeded | PartiallySucceeded | Failed)
    - ``properties.InputItemCount``, ``properties.TotalStepCount``,
      ``properties.SucceededStepCount``, ``properties.FailedStepCount``,
      ``properties.StoppedStepCount`` (Integers)
    """

    _AGGREGATE_PROPERTY_MEMBERS = (
        "Status",
        "InputItemCount",
        "TotalStepCount",
        "SucceededStepCount",
        "FailedStepCount",
        "StoppedStepCount",
    )

    def __init__(
        self,
        name: str,
        iterable_items: PipelineVariable = None,
        max_concurrency: int = None,
        for_each_body: List[Step] = None,
        selector: Optional[ForEachSelector] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step]]] = None,
    ):
        """Construct a ForEachStep.

        Args:
            name (str): The name of the `ForEachStep`.
            iterable_items (PipelineVariable): REQUIRED. The items to iterate
                over. A property reference (e.g. a `ParameterString` or a step
                output) whose value must resolve to a JSON list string at
                runtime, such as ``'["item-a", "item-b"]'``. The service
                currently only accepts the property-reference form; literal
                values (including literal lists) are rejected ("a literal value
                cannot resolve to a list"). Items are commonly JSON-object
                strings whose fields are addressed via ``CurrentItem().<field>``
                inside the body step.
            max_concurrency (int): REQUIRED. The maximum number of child steps
                that run concurrently. Range: [1, 50].
            for_each_body (List[Step]): REQUIRED. The step(s) to run once per
                item. v1 supports exactly ONE body step; the list signature is
                future-proofing for multi-step bodies. Disallowed body step
                types: Condition, Fail, ForEach (no nesting). Body steps may
                reference the current item via :class:`CurrentItem`.
            selector (ForEachSelector): Optional selector that picks a single
                child by comparing a ``ChildOutput`` field across children.
            display_name (str): The display name of the `ForEachStep`.
            description (str): The description of the `ForEachStep`.
            depends_on (List[Union[str, Step]]): The list of `Step` names or
                `Step` instances that the current `Step` depends on.
        """
        super(ForEachStep, self).__init__(
            name, display_name, description, StepTypeEnum.FOR_EACH, depends_on
        )

        self._validate_iterable_items(name, iterable_items)
        self._validate_max_concurrency(name, max_concurrency)
        self._validate_for_each_body(name, for_each_body)
        self._validate_selector(name, selector, for_each_body)

        self.iterable_items = iterable_items
        self.max_concurrency = max_concurrency
        self.for_each_body = list(for_each_body)
        self.selector = selector

        root_prop = Properties(step_name=name)
        for member in ForEachStep._AGGREGATE_PROPERTY_MEMBERS:
            root_prop.__dict__[member] = Properties(step_name=name, path=member)
        root_prop.__dict__["ChildOutputs"] = _ChildOutputsAccessor(self)
        root_prop.__dict__["Selected"] = _ForEachPropertyReference(f"Steps.{name}.Selected", [self])
        self._properties = root_prop

    @staticmethod
    def _validate_iterable_items(name: str, iterable_items: Any):
        """Validate the iterable_items argument."""
        if iterable_items is None:
            raise ValueError(
                f"ForEach step '{name}': iterable_items is required. Provide a "
                f"property reference that resolves to a JSON list string at "
                f"runtime."
            )
        if not isinstance(iterable_items, PipelineVariable):
            raise ValueError(
                f"ForEach step '{name}': iterable_items must be a property "
                f"reference that resolves to a JSON list string at runtime; "
                f"a literal value cannot resolve to a list. "
                f"Got: {type(iterable_items)}."
            )

    @staticmethod
    def _validate_max_concurrency(name: str, max_concurrency: Any):
        """Validate the max_concurrency argument."""
        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
            or not MAX_CONCURRENCY_MIN <= max_concurrency <= MAX_CONCURRENCY_MAX
        ):
            raise ValueError(
                f"ForEach step '{name}': max_concurrency is required and must be "
                f"an integer in the range "
                f"[{MAX_CONCURRENCY_MIN}, {MAX_CONCURRENCY_MAX}]. "
                f"Got: {max_concurrency}."
            )

    @staticmethod
    def _validate_for_each_body(name: str, for_each_body: Any):
        """Validate the for_each_body argument."""
        if not for_each_body:
            raise ValueError(
                f"ForEach step '{name}': for_each_body must be a non-empty list " f"of steps."
            )
        if len(for_each_body) != 1:
            raise ValueError(
                f"ForEach step '{name}': for_each_body currently supports exactly "
                f"one step; got {len(for_each_body)}. Multi-step bodies are not "
                f"yet supported by the service."
            )
        for body_step in for_each_body:
            if not isinstance(body_step, Step):
                raise ValueError(
                    f"ForEach step '{name}': for_each_body entries must be Step "
                    f"instances. Got: {type(body_step)}."
                )
            if body_step.step_type in _DISALLOWED_BODY_STEP_TYPES:
                raise ValueError(
                    f"ForEach step '{name}': step type "
                    f"'{body_step.step_type.value}' is not allowed in "
                    f"for_each_body. Disallowed types: Condition, Fail, ForEach."
                )

    @staticmethod
    def _validate_selector(
        name: str, selector: Optional[ForEachSelector], for_each_body: List[Step]
    ):
        """Validate that the selector field targets a body step of this ForEach."""
        if selector is None:
            return
        if not isinstance(selector, ForEachSelector):
            raise ValueError(
                f"ForEach step '{name}': selector must be a ForEachSelector "
                f"instance. Got: {type(selector)}."
            )
        field_path = selector.field.expr.get("Get")
        prefix = f"{_CHILD_OUTPUT_NAMESPACE}."
        if not isinstance(field_path, str) or not field_path.startswith(prefix):
            raise ValueError(
                f"ForEach step '{name}': Selector.Field must be a ChildOutput "
                f"reference (e.g. ChildOutput(body_step).<member>); other "
                f"namespaces are constant across children. Got: {field_path}."
            )
        remainder = field_path[len(prefix) :]
        match = re.match(r"^([^.\[\]]+)(.+)$", remainder)
        body_step_names = [step.name for step in for_each_body]
        if not match or match.group(1) not in body_step_names:
            raise ValueError(
                f"ForEach step '{name}': Selector.Field must target a body step "
                f"of this ForEach step ({body_step_names}) and drill at least "
                f"one member deep. Got: {field_path}."
            )

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to define the ForEach fan-out."""
        args = {
            "IterableItems": self.iterable_items,
            "MaxConcurrency": self.max_concurrency,
        }
        if self.selector is not None:
            args["Selector"] = self.selector.to_request()
        args["ForEachBody"] = list_to_request(self.for_each_body)
        return args

    @property
    def step_only_arguments(self) -> RequestType:
        """Argument dict pertaining to the step only, and not the body steps."""
        args = {
            "IterableItems": self.iterable_items,
            "MaxConcurrency": self.max_concurrency,
        }
        if self.selector is not None:
            args["Selector"] = self.selector.to_request()
        return args

    @property
    def properties(self):
        """A Properties object exposing the ForEach step's aggregate results.

        Members: ``Status``, ``InputItemCount``, ``TotalStepCount``,
        ``SucceededStepCount``, ``FailedStepCount``, ``StoppedStepCount``,
        ``ChildOutputs[i]`` and ``Selected``.
        """
        return self._properties
