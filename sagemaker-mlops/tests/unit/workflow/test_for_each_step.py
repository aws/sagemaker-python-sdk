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
"""Unit tests for workflow for_each_step."""

from __future__ import absolute_import

import json

import pytest
from unittest.mock import Mock

from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.properties import Properties
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.mlops.workflow.for_each_step import (
    ChildOutput,
    CurrentItem,
    ForEachSelector,
    ForEachSelectorPickType,
    ForEachStep,
)
from sagemaker.mlops.workflow.lambda_step import (
    LambdaOutput,
    LambdaOutputTypeEnum,
    LambdaStep,
)
from sagemaker.mlops.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


class CustomStep(Step):
    """A minimal concrete step for testing."""

    def __init__(self, name, step_type=StepTypeEnum.TRAINING, input_data=None, depends_on=None):
        super(CustomStep, self).__init__(name, None, None, step_type, depends_on)
        self._input_data = input_data
        self._properties = Properties(step_name=name, step=self)

    @property
    def arguments(self):
        return {"input_data": self._input_data} if self._input_data is not None else {}

    @property
    def properties(self):
        return self._properties


@pytest.fixture
def mock_session():
    session = Mock()
    session.boto_session.client.return_value = Mock()
    session.sagemaker_client = Mock()
    session.local_mode = False
    session.sagemaker_config = {}
    return session


@pytest.fixture
def item_list_parameter():
    return ParameterString(name="ItemList")


def make_lambda_body_step(function_arn="arn:aws:lambda:us-west-2:123456789012:function:body"):
    lambda_func = Mock()
    lambda_func.function_arn = function_arn
    lambda_func.zipped_code_dir = None
    lambda_func.script = None
    current_item = CurrentItem()
    return LambdaStep(
        name="ProcessItem",
        lambda_func=lambda_func,
        inputs={
            "item_name": current_item.name,
            "item_score": current_item.score,
        },
        outputs=[
            LambdaOutput(output_name="ProcessedScore", output_type=LambdaOutputTypeEnum.Integer),
            LambdaOutput(output_name="ResultPath", output_type=LambdaOutputTypeEnum.String),
        ],
    )


def make_for_each_step(body_step=None, selector=None, **kwargs):
    body_step = body_step or make_lambda_body_step()
    default_kwargs = dict(
        name="ForEachItems",
        iterable_items=ParameterString(name="ItemList"),
        max_concurrency=2,
        for_each_body=[body_step],
        selector=selector,
    )
    default_kwargs.update(kwargs)
    return ForEachStep(**default_kwargs)


# ---------------------------------------------------------------------------
# CurrentItem / ChildOutput references
# ---------------------------------------------------------------------------


def test_current_item_bare_expr():
    assert CurrentItem().expr == {"Get": "ForEach.CurrentItem"}


def test_current_item_drilled_expr():
    assert CurrentItem().name.expr == {"Get": "ForEach.CurrentItem.name"}


def test_current_item_multi_level_drill_rejected():
    # the service only supports one level of field access
    ref = CurrentItem().a
    with pytest.raises(AttributeError, match="Cannot drill deeper"):
        ref.b


def test_current_item_indexing_rejected():
    # the CurrentItem grammar only supports dot paths, not bracket indexing
    with pytest.raises(TypeError, match="does not support indexing"):
        CurrentItem()["score"]


def test_current_item_has_no_referenced_steps():
    assert CurrentItem().name._referenced_steps == []


def test_child_output_expr_from_step_and_str():
    body = make_lambda_body_step()
    ref = ChildOutput(body).OutputParameters["ProcessedScore"]
    assert ref.expr == {"Get": "ChildOutput.ProcessItem.OutputParameters['ProcessedScore']"}
    ref_from_str = ChildOutput("ProcessItem").Loss
    assert ref_from_str.expr == {"Get": "ChildOutput.ProcessItem.Loss"}
    assert ref._referenced_steps == []


# ---------------------------------------------------------------------------
# to_request serialization
# ---------------------------------------------------------------------------


def test_for_each_step_to_request_reference_items(item_list_parameter):
    step = make_for_each_step(iterable_items=item_list_parameter)
    request = step.to_request()
    assert request["Name"] == "ForEachItems"
    assert request["Type"] == "ForEach"
    args = request["Arguments"]
    assert args["IterableItems"] is item_list_parameter
    assert args["MaxConcurrency"] == 2
    assert "Selector" not in args
    assert len(args["ForEachBody"]) == 1
    assert args["ForEachBody"][0]["Name"] == "ProcessItem"
    assert args["ForEachBody"][0]["Type"] == "Lambda"


def test_for_each_step_literal_array_items_rejected():
    # the service currently only accepts the property-reference form
    literal_item = '{"name": "lit-c", "score": 5}'
    ref_item = ParameterString(name="ItemOne")
    with pytest.raises(ValueError, match="literal value cannot resolve to a list"):
        make_for_each_step(iterable_items=[ref_item, literal_item])


def test_for_each_step_to_request_with_selector():
    body = make_lambda_body_step()
    selector = ForEachSelector(
        pick="MaxBy",
        field=ChildOutput(body).OutputParameters["ProcessedScore"],
    )
    step = make_for_each_step(body_step=body, selector=selector)
    args = step.to_request()["Arguments"]
    assert args["Selector"]["Pick"] == "MaxBy"
    assert args["Selector"]["Field"].expr == {
        "Get": "ChildOutput.ProcessItem.OutputParameters['ProcessedScore']"
    }


def test_for_each_step_to_request_depends_on_and_metadata():
    upstream = CustomStep(name="UpstreamStep")
    step = make_for_each_step(
        depends_on=[upstream],
        display_name="display",
        description="desc",
    )
    request = step.to_request()
    assert request["DependsOn"] == [upstream]
    assert request["DisplayName"] == "display"
    assert request["Description"] == "desc"


def test_step_only_arguments_excludes_body():
    step = make_for_each_step()
    assert "ForEachBody" not in step.step_only_arguments
    assert "IterableItems" in step.step_only_arguments
    assert "MaxConcurrency" in step.step_only_arguments


# ---------------------------------------------------------------------------
# Parent properties
# ---------------------------------------------------------------------------


def test_for_each_step_aggregate_properties():
    step = make_for_each_step()
    props = step.properties
    assert props.Status.expr == {"Get": "Steps.ForEachItems.Status"}
    assert props.InputItemCount.expr == {"Get": "Steps.ForEachItems.InputItemCount"}
    assert props.TotalStepCount.expr == {"Get": "Steps.ForEachItems.TotalStepCount"}
    assert props.SucceededStepCount.expr == {"Get": "Steps.ForEachItems.SucceededStepCount"}
    assert props.FailedStepCount.expr == {"Get": "Steps.ForEachItems.FailedStepCount"}
    assert props.StoppedStepCount.expr == {"Get": "Steps.ForEachItems.StoppedStepCount"}


def test_for_each_step_child_outputs_property():
    step = make_for_each_step()
    ref = step.properties.ChildOutputs[3].OutputParameters["ResultPath"]
    assert ref.expr == {
        "Get": "Steps.ForEachItems.ChildOutputs[3].OutputParameters['ResultPath']"
    }
    assert ref._referenced_steps == [step]


def test_for_each_step_child_outputs_must_be_indexed():
    step = make_for_each_step()
    with pytest.raises(TypeError, match="must be indexed"):
        step.properties.ChildOutputs["ResultPath"]


def test_for_each_step_selected_property():
    step = make_for_each_step()
    ref = step.properties.Selected.OutputParameters["ProcessedScore"]
    assert ref.expr == {"Get": "Steps.ForEachItems.Selected.OutputParameters['ProcessedScore']"}
    assert ref._referenced_steps == [step]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_iterable_items_required():
    with pytest.raises(ValueError, match="iterable_items is required"):
        make_for_each_step(iterable_items=None)


def test_iterable_items_literal_scalar_rejected():
    with pytest.raises(ValueError, match="literal value cannot resolve to a list"):
        make_for_each_step(iterable_items="just-a-string")


@pytest.mark.parametrize("max_concurrency", [None, 0, 51, "2", 2.0, True])
def test_max_concurrency_invalid(max_concurrency):
    with pytest.raises(ValueError, match="max_concurrency is required"):
        make_for_each_step(max_concurrency=max_concurrency)


@pytest.mark.parametrize("max_concurrency", [1, 25, 50])
def test_max_concurrency_valid(max_concurrency):
    step = make_for_each_step(max_concurrency=max_concurrency)
    assert step.max_concurrency == max_concurrency


@pytest.mark.parametrize("body", [None, []])
def test_empty_body_rejected(body):
    with pytest.raises(ValueError, match="non-empty"):
        make_for_each_step(for_each_body=body)


def test_multi_step_body_rejected():
    with pytest.raises(ValueError, match="exactly one step"):
        ForEachStep(
            name="ForEachItems",
            iterable_items=ParameterString(name="ItemList"),
            max_concurrency=2,
            for_each_body=[CustomStep(name="A"), CustomStep(name="B")],
        )


def test_disallowed_body_step_types():
    condition_step = ConditionStep(name="Cond", conditions=[Mock()], if_steps=[], else_steps=[])
    fail_step = FailStep(name="Failure")
    nested_for_each = make_for_each_step(name="Inner")
    for bad_body in (condition_step, fail_step, nested_for_each):
        with pytest.raises(ValueError, match="not allowed in"):
            make_for_each_step(body_step=bad_body)


def test_selector_requires_pick_and_field():
    with pytest.raises(ValueError, match="requires 'pick'"):
        ForEachSelector(pick=None, field=ChildOutput("ProcessItem").Loss)
    with pytest.raises(ValueError, match="requires 'field'"):
        ForEachSelector(pick="MinBy", field=None)


def test_selector_invalid_pick():
    with pytest.raises(ValueError, match="Invalid ForEachSelector pick type"):
        ForEachSelector(pick="Median", field=ChildOutput("ProcessItem").Loss)


def test_selector_pick_enum():
    selector = ForEachSelector(
        pick=ForEachSelectorPickType.MIN_BY, field=ChildOutput("ProcessItem").Loss
    )
    assert selector.to_request()["Pick"] == "MinBy"


def test_selector_field_must_be_child_output():
    body = make_lambda_body_step()
    with pytest.raises(ValueError, match="must be a ChildOutput reference"):
        make_for_each_step(
            body_step=body,
            selector=ForEachSelector(pick="MaxBy", field=ParameterString(name="Other")),
        )


def test_selector_field_must_target_body_step():
    body = make_lambda_body_step()
    with pytest.raises(ValueError, match="must target a body step"):
        make_for_each_step(
            body_step=body,
            selector=ForEachSelector(pick="MaxBy", field=ChildOutput("OtherStep").Loss),
        )


def test_selector_field_must_drill_into_member():
    body = make_lambda_body_step()
    with pytest.raises(ValueError, match="drill at least"):
        make_for_each_step(
            body_step=body,
            selector=ForEachSelector(pick="MaxBy", field=ChildOutput(body)),
        )


# ---------------------------------------------------------------------------
# Pipeline definition (interpolated wire format)
# ---------------------------------------------------------------------------


def _definition_steps(pipeline):
    return json.loads(pipeline.definition())["Steps"]


def test_pipeline_definition_wire_format(mock_session):
    body = make_lambda_body_step()
    selector = ForEachSelector(
        pick="MaxBy",
        field=ChildOutput(body).OutputParameters["ProcessedScore"],
    )
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=ParameterString(name="ItemList"),
        max_concurrency=2,
        for_each_body=[body],
        selector=selector,
    )
    pipeline = Pipeline(
        name="test-pipeline",
        parameters=[ParameterString(name="ItemList")],
        steps=[for_each],
        sagemaker_session=mock_session,
    )
    steps = _definition_steps(pipeline)
    assert len(steps) == 1
    assert steps[0]["Name"] == "ForEachItems"
    assert steps[0]["Type"] == "ForEach"
    assert steps[0]["Arguments"]["IterableItems"] == {"Get": "Parameters.ItemList"}
    assert steps[0]["Arguments"]["MaxConcurrency"] == 2
    assert steps[0]["Arguments"]["Selector"] == {
        "Pick": "MaxBy",
        "Field": {"Get": "ChildOutput.ProcessItem.OutputParameters['ProcessedScore']"},
    }
    body_request = steps[0]["Arguments"]["ForEachBody"][0]
    assert body_request["Name"] == "ProcessItem"
    assert body_request["Type"] == "Lambda"
    assert body_request["Arguments"] == {
        "item_name": {"Get": "ForEach.CurrentItem.name"},
        "item_score": {"Get": "ForEach.CurrentItem.score"},
    }
    assert body_request["OutputParameters"] == [
        {"OutputName": "ProcessedScore", "OutputType": "Integer"},
        {"OutputName": "ResultPath", "OutputType": "String"},
    ]


def test_pipeline_definition_step_output_reference_items(mock_session):
    body = CustomStep(name="ProcessItem", input_data=CurrentItem())
    producer = CustomStep(name="ProduceItem")
    producer_output = Properties(
        step_name="ProduceItem", path="OutputParameters['ItemList']", step=producer
    )
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=producer_output,
        max_concurrency=1,
        for_each_body=[body],
    )
    pipeline = Pipeline(
        name="test-pipeline",
        steps=[for_each, producer],
        sagemaker_session=mock_session,
    )
    steps = _definition_steps(pipeline)
    for_each_request = next(s for s in steps if s["Name"] == "ForEachItems")
    assert for_each_request["Arguments"]["IterableItems"] == {
        "Get": "Steps.ProduceItem.OutputParameters['ItemList']"
    }
    # dependency is inferred by the service from the Get expression; the
    # compiler does not add an explicit DependsOn for property references
    assert "DependsOn" not in for_each_request
    # the referenced producer step must still be compiled as a top-level step
    assert {s["Name"] for s in steps} == {"ForEachItems", "ProduceItem"}
    body_request = for_each_request["Arguments"]["ForEachBody"][0]
    assert body_request["Arguments"] == {"input_data": {"Get": "ForEach.CurrentItem"}}


def test_pipeline_definition_downstream_references(mock_session):
    body = make_lambda_body_step()
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=ParameterString(name="ItemList"),
        max_concurrency=2,
        for_each_body=[body],
        selector=ForEachSelector(
            pick="MinBy", field=ChildOutput(body).OutputParameters["ProcessedScore"]
        ),
    )
    downstream = CustomStep(
        name="Consumer",
        input_data={
            "status": for_each.properties.Status,
            "input_item_count": for_each.properties.InputItemCount,
            "total": for_each.properties.TotalStepCount,
            "succeeded": for_each.properties.SucceededStepCount,
            "failed": for_each.properties.FailedStepCount,
            "stopped": for_each.properties.StoppedStepCount,
            "child_0": for_each.properties.ChildOutputs[0].OutputParameters["ResultPath"],
            "selected": for_each.properties.Selected.OutputParameters["ResultPath"],
        },
    )
    pipeline = Pipeline(
        name="test-pipeline",
        parameters=[ParameterString(name="ItemList")],
        steps=[for_each, downstream],
        sagemaker_session=mock_session,
    )
    steps = _definition_steps(pipeline)
    consumer = next(s for s in steps if s["Name"] == "Consumer")
    assert consumer["Arguments"]["input_data"] == {
        "status": {"Get": "Steps.ForEachItems.Status"},
        "input_item_count": {"Get": "Steps.ForEachItems.InputItemCount"},
        "total": {"Get": "Steps.ForEachItems.TotalStepCount"},
        "succeeded": {"Get": "Steps.ForEachItems.SucceededStepCount"},
        "failed": {"Get": "Steps.ForEachItems.FailedStepCount"},
        "stopped": {"Get": "Steps.ForEachItems.StoppedStepCount"},
        "child_0": {"Get": "Steps.ForEachItems.ChildOutputs[0].OutputParameters['ResultPath']"},
        "selected": {"Get": "Steps.ForEachItems.Selected.OutputParameters['ResultPath']"},
    }
    # dependency is inferred by the service from the Get expressions;
    # no explicit DependsOn is added for property references
    assert "DependsOn" not in consumer


def test_pipeline_definition_for_each_inside_condition_step(mock_session):
    body = CustomStep(name="ProcessItem", input_data=CurrentItem().name)
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=ParameterString(name="ItemList"),
        max_concurrency=2,
        for_each_body=[body],
    )
    condition = Mock()
    condition.to_request.return_value = {"Type": "Equals", "LeftValue": 1, "RightValue": 1}
    condition._referenced_steps = []
    condition_step = ConditionStep(
        name="CheckSomething",
        conditions=[condition],
        if_steps=[for_each],
        else_steps=[],
    )
    pipeline = Pipeline(
        name="test-pipeline",
        parameters=[ParameterString(name="ItemList")],
        steps=[condition_step],
        sagemaker_session=mock_session,
    )
    steps = _definition_steps(pipeline)
    assert len(steps) == 1
    assert steps[0]["Name"] == "CheckSomething"
    if_steps = steps[0]["Arguments"]["IfSteps"]
    assert len(if_steps) == 1
    assert if_steps[0]["Type"] == "ForEach"
    assert if_steps[0]["Arguments"]["ForEachBody"][0]["Name"] == "ProcessItem"


def test_body_steps_are_not_top_level_steps(mock_session):
    body = CustomStep(name="ProcessItem", input_data=CurrentItem())
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=ParameterString(name="ItemList"),
        max_concurrency=2,
        for_each_body=[body],
    )
    pipeline = Pipeline(
        name="test-pipeline",
        parameters=[ParameterString(name="ItemList")],
        steps=[for_each],
        sagemaker_session=mock_session,
    )
    steps = _definition_steps(pipeline)
    assert [s["Name"] for s in steps] == ["ForEachItems"]


def test_duplicate_body_step_name_rejected(mock_session):
    body = CustomStep(name="ProcessItem")
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=ParameterString(name="ItemList"),
        max_concurrency=2,
        for_each_body=[body],
    )
    duplicate = CustomStep(name="ProcessItem")
    pipeline = Pipeline(
        name="test-pipeline",
        steps=[for_each, duplicate],
        sagemaker_session=mock_session,
    )
    with pytest.raises(ValueError, match="duplicate names"):
        pipeline.definition()


def test_pipeline_graph_includes_body_step_edges(mock_session):
    body = CustomStep(name="ProcessItem")
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=ParameterString(name="ItemList"),
        max_concurrency=2,
        for_each_body=[body],
    )
    graph = PipelineGraph([for_each])
    assert "ProcessItem" in graph.step_map
    assert "ProcessItem" in graph.adjacency_list["ForEachItems"]
