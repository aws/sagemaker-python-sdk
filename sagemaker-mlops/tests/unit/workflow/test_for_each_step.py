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
    CurrentItem,
    ForEachSelector,
    ForEachSelectorPickType,
    ForEachStep,
    IterationOutput,
)
from sagemaker.mlops.workflow.lambda_step import (
    LambdaOutput,
    LambdaOutputTypeEnum,
    LambdaStep,
)
from sagemaker.mlops.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.mlops.workflow.step_dependency import StepDependency, TriggerCondition
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
# CurrentItem / IterationOutput references
# ---------------------------------------------------------------------------


def test_current_item_bare_expr():
    assert CurrentItem().expr == {"Get": "ForEach.CurrentItem"}


def test_current_item_drilled_expr():
    assert CurrentItem().name.expr == {"Get": "ForEach.CurrentItem.name"}
    assert CurrentItem().a.b.c.expr == {"Get": "ForEach.CurrentItem.a.b.c"}


def test_current_item_bracket_indexing():
    assert CurrentItem()["score"].expr == {"Get": "ForEach.CurrentItem['score']"}
    assert CurrentItem().items[0].expr == {"Get": "ForEach.CurrentItem.items[0]"}
    assert CurrentItem()["nested"]["key"].expr == {"Get": "ForEach.CurrentItem['nested']['key']"}


def test_current_item_has_no_referenced_steps():
    assert CurrentItem().name._referenced_steps == []


# ---------------------------------------------------------------------------
# IterationOutput references
# ---------------------------------------------------------------------------


def test_iteration_output_expr_from_step_and_str():
    body = make_lambda_body_step()
    ref = IterationOutput(body).OutputParameters["ProcessedScore"]
    assert ref.expr == {"Get": "IterationOutput.ProcessItem.OutputParameters['ProcessedScore']"}
    ref_from_str = IterationOutput("ProcessItem").Loss
    assert ref_from_str.expr == {"Get": "IterationOutput.ProcessItem.Loss"}
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


def test_for_each_step_literal_list_items_accepted():
    ref_item = ParameterString(name="ItemOne")
    items = [ref_item, '{"name": "lit-c", "score": 5}', 42, 1.5, True]
    step = make_for_each_step(iterable_items=items)
    assert step.to_request()["Arguments"]["IterableItems"] == items


@pytest.mark.parametrize("items", [[], ["x"] * 101])
def test_for_each_step_literal_list_size_out_of_range(items):
    with pytest.raises(ValueError, match="between 1 and 100 elements"):
        make_for_each_step(iterable_items=items)


@pytest.mark.parametrize("element", [None, {"a": 1}, ["nested"]])
def test_for_each_step_literal_list_invalid_element(element):
    with pytest.raises(ValueError, match="literal scalars"):
        make_for_each_step(iterable_items=["ok", element])


def test_for_each_step_to_request_with_selector():
    body = make_lambda_body_step()
    selector = ForEachSelector(
        pick="MaxBy",
        field=IterationOutput(body).OutputParameters["ProcessedScore"],
    )
    step = make_for_each_step(body_step=body, selector=selector)
    args = step.to_request()["Arguments"]
    assert args["Selector"]["Pick"] == "MaxBy"
    assert args["Selector"]["Field"].expr == {
        "Get": "IterationOutput.ProcessItem.OutputParameters['ProcessedScore']"
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


def test_for_each_step_iteration_outputs_property():
    step = make_for_each_step()
    ref = step.properties.IterationOutputs[3].ProcessItem.OutputParameters["ResultPath"]
    assert ref.expr == {
        "Get": "Steps.ForEachItems.IterationOutputs[3].ProcessItem.OutputParameters['ResultPath']"
    }
    assert ref._referenced_steps == [step]


@pytest.mark.parametrize(
    "member",
    [
        "Status",
        "InputItemCount",
        "TotalStepCount",
        "SucceededStepCount",
        "FailedStepCount",
        "StoppedStepCount",
    ],
)
def test_for_each_step_aggregate_properties(member):
    step = make_for_each_step()
    ref = getattr(step.properties, member)
    assert ref.expr == {"Get": f"Steps.ForEachItems.{member}"}
    assert ref._referenced_steps == [step]


def test_for_each_step_iteration_outputs_must_be_indexed():
    step = make_for_each_step()
    with pytest.raises(TypeError, match="must be indexed"):
        step.properties.IterationOutputs["ResultPath"]


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


def test_multi_step_body_accepted():
    step_a = CustomStep(name="A")
    step_b = CustomStep(name="B")
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=ParameterString(name="ItemList"),
        max_concurrency=2,
        for_each_body=[step_a, step_b],
    )
    assert for_each.for_each_body == [step_a, step_b]
    request_body = for_each.to_request()["Arguments"]["ForEachBody"]
    assert [entry["Name"] for entry in request_body] == ["A", "B"]


def test_disallowed_body_step_types():
    condition_step = ConditionStep(name="Cond", conditions=[Mock()], if_steps=[], else_steps=[])
    fail_step = FailStep(name="Failure")
    nested_for_each = make_for_each_step(name="Inner")
    for bad_body in (condition_step, fail_step, nested_for_each):
        with pytest.raises(ValueError, match="not allowed in"):
            make_for_each_step(body_step=bad_body)


def test_selector_requires_pick_and_field():
    with pytest.raises(ValueError, match="requires 'pick'"):
        ForEachSelector(pick=None, field=IterationOutput("ProcessItem").Loss)
    with pytest.raises(ValueError, match="requires 'field'"):
        ForEachSelector(pick="MinBy", field=None)


def test_selector_invalid_pick():
    with pytest.raises(ValueError, match="Invalid ForEachSelector pick type"):
        ForEachSelector(pick="Median", field=IterationOutput("ProcessItem").Loss)


def test_selector_pick_enum():
    selector = ForEachSelector(
        pick=ForEachSelectorPickType.MIN_BY, field=IterationOutput("ProcessItem").Loss
    )
    assert selector.to_request()["Pick"] == "MinBy"


def test_selector_field_must_be_iteration_output():
    body = make_lambda_body_step()
    with pytest.raises(ValueError, match="must be an IterationOutput reference"):
        make_for_each_step(
            body_step=body,
            selector=ForEachSelector(pick="MaxBy", field=ParameterString(name="Other")),
        )


def test_selector_field_must_target_body_step():
    body = make_lambda_body_step()
    with pytest.raises(ValueError, match="must target a body step"):
        make_for_each_step(
            body_step=body,
            selector=ForEachSelector(pick="MaxBy", field=IterationOutput("OtherStep").Loss),
        )


def test_selector_field_must_drill_into_member():
    body = make_lambda_body_step()
    with pytest.raises(ValueError, match="drill at least"):
        make_for_each_step(
            body_step=body,
            selector=ForEachSelector(pick="MaxBy", field=IterationOutput(body)),
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
        field=IterationOutput(body).OutputParameters["ProcessedScore"],
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
        "Field": {"Get": "IterationOutput.ProcessItem.OutputParameters['ProcessedScore']"},
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
            pick="MinBy", field=IterationOutput(body).OutputParameters["ProcessedScore"]
        ),
    )
    downstream = CustomStep(
        name="Consumer",
        input_data={
            "iter_0": for_each.properties.IterationOutputs[0].ProcessItem.OutputParameters[
                "ResultPath"
            ],
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
        "iter_0": {
            "Get": "Steps.ForEachItems.IterationOutputs[0].ProcessItem"
            ".OutputParameters['ResultPath']"
        },
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


def test_pipeline_definition_literal_list_items_and_aggregate_reference(mock_session):
    body = CustomStep(name="ProcessItem", input_data=CurrentItem())
    producer = CustomStep(name="ProduceItem")
    producer_output = Properties(
        step_name="ProduceItem", path="OutputParameters['ItemA']", step=producer
    )
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=[producer_output, "s3://my-bucket/item-b", 42, True],
        max_concurrency=2,
        for_each_body=[body],
    )
    consumer = CustomStep(
        name="Consumer",
        input_data={
            "status": for_each.properties.Status,
            "total": for_each.properties.TotalStepCount,
        },
    )
    pipeline = Pipeline(
        name="test-pipeline",
        steps=[producer, for_each, consumer],
        sagemaker_session=mock_session,
    )
    steps = _definition_steps(pipeline)
    for_each_request = next(s for s in steps if s["Name"] == "ForEachItems")
    # same shape as the service's iterable_literal_list.json fixture
    assert for_each_request["Arguments"]["IterableItems"] == [
        {"Get": "Steps.ProduceItem.OutputParameters['ItemA']"},
        "s3://my-bucket/item-b",
        42,
        True,
    ]
    consumer_request = next(s for s in steps if s["Name"] == "Consumer")
    assert consumer_request["Arguments"]["input_data"] == {
        "status": {"Get": "Steps.ForEachItems.Status"},
        "total": {"Get": "Steps.ForEachItems.TotalStepCount"},
    }


def test_pipeline_definition_run_conditions(mock_session):
    producer = CustomStep(name="ProduceItem")
    first = CustomStep(name="FirstBodyStep", input_data=CurrentItem())
    second = CustomStep(
        name="SecondBodyStep",
        depends_on=[StepDependency(first, TriggerCondition.ANY)],
    )
    for_each = ForEachStep(
        name="ForEachItems",
        iterable_items=ParameterString(name="ItemList"),
        max_concurrency=2,
        for_each_body=[first, second],
        depends_on=[StepDependency(producer, TriggerCondition.FAILED)],
    )
    notify = CustomStep(
        name="NotifyOnFailure",
        depends_on=[StepDependency(for_each, TriggerCondition.FAILED)],
    )
    pipeline = Pipeline(
        name="test-pipeline",
        parameters=[ParameterString(name="ItemList")],
        steps=[producer, for_each, notify],
        sagemaker_session=mock_session,
    )
    steps = {s["Name"]: s for s in _definition_steps(pipeline)}
    assert steps["ForEachItems"]["DependsOn"] == [{"Step": "ProduceItem", "Trigger": "Failed"}]
    assert steps["NotifyOnFailure"]["DependsOn"] == [{"Step": "ForEachItems", "Trigger": "Failed"}]
    body = {s["Name"]: s for s in steps["ForEachItems"]["Arguments"]["ForEachBody"]}
    assert "DependsOn" not in body["FirstBodyStep"]
    assert body["SecondBodyStep"]["DependsOn"] == [{"Step": "FirstBodyStep", "Trigger": "Any"}]
