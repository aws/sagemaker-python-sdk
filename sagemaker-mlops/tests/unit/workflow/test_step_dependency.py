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
from __future__ import absolute_import

import json
from unittest.mock import Mock

import pytest

from sagemaker.core.workflow.conditions import ConditionEquals
from sagemaker.mlops.local.pipeline import LocalPipelineExecutor
from sagemaker.mlops.workflow import StepDependency, TriggerCondition
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


class CustomStep(Step):
    def __init__(self, name, depends_on=None):
        super(CustomStep, self).__init__(
            name=name, step_type=StepTypeEnum.TRAINING, depends_on=depends_on
        )

    @property
    def arguments(self):
        return {}

    @property
    def properties(self):
        return None


@pytest.fixture
def session():
    session = Mock()
    session.local_mode = False
    session.sagemaker_config = {}
    return session


def _depends_on_by_step(steps, session):
    definition = json.loads(Pipeline(name="P", steps=steps, sagemaker_session=session).definition())
    return {step["Name"]: step.get("DependsOn") for step in definition["Steps"]}


def test_to_request_uses_plain_name_for_completed():
    assert StepDependency("A").to_request() == "A"
    assert StepDependency("A", TriggerCondition.COMPLETED).to_request() == "A"


def test_to_request_uses_object_form_for_other_triggers():
    assert StepDependency("A", TriggerCondition.FAILED).to_request() == {
        "Step": "A",
        "Trigger": "Failed",
    }
    assert StepDependency("A", TriggerCondition.STOPPED).to_request()["Trigger"] == "Stopped"
    assert StepDependency("A", TriggerCondition.ANY).to_request()["Trigger"] == "Any"


def test_step_name_from_step_object():
    assert StepDependency(CustomStep("A"), TriggerCondition.ANY).step_name == "A"


def test_rejects_non_enum_trigger():
    with pytest.raises(TypeError):
        StepDependency("A", "Failed")


def test_completed_dependency_produces_same_definition_as_plain_entry(session):
    a = CustomStep("A")
    plain = Pipeline(
        name="P", steps=[a, CustomStep("B", depends_on=[a, "A"])], sagemaker_session=session
    ).definition()

    a = CustomStep("A")
    wrapped = Pipeline(
        name="P",
        steps=[a, CustomStep("B", depends_on=[StepDependency(a), StepDependency("A")])],
        sagemaker_session=session,
    ).definition()

    assert wrapped == plain


def test_definition_mixes_plain_and_object_entries(session):
    a = CustomStep("A")
    b = CustomStep("B", depends_on=[StepDependency(a, TriggerCondition.FAILED)])
    c = CustomStep("C", depends_on=["A", StepDependency(b, TriggerCondition.ANY)])

    assert _depends_on_by_step([a, b, c], session) == {
        "A": None,
        "B": [{"Step": "A", "Trigger": "Failed"}],
        "C": ["A", {"Step": "B", "Trigger": "Any"}],
    }


def test_upstream_step_is_compiled_when_only_referenced_by_step_dependency(session):
    a = CustomStep("A")
    b = CustomStep("B", depends_on=[StepDependency(a, TriggerCondition.FAILED)])

    assert set(_depends_on_by_step([b], session)) == {"A", "B"}


def test_condition_step_depends_on_object_form(session):
    a = CustomStep("A")
    condition = ConditionStep(
        name="Check",
        conditions=[ConditionEquals(left=1, right=1)],
        if_steps=[CustomStep("IfStep")],
        else_steps=[],
        depends_on=[StepDependency(a, TriggerCondition.STOPPED)],
    )

    assert _depends_on_by_step([a, condition], session)["Check"] == [
        {"Step": "A", "Trigger": "Stopped"}
    ]


def test_same_trigger_listed_twice_is_deduplicated(session):
    a = CustomStep("A")
    b = CustomStep(
        "B",
        depends_on=[
            StepDependency(a, TriggerCondition.FAILED),
            StepDependency("A", TriggerCondition.FAILED),
        ],
    )

    assert _depends_on_by_step([a, b], session)["B"] == [{"Step": "A", "Trigger": "Failed"}]


def test_conflicting_triggers_raise(session):
    a = CustomStep("A")
    b = CustomStep(
        "B",
        depends_on=[
            StepDependency(a, TriggerCondition.FAILED),
            StepDependency(a, TriggerCondition.ANY),
        ],
    )

    with pytest.raises(ValueError, match="conflicting triggers"):
        Pipeline(name="P", steps=[a, b], sagemaker_session=session).definition()


FAILED_A = {"Step": "A", "Trigger": "Failed"}


@pytest.mark.parametrize(
    "depends_on, expected",
    [
        (lambda a: ["A"], ["A"]),
        (lambda a: [a], ["A"]),
        (lambda a: [a, "A"], ["A"]),
        (lambda a: [StepDependency(a)], ["A"]),
        (lambda a: [StepDependency(a, TriggerCondition.COMPLETED)], ["A"]),
        (lambda a: [StepDependency(a), "A"], ["A"]),
        (lambda a: [StepDependency(a, TriggerCondition.FAILED)], [FAILED_A]),
        (lambda a: [StepDependency(a, TriggerCondition.ANY)], [{"Step": "A", "Trigger": "Any"}]),
        (
            lambda a: [
                StepDependency(a, TriggerCondition.FAILED),
                StepDependency("A", TriggerCondition.FAILED),
            ],
            [FAILED_A],
        ),
    ],
)
def test_depends_on_combinations(session, depends_on, expected):
    a = CustomStep("A")
    b = CustomStep("B", depends_on=depends_on(a))

    assert _depends_on_by_step([a, b], session)["B"] == expected


@pytest.mark.parametrize(
    "depends_on",
    [
        lambda a: [
            StepDependency(a, TriggerCondition.COMPLETED),
            StepDependency(a, TriggerCondition.FAILED),
        ],
        lambda a: ["A", StepDependency(a, TriggerCondition.FAILED)],
        lambda a: [a, StepDependency(a, TriggerCondition.FAILED)],
        lambda a: [StepDependency(a, TriggerCondition.FAILED), "A"],
        lambda a: [StepDependency("A", TriggerCondition.ANY), a],
    ],
)
def test_plain_entry_conflicts_with_non_completed_trigger(session, depends_on):
    a = CustomStep("A")
    b = CustomStep("B", depends_on=depends_on(a))

    with pytest.raises(ValueError, match="conflicting triggers: (Completed|Failed|Any)"):
        Pipeline(name="P", steps=[a, b], sagemaker_session=session).definition()


def test_pipeline_graph_keeps_object_form_edges(session):
    a = CustomStep("A")
    b = CustomStep("B", depends_on=[StepDependency(a, TriggerCondition.ANY)])
    pipeline = Pipeline(name="P", steps=[a, b], sagemaker_session=session)

    assert PipelineGraph.from_pipeline(pipeline).adjacency_list["A"] == ["B"]


def test_local_mode_rejects_run_conditions(session):
    a = CustomStep("A")
    b = CustomStep("B", depends_on=[StepDependency(a, TriggerCondition.FAILED)])
    execution = Mock()
    execution.pipeline = Pipeline(name="P", steps=[a, b], sagemaker_session=session)

    with pytest.raises(ValueError, match="not supported in local mode"):
        LocalPipelineExecutor(execution, Mock())


def test_local_mode_allows_completed_step_dependency(session):
    a = CustomStep("A")
    b = CustomStep("B", depends_on=[StepDependency(a)])
    execution = Mock()
    execution.pipeline = Pipeline(name="P", steps=[a, b], sagemaker_session=session)

    LocalPipelineExecutor(execution, Mock())
