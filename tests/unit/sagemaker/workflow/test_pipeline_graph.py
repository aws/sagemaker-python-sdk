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

import pytest

from mock import Mock

from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import (
    ConditionEquals,
    ConditionIn,
    ConditionNot,
    ConditionOr,
    ConditionGreaterThan,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from tests.unit.sagemaker.workflow.helpers import ordered, CustomStep, CustomStepCollection


@pytest.fixture
def sagemaker_session_mock():
    session_mock = Mock()
    session_mock.default_bucket = Mock(name="default_bucket", return_value="s3_bucket")
    return session_mock


@pytest.fixture
def role_arn():
    return "arn:role"


def test_pipeline_duplicate_step_name(sagemaker_session_mock):
    step1 = CustomStep(name="foo")
    step2 = CustomStep(name="foo")
    with pytest.raises(ValueError) as error:
        pipeline = Pipeline(
            name="MyPipeline", steps=[step1, step2], sagemaker_session=sagemaker_session_mock
        )
        PipelineGraph.from_pipeline(pipeline)
    assert "Pipeline steps cannot have duplicate names." in str(error.value)


def test_pipeline_duplicate_step_name_in_condition_step(sagemaker_session_mock):
    param = ParameterInteger(name="MyInt", default_value=2)
    cond = ConditionEquals(left=param, right=1)
    custom_step = CustomStep(name="foo")
    custom_step2 = CustomStep(name="foo")
    condition_step = ConditionStep(
        name="condStep", conditions=[cond], depends_on=[custom_step], if_steps=[custom_step2]
    )
    with pytest.raises(ValueError) as error:
        pipeline = Pipeline(
            name="MyPipeline",
            steps=[custom_step, condition_step],
            sagemaker_session=sagemaker_session_mock,
        )
        PipelineGraph.from_pipeline(pipeline)
    assert "Pipeline steps cannot have duplicate names." in str(error.value)


def test_pipeline_duplicate_step_name_in_step_collection(sagemaker_session_mock):
    custom_step = CustomStep(name="foo-1")
    custom_step_collection = CustomStepCollection(name="foo", depends_on=[custom_step])
    with pytest.raises(ValueError) as error:
        pipeline = Pipeline(
            name="MyPipeline",
            steps=[custom_step, custom_step_collection],
            sagemaker_session=sagemaker_session_mock,
        )
        PipelineGraph.from_pipeline(pipeline)
    assert "Pipeline steps cannot have duplicate names." in str(error.value)


def test_pipeline_graph_acyclic(sagemaker_session_mock):
    step_a = CustomStep(name="stepA")
    step_b = CustomStep(name="stepB")
    step_c = CustomStep(name="stepC", depends_on=[step_a])
    step_d = CustomStep(name="stepD", depends_on=[step_c])
    step_e = CustomStep(name="stepE", depends_on=[step_a, step_b, step_d])

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step_a, step_b, step_c, step_d, step_e],
        sagemaker_session=sagemaker_session_mock,
    )

    pipeline_graph = PipelineGraph.from_pipeline(pipeline)
    adjacency_list = pipeline_graph.adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "stepA": ["stepC", "stepE"],
            "stepB": ["stepE"],
            "stepC": ["stepD"],
            "stepD": ["stepE"],
            "stepE": [],
        }
    )
    _verify_pipeline_graph_traversal(pipeline_graph)


def test_pipeline_graph_acyclic_with_condition_step_explicit_dependency(sagemaker_session_mock):
    custom_step = CustomStep(name="TestStep")
    if_step = CustomStep(name="IfStep")
    else_step = CustomStep(name="ElseStep")
    param = ParameterInteger(name="MyInt", default_value=2)
    cond = ConditionEquals(left=param, right=1)
    condition_step = ConditionStep(
        name="condStep",
        conditions=[cond],
        depends_on=[custom_step],
        if_steps=[if_step],
        else_steps=[else_step],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[custom_step, condition_step],
        sagemaker_session=sagemaker_session_mock,
    )

    pipeline_graph = PipelineGraph.from_pipeline(pipeline)
    adjacency_list = pipeline_graph.adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"condStep": ["ElseStep", "IfStep"], "ElseStep": [], "IfStep": [], "TestStep": ["condStep"]}
    )
    _verify_pipeline_graph_traversal(pipeline_graph)


def test_pipeline_graph_acyclic_with_condition_step_property_reference_dependency(
    sagemaker_session_mock,
):
    custom_step = CustomStep(name="TestStep")
    if_step = CustomStep(name="IfStep")
    else_step = CustomStep(name="ElseStep")
    cond = ConditionEquals(left=custom_step.properties.TrainingJobStatus, right="Succeeded")
    condition_step = ConditionStep(
        name="condStep", conditions=[cond], if_steps=[if_step], else_steps=[else_step]
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[custom_step, condition_step],
        sagemaker_session=sagemaker_session_mock,
    )

    pipeline_graph = PipelineGraph.from_pipeline(pipeline)
    adjacency_list = pipeline_graph.adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"condStep": ["ElseStep", "IfStep"], "ElseStep": [], "IfStep": [], "TestStep": ["condStep"]}
    )
    _verify_pipeline_graph_traversal(pipeline_graph)


def test_pipeline_graph_acyclic_with_step_collection_explicit_dependency(sagemaker_session_mock):
    custom_step1 = CustomStep(name="TestStep")
    custom_step_collection = CustomStepCollection(
        name="TestStepCollection", depends_on=[custom_step1]
    )
    custom_step2 = CustomStep(name="TestStep2", depends_on=[custom_step_collection])
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[custom_step1, custom_step_collection, custom_step2],
        sagemaker_session=sagemaker_session_mock,
    )

    pipeline_graph = PipelineGraph.from_pipeline(pipeline)
    adjacency_list = pipeline_graph.adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "TestStep": ["TestStepCollection-0"],
            "TestStep2": [],
            "TestStepCollection-0": ["TestStepCollection-1"],
            "TestStepCollection-1": ["TestStep2"],
        }
    )
    _verify_pipeline_graph_traversal(pipeline_graph)


def test_pipeline_graph_acyclic_with_step_collection_property_reference_dependency(
    sagemaker_session_mock,
):
    custom_step_collection = CustomStepCollection(name="TestStepCollection")
    custom_step = CustomStep(
        name="TestStep",
        input_data=custom_step_collection.properties.AlgorithmSpecification.AlgorithmName,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[custom_step_collection, custom_step],
        sagemaker_session=sagemaker_session_mock,
    )

    pipeline_graph = PipelineGraph.from_pipeline(pipeline)
    adjacency_list = pipeline_graph.adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "TestStep": [],
            "TestStepCollection-0": ["TestStepCollection-1"],
            "TestStepCollection-1": ["TestStep"],
        }
    )
    _verify_pipeline_graph_traversal(pipeline_graph)


def test_pipeline_graph_cyclic(sagemaker_session_mock):
    step_a = CustomStep(name="stepA", depends_on=["stepC"])
    step_b = CustomStep(name="stepB", depends_on=["stepA"])
    step_c = CustomStep(name="stepC", depends_on=["stepB"])

    pipeline = Pipeline(
        name="MyPipeline", steps=[step_a, step_b, step_c], sagemaker_session=sagemaker_session_mock
    )

    with pytest.raises(ValueError) as error:
        PipelineGraph.from_pipeline(pipeline)
    assert "Cycle detected in pipeline step graph." in str(error.value)


def test_condition_comparison(sagemaker_session):
    param = ParameterInteger(name="MyInt")
    cond = ConditionEquals(left=param, right=1)
    if_step = CustomStep(name="IfStep")
    else_step = CustomStep(name="ElseStep")
    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond],
        if_steps=[if_step],
        else_steps=[else_step],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session,
        parameters=[param],
    )
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"MyConditionStep": ["ElseStep", "IfStep"], "ElseStep": [], "IfStep": []}
    )


def test_condition_not(sagemaker_session):
    param = ParameterInteger(name="MyInt")
    cond = ConditionEquals(left=param, right=1)
    cond_not = ConditionNot(expression=cond)
    if_step = CustomStep(name="IfStep")
    else_step = CustomStep(name="ElseStep")
    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_not],
        if_steps=[if_step],
        else_steps=[else_step],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session,
        parameters=[param],
    )
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"MyConditionStep": ["ElseStep", "IfStep"], "ElseStep": [], "IfStep": []}
    )


def test_condition_in(sagemaker_session):
    param = ParameterString(name="MyStr")
    cond_in = ConditionIn(value=param, in_values=["abc", "def"])
    if_step = CustomStep(name="IfStep")
    else_step = CustomStep(name="ElseStep")
    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_in],
        if_steps=[if_step],
        else_steps=[else_step],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session,
        parameters=[param],
    )
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyConditionStep": ["ElseStep", "IfStep"],
            "ElseStep": [],
            "IfStep": [],
        }
    )


def test_condition_or(sagemaker_session):
    param = ParameterString(name="MyStr")
    cond1 = ConditionGreaterThan(left=ExecutionVariables.START_DATETIME, right="2020-12-01")
    cond2 = ConditionEquals(left=param, right="Success")
    cond_or = ConditionOr(conditions=[cond1, cond2])
    if_step = CustomStep(name="IfStep")
    else_step = CustomStep(name="ElseStep")
    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_or],
        if_steps=[if_step],
        else_steps=[else_step],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session,
        parameters=[param],
    )
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyConditionStep": ["ElseStep", "IfStep"],
            "ElseStep": [],
            "IfStep": [],
        }
    )


def _verify_pipeline_graph_traversal(pipeline_graph):
    adjacency_list = pipeline_graph.adjacency_list
    traversed_steps = []
    for step in pipeline_graph:
        # the traversal order of a PipelineGraph needs to be a topological sort traversal
        # i.e. parent steps are always traversed before their children steps
        assert step not in traversed_steps
        for children_steps in adjacency_list[step.name]:
            assert children_steps not in traversed_steps
        traversed_steps.append(step)
