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
"""Unit tests for _steps_compiler module."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, MagicMock, patch

from sagemaker.mlops.workflow._steps_compiler import (
    CompiledStep,
    _StepsSet,
    _BuildQueue,
    StepsCompiler,
)
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum, PropertyFile
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.core.workflow.step_outputs import StepOutput


class TestCompiledStep:
    """Tests for CompiledStep class."""

    def test_init(self):
        """Test CompiledStep initialization."""
        request_dict = {
            "Name": "test-step",
            "Type": "Training",
            "Arguments": {"TrainingJobName": "job-123"},
            "DependsOn": ["step1"],
            "Description": "Test description",
            "DisplayName": "Test Display",
        }

        compiled_step = CompiledStep(request_dict)

        assert compiled_step.name == "test-step"
        assert compiled_step.step_type == StepTypeEnum.TRAINING
        assert compiled_step.depends_on == ["step1"]
        assert compiled_step.description == "Test description"
        assert compiled_step.display_name == "Test Display"

    def test_arguments_property(self):
        """Test arguments property returns step arguments."""
        request_dict = {
            "Name": "test-step",
            "Type": "Training",
            "Arguments": {"TrainingJobName": "job-123", "RoleArn": "arn:aws:iam::123"},
        }

        compiled_step = CompiledStep(request_dict)

        assert compiled_step.arguments == {
            "TrainingJobName": "job-123",
            "RoleArn": "arn:aws:iam::123",
        }

    def test_property_files(self):
        """Test property_files returns list of PropertyFile objects."""
        request_dict = {
            "Name": "test-step",
            "Type": "Processing",
            "Arguments": {},
            "PropertyFiles": [
                {
                    "PropertyFileName": "metrics",
                    "OutputName": "output1",
                    "FilePath": "metrics.json",
                }
            ],
        }

        compiled_step = CompiledStep(request_dict)

        assert len(compiled_step.property_files) == 1
        assert compiled_step.property_files[0].name == "metrics"
        assert compiled_step.property_files[0].output_name == "output1"
        assert compiled_step.property_files[0].path == "metrics.json"

    def test_property_files_empty(self):
        """Test property_files returns empty list when no property files."""
        request_dict = {
            "Name": "test-step",
            "Type": "Training",
            "Arguments": {},
        }

        compiled_step = CompiledStep(request_dict)

        assert compiled_step.property_files == []

    def test_properties_raises_not_implemented(self):
        """Test properties raises NotImplementedError."""
        request_dict = {
            "Name": "test-step",
            "Type": "Training",
            "Arguments": {},
        }

        compiled_step = CompiledStep(request_dict)

        with pytest.raises(NotImplementedError):
            _ = compiled_step.properties

    def test_add_depends_on_raises_not_implemented(self):
        """Test add_depends_on raises NotImplementedError."""
        request_dict = {
            "Name": "test-step",
            "Type": "Training",
            "Arguments": {},
        }

        compiled_step = CompiledStep(request_dict)

        with pytest.raises(NotImplementedError):
            compiled_step.add_depends_on(["step1"])

    def test_to_request(self):
        """Test to_request returns the request dictionary."""
        request_dict = {
            "Name": "test-step",
            "Type": "Training",
            "Arguments": {"TrainingJobName": "job-123"},
        }

        compiled_step = CompiledStep(request_dict)

        assert compiled_step.to_request() == request_dict


class TestStepsSet:
    """Tests for _StepsSet class."""

    def test_init(self):
        """Test _StepsSet initialization."""
        steps_set = _StepsSet()

        assert len(steps_set) == 0

    def test_add_single_step(self):
        """Test adding a single step."""
        steps_set = _StepsSet()
        step = Mock(spec=Step)

        steps_set.add(step)

        assert len(steps_set) == 1
        assert step in steps_set

    def test_add_duplicate_step(self):
        """Test adding duplicate step doesn't increase size."""
        steps_set = _StepsSet()
        step = Mock(spec=Step)

        steps_set.add(step)
        steps_set.add(step)

        assert len(steps_set) == 1

    def test_add_list(self):
        """Test adding a list of steps."""
        steps_set = _StepsSet()
        step1 = Mock(spec=Step)
        step2 = Mock(spec=Step)

        steps_set.add_list([step1, step2])

        assert len(steps_set) == 2
        assert step1 in steps_set
        assert step2 in steps_set

    def test_contains(self):
        """Test __contains__ method."""
        steps_set = _StepsSet()
        step1 = Mock(spec=Step)
        step2 = Mock(spec=Step)

        steps_set.add(step1)

        assert step1 in steps_set
        assert step2 not in steps_set

    def test_getitem(self):
        """Test __getitem__ method."""
        steps_set = _StepsSet()
        step1 = Mock(spec=Step)
        step2 = Mock(spec=Step)

        steps_set.add(step1)
        steps_set.add(step2)

        assert steps_set[0] == step1
        assert steps_set[1] == step2


class TestBuildQueue:
    """Tests for _BuildQueue class."""

    def test_init(self):
        """Test _BuildQueue initialization."""
        queue = _BuildQueue()

        assert len(queue) == 0

    def test_push_single_step(self):
        """Test pushing a single step."""
        queue = _BuildQueue()
        step = Mock(spec=Step)

        queue.push([step])

        assert len(queue) == 1

    def test_push_multiple_steps(self):
        """Test pushing multiple steps."""
        queue = _BuildQueue()
        step1 = Mock(spec=Step)
        step2 = Mock(spec=Step)

        queue.push([step1, step2])

        assert len(queue) == 2

    def test_pop(self):
        """Test popping a step from queue."""
        queue = _BuildQueue()
        step1 = Mock(spec=Step)
        step2 = Mock(spec=Step)

        queue.push([step1, step2])

        popped = queue.pop()
        assert popped == step1
        assert len(queue) == 1

    def test_pop_empty_queue(self):
        """Test popping from empty queue returns None."""
        queue = _BuildQueue()

        popped = queue.pop()

        assert popped is None

    def test_fifo_order(self):
        """Test queue maintains FIFO order."""
        queue = _BuildQueue()
        step1 = Mock(spec=Step)
        step2 = Mock(spec=Step)
        step3 = Mock(spec=Step)

        queue.push([step1, step2, step3])

        assert queue.pop() == step1
        assert queue.pop() == step2
        assert queue.pop() == step3


class TestStepsCompiler:
    """Tests for StepsCompiler class."""

    @pytest.fixture
    def mock_session(self):
        return Mock()

    @pytest.fixture
    def mock_step(self):
        step = Mock(spec=Step)
        step.name = "test-step"
        step.step_type = StepTypeEnum.TRAINING
        step.depends_on = []
        step.to_request = Mock(
            return_value={
                "Name": "test-step",
                "Type": "Training",
                "Arguments": {"TrainingJobName": "job-123"},
            }
        )
        return step

    def test_init(self, mock_session, mock_step):
        """Test StepsCompiler initialization."""
        compiler = StepsCompiler(
            pipeline_name="test-pipeline",
            sagemaker_session=mock_session,
            steps=[mock_step],
        )

        assert compiler.pipeline_name == "test-pipeline"
        assert compiler.sagemaker_session == mock_session
        assert len(compiler._input_steps) == 1

    def test_generate_step_map(self, mock_step):
        """Test _generate_step_map creates correct mapping."""
        step_map = {}
        StepsCompiler._generate_step_map([mock_step], step_map)

        assert "test-step" in step_map
        assert step_map["test-step"] == mock_step

    def test_generate_step_map_duplicate_names_raises_error(self):
        """Test _generate_step_map raises error for duplicate names."""
        step1 = Mock(spec=Step)
        step1.name = "duplicate"

        step2 = Mock(spec=Step)
        step2.name = "duplicate"

        step_map = {}

        with pytest.raises(ValueError) as exc_info:
            StepsCompiler._generate_step_map([step1, step2], step_map)

        assert "duplicate names" in str(exc_info.value)

    def test_simplify_step_list_with_steps(self, mock_session, mock_step):
        """Test _simplify_step_list with Step objects."""
        compiler = StepsCompiler(
            pipeline_name="test-pipeline",
            sagemaker_session=mock_session,
            steps=[mock_step],
        )

        result = compiler._simplify_step_list([mock_step])

        assert len(result) == 1
        assert result[0] == mock_step

    def test_simplify_step_list_with_string(self, mock_session, mock_step):
        """Test _simplify_step_list with string step names."""
        compiler = StepsCompiler(
            pipeline_name="test-pipeline",
            sagemaker_session=mock_session,
            steps=[mock_step],
        )

        result = compiler._simplify_step_list(["test-step"])

        assert len(result) == 1
        assert result[0] == mock_step

    def test_simplify_step_list_unknown_step_name_raises_error(self, mock_session, mock_step):
        """Test _simplify_step_list raises error for unknown step name."""
        compiler = StepsCompiler(
            pipeline_name="test-pipeline",
            sagemaker_session=mock_session,
            steps=[mock_step],
        )

        with pytest.raises(ValueError) as exc_info:
            compiler._simplify_step_list(["unknown-step"])

        assert "unknown-step" in str(exc_info.value)

    def test_simplify_step_list_empty(self, mock_session, mock_step):
        """Test _simplify_step_list with empty list."""
        compiler = StepsCompiler(
            pipeline_name="test-pipeline",
            sagemaker_session=mock_session,
            steps=[mock_step],
        )

        result = compiler._simplify_step_list([])

        assert result == []

    def test_simplify_step_list_removes_duplicates(self, mock_session, mock_step):
        """Test _simplify_step_list removes duplicate steps."""
        compiler = StepsCompiler(
            pipeline_name="test-pipeline",
            sagemaker_session=mock_session,
            steps=[mock_step],
        )

        result = compiler._simplify_step_list([mock_step, mock_step, "test-step"])

        assert len(result) == 1

    def test_get_upstream_steps_from_step_arguments(self):
        """Test get_upstream_steps_from_step_arguments."""
        with patch.object(Step, "_find_pipeline_variables_in_step_arguments") as mock_find:
            mock_var = Mock()
            mock_var._referenced_steps = [Mock(spec=Step)]
            mock_find.return_value = [mock_var]

            result = StepsCompiler.get_upstream_steps_from_step_arguments({"arg": "value"})

            assert len(result) == 1
            mock_find.assert_called_once()

    def test_build_raises_error_on_second_call(self, mock_session, mock_step):
        """Test build raises error when called more than once."""
        with patch("sagemaker.mlops.workflow._steps_compiler.step_compilation_context_manager"):
            compiler = StepsCompiler(
                pipeline_name="test-pipeline",
                sagemaker_session=mock_session,
                steps=[mock_step],
            )

            compiler.build()

            with pytest.raises(RuntimeError) as exc_info:
                compiler.build()

            assert "more than once" in str(exc_info.value)

    def test_flatten_condition_step(self, mock_session):
        """Test _flatten_condition_step flattens nested condition steps."""
        step1 = Mock(spec=Step)
        step1.name = "step1"

        condition_step = Mock(spec=ConditionStep)
        condition_step.name = "condition"
        condition_step.if_steps = [step1]
        condition_step.else_steps = []

        compiler = StepsCompiler(
            pipeline_name="test-pipeline",
            sagemaker_session=mock_session,
            steps=[condition_step],
        )

        result = compiler._flatten_condition_step(condition_step)

        assert len(result) >= 1
        assert condition_step in result

    def test_push_to_build_queue(self, mock_session, mock_step):
        """Test _push_to_build_queue adds steps to queue."""
        compiler = StepsCompiler(
            pipeline_name="test-pipeline",
            sagemaker_session=mock_session,
            steps=[],
        )

        compiler._push_to_build_queue([mock_step])

        assert len(compiler._build_queue) == 1
        assert mock_step in compiler._all_known_steps

    def test_push_to_build_queue_duplicate(self, mock_session, mock_step):
        """Test _push_to_build_queue doesn't add duplicate steps to queue."""
        compiler = StepsCompiler(
            pipeline_name="test-pipeline",
            sagemaker_session=mock_session,
            steps=[],
        )

        compiler._push_to_build_queue([mock_step])
        compiler._push_to_build_queue([mock_step])

        # Should only be added once to queue
        assert len(compiler._build_queue) == 1
