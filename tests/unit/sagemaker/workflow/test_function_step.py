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

from copy import deepcopy

import pytest
import re
from mock import patch, Mock, ANY
from typing import List, Tuple

from mock.mock import MagicMock

from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.step_outputs import get_step
from sagemaker.workflow.function_step import DelayedReturn, step
from sagemaker.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum
from sagemaker.workflow.pipeline_context import _PipelineConfig
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig

SOME_TRAINING_REQUEST = {
    "AlgorithmSpecification": {"TrainingImage": "IMAGE_URI", "TrainingInputMode": "File"},
    "InputDataConfig": [
        {
            "ChannelName": "Bootstrap",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://some_s3_path",
                }
            },
        }
    ],
    "OutputDataConfig": {"S3OutputPath": {"s3://some_s3_path"}},
    "ResourceConfig": {
        "InstanceCount": {"Get": "Parameters.InstanceCount"},
        "InstanceType": {"Get": "Parameters.InstanceType"},
        "VolumeSizeInGB": 30,
    },
    "RoleArn": "ROLE",
    "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
}

_DEFINITION_CONFIG = PipelineDefinitionConfig(use_custom_job_prefix=False)
MOCKED_PIPELINE_CONFIG = _PipelineConfig(
    "test-pipeline",
    "test-function-step",
    Mock(),
    "code-hash-0123456789",
    "config-hash-0123456789",
    _DEFINITION_CONFIG,
    "build_time",
    True,
    True,
)


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@patch("sagemaker.remote_function.job._JobSettings")
def test_step_decorator(mock_job_settings):

    retry_policy = StepRetryPolicy(
        exception_types=[StepExceptionTypeEnum.THROTTLING],
        interval_seconds=5,
        max_attempts=3,
    )

    @step(
        name="step_name",
        display_name="step_display_name",
        description="step_description",
        retry_policies=[retry_policy],
        instance_type="ml.m5.large",
        image_uri="test_image_uri",
    )
    def sum(a, b, c, d):
        return a + b + c + d

    step_output = sum(2, 3, c=3, d=5)

    assert isinstance(step_output, DelayedReturn)

    function_step = get_step(step_output)

    assert function_step is not None
    assert "step_name" in function_step.name
    assert function_step.display_name == "step_display_name"
    assert function_step.description == "step_description"
    assert function_step.retry_policies == [retry_policy]
    assert function_step.depends_on == []
    assert function_step.func_args == (2, 3)
    assert function_step.func_kwargs == {"c": 3, "d": 5}
    assert function_step.func is not None

    assert function_step._job_settings is not None
    assert mock_job_settings.call_args[1]["image_uri"] == "test_image_uri"
    assert function_step._serialized_data.func is not None
    assert function_step._serialized_data.args is not None


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@patch("sagemaker.remote_function.job._JobSettings")
def test_step_decorator_with_default_step_configs(mock_job_settings):
    @step
    def sum(a, b, c, d):
        """Returns sum of numbers"""
        return a + b + c + d

    step_output = sum(2, 3, c=3, d=5)

    assert isinstance(step_output, DelayedReturn)

    function_step = get_step(step_output)

    assert function_step is not None
    assert "sum" in function_step.name
    assert function_step.display_name == "tests.unit.sagemaker.workflow.test_function_step.sum"
    assert function_step.description == "Returns sum of numbers"
    assert function_step.retry_policies == []
    assert function_step.depends_on == []
    assert function_step._serialized_data.func is not None
    assert function_step._serialized_data.args is not None


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@patch("sagemaker.remote_function.job._JobSettings")
def test_step_decorator_default_name_collision(mock_job_settings):
    @step
    def sum(a, b, c, d):
        """Returns sum of numbers"""
        return a + b + c + d

    step_output1 = sum(2, 3, c=3, d=5)
    uuid_pattern = (
        r"^(.+)-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    )

    assert isinstance(step_output1, DelayedReturn)
    function_step1 = get_step(step_output1)
    assert function_step1 is not None
    assert "sum" in function_step1.name
    assert len(function_step1.name) < 63
    assert re.match(uuid_pattern, function_step1.name)

    step_output2 = sum(2, 3, c=3, d=5)

    function_step2 = get_step(step_output2)
    assert "sum" in function_step2.name
    assert len(function_step2.name) < 63
    assert re.match(uuid_pattern, function_step2.name)

    assert function_step2.name != function_step1.name


def test_step_decorator_auto_capture_dependencies_error():

    with pytest.raises(
        ValueError, match=r"Auto Capture of dependencies is not supported for pipeline steps."
    ):

        @step(dependencies="auto_capture")
        def sum(a, b):
            return a + b


@pytest.mark.parametrize(
    "args, kwargs, error_message",
    [
        (
            [1, 2, 3],
            {},
            "decorated_function() missing 2 required keyword-only arguments: 'd', and 'e'",
        ),
        ([1, 2, 3], {"d": 4}, "decorated_function() missing 1 required keyword-only argument: 'e'"),
        (
            [1, 2, 3],
            {"d": 3, "e": 4, "g": "extra_arg"},
            "decorated_function() got an unexpected keyword argument 'g'",
        ),
        (
            [],
            {"c": 3, "d": 4},
            "decorated_function() missing 2 required positional arguments: 'a', and 'b'",
        ),
        ([1], {"c": 3, "d": 4}, "decorated_function() missing 1 required positional argument: 'b'"),
        (
            [1, 2, 3, "extra_arg"],
            {"d": 3, "e": 4},
            "decorated_function() takes 3 positional arguments but 4 were given.",
        ),
        ([], {"a": 1, "b": 2, "d": 3, "e": 2}, None),
        (
            (1, 2),
            {"a": 1, "c": 3, "d": 2},
            "decorated_function() got multiple values for argument 'a'",
        ),
        (
            (1, 2),
            {"b": 1, "c": 3, "d": 2},
            "decorated_function() got multiple values for argument 'b'",
        ),
    ],
)
@patch("sagemaker.remote_function.job._JobSettings")
def test_step_decorator_invalid_function_args(mock_job_settings, args, kwargs, error_message):
    @step
    def decorated_function(a, b, c=1, *, d, e, f=3):
        return a * b * c * d * e * f

    if error_message:
        with pytest.raises(TypeError) as e:
            decorated_function(*args, **kwargs)
        assert error_message in str(e.value)
    else:
        try:
            decorated_function(*args, **kwargs)
        except Exception as ex:
            pytest.fail("Unexpected Exception: " + str(ex))


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@patch("sagemaker.remote_function.job._Job.compile", return_value=SOME_TRAINING_REQUEST)
@patch("sagemaker.remote_function.job._JobSettings")
def test_function_step_to_request(mock_job_settings_ctr, mock_compile, *args):
    s3_root_uri = "s3://bucket"
    mock_job_settings = Mock()
    mock_job_settings.s3_root_uri = s3_root_uri
    mock_job_settings.job_name_prefix = "sum"

    mock_job_settings_ctr.return_value = mock_job_settings

    @step(
        name="step_name",
        display_name="step_display_name",
        description="step_description",
    )
    def sum(a, b):
        return a + b

    step_output = sum(2, 3)

    assert step_output._step.to_request() == {
        "Name": "step_name",
        "Type": "Training",
        "Description": "step_description",
        "DisplayName": "step_display_name",
        "Arguments": SOME_TRAINING_REQUEST,
    }

    mock_compile.assert_called_once_with(
        job_settings=mock_job_settings,
        job_name="sum",
        s3_base_uri=s3_root_uri + "/" + MOCKED_PIPELINE_CONFIG.pipeline_name,
        func=ANY,
        func_args=(2, 3),
        func_kwargs={},
        serialized_data=step_output._step._serialized_data,
    )

    mock_job_settings_ctr.assert_called_once()


@patch("sagemaker.remote_function.job._JobSettings", MagicMock())
@pytest.mark.parametrize(
    "type_hint",
    [
        list,
        List,
        List[int],
        tuple,
        Tuple,
        Tuple[int, int, int],
        Tuple[int, ...],
    ],
)
def test_step_function_with_sequence_return_value(type_hint):
    @step
    def func() -> type_hint:
        return 1, 2, 3

    step_output = func()

    assert step_output[0]._reference_path == (("__getitem__", 0),)
    with pytest.raises(TypeError):
        step_output["some_key"]

    with pytest.raises(NotImplementedError):
        for _ in step_output:
            pass


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
def test_step_function_with_no_hint_on_return_values():
    @step(name="step_name")
    def func():
        return 1, 2, 3

    step_output = func()

    assert step_output["some_key"][0]._reference_path == (
        ("__getitem__", "some_key"),
        ("__getitem__", 0),
    )

    assert step_output["some_key"][0].expr == {
        "Std:JsonGet": {
            "Path": "Result['some_key'][0]",
            "S3Uri": {
                "Std:Join": {
                    "On": "/",
                    "Values": [
                        {"Get": "Steps.step_name.OutputDataConfig.S3OutputPath"},
                        "results.json",
                    ],
                }
            },
        }
    }

    assert step_output[0]["some_key"]._reference_path == (
        ("__getitem__", 0),
        ("__getitem__", "some_key"),
    )

    assert step_output[0]["some_key"].expr == {
        "Std:JsonGet": {
            "Path": "Result[0]['some_key']",
            "S3Uri": {
                "Std:Join": {
                    "On": "/",
                    "Values": [
                        {"Get": "Steps.step_name.OutputDataConfig.S3OutputPath"},
                        "results.json",
                    ],
                }
            },
        }
    }

    with pytest.raises(NotImplementedError):
        for _ in step_output:
            pass


@patch("sagemaker.remote_function.core.serialization.CloudpickleSerializer.serialize", MagicMock())
@patch("sagemaker.remote_function.job._JobSettings", MagicMock())
def test_step_function_take_in_delayed_return_as_positional_arguments():
    @step
    def func_1() -> Tuple:
        return 1, 2, 3

    @step
    def func_2(a, b, c, param_1, param_2):
        return a, b, c

    func_1_output = func_1()
    func_2_output = func_2(
        func_1_output[0],
        func_1_output[1],
        func_1_output[2],
        param_1=ExecutionVariables.PIPELINE_EXECUTION_ID,
        param_2=ParameterInteger("param_2"),
    )

    assert get_step(func_2_output).depends_on == [get_step(func_1_output)]
    with pytest.raises(ValueError):
        get_step(func_2_output).depends_on = []


@patch("sagemaker.remote_function.core.serialization.CloudpickleSerializer.serialize", MagicMock())
@patch("sagemaker.remote_function.job._JobSettings", MagicMock())
def test_step_function_take_in_delayed_return_as_keyword_arguments():
    @step
    def func_1() -> Tuple:
        return 1, 2, 3

    @step
    def func_2(a, b, c, param_1, param_2):
        return a, b, c

    func_1_output = func_1()
    func_2_output = func_2(
        a=func_1_output[0],
        b=func_1_output[1],
        c=func_1_output[2],
        param_1=ExecutionVariables.PIPELINE_EXECUTION_ID,
        param_2=ParameterInteger("param_2"),
    )

    assert get_step(func_2_output).depends_on == [get_step(func_1_output)]
    with pytest.raises(ValueError):
        get_step(func_2_output).depends_on = []


@patch("sagemaker.remote_function.core.serialization.CloudpickleSerializer.serialize", MagicMock())
@patch("sagemaker.remote_function.job._JobSettings", MagicMock())
def test_delayed_returns_in_nested_object_are_ignored():
    @step
    def func_1() -> Tuple:
        return 1, 2, 3

    @step
    def func_2(data, param_1, param_2):
        return data

    func_1_output = func_1()
    func_2_output = func_2(
        dict(
            a=func_1_output[0],
            b=func_1_output[1],
            c=func_1_output[2],
        ),
        param_1=ExecutionVariables.PIPELINE_EXECUTION_ID,
        param_2=ParameterInteger("param_2"),
    )

    assert get_step(func_2_output).depends_on == []


@patch("sagemaker.remote_function.core.serialization.CloudpickleSerializer.serialize", MagicMock())
@patch("sagemaker.remote_function.job._JobSettings", MagicMock())
def test_unsupported_pipeline_variables_as_function_arguments():
    @step
    def func_1() -> Tuple:
        return 1, 2, 3

    @step
    def func_2(a, b, c, param_1, param_2):
        return a, b, c

    func_1_output = func_1()

    with pytest.raises(NotImplementedError) as e:
        func_2(
            func_1_output[0],
            func_1_output[1],
            func_1_output[2],
            Join(values=[ExecutionVariables.PIPELINE_EXECUTION_ID]),
            param_2=get_step(func_1_output).properties.TrainingJobName,
        )
        assert "Properties attribute is not supported for _FunctionStep" in str(e.value)


@patch("sagemaker.remote_function.core.serialization.CloudpickleSerializer.serialize", MagicMock())
@patch("sagemaker.remote_function.job._JobSettings", MagicMock())
def test_both_data_and_execution_dependency_between_steps():
    @step
    def func_0() -> None:
        pass

    @step
    def func_1() -> Tuple:
        return 1, 2, 3

    @step
    def func_2(a, b, c, param_1, param_2):
        return a, b, c

    func_0_output = func_0()
    func_1_output = func_1()
    func_2_output = func_2(
        a=func_1_output[0],
        b=func_1_output[1],
        c=func_1_output[2],
        param_1=ExecutionVariables.PIPELINE_EXECUTION_ID,
        param_2=ParameterInteger("param_2"),
    )
    get_step(func_2_output).add_depends_on([func_0_output])

    assert get_step(func_2_output).depends_on == [get_step(func_1_output), func_0_output]
    with pytest.raises(ValueError):
        get_step(func_2_output).depends_on = []


@patch("sagemaker.remote_function.job._JobSettings", MagicMock())
def test_disable_deepcopy_of_delayed_return():
    @step
    def func():
        return 1

    func_output = func()

    assert id(func_output) == id(deepcopy(func_output))
