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
from mock import patch, Mock
from sagemaker.remote_function.core.pipeline_variables import (
    _ParameterInteger,
    _ParameterFloat,
    _ParameterString,
    _ParameterBoolean,
    _ExecutionVariable,
    _Properties,
    _DelayedReturn,
    Context,
    _ParameterResolver,
    _ExecutionVariableResolver,
    _DelayedReturnResolver,
    resolve_pipeline_variables,
    convert_pipeline_variables_to_pickleable,
    _PropertiesResolver,
    _S3BaseUriIdentifier,
)

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterFloat,
    ParameterString,
    ParameterBoolean,
)
from sagemaker.workflow.function_step import DelayedReturn
from sagemaker.workflow.properties import Properties

PIPELINE_NAME = "some-pipeline"


@patch("sagemaker.remote_function.core.pipeline_variables.deserialize_obj_from_s3")
def test_resolve_delayed_returns(mock_deserializer):
    delayed_returns = [
        _DelayedReturn(
            uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
            reference_path=(("__getitem__", 0),),
        ),
        _DelayedReturn(
            uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
            reference_path=(("__getitem__", 1),),
        ),
        _DelayedReturn(
            uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
            reference_path=(("__getitem__", 2),),
        ),
        _DelayedReturn(
            uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
            reference_path=(("__getitem__", 2), ("__getitem__", "key")),
        ),
        # index out of bounds
        _DelayedReturn(
            uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
            reference_path=(("__getitem__", 3),),
        ),
        _DelayedReturn(uri=_Properties("Steps.func2.OutputDataConfig.S3OutputPath")),
        # the obsoleted uri schema in old SDK version
        _DelayedReturn(
            uri=["s3://my-bucket/", "sub-folder-1/"], reference_path=(("__getitem__", 0),)
        ),
    ]

    mock_deserializer.return_value = (1, 2, {"key": 3})
    context = Context(
        property_references={
            "Steps.func1.OutputDataConfig.S3OutputPath": "s3://my_bucket/exe_id/sub_folder1",
            "Steps.func2.OutputDataConfig.S3OutputPath": "s3://my_bucket/exe_id/sub_folder2",
        }
    )
    resolver = _DelayedReturnResolver(
        delayed_returns,
        "1234",
        properties_resolver=_PropertiesResolver(context),
        parameter_resolver=_ParameterResolver(context),
        execution_variable_resolver=_ExecutionVariableResolver(context),
        sagemaker_session=None,
        s3_base_uri=f"s3://my-bucket/{PIPELINE_NAME}",
    )

    assert resolver.resolve(delayed_returns[0]) == 1
    assert resolver.resolve(delayed_returns[1]) == 2
    assert resolver.resolve(delayed_returns[2]) == {"key": 3}
    assert resolver.resolve(delayed_returns[3]) == 3
    with pytest.raises(IndexError):
        resolver.resolve(delayed_returns[4])
    assert resolver.resolve(delayed_returns[5]) == (1, 2, {"key": 3})
    assert resolver.resolve(delayed_returns[6]) == 1
    assert mock_deserializer.call_count == 3


@patch("sagemaker.remote_function.core.pipeline_variables.deserialize_obj_from_s3")
def test_deserializer_fails(mock_deserializer):
    delayed_returns = [
        _DelayedReturn(
            uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
            reference_path=(("__getitem__", 0),),
        ),
        _DelayedReturn(uri=_Properties("Steps.func2.OutputDataConfig.S3OutputPath")),
    ]

    mock_deserializer.side_effect = Exception("Something went wrong")
    context = Context(
        property_references={
            "Steps.func1.OutputDataConfig.S3OutputPath": "s3://my_bucket/exe_id/sub_folder1",
            "Steps.func2.OutputDataConfig.S3OutputPath": "s3://my_bucket/exe_id/sub_folder2",
        }
    )
    with pytest.raises(Exception, match="Something went wrong"):
        _DelayedReturnResolver(
            delayed_returns,
            "1234",
            properties_resolver=_PropertiesResolver(context),
            parameter_resolver=_ParameterResolver(context),
            execution_variable_resolver=_ExecutionVariableResolver(context),
            sagemaker_session=None,
            s3_base_uri=f"s3://my-bucket/{PIPELINE_NAME}",
        )


@pytest.mark.parametrize(
    "func_args, func_kwargs",
    [
        (None, None),
        ((), {}),
        ((1, 2, 3), {"a": 1, "b": 2}),
    ],
)
@patch("sagemaker.remote_function.core.pipeline_variables.deserialize_obj_from_s3")
def test_no_pipeline_variables_to_resolve(mock_deserializer, func_args, func_kwargs, monkeypatch):

    mock_deserializer.return_value = (1.0, 2.0, 3.0)

    resolved_args, resolved_kwargs = resolve_pipeline_variables(
        Context(),
        func_args,
        func_kwargs,
        hmac_key="1234",
        s3_base_uri="s3://my-bucket",
        sagemaker_session=None,
    )

    assert resolved_args == func_args
    assert resolved_kwargs == func_kwargs


@pytest.mark.parametrize(
    "func_args, func_kwargs, expected_resolved_args, expected_resolved_kwargs",
    [
        (
            (
                _ParameterInteger("parameter_1"),
                _ParameterString("parameter_3"),
                _ParameterFloat("parameter_2"),
                _ParameterBoolean("parameter_4"),
                _DelayedReturn(
                    uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
                    reference_path=(("__getitem__", 0),),
                ),
                _DelayedReturn(
                    uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
                    reference_path=(("__getitem__", 1),),
                ),
                # obsolete uri schema in old SDK version
                _DelayedReturn(
                    uri=[
                        _S3BaseUriIdentifier(),
                        _ExecutionVariable("ExecutionId"),
                        "sub-folder-1/",
                    ],
                    reference_path=(("__getitem__", 0),),
                ),
                _DelayedReturn(
                    uri=[
                        _S3BaseUriIdentifier(),
                        _ExecutionVariable("ExecutionId"),
                        "sub-folder-1/",
                    ],
                    reference_path=(("__getitem__", 1),),
                ),
                _Properties("Steps.step_name.TrainingJobName"),
            ),
            {},
            (1, "string", 2.0, True, 1.0, 2.0, 1.0, 2.0, "a-cool-name"),
            {},
        ),
        (
            (),
            {
                "a": _ParameterInteger("parameter_1"),
                "b": _ParameterString("parameter_3"),
                "c": _ParameterFloat("parameter_2"),
                "d": _ParameterBoolean("parameter_4"),
                "e": _DelayedReturn(
                    uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
                    reference_path=(("__getitem__", 0),),
                ),
                "f": _DelayedReturn(
                    uri=_Properties("Steps.func1.OutputDataConfig.S3OutputPath"),
                    reference_path=(("__getitem__", 1),),
                ),
                "g": _Properties("Steps.step_name.TrainingJobName"),
                # obsolete uri schema in old SDK version
                "h": _DelayedReturn(
                    uri=[
                        _S3BaseUriIdentifier(),
                        _ExecutionVariable("ExecutionId"),
                        "sub-folder-1/",
                    ],
                    reference_path=(("__getitem__", 0),),
                ),
                "i": _DelayedReturn(
                    uri=[
                        _S3BaseUriIdentifier(),
                        _ExecutionVariable("ExecutionId"),
                        "sub-folder-1/",
                    ],
                    reference_path=(("__getitem__", 1),),
                ),
            },
            (),
            {
                "a": 1,
                "b": "string",
                "c": 2.0,
                "d": True,
                "e": 1.0,
                "f": 2.0,
                "g": "a-cool-name",
                "h": 1.0,
                "i": 2.0,
            },
        ),
    ],
)
@patch("sagemaker.remote_function.core.pipeline_variables.deserialize_obj_from_s3")
def test_resolve_pipeline_variables(
    mock_deserializer,
    func_args,
    func_kwargs,
    expected_resolved_args,
    expected_resolved_kwargs,
):
    s3_base_uri = f"s3://my-bucket/{PIPELINE_NAME}"
    s3_results_uri = f"{s3_base_uri}/execution-id/sub-folder-1"
    context = Context(
        property_references={
            "Parameters.parameter_1": "1",
            "Parameters.parameter_2": "2.0",
            "Parameters.parameter_3": "string",
            "Parameters.parameter_4": "true",
            "Execution.ExecutionId": "execution-id",
            "Steps.func1.OutputDataConfig.S3OutputPath": s3_results_uri,
            "Steps.step_name.TrainingJobName": "a-cool-name",
        },
    )

    mock_deserializer.return_value = (1.0, 2.0, 3.0)

    resolved_args, resolved_kwargs = resolve_pipeline_variables(
        context,
        func_args,
        func_kwargs,
        hmac_key="1234",
        s3_base_uri=s3_base_uri,
        sagemaker_session=None,
    )

    assert resolved_args == expected_resolved_args
    assert resolved_kwargs == expected_resolved_kwargs
    mock_deserializer.assert_called_once_with(
        sagemaker_session=None,
        s3_uri=s3_results_uri,
        hmac_key="1234",
    )


def test_convert_pipeline_variables_to_pickleable():
    function_step = Mock()
    function_step.name = "parent_step"
    function_step._properties.OutputDataConfig.S3OutputPath = Properties(
        step_name=function_step.name, path="OutputDataConfig.S3OutputPath"
    )
    func_args = (
        DelayedReturn(function_step, reference_path=("__getitem__", 0)),
        ParameterBoolean("parameter_1"),
        ParameterInteger("parameter_2"),
        ParameterFloat("parameter_3"),
        ParameterString("parameter_4"),
        Properties(step_name="step_name", shape_name="DescribeTrainingJobResponse").TrainingJobName,
        1,
        2.0,
    )
    func_kwargs = {
        "a": DelayedReturn(function_step, reference_path=("__getitem__", 1)),
        "b": ParameterBoolean("parameter_1"),
        "c": ParameterInteger("parameter_2"),
        "d": ParameterFloat("parameter_3"),
        "e": ParameterString("parameter_4"),
        "f": Properties(
            step_name="step_name", shape_name="DescribeTrainingJobResponse"
        ).TrainingJobName,
        "g": 1,
        "h": 2.0,
    }

    converted_args, converted_kwargs = convert_pipeline_variables_to_pickleable(
        func_args, func_kwargs
    )

    assert converted_args == (
        _DelayedReturn(
            uri=_Properties(f"Steps.{function_step.name}.OutputDataConfig.S3OutputPath"),
            reference_path=("__getitem__", 0),
        ),
        _ParameterBoolean(name="parameter_1"),
        _ParameterInteger(name="parameter_2"),
        _ParameterFloat(name="parameter_3"),
        _ParameterString(name="parameter_4"),
        _Properties(path="Steps.step_name.TrainingJobName"),
        1,
        2.0,
    )

    assert converted_kwargs == {
        "a": _DelayedReturn(
            uri=_Properties(f"Steps.{function_step.name}.OutputDataConfig.S3OutputPath"),
            reference_path=("__getitem__", 1),
        ),
        "b": _ParameterBoolean(name="parameter_1"),
        "c": _ParameterInteger(name="parameter_2"),
        "d": _ParameterFloat(name="parameter_3"),
        "e": _ParameterString(name="parameter_4"),
        "f": _Properties(path="Steps.step_name.TrainingJobName"),
        "g": 1,
        "h": 2.0,
    }
