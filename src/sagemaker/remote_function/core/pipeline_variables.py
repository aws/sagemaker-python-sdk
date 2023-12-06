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
"""SageMaker remote function data serializer/deserializer."""
from __future__ import absolute_import

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Union, Dict, List, Tuple

from sagemaker.s3 import s3_path_join
from sagemaker.remote_function.core.serialization import deserialize_obj_from_s3


@dataclass
class Context:
    """Context for an execution."""

    step_name: str = None
    execution_id: str = None
    property_references: Dict[str, str] = field(default_factory=dict)
    serialize_output_to_json: bool = False
    func_step_s3_dir: str = None


@dataclass
class _Parameter:
    """Parameter to a function."""

    name: str


class _ParameterInteger(_Parameter):
    """Integer parameter to a function."""

    ...


class _ParameterFloat(_Parameter):
    """Float parameter to a function."""

    ...


class _ParameterString(_Parameter):
    """String parameter to a function."""

    ...


class _ParameterBoolean(_Parameter):
    """Boolean parameter to a function."""

    ...


@dataclass
class _Properties:
    """Properties of classic steps."""

    path: str


@dataclass
class _ExecutionVariable:
    """Execution variable."""

    name: str


@dataclass
class _DelayedReturn:
    """Delayed return from a function."""

    uri: List[Union[str, _Parameter, _ExecutionVariable]]
    reference_path: Tuple = field(default_factory=tuple)


class _ExecutionVariableResolver:
    """Resolve execution variables."""

    def __init__(self, context: Context):
        """Resolve execution variables."""
        self._context = context

    def resolve(self, execution_variable: _ExecutionVariable):
        """Resolve a single execution variable.

        Args:
            execution_variable: execution variable to resolve.
        Returns:
            resolved value
        """
        return self._context.property_references[f"Execution.{execution_variable.name}"]


class _ParameterResolver:
    """Resolve parameters."""

    def __init__(self, context: Context):
        """Resolve parameters."""
        self._context = context

    def resolve(self, parameter: _Parameter):
        """Resolve a single property reference.

        Args:
            parameter: parameter to resolve.
        Returns:
            resolved value
        """
        if isinstance(parameter, _ParameterInteger):
            return int(self._context.property_references[f"Parameters.{parameter.name}"])
        if isinstance(parameter, _ParameterFloat):
            return float(self._context.property_references[f"Parameters.{parameter.name}"])
        if isinstance(parameter, _ParameterString):
            return self._context.property_references[f"Parameters.{parameter.name}"]

        return self._context.property_references[f"Parameters.{parameter.name}"] == "true"


class _PropertiesResolver:
    """Resolve classic step properties."""

    def __init__(self, context: Context):
        """Resolve classic step properties."""
        self._context = context

    def resolve(self, properties: _Properties):
        """Resolve classic step properties.

        Args:
            properties: classic step properties.
        Returns:
            resolved value
        """
        return self._context.property_references[properties.path]


class _DelayedReturnResolver:
    """Resolve delayed returns."""

    def __init__(
        self,
        delayed_returns: List[_DelayedReturn],
        hmac_key: str,
        parameter_resolver: _ParameterResolver,
        execution_variable_resolver: _ExecutionVariableResolver,
        **settings,
    ):
        """Resolve delayed return.

        Args:
            delayed_returns: list of delayed returns to resolve.
            hmac_key: key used to encrypt serialized and deserialized function and arguments.
            parameter_resolver: resolver used to pipeline parameters.
            execution_variable_resolver: resolver used to resolve execution variables.
            **settings: settings to pass to the deserialization function.
        """
        self._parameter_resolver = parameter_resolver
        self._execution_variable_resolver = execution_variable_resolver
        # different delayed returns can have the same uri, so we need to dedupe
        uris = {
            self._resolve_delayed_return_uri(delayed_return) for delayed_return in delayed_returns
        }

        def deserialization_task(uri):
            return uri, deserialize_obj_from_s3(
                sagemaker_session=settings["sagemaker_session"],
                s3_uri=uri,
                hmac_key=hmac_key,
            )

        with ThreadPoolExecutor() as executor:
            self._deserialized_objects = dict(executor.map(deserialization_task, uris))

    def resolve(self, delayed_return: _DelayedReturn) -> Any:
        """Resolve a single delayed return.

        Args:
            delayed_return: delayed return to resolve.
        Returns:
            resolved delayed return.
        """
        deserialized_obj = self._deserialized_objects[
            self._resolve_delayed_return_uri(delayed_return)
        ]
        return _retrieve_child_item(delayed_return, deserialized_obj)

    def _resolve_delayed_return_uri(self, delayed_return: _DelayedReturn):
        """Resolve the s3 uri of the delayed return."""

        uri = []
        for component in delayed_return.uri:
            if isinstance(component, _Parameter):
                uri.append(self._parameter_resolver.resolve(component))
            elif isinstance(component, _ExecutionVariable):
                uri.append(self._execution_variable_resolver.resolve(component))
            else:
                uri.append(component)
        return s3_path_join(*uri)


def _retrieve_child_item(delayed_return: _DelayedReturn, deserialized_obj: Any):
    """Retrieve child item from deserialized object."""
    result = deserialized_obj
    for component in delayed_return.reference_path:
        result = result[component[1]]
    return result


def resolve_pipeline_variables(
    context: Context, func_args: Tuple, func_kwargs: Dict, hmac_key: str, **settings
):
    """Resolve pipeline variables.

    Args:
        context: context for the execution.
        func_args: function args.
        func_kwargs: function kwargs.
        hmac_key: key used to encrypt serialized and deserialized function and arguments.
        **settings: settings to pass to the deserialization function.
    """

    delayed_returns = []

    if func_args is not None:
        for arg in func_args:
            if isinstance(arg, _DelayedReturn):
                delayed_returns.append(arg)
    if func_kwargs is not None:
        for arg in func_kwargs.values():
            if isinstance(arg, _DelayedReturn):
                delayed_returns.append(arg)

    # build the resolvers
    parameter_resolver = _ParameterResolver(context)
    execution_variable_resolver = _ExecutionVariableResolver(context)
    properties_resolver = _PropertiesResolver(context)
    delayed_return_resolver = _DelayedReturnResolver(
        delayed_returns=delayed_returns,
        hmac_key=hmac_key,
        parameter_resolver=parameter_resolver,
        execution_variable_resolver=execution_variable_resolver,
        **settings,
    )

    # resolve the pipeline variables
    resolved_func_args = None
    if func_args is not None:
        resolved_func_args = []
        for arg in func_args:
            if isinstance(arg, _Parameter):
                resolved_func_args.append(parameter_resolver.resolve(arg))
            elif isinstance(arg, _ExecutionVariable):
                resolved_func_args.append(execution_variable_resolver.resolve(arg))
            elif isinstance(arg, _Properties):
                resolved_func_args.append(properties_resolver.resolve(arg))
            elif isinstance(arg, _DelayedReturn):
                resolved_func_args.append(delayed_return_resolver.resolve(arg))
            else:
                resolved_func_args.append(arg)
        resolved_func_args = tuple(resolved_func_args)

    resolved_func_kwargs = None
    if func_kwargs is not None:
        resolved_func_kwargs = {}
        for key, value in func_kwargs.items():
            if isinstance(value, _Parameter):
                resolved_func_kwargs[key] = parameter_resolver.resolve(value)
            elif isinstance(value, _ExecutionVariable):
                resolved_func_kwargs[key] = execution_variable_resolver.resolve(value)
            elif isinstance(value, _Properties):
                resolved_func_kwargs[key] = properties_resolver.resolve(value)
            elif isinstance(value, _DelayedReturn):
                resolved_func_kwargs[key] = delayed_return_resolver.resolve(value)
            else:
                resolved_func_kwargs[key] = value

    return resolved_func_args, resolved_func_kwargs


def convert_pipeline_variables_to_pickleable(s3_base_uri: str, func_args: Tuple, func_kwargs: Dict):
    """Convert pipeline variables to pickleable.

    Args:
        s3_base_uri: s3 base uri where artifacts are stored.
        func_args: function args.
        func_kwargs: function kwargs.
    """

    from sagemaker.workflow.entities import PipelineVariable

    from sagemaker.workflow.execution_variables import ExecutionVariables

    from sagemaker.workflow.function_step import DelayedReturn

    def convert(arg):
        if isinstance(arg, DelayedReturn):
            return _DelayedReturn(
                uri=[
                    s3_base_uri,
                    ExecutionVariables.PIPELINE_EXECUTION_ID._pickleable,
                    arg._step.name,
                    "results",
                ],
                reference_path=arg._reference_path,
            )

        if isinstance(arg, PipelineVariable):
            return arg._pickleable

        return arg

    converted_func_args = tuple(convert(arg) for arg in func_args)
    converted_func_kwargs = {key: convert(arg) for key, arg in func_kwargs.items()}

    return converted_func_args, converted_func_kwargs
