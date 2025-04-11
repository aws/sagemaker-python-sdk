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
"""The step definitions for workflow."""
from __future__ import absolute_import

from typing import List, Union, Optional, TYPE_CHECKING

import attr

from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.execution_variables import ExecutionVariable
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.properties import PropertyFile, Properties

if TYPE_CHECKING:
    from sagemaker.workflow.steps import Step


@attr.s
class Join(PipelineVariable):
    """Join together properties.

    Examples:
    Build a Amazon S3 Uri with bucket name parameter and pipeline execution Id and use it
    as training input::

        bucket = ParameterString('bucket', default_value='my-bucket')

        TrainingInput(
            s3_data=Join(
                on='/',
                values=['s3:/', bucket, ExecutionVariables.PIPELINE_EXECUTION_ID]
            ),
            content_type="text/csv")

    Attributes:
        values (List[Union[PrimitiveType, Parameter, Expression]]):
            The primitive type values, parameters, step properties, expressions to join.
        on (str): The string to join the values on (Defaults to "").
    """

    on: str = attr.ib(factory=str)
    values: List = attr.ib(factory=list)

    def to_string(self) -> PipelineVariable:
        """Prompt the pipeline to convert the pipeline variable to String in runtime

        As Join is treated as String in runtime, no extra actions are needed.
        """
        return self

    @property
    def expr(self):
        """The expression dict for a `Join` function."""

        return {
            "Std:Join": {
                "On": self.on,
                "Values": [
                    value.expr if hasattr(value, "expr") else value for value in self.values
                ],
            },
        }

    @property
    def _referenced_steps(self) -> List[Union["Step", str]]:
        """List of step names that this function depends on."""
        steps = []
        for value in self.values:
            if isinstance(value, PipelineVariable):
                steps.extend(value._referenced_steps)
        return steps


@attr.s
class JsonGet(PipelineVariable):
    """Get JSON properties from PropertyFiles or S3 location.

    Attributes:
        step_name (str): The step name from which to get the property file.
        property_file (Optional[Union[PropertyFile, str]]): Either a PropertyFile instance
            or the name of a property file.
        json_path (str): The JSON path expression to the requested value.
        s3_uri (Optional[sagemaker.workflow.functions.Join]): The S3 location from which to fetch
            a Json file. The Json file is the output of a step defined with ``@step`` decorator.
        step (Step): The upstream step object which the s3_uri is associated to.
    """

    # pylint: disable=W0613
    def _check_property_file_s3_uri(self, attribute, value):
        """Validate mutually exclusive property file / s3uri"""
        if self.property_file and self.s3_uri:
            raise ValueError(
                "Please specify either a property file or s3 uri as an input, but not both."
            )
        if not self.property_file and not self.s3_uri:
            raise ValueError(
                "Missing s3uri or property file as a required input to JsonGet."
                "Please specify either a property file or s3 uri as an input, but not both."
            )
        if self.s3_uri:
            self._validate_json_get_s3_uri()

    step_name: str = attr.ib(default=None)
    property_file: Optional[Union[PropertyFile, str]] = attr.ib(
        default=None, validator=_check_property_file_s3_uri
    )
    json_path: str = attr.ib(default=None)
    s3_uri: Optional[Join] = attr.ib(default=None, validator=_check_property_file_s3_uri)
    step: "Step" = attr.ib(default=None)

    # pylint: disable=R1710
    @property
    def expr(self):
        """The expression dict for a `JsonGet` function."""

        if self.property_file:
            if not isinstance(self.step_name, str) or not self.step_name:
                raise ValueError("Please give a valid step name as a string.")
            if isinstance(self.property_file, PropertyFile):
                name = self.property_file.name
            else:
                name = self.property_file
            return {
                "Std:JsonGet": {
                    "PropertyFile": {"Get": f"Steps.{self.step_name}.PropertyFiles.{name}"},
                    "Path": self.json_path,
                }
            }

        # ConditionStep uses a JoinFunction to provide this non-static, built s3Uri in
        # the case of Lightsaber steps.
        if self.s3_uri:
            return {
                "Std:JsonGet": {
                    "S3Uri": (
                        self.s3_uri.expr
                        if isinstance(self.s3_uri, PipelineVariable)
                        else self.s3_uri
                    ),
                    "Path": self.json_path,
                }
            }

    @property
    def _referenced_steps(self) -> List[Union["Step", str]]:
        """List of step that this function depends on."""
        if self.step:
            return [self.step]
        if self.step_name:
            return [self.step_name]

        return []

    def _validate_json_get_s3_uri(self):
        """Validate the s3 uri in JsonGet"""
        s3_uri = self.s3_uri
        if not isinstance(s3_uri, Join):
            raise ValueError(
                f"Invalid JsonGet function {self.expr}. JsonGet "
                "function's s3_uri can only be a sagemaker.workflow.functions.Join object."
            )
        for join_arg in s3_uri.values:
            if not is_pipeline_variable(join_arg):
                continue
            if not isinstance(join_arg, (Parameter, ExecutionVariable, Properties)):
                raise ValueError(
                    f"Invalid JsonGet function {self.expr}. "
                    f"The Join values in JsonGet's s3_uri can only be a primitive object, "
                    f"Parameter, ExecutionVariable or Properties."
                )
