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
"""The Pipeline entity for workflow."""
from __future__ import absolute_import

import json

from copy import deepcopy
from typing import Any, Dict, List, Sequence, Union, Optional

import attr
import botocore
from botocore.exceptions import ClientError

from sagemaker import s3
from sagemaker._studio import _append_project_tags
from sagemaker.session import Session
from sagemaker.workflow.callback_step import CallbackOutput, CallbackStep
from sagemaker.workflow.lambda_step import LambdaOutput, LambdaStep
from sagemaker.workflow.entities import (
    Entity,
    Expression,
    RequestType,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.parallelism_config import ParallelismConfiguration
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import Step
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.utilities import list_to_request


@attr.s
class Pipeline(Entity):
    """Pipeline for workflow.

    Attributes:
        name (str): The name of the pipeline.
        parameters (Sequence[Parameter]): The list of the parameters.
        pipeline_experiment_config (Optional[PipelineExperimentConfig]): If set,
            the workflow will attempt to create an experiment and trial before
            executing the steps. Creation will be skipped if an experiment or a trial with
            the same name already exists. By default, pipeline name is used as
            experiment name and execution id is used as the trial name.
            If set to None, no experiment or trial will be created automatically.
        steps (Sequence[Union[Step, StepCollection]]): The list of the non-conditional steps
            associated with the pipeline. Any steps that are within the
            `if_steps` or `else_steps` of a `ConditionStep` cannot be listed in the steps of a
            pipeline. Of particular note, the workflow service rejects any pipeline definitions that
            specify a step in the list of steps of a pipeline and that step in the `if_steps` or
            `else_steps` of any `ConditionStep`.
        sagemaker_session (sagemaker.session.Session): Session object that manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            pipeline creates one using the default AWS configuration chain.
    """

    name: str = attr.ib(factory=str)
    parameters: Sequence[Parameter] = attr.ib(factory=list)
    pipeline_experiment_config: Optional[PipelineExperimentConfig] = attr.ib(
        default=PipelineExperimentConfig(
            ExecutionVariables.PIPELINE_NAME, ExecutionVariables.PIPELINE_EXECUTION_ID
        )
    )
    steps: Sequence[Union[Step, StepCollection]] = attr.ib(factory=list)
    sagemaker_session: Session = attr.ib(factory=Session)

    _version: str = "2020-12-01"
    _metadata: Dict[str, Any] = dict()

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        return {
            "Version": self._version,
            "Metadata": self._metadata,
            "Parameters": list_to_request(self.parameters),
            "PipelineExperimentConfig": self.pipeline_experiment_config.to_request()
            if self.pipeline_experiment_config is not None
            else None,
            "Steps": list_to_request(self.steps),
        }

    def create(
        self,
        role_arn: str,
        description: str = None,
        tags: List[Dict[str, str]] = None,
        parallelism_config: ParallelismConfiguration = None,
    ) -> Dict[str, Any]:
        """Creates a Pipeline in the Pipelines service.

        Args:
            role_arn (str): The role arn that is assumed by the pipeline to create step artifacts.
            description (str): A description of the pipeline.
            tags (List[Dict[str, str]]): A list of {"Key": "string", "Value": "string"} dicts as
                tags.
            parallelism_config (Optional[ParallelismConfiguration]): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline.

        Returns:
            A response dict from the service.
        """
        tags = _append_project_tags(tags)
        kwargs = self._create_args(role_arn, description, parallelism_config)
        update_args(
            kwargs,
            Tags=tags,
        )
        return self.sagemaker_session.sagemaker_client.create_pipeline(**kwargs)

    def _create_args(
        self, role_arn: str, description: str, parallelism_config: ParallelismConfiguration
    ):
        """Constructs the keyword argument dict for a create_pipeline call.

        Args:
            role_arn (str): The role arn that is assumed by pipelines to create step artifacts.
            description (str): A description of the pipeline.
            parallelism_config (Optional[ParallelismConfiguration]): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline.

        Returns:
            A keyword argument dict for calling create_pipeline.
        """
        pipeline_definition = self.definition()
        kwargs = dict(
            PipelineName=self.name,
            RoleArn=role_arn,
        )

        # If pipeline definition is large, upload to S3 bucket and
        # provide PipelineDefinitionS3Location to request instead.
        if len(pipeline_definition.encode("utf-8")) < 1024 * 100:
            kwargs["PipelineDefinition"] = pipeline_definition
        else:
            desired_s3_uri = s3.s3_path_join(
                "s3://", self.sagemaker_session.default_bucket(), self.name
            )
            s3.S3Uploader.upload_string_as_file_body(
                body=pipeline_definition,
                desired_s3_uri=desired_s3_uri,
                sagemaker_session=self.sagemaker_session,
            )
            kwargs["PipelineDefinitionS3Location"] = {
                "Bucket": self.sagemaker_session.default_bucket(),
                "ObjectKey": self.name,
            }

        update_args(
            kwargs, PipelineDescription=description, ParallelismConfiguration=parallelism_config
        )
        return kwargs

    def describe(self) -> Dict[str, Any]:
        """Describes a Pipeline in the Workflow service.

        Returns:
            Response dict from the service. See `boto3 client documentation
            <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/\
sagemaker.html#SageMaker.Client.describe_pipeline>`_
        """
        return self.sagemaker_session.sagemaker_client.describe_pipeline(PipelineName=self.name)

    def update(
        self,
        role_arn: str,
        description: str = None,
        parallelism_config: ParallelismConfiguration = None,
    ) -> Dict[str, Any]:
        """Updates a Pipeline in the Workflow service.

        Args:
            role_arn (str): The role arn that is assumed by pipelines to create step artifacts.
            description (str): A description of the pipeline.
            parallelism_config (Optional[ParallelismConfiguration]): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline.

        Returns:
            A response dict from the service.
        """
        kwargs = self._create_args(role_arn, description, parallelism_config)
        return self.sagemaker_session.sagemaker_client.update_pipeline(**kwargs)

    def upsert(
        self,
        role_arn: str,
        description: str = None,
        tags: List[Dict[str, str]] = None,
        parallelism_config: ParallelismConfiguration = None,
    ) -> Dict[str, Any]:
        """Creates a pipeline or updates it, if it already exists.

        Args:
            role_arn (str): The role arn that is assumed by workflow to create step artifacts.
            description (str): A description of the pipeline.
            tags (List[Dict[str, str]]): A list of {"Key": "string", "Value": "string"} dicts as
                tags.
            parallelism_config (Optional[Config for parallel steps, Parallelism configuration that
                is applied to each of. the executions

        Returns:
            response dict from service
        """
        try:
            response = self.create(role_arn, description, tags, parallelism_config)
        except ClientError as e:
            error = e.response["Error"]
            if (
                error["Code"] == "ValidationException"
                and "Pipeline names must be unique within" in error["Message"]
            ):
                response = self.update(role_arn, description)
                if tags is not None:
                    old_tags = self.sagemaker_session.sagemaker_client.list_tags(
                        ResourceArn=response["PipelineArn"]
                    )["Tags"]

                    tag_keys = [tag["Key"] for tag in tags]
                    for old_tag in old_tags:
                        if old_tag["Key"] not in tag_keys:
                            tags.append(old_tag)

                    self.sagemaker_session.sagemaker_client.add_tags(
                        ResourceArn=response["PipelineArn"], Tags=tags
                    )
            else:
                raise
        return response

    def delete(self) -> Dict[str, Any]:
        """Deletes a Pipeline in the Workflow service.

        Returns:
            A response dict from the service.
        """
        return self.sagemaker_session.sagemaker_client.delete_pipeline(PipelineName=self.name)

    def start(
        self,
        parameters: Dict[str, Union[str, bool, int, float]] = None,
        execution_display_name: str = None,
        execution_description: str = None,
        parallelism_config: ParallelismConfiguration = None,
    ):
        """Starts a Pipeline execution in the Workflow service.

        Args:
            parameters (Dict[str, Union[str, bool, int, float]]): values to override
                pipeline parameters.
            execution_display_name (str): The display name of the pipeline execution.
            execution_description (str): A description of the execution.
            parallelism_config (Optional[ParallelismConfiguration]): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline.

        Returns:
            A `_PipelineExecution` instance, if successful.
        """
        exists = True
        try:
            self.describe()
        except ClientError:
            exists = False

        if not exists:
            raise ValueError(
                "This pipeline is not associated with a Pipeline in SageMaker. "
                "Please invoke create() first before attempting to invoke start()."
            )

        kwargs = dict(PipelineName=self.name)
        update_args(
            kwargs,
            PipelineParameters=format_start_parameters(parameters),
            PipelineExecutionDescription=execution_description,
            PipelineExecutionDisplayName=execution_display_name,
            ParallelismConfiguration=parallelism_config,
        )
        response = self.sagemaker_session.sagemaker_client.start_pipeline_execution(**kwargs)
        return _PipelineExecution(
            arn=response["PipelineExecutionArn"],
            sagemaker_session=self.sagemaker_session,
        )

    def definition(self) -> str:
        """Converts a request structure to string representation for workflow service calls."""
        request_dict = self.to_request()
        request_dict["PipelineExperimentConfig"] = interpolate(
            request_dict["PipelineExperimentConfig"], {}, {}
        )
        callback_output_to_step_map = _map_callback_outputs(self.steps)
        lambda_output_to_step_name = _map_lambda_outputs(self.steps)
        request_dict["Steps"] = interpolate(
            request_dict["Steps"],
            callback_output_to_step_map=callback_output_to_step_map,
            lambda_output_to_step_map=lambda_output_to_step_name,
        )

        return json.dumps(request_dict)


def format_start_parameters(parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Formats start parameter overrides as a list of dicts.

    This list of dicts adheres to the request schema of:

        `{"Name": "MyParameterName", "Value": "MyValue"}`

    Args:
        parameters (Dict[str, Any]): A dict of named values where the keys are
            the names of the parameters to pass values into.
    """
    if parameters is None:
        return None
    return [{"Name": name, "Value": str(value)} for name, value in parameters.items()]


def interpolate(
    request_obj: RequestType,
    callback_output_to_step_map: Dict[str, str],
    lambda_output_to_step_map: Dict[str, str],
) -> RequestType:
    """Replaces Parameter values in a list of nested Dict[str, Any] with their workflow expression.

    Args:
        request_obj (RequestType): The request dict.
        callback_output_to_step_map (Dict[str, str]): A dict of output name -> step name.
        lambda_output_to_step_map (Dict[str, str]): A dict of output name -> step name.

    Returns:
        RequestType: The request dict with Parameter values replaced by their expression.
    """
    try:
        request_obj_copy = deepcopy(request_obj)
        return _interpolate(
            request_obj_copy,
            callback_output_to_step_map=callback_output_to_step_map,
            lambda_output_to_step_map=lambda_output_to_step_map,
        )
    except TypeError as type_err:
        raise TypeError("Not able to interpolate Pipeline definition: %s" % type_err)


def _interpolate(
    obj: Union[RequestType, Any],
    callback_output_to_step_map: Dict[str, str],
    lambda_output_to_step_map: Dict[str, str],
):
    """Walks the nested request dict, replacing Parameter type values with workflow expressions.

    Args:
        obj (Union[RequestType, Any]): The request dict.
        callback_output_to_step_map (Dict[str, str]): A dict of output name -> step name.
    """
    if isinstance(obj, (Expression, Parameter, Properties)):
        return obj.expr

    if isinstance(obj, CallbackOutput):
        step_name = callback_output_to_step_map[obj.output_name]
        return obj.expr(step_name)
    if isinstance(obj, LambdaOutput):
        step_name = lambda_output_to_step_map[obj.output_name]
        return obj.expr(step_name)
    if isinstance(obj, dict):
        new = obj.__class__()
        for key, value in obj.items():
            new[key] = interpolate(value, callback_output_to_step_map, lambda_output_to_step_map)
    elif isinstance(obj, (list, set, tuple)):
        new = obj.__class__(
            interpolate(value, callback_output_to_step_map, lambda_output_to_step_map)
            for value in obj
        )
    else:
        return obj
    return new


def _map_callback_outputs(steps: List[Step]):
    """Iterate over the provided steps, building a map of callback output parameters to step names.

    Args:
        step (List[Step]): The steps list.
    """

    callback_output_map = {}
    for step in steps:
        if isinstance(step, CallbackStep):
            if step.outputs:
                for output in step.outputs:
                    callback_output_map[output.output_name] = step.name

    return callback_output_map


def _map_lambda_outputs(steps: List[Step]):
    """Iterate over the provided steps, building a map of lambda output parameters to step names.

    Args:
        step (List[Step]): The steps list.
    """

    lambda_output_map = {}
    for step in steps:
        if isinstance(step, LambdaStep):
            if step.outputs:
                for output in step.outputs:
                    lambda_output_map[output.output_name] = step.name

    return lambda_output_map


def update_args(args: Dict[str, Any], **kwargs):
    """Updates the request arguments dict with a value, if populated.

    This handles the case when the service API doesn't like NoneTypes for argument values.

    Args:
        request_args (Dict[str, Any]): The request arguments dict.
        kwargs: key, value pairs to update the args dict with.
    """
    for key, value in kwargs.items():
        if value is not None:
            args.update({key: value})


@attr.s
class _PipelineExecution:
    """Internal class for encapsulating pipeline execution instances.

    Attributes:
        arn (str): The arn of the pipeline execution.
        sagemaker_session (sagemaker.session.Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            pipeline creates one using the default AWS configuration chain.
    """

    arn: str = attr.ib()
    sagemaker_session: Session = attr.ib(factory=Session)

    def stop(self):
        """Stops a pipeline execution."""
        return self.sagemaker_session.sagemaker_client.stop_pipeline_execution(
            PipelineExecutionArn=self.arn
        )

    def describe(self):
        """Describes a pipeline execution.

        Returns:
             Information about the pipeline execution. See
             `boto3 client describe_pipeline_execution
             <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/\
sagemaker.html#SageMaker.Client.describe_pipeline_execution>`_.
        """
        return self.sagemaker_session.sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=self.arn,
        )

    def list_steps(self):
        """Describes a pipeline execution's steps.

        Returns:
             Information about the steps of the pipeline execution. See
             `boto3 client list_pipeline_execution_steps
             <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/\
sagemaker.html#SageMaker.Client.list_pipeline_execution_steps>`_.
        """
        response = self.sagemaker_session.sagemaker_client.list_pipeline_execution_steps(
            PipelineExecutionArn=self.arn
        )
        return response["PipelineExecutionSteps"]

    def wait(self, delay=30, max_attempts=60):
        """Waits for a pipeline execution.

        Args:
            delay (int): The polling interval. (Defaults to 30 seconds)
            max_attempts (int): The maximum number of polling attempts.
                (Defaults to 60 polling attempts)
        """
        waiter_id = "PipelineExecutionComplete"
        # TODO: this waiter should be included in the botocore
        model = botocore.waiter.WaiterModel(
            {
                "version": 2,
                "waiters": {
                    waiter_id: {
                        "delay": delay,
                        "maxAttempts": max_attempts,
                        "operation": "DescribePipelineExecution",
                        "acceptors": [
                            {
                                "expected": "Succeeded",
                                "matcher": "path",
                                "state": "success",
                                "argument": "PipelineExecutionStatus",
                            },
                            {
                                "expected": "Failed",
                                "matcher": "path",
                                "state": "failure",
                                "argument": "PipelineExecutionStatus",
                            },
                        ],
                    }
                },
            }
        )
        waiter = botocore.waiter.create_waiter_with_client(
            waiter_id, model, self.sagemaker_session.sagemaker_client
        )
        waiter.wait(PipelineExecutionArn=self.arn)
