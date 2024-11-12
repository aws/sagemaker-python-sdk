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

import logging
from datetime import datetime
from typing import Any, Dict, List, Set, Sequence, Union, Optional

import attr
import botocore
import pytz
from botocore.exceptions import ClientError, WaiterError

from sagemaker import s3, LocalSession
from sagemaker._studio import _append_project_tags
from sagemaker.config import PIPELINE_ROLE_ARN_PATH, PIPELINE_TAGS_PATH
from sagemaker.remote_function.core.serialization import deserialize_obj_from_s3
from sagemaker.remote_function.core.stored_function import RESULTS_FOLDER
from sagemaker.remote_function.errors import RemoteFunctionError
from sagemaker.remote_function.job import JOBS_CONTAINER_ENTRYPOINT
from sagemaker.s3_utils import s3_path_join
from sagemaker.session import Session
from sagemaker.utils import resolve_value_from_config, retry_with_backoff, format_tags, Tags
from sagemaker.workflow.callback_step import CallbackOutput, CallbackStep
from sagemaker.workflow._event_bridge_client_helper import (
    EventBridgeSchedulerHelper,
    RESOURCE_NOT_FOUND,
    RESOURCE_NOT_FOUND_EXCEPTION,
    EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT,
)
from sagemaker.workflow.lambda_step import LambdaOutput, LambdaStep
from sagemaker.workflow.entities import (
    Expression,
    RequestType,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.parallelism_config import ParallelismConfiguration
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.selective_execution_config import SelectiveExecutionConfig
from sagemaker.workflow.step_outputs import StepOutput
from sagemaker.workflow.steps import Step
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.triggers import (
    PipelineSchedule,
    Trigger,
    validate_default_parameters_for_schedules,
)
from sagemaker.workflow.utilities import list_to_request
from sagemaker.workflow._steps_compiler import StepsCompiler

logger = logging.getLogger(__name__)

_DEFAULT_EXPERIMENT_CFG = PipelineExperimentConfig(
    ExecutionVariables.PIPELINE_NAME, ExecutionVariables.PIPELINE_EXECUTION_ID
)

_DEFAULT_DEFINITION_CFG = PipelineDefinitionConfig(use_custom_job_prefix=False)


class Pipeline:
    """Pipeline for workflow."""

    def __init__(
        self,
        name: str = "",
        parameters: Optional[Sequence[Parameter]] = None,
        pipeline_experiment_config: Optional[PipelineExperimentConfig] = _DEFAULT_EXPERIMENT_CFG,
        steps: Optional[Sequence[Union[Step, StepCollection, StepOutput]]] = None,
        sagemaker_session: Optional[Session] = None,
        pipeline_definition_config: Optional[PipelineDefinitionConfig] = _DEFAULT_DEFINITION_CFG,
    ):
        """Initialize a Pipeline

        Args:
            name (str): The name of the pipeline.
            parameters (Sequence[Parameter]): The list of the parameters.
            pipeline_experiment_config (Optional[PipelineExperimentConfig]): If set,
                the workflow will attempt to create an experiment and trial before
                executing the steps. Creation will be skipped if an experiment or a trial with
                the same name already exists. By default, pipeline name is used as
                experiment name and execution id is used as the trial name.
                If set to None, no experiment or trial will be created automatically.
            steps (Sequence[Union[Step, StepCollection, StepOutput]]): The list of the
                non-conditional steps associated with the pipeline. Any steps that are within the
                `if_steps` or `else_steps` of a `ConditionStep` cannot be listed in the steps of a
                pipeline. Of particular note, the workflow service rejects any pipeline definitions
                that specify a step in the list of steps of a pipeline and that step in the
                `if_steps` or `else_steps` of any `ConditionStep`.
            sagemaker_session (sagemaker.session.Session): Session object that manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                pipeline creates one using the default AWS configuration chain.
            pipeline_definition_config (Optional[PipelineDefinitionConfig]): If set,
                the workflow customizes the pipeline definition using the configurations
                specified. By default, custom job-prefixing is turned off.
        """
        self.name = name
        self.parameters = parameters if parameters else []
        self.pipeline_experiment_config = pipeline_experiment_config
        self.steps = steps if steps else []
        self.sagemaker_session = sagemaker_session if sagemaker_session else Session()
        self.pipeline_definition_config = pipeline_definition_config

        self._version = "2020-12-01"
        self._metadata = dict()

        # EventBridge helper for client.create() calls
        self._event_bridge_scheduler_helper = EventBridgeSchedulerHelper(
            self.sagemaker_session.boto_session.client("scheduler"),
        )

    def create(
        self,
        role_arn: str = None,
        description: str = None,
        tags: Optional[Tags] = None,
        parallelism_config: ParallelismConfiguration = None,
    ) -> Dict[str, Any]:
        """Creates a Pipeline in the Pipelines service.

        Args:
            role_arn (str): The role arn that is assumed by the pipeline to create step artifacts.
            description (str): A description of the pipeline.
            tags (Optional[Tags]): Tags to be passed to the pipeline.
            parallelism_config (Optional[ParallelismConfiguration]): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline.

        Returns:
            A response dict from the service.
        """
        role_arn = resolve_value_from_config(
            role_arn, PIPELINE_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        if not role_arn:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create a Pipeline.")
        if self.sagemaker_session.local_mode:
            if parallelism_config:
                logger.warning("Pipeline parallelism config is not supported in the local mode.")
            return self.sagemaker_session.sagemaker_client.create_pipeline(self, description)
        tags = format_tags(tags)
        tags = _append_project_tags(tags)
        tags = self.sagemaker_session._append_sagemaker_config_tags(tags, PIPELINE_TAGS_PATH)
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
            bucket, object_key = s3.determine_bucket_and_prefix(
                bucket=None, key_prefix=self.name, sagemaker_session=self.sagemaker_session
            )

            desired_s3_uri = s3.s3_path_join("s3://", bucket, object_key)
            s3.S3Uploader.upload_string_as_file_body(
                body=pipeline_definition,
                desired_s3_uri=desired_s3_uri,
                sagemaker_session=self.sagemaker_session,
            )
            kwargs["PipelineDefinitionS3Location"] = {
                "Bucket": bucket,
                "ObjectKey": object_key,
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
        role_arn: str = None,
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
        role_arn = resolve_value_from_config(
            role_arn, PIPELINE_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        if not role_arn:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to update a Pipeline.")
        if self.sagemaker_session.local_mode:
            if parallelism_config:
                logger.warning("Pipeline parallelism config is not supported in the local mode.")
            return self.sagemaker_session.sagemaker_client.update_pipeline(self, description)

        kwargs = self._create_args(role_arn, description, parallelism_config)
        return self.sagemaker_session.sagemaker_client.update_pipeline(**kwargs)

    def upsert(
        self,
        role_arn: str = None,
        description: str = None,
        tags: Optional[Tags] = None,
        parallelism_config: ParallelismConfiguration = None,
    ) -> Dict[str, Any]:
        """Creates a pipeline or updates it, if it already exists.

        Args:
            role_arn (str): The role arn that is assumed by workflow to create step artifacts.
            description (str): A description of the pipeline.
            tags (Optional[Tags]): Tags to be passed.
            parallelism_config (Optional[Config for parallel steps, Parallelism configuration that
                is applied to each of the executions

        Returns:
            response dict from service
        """
        role_arn = resolve_value_from_config(
            role_arn, PIPELINE_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        tags = format_tags(tags)
        if not role_arn:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create or update a Pipeline.")
        try:
            response = self.create(role_arn, description, tags, parallelism_config)
        except ClientError as ce:
            error_code = ce.response["Error"]["Code"]
            error_message = ce.response["Error"]["Message"]
            if not (error_code == "ValidationException" and "already exists" in error_message):
                raise ce
            # already exists
            response = self.update(role_arn, description, parallelism_config=parallelism_config)
            # add new tags to existing resource
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
        return response

    def delete(self) -> Dict[str, Any]:
        """Deletes a Pipeline in the Workflow service.

        Returns:
            A response dict from the service.
        """
        logger.info(
            "If triggers have been setup for this target, they will become orphaned."
            "You will need to clean them up manually via the CLI or EventBridge console."
        )
        return self.sagemaker_session.sagemaker_client.delete_pipeline(PipelineName=self.name)

    def start(
        self,
        parameters: Dict[str, Union[str, bool, int, float]] = None,
        execution_display_name: str = None,
        execution_description: str = None,
        parallelism_config: ParallelismConfiguration = None,
        selective_execution_config: SelectiveExecutionConfig = None,
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
            selective_execution_config (Optional[SelectiveExecutionConfig]): The configuration for
                selective step execution.

        Returns:
            A `_PipelineExecution` instance, if successful.
        """
        if selective_execution_config is not None:
            if (
                selective_execution_config.source_pipeline_execution_arn is None
                and selective_execution_config.reference_latest_execution
            ):
                selective_execution_config.source_pipeline_execution_arn = (
                    self._get_latest_execution_arn()
                )
            selective_execution_config = selective_execution_config.to_request()

        kwargs = dict(PipelineName=self.name)
        update_args(
            kwargs,
            PipelineExecutionDescription=execution_description,
            PipelineExecutionDisplayName=execution_display_name,
            ParallelismConfiguration=parallelism_config,
            SelectiveExecutionConfig=selective_execution_config,
        )
        if self.sagemaker_session.local_mode:
            update_args(kwargs, PipelineParameters=parameters)
            return self.sagemaker_session.sagemaker_client.start_pipeline_execution(**kwargs)
        update_args(kwargs, PipelineParameters=format_start_parameters(parameters))

        # retry on AccessDeniedException to cover case of tag propagation delay
        response = retry_with_backoff(
            lambda: self.sagemaker_session.sagemaker_client.start_pipeline_execution(**kwargs),
            botocore_client_error_code="AccessDeniedException",
        )
        return _PipelineExecution(
            arn=response["PipelineExecutionArn"],
            sagemaker_session=self.sagemaker_session,
        )

    def definition(self) -> str:
        """Converts a request structure to string representation for workflow service calls."""
        compiled_steps = StepsCompiler(
            pipeline_name=self.name,
            sagemaker_session=self.sagemaker_session,
            steps=self.steps,
            pipeline_definition_config=self.pipeline_definition_config,
        ).build()

        request_dict = {
            "Version": self._version,
            "Metadata": self._metadata,
            "Parameters": list_to_request(self.parameters),
            "PipelineExperimentConfig": (
                self.pipeline_experiment_config.to_request()
                if self.pipeline_experiment_config is not None
                else None
            ),
            "Steps": list_to_request(compiled_steps),
        }

        request_dict["PipelineExperimentConfig"] = interpolate(
            request_dict["PipelineExperimentConfig"], {}, {}, pipeline_name=self.name
        )
        callback_output_to_step_map = _map_callback_outputs(self.steps)
        lambda_output_to_step_name = _map_lambda_outputs(self.steps)
        request_dict["Steps"] = interpolate(
            request_dict["Steps"],
            callback_output_to_step_map=callback_output_to_step_map,
            lambda_output_to_step_map=lambda_output_to_step_name,
            pipeline_name=self.name,
        )

        return json.dumps(request_dict)

    def list_executions(
        self,
        sort_by: str = None,
        sort_order: str = None,
        max_results: int = None,
        next_token: str = None,
    ) -> Dict[str, Any]:
        """Lists a pipeline's executions.

        Args:
            sort_by (str): The field by which to sort results(CreationTime/PipelineExecutionArn).
            sort_order (str): The sort order for results (Ascending/Descending).
            max_results (int): The maximum number of pipeline executions to return in the response.
            next_token (str):  If the result of the previous `ListPipelineExecutions` request was
                truncated, the response includes a `NextToken`. To retrieve the next set of pipeline
                executions, use the token in the next request.

        Returns:
            List of Pipeline Execution Summaries. See
            boto3 client list_pipeline_executions
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.list_pipeline_executions
        """
        kwargs = dict(PipelineName=self.name)
        update_args(
            kwargs,
            SortBy=sort_by,
            SortOrder=sort_order,
            NextToken=next_token,
            MaxResults=max_results,
        )
        response = self.sagemaker_session.sagemaker_client.list_pipeline_executions(**kwargs)

        # Return only PipelineExecutionSummaries and NextToken from the list_pipeline_executions
        # response
        return {
            key: response[key]
            for key in ["PipelineExecutionSummaries", "NextToken"]
            if key in response
        }

    def _get_latest_execution_arn(self):
        """Retrieves the latest execution of this pipeline"""
        response = self.list_executions(
            sort_by="CreationTime",
            sort_order="Descending",
            max_results=1,
        )
        if response["PipelineExecutionSummaries"]:
            return response["PipelineExecutionSummaries"][0]["PipelineExecutionArn"]
        return None

    def build_parameters_from_execution(
        self,
        pipeline_execution_arn: str,
        parameter_value_overrides: Dict[str, Union[str, bool, int, float]] = None,
    ) -> Dict[str, Union[str, bool, int, float]]:
        """Gets the parameters from an execution, update with optional parameter value overrides.

        Args:
            pipeline_execution_arn (str): The arn of the reference pipeline execution.
            parameter_value_overrides (Dict[str, Union[str, bool, int, float]]): Parameter dict
                to be updated with the parameters from the referenced execution.

        Returns:
            A parameter dict built from an execution and provided parameter value overrides.
        """
        execution_parameters = self._get_parameters_for_execution(pipeline_execution_arn)
        if parameter_value_overrides is not None:
            self._validate_parameter_overrides(
                pipeline_execution_arn, execution_parameters, parameter_value_overrides
            )
            execution_parameters.update(parameter_value_overrides)
        return execution_parameters

    def _get_parameters_for_execution(self, pipeline_execution_arn: str) -> Dict[str, str]:
        """Gets all the parameters from an execution.

        Args:
            pipeline_execution_arn (str): The arn of the pipeline execution.

        Returns:
            A parameter dict from the execution.
        """
        pipeline_execution = _PipelineExecution(
            arn=pipeline_execution_arn,
            sagemaker_session=self.sagemaker_session,
        )

        response = pipeline_execution.list_parameters()
        parameter_list = response["PipelineParameters"]
        while response.get("NextToken") is not None:
            response = pipeline_execution.list_parameters(next_token=response["NextToken"])
            parameter_list.extend(response["PipelineParameters"])

        return {parameter["Name"]: parameter["Value"] for parameter in parameter_list}

    @staticmethod
    def _validate_parameter_overrides(
        pipeline_execution_arn: str,
        execution_parameters: Dict[str, str],
        parameter_overrides: Dict[str, Union[str, bool, int, float]],
    ):
        """Validates the parameter overrides are present in the execution parameters.

        Args:
            pipeline_execution_arn (str): The arn of the pipeline execution.
            execution_parameters (Dict[str, str]): A parameter dict from the execution.
            parameter_overrides (Dict[str, Union[str, bool, int, float]]): Parameter dict to be
                updated in the parameters from the referenced execution.

        Raises:
            ValueError: If any parameters in parameter overrides is not present in the
                execution parameters.
        """
        invalid_parameters = set(parameter_overrides) - set(execution_parameters)
        if invalid_parameters:
            raise ValueError(
                f"The following parameter overrides provided: {str(invalid_parameters)} "
                + f"are not present in the pipeline execution: {pipeline_execution_arn}"
            )

    def put_triggers(
        self,
        triggers: List[Trigger],
        role_arn: str = None,
    ) -> List[str]:
        """Attach triggers to a parent SageMaker Pipeline.

        Args:
            triggers (List[Trigger]): List of supported triggers. Currently, this can only be of
                type PipelineSchedule.
            role_arn (str): The role arn that is assumed by EventBridge service.

        Returns:
            List[str]: Successfully created trigger Arn(s). Currently, the pythonSDK only supports
                PipelineSchedule triggers, thus, this is a list of EventBridge Schedule Arn(s)
                that were created/upserted.
        """
        _role_arn = role_arn or resolve_value_from_config(
            role_arn, PIPELINE_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        if not _role_arn:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create triggers for a pipeline.")
        if not triggers:
            raise TypeError(
                "No Triggers provided. Please specify at least one to setup pipeline triggers."
            )

        # Ensure pipeline exists first
        try:
            describe_pipeline_response = self.describe()
        except ClientError as e:
            if RESOURCE_NOT_FOUND == e.response["Error"]["Code"]:
                raise RuntimeError(
                    f"Cannot create triggers for pipeline {self.name} that does not exist. "
                    f"Please create the pipeline before assigning triggers to it."
                )
            raise e

        validate_default_parameters_for_schedules(self.parameters)
        created_triggers = set()
        for trigger in triggers:
            if isinstance(trigger, PipelineSchedule):
                _start_date = trigger.start_date or datetime.now(tz=pytz.utc)
                _schedule_expression = trigger.resolve_schedule_expression()
                _state = trigger.resolve_trigger_state()
                _schedule_name = trigger.resolve_trigger_name(self.name)

                logger.info("Creating/Updating EventBridge Schedule for pipeline: %s.", self.name)
                event_bridge_schedule_arn = self._event_bridge_scheduler_helper.upsert_schedule(
                    schedule_name=_schedule_name,
                    pipeline_arn=describe_pipeline_response["PipelineArn"],
                    schedule_expression=_schedule_expression,
                    state=_state,
                    start_date=_start_date,
                    role=_role_arn,
                )
                logger.info(
                    "Created/Updated EventBridge Schedule for pipeline: %s with ScheduleName: %s",
                    self.name,
                    _schedule_name,
                )
                created_triggers.add(event_bridge_schedule_arn["ScheduleArn"])
            else:
                raise TypeError(f"Unsupported TriggerType: {trigger.__class__.__name__}")

        return list(created_triggers)

    def describe_trigger(self, trigger_name: str) -> Dict[str, Any]:
        """Describe Trigger for a parent SageMaker Pipeline.

        Args:
            trigger_name (str): Trigger name to be described. Currently, this can only
                be an EventBridge schedule name.

        Returns:
            Dict[str, str]: Trigger describe responses from EventBridge.
        """
        if not trigger_name:
            raise TypeError(
                "No trigger name provided. Please specify at least one to describe pipeline "
                "triggers."
            )

        event_bridge_schedule = self._event_bridge_scheduler_helper.describe_schedule(
            schedule_name=trigger_name
        )
        describe_response_dict = {}
        if event_bridge_schedule:
            describe_response_dict.update(
                dict(
                    Schedule_Arn=event_bridge_schedule["Arn"],
                    Schedule_Expression=event_bridge_schedule["ScheduleExpression"],
                    Schedule_State=event_bridge_schedule["State"],
                    Schedule_Start_Date=event_bridge_schedule["StartDate"].strftime(
                        EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT
                    ),
                    Schedule_Role=event_bridge_schedule["Target"]["RoleArn"],
                )
            )
        return describe_response_dict

    def delete_triggers(self, trigger_names: List[str]):
        """Delete Triggers for a parent SageMaker Pipeline if they exist.

        Args:
            trigger_names (List[str]): List of trigger names to be deleted. Currently, these can
                only be EventBridge schedule names.
        """
        for trigger_name in trigger_names:
            #  /default group is used
            logger.info("Deleting Pipeline Schedule: %s ...", trigger_name)
            try:
                self._event_bridge_scheduler_helper.delete_schedule(schedule_name=trigger_name)
            except ClientError as e:
                if RESOURCE_NOT_FOUND_EXCEPTION == e.response["Error"]["Code"]:
                    logger.warning(
                        "Pipeline Schedule %s does not exist. The schedule could have "
                        "been already deleted or never created in the first place.",
                        trigger_name,
                    )
                    continue
                raise e
            logger.info("Deleted Pipeline Schedule: %s ...", trigger_name)


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
    pipeline_name: str,  # TODO: remove it once its ExecutionVariable fixed in backend
) -> RequestType:
    """Replaces Parameter values in a list of nested Dict[str, Any] with their workflow expression.

    Args:
        request_obj (RequestType): The request dict.
        callback_output_to_step_map (Dict[str, str]): A dict of output name -> step name.
        lambda_output_to_step_map (Dict[str, str]): A dict of output name -> step name.
        pipeline_name (str): The name of the pipeline to be interpolated.

    Returns:
        RequestType: The request dict with Parameter values replaced by their expression.
    """
    try:
        return _interpolate(
            request_obj,
            callback_output_to_step_map=callback_output_to_step_map,
            lambda_output_to_step_map=lambda_output_to_step_map,
            pipeline_name=pipeline_name,
        )
    except TypeError as type_err:
        raise TypeError("Not able to interpolate Pipeline definition: %s" % type_err)


def _interpolate(
    obj: Union[RequestType, Any],
    callback_output_to_step_map: Dict[str, str],
    lambda_output_to_step_map: Dict[str, str],
    pipeline_name: str,
):
    """Walks the nested request dict, replacing Parameter type values with workflow expressions.

    Args:
        obj (Union[RequestType, Any]): The request dict.
        callback_output_to_step_map (Dict[str, str]): A dict of output name -> step name.
        lambda_output_to_step_map (Dict[str, str]): A dict of output name -> step name.
        pipeline_name (str): The name of the pipeline to be interpolated.
    """
    if isinstance(obj, (Expression, Parameter, Properties, StepOutput)):
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
            new[key] = interpolate(
                value,
                callback_output_to_step_map,
                lambda_output_to_step_map,
                pipeline_name=pipeline_name,
            )
    elif isinstance(obj, (list, set, tuple)):
        new = obj.__class__(
            interpolate(
                value,
                callback_output_to_step_map,
                lambda_output_to_step_map,
                pipeline_name=pipeline_name,
            )
            for value in obj
        )
    else:
        return obj
    return new


def _map_callback_outputs(steps: List[Step]):
    """Iterate over the provided steps, building a map of callback output parameters to step names.

    Args:
        steps (List[Step]): The steps list.
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
        steps (List[Step]): The steps list.
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


def _generate_step_map(steps: Sequence[Union[Step, StepCollection]], step_map: dict):
    """Helper method to create a mapping from Step/Step Collection name to itself."""
    for step in steps:
        if step.name in step_map:
            raise ValueError(
                "Pipeline steps cannot have duplicate names. In addition, steps added in "
                "the ConditionStep cannot be added in the Pipeline steps list."
            )
        step_map[step.name] = step
        if isinstance(step, ConditionStep):
            _generate_step_map(step.if_steps + step.else_steps, step_map)
        if isinstance(step, StepCollection):
            _generate_step_map(step.steps, step_map)


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

    def list_parameters(self, max_results: int = None, next_token: str = None):
        """Gets a list of parameters for a pipeline execution.

        Args:
            max_results (int): The maximum number of parameters to return in the response.
            next_token (str):  If the result of the previous `ListPipelineParametersForExecution`
                request was truncated, the response includes a `NextToken`. To retrieve the next
                set of parameters, use the token in the next request.

        Returns:
            Information about the parameters of the pipeline execution. This function is also
            a wrapper for `list_pipeline_parameters_for_execution
            <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.list_pipeline_parameters_for_execution>`_.
        """
        kwargs = dict(PipelineExecutionArn=self.arn)
        update_args(
            kwargs,
            MaxResults=max_results,
            NextToken=next_token,
        )
        return self.sagemaker_session.sagemaker_client.list_pipeline_parameters_for_execution(
            **kwargs
        )

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

    def result(self, step_name: str):
        """Retrieves the output of the provided step if it is a ``@step`` decorated function.

        Args:
            step_name (str): The name of the pipeline step.
        Returns:
            The step output.

        Raises:
              ValueError if the provided step is not a ``@step`` decorated function.
              RemoteFunctionError if the provided step is not in "Completed" status
        """
        try:
            self.wait()
        except WaiterError as e:
            if "Waiter encountered a terminal failure state" not in str(e):
                raise

        return get_function_step_result(
            step_name=step_name,
            step_list=self.list_steps(),
            execution_id=self.arn.split("/")[-1],
            sagemaker_session=self.sagemaker_session,
        )


def get_function_step_result(
    step_name: str,
    step_list: list,
    execution_id: str,
    sagemaker_session: Session,
):
    """Helper function to retrieve the output of a ``@step`` decorated function.

    Args:
        step_name (str): The name of the pipeline step.
        step_list (list): A list of executed pipeline steps of the specified execution.
        execution_id (str): The specified id of the pipeline execution.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed.
    Returns:
        The step output.

    Raises:
        ValueError if the provided step is not a ``@step`` decorated function.
        RemoteFunctionError if the provided step is not in "Completed" status
    """
    _ERROR_MSG_OF_WRONG_STEP_TYPE = (
        "This method can only be used on pipeline steps created using @step decorator."
    )
    _ERROR_MSG_OF_STEP_INCOMPLETE = (
        f"Unable to retrieve step output as the step {step_name} is not in Completed status."
    )

    step = next(filter(lambda x: x["StepName"] == step_name, step_list), None)
    if not step:
        raise ValueError(f"Invalid step name {step_name}")

    if isinstance(sagemaker_session, LocalSession) and not step.get("Metadata", None):
        # In local mode, if the training job failed,
        # it's not tracked in LocalSagemakerClient and it's not describable.
        # Thus, the step Metadata is not set.
        raise RuntimeError(_ERROR_MSG_OF_STEP_INCOMPLETE)

    step_type = next(iter(step["Metadata"]))
    step_metadata = next(iter(step["Metadata"].values()))
    if step_type != "TrainingJob":
        raise ValueError(_ERROR_MSG_OF_WRONG_STEP_TYPE)

    job_arn = step_metadata["Arn"]
    job_name = job_arn.split("/")[-1]

    if isinstance(sagemaker_session, LocalSession):
        describe_training_job_response = sagemaker_session.sagemaker_client.describe_training_job(
            job_name
        )
    else:
        describe_training_job_response = sagemaker_session.describe_training_job(job_name)
    container_args = describe_training_job_response["AlgorithmSpecification"]["ContainerEntrypoint"]
    if container_args != JOBS_CONTAINER_ENTRYPOINT:
        raise ValueError(_ERROR_MSG_OF_WRONG_STEP_TYPE)
    s3_output_path = describe_training_job_response["OutputDataConfig"]["S3OutputPath"]

    s3_uri_suffix = s3_path_join(execution_id, step_name, RESULTS_FOLDER)
    if s3_output_path.endswith(s3_uri_suffix) or s3_output_path[0:-1].endswith(s3_uri_suffix):
        s3_uri = s3_output_path
    else:
        # This is the obsoleted version of s3_output_path
        # Keeping it for backward compatible
        s3_uri = s3_path_join(s3_output_path, s3_uri_suffix)

    job_status = describe_training_job_response["TrainingJobStatus"]
    if job_status == "Completed":
        return deserialize_obj_from_s3(
            sagemaker_session=sagemaker_session,
            s3_uri=s3_uri,
            hmac_key=describe_training_job_response["Environment"]["REMOTE_FUNCTION_SECRET_KEY"],
        )

    raise RemoteFunctionError(_ERROR_MSG_OF_STEP_INCOMPLETE)


class PipelineGraph:
    """Helper class representing the Pipeline Directed Acyclic Graph (DAG)

    Attributes:
        steps (Sequence[Union[Step, StepCollection]]): Sequence of `Step`s and/or `StepCollection`s
            that represent each node in the pipeline DAG
    """

    def __init__(self, steps: Sequence[Union[Step, StepCollection]]):
        self.step_map = {}
        _generate_step_map(steps, self.step_map)
        self.adjacency_list = self._initialize_adjacency_list()
        if self.is_cyclic():
            raise ValueError("Cycle detected in pipeline step graph.")

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline):
        """Create a PipelineGraph object from the Pipeline object."""
        compiled_steps = StepsCompiler(
            pipeline_name=pipeline.name,
            sagemaker_session=pipeline.sagemaker_session,
            pipeline_definition_config=pipeline.pipeline_definition_config,
            steps=pipeline.steps,
        ).build()
        return cls(compiled_steps)

    def _initialize_adjacency_list(self) -> Dict[str, List[str]]:
        """Generate an adjacency list representing the step dependency DAG in this pipeline."""
        from collections import defaultdict

        dependency_list = defaultdict(set)
        for step in self.step_map.values():
            if isinstance(step, Step):
                dependency_list[step.name].update(step._find_step_dependencies(self.step_map))

            if isinstance(step, ConditionStep):
                for child_step in step.if_steps + step.else_steps:
                    if isinstance(child_step, Step):
                        dependency_list[child_step.name].add(step.name)
                    elif isinstance(child_step, StepCollection):
                        child_first_step = self.step_map[child_step.name].steps[0].name
                        dependency_list[child_first_step].add(step.name)

        adjacency_list = {}
        for step in dependency_list:
            for step_dependency in dependency_list[step]:
                adjacency_list[step_dependency] = list(
                    set(adjacency_list.get(step_dependency, []) + [step])
                )
        for step in dependency_list:
            if step not in adjacency_list:
                adjacency_list[step] = []
        return adjacency_list

    def is_cyclic(self) -> bool:
        """Check if this pipeline graph is cyclic.

        Returns true if it is cyclic, false otherwise.
        """

        def is_cyclic_helper(current_step):
            visited_steps.add(current_step)
            recurse_steps.add(current_step)
            for child_step in self.adjacency_list[current_step]:
                if child_step in recurse_steps:
                    return True
                if child_step not in visited_steps:
                    if is_cyclic_helper(child_step):
                        return True
            recurse_steps.remove(current_step)
            return False

        visited_steps = set()
        recurse_steps = set()
        for step in self.adjacency_list:
            if step not in visited_steps:
                if is_cyclic_helper(step):
                    return True
        return False

    def get_steps_in_sub_dag(
        self, current_step: Union[Step, StepCollection], sub_dag_steps: Set[str] = None
    ) -> Set[str]:
        """Get names of all steps (including current step) in the sub dag of current step.

        Returns a set of step names in the sub dag.
        """
        if sub_dag_steps is None:
            sub_dag_steps = set()

        if isinstance(current_step, StepCollection):
            current_steps = current_step.steps
        else:
            current_steps = [current_step]

        for step in current_steps:
            if step.name not in self.adjacency_list:
                raise ValueError("Step: %s does not exist in the pipeline." % step.name)
            sub_dag_steps.add(step.name)
            for sub_step in self.adjacency_list[step.name]:
                self.get_steps_in_sub_dag(self.step_map.get(sub_step), sub_dag_steps)
        return sub_dag_steps

    def __iter__(self):
        """Perform topological sort traversal of the Pipeline Graph."""

        def topological_sort(current_step):
            visited_steps.add(current_step)
            for child_step in self.adjacency_list[current_step]:
                if child_step not in visited_steps:
                    topological_sort(child_step)
            self.stack.append(current_step)

        visited_steps = set()
        self.stack = []  # pylint: disable=W0201
        for step in self.adjacency_list:
            if step not in visited_steps:
                topological_sort(step)
        return self

    def __next__(self) -> Step:
        """Return the next Step node from the Topological sort order."""

        while self.stack:
            return self.step_map.get(self.stack.pop())
        raise StopIteration
