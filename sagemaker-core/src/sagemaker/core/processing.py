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
"""This module contains code related to the ``Processor`` class.

which is used for Amazon SageMaker Processing Jobs. These jobs let users perform
data pre-processing, post-processing, feature engineering, data validation, and model evaluation,
and interpretation on Amazon SageMaker.
"""
from __future__ import absolute_import

import json
import logging
import os
import pathlib
import re
from typing import Dict, List, Optional, Union
import time
from copy import copy
from textwrap import dedent
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import url2pathname
from sagemaker.core.network import NetworkConfig
from sagemaker.core import s3
from sagemaker.core.apiutils._base_types import ApiObject
from sagemaker.core.config.config_schema import (
    PROCESSING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
    PROCESSING_JOB_ENVIRONMENT_PATH,
    PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
    PROCESSING_JOB_KMS_KEY_ID_PATH,
    PROCESSING_JOB_ROLE_ARN_PATH,
    PROCESSING_JOB_SECURITY_GROUP_IDS_PATH,
    PROCESSING_JOB_SUBNETS_PATH,
    PROCESSING_JOB_VOLUME_KMS_KEY_ID_PATH,
    PROCESSING_JOB_INPUTS_PATH,
    PROCESSING_JOB_NETWORK_CONFIG_PATH,
    PROCESSING_OUTPUT_CONFIG_PATH,
    PROCESSING_JOB_PROCESSING_RESOURCES_PATH,
    SAGEMAKER,
    PROCESSING_JOB,
    TAGS,
)
from sagemaker.core.local.local_session import LocalSession
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.shapes import ProcessingInput, ProcessingOutput, ProcessingS3Input
from sagemaker.core.resources import ProcessingJob
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.common_utils import (
    Tags,
    base_name_from_image,
    check_and_get_run_experiment_config,
    format_tags,
    name_from_base,
    resolve_class_attribute_from_config,
    resolve_value_from_config,
    resolve_nested_dict_value_from_config,
    update_list_of_dicts_with_values_from_config,
    update_nested_dictionary_with_values_from_config,
    _get_initial_job_state,
    _wait_until,
    _flush_log_streams,
    _logs_init,
    LogState,
    _check_job_status,
)
from sagemaker.core.workflow import is_pipeline_variable
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.pipeline_context import runnable_by_pipeline

from sagemaker.core._studio import _append_project_tags
from sagemaker.core.config.config_utils import _append_sagemaker_config_tags
from sagemaker.core.utils.utils import serialize

logger = logging.getLogger(__name__)


class Processor(object):
    """Handles Amazon SageMaker Processing tasks."""

    JOB_CLASS_NAME = "processing-job"

    def __init__(
        self,
        role: str = None,
        image_uri: Union[str, PipelineVariable] = None,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        entrypoint: Optional[List[Union[str, PipelineVariable]]] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[Tags] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """Initializes a ``Processor`` instance.

        The ``Processor`` handles Amazon SageMaker Processing tasks.

        Args:
            role (str or PipelineVariable): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            image_uri (str or PipelineVariable): The URI of the Docker image to use for the
                processing jobs.
            instance_count (int or PipelineVariable): The number of instances to run
                a processing job with.
            instance_type (str or PipelineVariable): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            entrypoint (list[str] or list[PipelineVariable]): The entrypoint for the
                processing job (default: None). This is in the form of a list of strings
                that make a command.
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing
                volume (default: None).
            output_kms_key (str or PipelineVariable): The KMS key ID for processing job
                outputs (default: None).
            max_runtime_in_seconds (int or PipelineVariable): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            base_job_name (str): Prefix for processing job name. If not specified,
                the processor generates a default job name, based on the
                processing image name and current timestamp.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables
                to be passed to the processing jobs (default: None).
            tags (Optional[Tags]): Tags to be passed to the processing job (default: None).
                For more, see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        self.image_uri = image_uri
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.entrypoint = entrypoint
        self.volume_size_in_gb = volume_size_in_gb
        self.max_runtime_in_seconds = max_runtime_in_seconds
        self.base_job_name = base_job_name
        self.tags = format_tags(tags)

        self.jobs = []
        self.latest_job = None
        self._current_job_name = None
        self.arguments = None

        if self.instance_type in ("local", "local_gpu"):
            if not isinstance(sagemaker_session, LocalSession):
                # Until Local Mode Processing supports local code, we need to disable it:
                sagemaker_session = LocalSession(disable_local_code=True)

        self.sagemaker_session = sagemaker_session or Session()
        self.output_kms_key = resolve_value_from_config(
            output_kms_key, PROCESSING_JOB_KMS_KEY_ID_PATH, sagemaker_session=self.sagemaker_session
        )
        self.volume_kms_key = resolve_value_from_config(
            volume_kms_key,
            PROCESSING_JOB_VOLUME_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            network_config,
            "subnets",
            PROCESSING_JOB_SUBNETS_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            self.network_config,
            "security_group_ids",
            PROCESSING_JOB_SECURITY_GROUP_IDS_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            self.network_config,
            "enable_network_isolation",
            PROCESSING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            self.network_config,
            "encrypt_inter_container_traffic",
            PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.role = resolve_value_from_config(
            role, PROCESSING_JOB_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        if not self.role:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create a Processing job.")

        self.env = resolve_value_from_config(
            env, PROCESSING_JOB_ENVIRONMENT_PATH, sagemaker_session=self.sagemaker_session
        )

    @runnable_by_pipeline
    def run(
        self,
        inputs: Optional[List[ProcessingInput]] = None,
        outputs: Optional[List[ProcessingOutput]] = None,
        arguments: Optional[List[Union[str, PipelineVariable]]] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
    ):
        """Runs a processing job.

        Args:
            inputs (list[:class:`~sagemaker.core.shapes.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.core.shapes.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.core.shapes.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.core.shapes.ProcessingOutput` objects (default: None).
            arguments (list[str] or list[PipelineVariable]): A list of string arguments
                to be passed to a processing job (default: None).
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the base job name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
                * Both `ExperimentName` and `TrialName` will be ignored if the Processor instance
                is built with :class:`~sagemaker.workflow.pipeline_context.PipelineSession`.
                However, the value of `TrialComponentDisplayName` is honored for display in Studio.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        Returns:
            None or pipeline step arguments in case the Processor instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        Raises:
            ValueError: if ``logs`` is True but ``wait`` is False.
        """
        if logs and not wait:
            raise ValueError(
                """Logs can only be shown if wait is set to True.
                Please either set wait to True or set logs to False."""
            )

        normalized_inputs, normalized_outputs = self._normalize_args(
            job_name=job_name,
            arguments=arguments,
            inputs=inputs,
            kms_key=kms_key,
            outputs=outputs,
        )

        experiment_config = check_and_get_run_experiment_config(experiment_config)
        self.latest_job = self._start_new(
            inputs=normalized_inputs,
            outputs=normalized_outputs,
            experiment_config=experiment_config,
        )

        if not isinstance(self.sagemaker_session, PipelineSession):
            self.jobs.append(self.latest_job)
            if wait:
                self.latest_job.wait(logs=logs)

    def _extend_processing_args(self, inputs, outputs, **kwargs):  # pylint: disable=W0613
        """Extend inputs and outputs based on extra parameters"""
        return inputs, outputs

    def _normalize_args(
        self,
        job_name=None,
        arguments=None,
        inputs=None,
        outputs=None,
        code=None,
        kms_key=None,
    ):
        """Normalizes the arguments so that they can be passed to the job run

        Args:
            job_name (str): Name of the processing job to be created. If not specified, one
                is generated, using the base name given to the constructor, if applicable
                (default: None).
            arguments (list[str]): A list of string arguments to be passed to a
                processing job (default: None).
            inputs (list[:class:`~sagemaker.core.shapes.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.core.shapes.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.core.shapes.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.core.shapes.ProcessingOutput` objects (default: None).
            code (str): This can be an S3 URI or a local path to a file with the framework
                script to run (default: None). A no op in the base class.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        """
        if code and is_pipeline_variable(code):
            raise ValueError(
                "code argument has to be a valid S3 URI or local file path "
                + "rather than a pipeline variable"
            )

        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        inputs_with_code = self._include_code_in_inputs(inputs, code, kms_key)
        normalized_inputs = self._normalize_inputs(inputs_with_code, kms_key)
        normalized_outputs = self._normalize_outputs(outputs)
        self.arguments = arguments

        return normalized_inputs, normalized_outputs

    def _include_code_in_inputs(self, inputs, _code, _kms_key):
        """A no op in the base class to include code in the processing job inputs.

        Args:
            inputs (list[:class:`~sagemaker.core.shapes.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.core.shapes.ProcessingInput` objects.
            _code (str): This can be an S3 URI or a local path to a file with the framework
                script to run (default: None). A no op in the base class.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            list[:class:`~sagemaker.core.shapes.ProcessingInput`]: inputs
        """
        return inputs

    def _generate_current_job_name(self, job_name=None):
        """Generates the job name before running a processing job.

        Args:
            job_name (str): Name of the processing job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.

        Returns:
            str: The supplied or generated job name.
        """
        if job_name is not None:
            return job_name
        # Honor supplied base_job_name or generate it.
        if self.base_job_name:
            base_name = self.base_job_name
        else:
            base_name = base_name_from_image(
                self.image_uri, default_base_name=Processor.JOB_CLASS_NAME
            )

        # Replace invalid characters with hyphens to comply with AWS naming constraints
        base_name = re.sub(r"[^a-zA-Z0-9-]", "-", base_name)
        return name_from_base(base_name)

    def _normalize_inputs(self, inputs=None, kms_key=None):
        """Ensures that all the ``ProcessingInput`` objects have names and S3 URIs.

        Args:
            inputs (list[sagemaker.core.shapes.ProcessingInput]): A list of ``ProcessingInput``
                objects to be normalized (default: None). If not specified,
                an empty list is returned.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            list[sagemaker.core.shapes.ProcessingInput]: The list of normalized
                ``ProcessingInput`` objects.

        Raises:
            TypeError: if the inputs are not ``ProcessingInput`` objects.
        """
        from sagemaker.core.workflow.utilities import _pipeline_config

        # Initialize a list of normalized ProcessingInput objects.
        normalized_inputs = []
        if inputs is not None:
            # Iterate through the provided list of inputs.
            for count, file_input in enumerate(inputs, 1):
                if not isinstance(file_input, ProcessingInput):
                    raise TypeError("Your inputs must be provided as ProcessingInput objects.")
                # Generate a name for the ProcessingInput if it doesn't have one.
                if file_input.input_name is None:
                    file_input.input_name = "input-{}".format(count)

                if file_input.dataset_definition:
                    normalized_inputs.append(file_input)
                    continue
                if file_input.s3_input and is_pipeline_variable(file_input.s3_input.s3_uri):
                    normalized_inputs.append(file_input)
                    continue
                # If the s3_uri is not an s3_uri, create one.
                parse_result = urlparse(file_input.s3_input.s3_uri)
                if parse_result.scheme != "s3":
                    if _pipeline_config:
                        desired_s3_uri = s3.s3_path_join(
                            "s3://",
                            self.sagemaker_session.default_bucket(),
                            self.sagemaker_session.default_bucket_prefix,
                            _pipeline_config.pipeline_name,
                            _pipeline_config.step_name,
                            "input",
                            file_input.input_name,
                        )
                    else:
                        desired_s3_uri = s3.s3_path_join(
                            "s3://",
                            self.sagemaker_session.default_bucket(),
                            self.sagemaker_session.default_bucket_prefix,
                            self._current_job_name,
                            "input",
                            file_input.input_name,
                        )
                    s3_uri = s3.S3Uploader.upload(
                        local_path=file_input.s3_input.s3_uri,
                        desired_s3_uri=desired_s3_uri,
                        sagemaker_session=self.sagemaker_session,
                        kms_key=kms_key,
                    )
                    file_input.s3_input.s3_uri = s3_uri
                normalized_inputs.append(file_input)
        return normalized_inputs

    def _normalize_outputs(self, outputs=None):
        """Ensures that all the outputs are ``ProcessingOutput`` objects with names and S3 URIs.

        Args:
            outputs (list[sagemaker.core.shapes.ProcessingOutput]): A list
                of outputs to be normalized (default: None). Can be either strings or
                ``ProcessingOutput`` objects. If not specified,
                an empty list is returned.

        Returns:
            list[sagemaker.core.shapes.ProcessingOutput]: The list of normalized
                ``ProcessingOutput`` objects.

        Raises:
            TypeError: if the outputs are not ``ProcessingOutput`` objects.
        """
        # Initialize a list of normalized ProcessingOutput objects.
        from sagemaker.core.workflow.utilities import _pipeline_config

        normalized_outputs = []
        if outputs is not None:
            # Iterate through the provided list of outputs.
            for count, output in enumerate(outputs, 1):
                if not isinstance(output, ProcessingOutput):
                    raise TypeError("Your outputs must be provided as ProcessingOutput objects.")
                # Generate a name for the ProcessingOutput if it doesn't have one.
                if output.output_name is None:
                    output.output_name = "output-{}".format(count)
                if output.s3_output and is_pipeline_variable(output.s3_output.s3_uri):
                    normalized_outputs.append(output)
                    continue
                # If the output's s3_uri is not an s3_uri, create one.
                parse_result = urlparse(output.s3_output.s3_uri)
                if parse_result.scheme != "s3":
                    if _pipeline_config:
                        s3_uri = Join(
                            on="/",
                            values=[
                                "s3:/",
                                self.sagemaker_session.default_bucket(),
                                *(
                                    # don't include default_bucket_prefix if it is None or ""
                                    [self.sagemaker_session.default_bucket_prefix]
                                    if self.sagemaker_session.default_bucket_prefix
                                    else []
                                ),
                                _pipeline_config.pipeline_name,
                                ExecutionVariables.PIPELINE_EXECUTION_ID,
                                _pipeline_config.step_name,
                                "output",
                                output.output_name,
                            ],
                        )
                    else:
                        s3_uri = s3.s3_path_join(
                            "s3://",
                            self.sagemaker_session.default_bucket(),
                            self.sagemaker_session.default_bucket_prefix,
                            self._current_job_name,
                            "output",
                            output.output_name,
                        )
                    output.s3_output.s3_uri = s3_uri
                normalized_outputs.append(output)
        return normalized_outputs

    def _start_new(self, inputs, outputs, experiment_config):
        """Starts a new processing job and returns ProcessingJob instance."""
        from sagemaker.core.workflow.pipeline_context import PipelineSession

        process_args = self._get_process_args(inputs, outputs, experiment_config)

        logger.debug("Job Name: %s", process_args["job_name"])
        logger.debug("Inputs: %s", process_args["inputs"])
        logger.debug("Outputs: %s", process_args["output_config"]["Outputs"])

        tags = _append_project_tags(format_tags(process_args["tags"]))
        tags = _append_sagemaker_config_tags(
            self.sagemaker_session, tags, "{}.{}.{}".format(SAGEMAKER, PROCESSING_JOB, TAGS)
        )

        network_config = resolve_nested_dict_value_from_config(
            process_args["network_config"],
            ["EnableInterContainerTrafficEncryption"],
            PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
            sagemaker_session=self.sagemaker_session,
        )

        union_key_paths_for_dataset_definition = [
            ["DatasetDefinition", "S3Input"],
            [
                "DatasetDefinition.AthenaDatasetDefinition",
                "DatasetDefinition.RedshiftDatasetDefinition",
            ],
        ]
        update_list_of_dicts_with_values_from_config(
            process_args["inputs"],
            PROCESSING_JOB_INPUTS_PATH,
            union_key_paths=union_key_paths_for_dataset_definition,
            sagemaker_session=self.sagemaker_session,
        )

        role_arn = resolve_value_from_config(
            process_args["role_arn"],
            PROCESSING_JOB_ROLE_ARN_PATH,
            sagemaker_session=self.sagemaker_session,
        )

        inferred_network_config = update_nested_dictionary_with_values_from_config(
            network_config,
            PROCESSING_JOB_NETWORK_CONFIG_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        inferred_output_config = update_nested_dictionary_with_values_from_config(
            process_args["output_config"],
            PROCESSING_OUTPUT_CONFIG_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        inferred_resources_config = update_nested_dictionary_with_values_from_config(
            process_args["resources"],
            PROCESSING_JOB_PROCESSING_RESOURCES_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        environment = resolve_value_from_config(
            direct_input=process_args["environment"],
            config_path=PROCESSING_JOB_ENVIRONMENT_PATH,
            default_value=None,
            sagemaker_session=self.sagemaker_session,
        )

        process_request = _get_process_request(
            inputs=process_args["inputs"],
            output_config=inferred_output_config,
            job_name=process_args["job_name"],
            resources=inferred_resources_config,
            stopping_condition=process_args["stopping_condition"],
            app_specification=process_args["app_specification"],
            environment=environment,
            network_config=inferred_network_config,
            role_arn=role_arn,
            tags=tags,
            experiment_config=experiment_config,
        )

        # convert Unassigned() type in sagemaker-core to None
        serialized_request = serialize(process_request)

        if isinstance(self.sagemaker_session, PipelineSession):
            self.sagemaker_session._intercept_create_request(serialized_request, None, "process")
            return

        def submit(request):
            try:
                logger.info("Creating processing-job with name %s", process_args["job_name"])
                logger.debug("process request: %s", json.dumps(request, indent=4))
                self.sagemaker_session.sagemaker_client.create_processing_job(**request)
            except Exception as e:
                troubleshooting = (
                    "https://docs.aws.amazon.com/sagemaker/latest/dg/"
                    "sagemaker-python-sdk-troubleshooting.html"
                    "#sagemaker-python-sdk-troubleshooting-create-processing-job"
                )
                logger.error(
                    "Please check the troubleshooting guide for common errors: %s", troubleshooting
                )
                raise e

        self.sagemaker_session._intercept_create_request(serialized_request, submit, "process")

        from sagemaker.core.utils.code_injection.codec import transform

        transformed = transform(serialized_request, "CreateProcessingJobRequest")
        return ProcessingJob(**transformed)

    def _get_process_args(self, inputs, outputs, experiment_config):
        """Gets a dict of arguments for a new Amazon SageMaker processing job."""
        process_request_args = {}
        process_request_args["inputs"] = [_processing_input_to_request_dict(inp) for inp in inputs]
        process_request_args["output_config"] = {
            "Outputs": [_processing_output_to_request_dict(output) for output in outputs]
        }
        if self.output_kms_key is not None:
            process_request_args["output_config"]["KmsKeyId"] = self.output_kms_key
        process_request_args["experiment_config"] = experiment_config
        process_request_args["job_name"] = self._current_job_name
        process_request_args["resources"] = {
            "ClusterConfig": {
                "InstanceType": self.instance_type,
                "InstanceCount": self.instance_count,
                "VolumeSizeInGB": self.volume_size_in_gb,
            }
        }
        if self.volume_kms_key is not None:
            process_request_args["resources"]["ClusterConfig"][
                "VolumeKmsKeyId"
            ] = self.volume_kms_key
        if self.max_runtime_in_seconds is not None:
            process_request_args["stopping_condition"] = {
                "MaxRuntimeInSeconds": self.max_runtime_in_seconds
            }
        else:
            process_request_args["stopping_condition"] = None
        process_request_args["app_specification"] = {"ImageUri": self.image_uri}
        if self.arguments is not None:
            process_request_args["app_specification"]["ContainerArguments"] = self.arguments
        if self.entrypoint is not None:
            process_request_args["app_specification"]["ContainerEntrypoint"] = self.entrypoint
        process_request_args["environment"] = self.env
        if self.network_config is not None:
            process_request_args["network_config"] = self.network_config._to_request_dict()
        else:
            process_request_args["network_config"] = None
        process_request_args["role_arn"] = (
            self.role
            if is_pipeline_variable(self.role)
            else self.sagemaker_session.expand_role(self.role)
        )
        process_request_args["tags"] = self.tags
        return process_request_args


class ScriptProcessor(Processor):
    """Handles Amazon SageMaker processing tasks for jobs using a machine learning framework."""

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        image_uri: Union[str, PipelineVariable] = None,
        command: List[str] = None,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[Tags] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """Initializes a ``ScriptProcessor`` instance.

        The ``ScriptProcessor`` handles Amazon SageMaker Processing tasks for jobs
        using a machine learning framework, which allows for providing a script to be
        run as part of the Processing Job.

        Args:
            role (str or PipelineVariable): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            image_uri (str or PipelineVariable): The URI of the Docker image to use for the
                processing jobs.
            command ([str]): The command to run, along with any command-line flags.
                Example: ["python3", "-v"].
            instance_count (int or PipelineVariable): The number of instances to run
                a processing job with.
            instance_type (str or PipelineVariable): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing
                volume (default: None).
            output_kms_key (str or PipelineVariable): The KMS key ID for processing
                job outputs (default: None).
            max_runtime_in_seconds (int or PipelineVariable): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                processing image name and current timestamp.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
            env (dict[str, str] or dict[str, PipelineVariable])): Environment variables to
                be passed to the processing jobs (default: None).
            tags (Optional[Tags]): Tags to be passed to the processing job (default: None).
                For more, see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        self._CODE_CONTAINER_BASE_PATH = "/opt/ml/processing/input/"
        self._CODE_CONTAINER_INPUT_NAME = "code"

        if (
            not command
            and image_uri
            and ("sklearn" in str(image_uri) or "scikit-learn" in str(image_uri))
        ):
            command = ["python3"]

        self.command = command

        super(ScriptProcessor, self).__init__(
            role=role,
            image_uri=image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=format_tags(tags),
            network_config=network_config,
        )

    @runnable_by_pipeline
    def run(
        self,
        code: str,
        inputs: Optional[List[ProcessingInput]] = None,
        outputs: Optional[List[ProcessingOutput]] = None,
        arguments: Optional[List[Union[str, PipelineVariable]]] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
    ):
        """Runs a processing job.

        Args:
            code (str): This can be an S3 URI or a local path to
                a file with the framework script to run.
            inputs (list[:class:`~sagemaker.core.shapes.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.core.shapes.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.core.shapes.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.core.shapes.ProcessingOutput` objects (default: None).
            arguments (list[str]): A list of string arguments to be passed to a
                processing job (default: None).
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the base job name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
                * Both `ExperimentName` and `TrialName` will be ignored if the Processor instance
                is built with :class:`~sagemaker.workflow.pipeline_context.PipelineSession`.
                However, the value of `TrialComponentDisplayName` is honored for display in Studio.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        Returns:
            None or pipeline step arguments in case the Processor instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        """
        normalized_inputs, normalized_outputs = self._normalize_args(
            job_name=job_name,
            arguments=arguments,
            inputs=inputs,
            outputs=outputs,
            code=code,
            kms_key=kms_key,
        )

        experiment_config = check_and_get_run_experiment_config(experiment_config)
        self.latest_job = self._start_new(
            inputs=normalized_inputs,
            outputs=normalized_outputs,
            experiment_config=experiment_config,
        )

        from sagemaker.core.workflow.pipeline_context import PipelineSession

        if not isinstance(self.sagemaker_session, PipelineSession):
            self.jobs.append(self.latest_job)
            if wait:
                self.latest_job.wait(logs=logs)

    def _include_code_in_inputs(self, inputs, code, kms_key=None):
        """Converts code to appropriate input and includes in input list.

        Side effects include:
            * uploads code to S3 if the code is a local file.
            * sets the entrypoint attribute based on the command and user script name from code.

        Args:
            inputs (list[:class:`~sagemaker.core.shapes.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.core.shapes.ProcessingInput` objects.
            code (str): This can be an S3 URI or a local path to a file with the framework
                script to run (default: None).
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            list[:class:`~sagemaker.core.shapes.ProcessingInput`]: inputs together with the
                code as `ProcessingInput`.
        """
        user_code_s3_uri = self._handle_user_code_url(code, kms_key)
        user_script_name = self._get_user_code_name(code)

        inputs_with_code = self._convert_code_and_add_to_inputs(inputs, user_code_s3_uri)

        self._set_entrypoint(self.command, user_script_name)
        return inputs_with_code

    def _get_user_code_name(self, code):
        """Gets the basename of the user's code from the URL the customer provided.

        Args:
            code (str): A URL to the user's code.

        Returns:
            str: The basename of the user's code.

        """
        code_url = urlparse(code)
        return os.path.basename(code_url.path)

    def _handle_user_code_url(self, code, kms_key=None):
        """Gets the S3 URL containing the user's code.

           Inspects the scheme the customer passed in ("s3://" for code in S3, "file://" or nothing
           for absolute or local file paths. Uploads the code to S3 if the code is a local file.

        Args:
            code (str): A URL to the customer's code.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            str: The S3 URL to the customer's code.

        Raises:
            ValueError: if the code isn't found, is a directory, or
                does not have a valid URL scheme.
        """
        code_url = urlparse(code)
        if code_url.scheme == "s3":
            user_code_s3_uri = code
        elif code_url.scheme == "" or code_url.scheme == "file":
            # Validate that the file exists locally and is not a directory.
            code_path = url2pathname(code_url.path)
            if not os.path.exists(code_path):
                raise ValueError(
                    """code {} wasn't found. Please make sure that the file exists.
                    """.format(
                        code
                    )
                )
            if not os.path.isfile(code_path):
                raise ValueError(
                    """code {} must be a file, not a directory. Please pass a path to a file.
                    """.format(
                        code
                    )
                )
            user_code_s3_uri = self._upload_code(code_path, kms_key)
        else:
            raise ValueError(
                "code {} url scheme {} is not recognized. Please pass a file path or S3 url".format(
                    code, code_url.scheme
                )
            )
        return user_code_s3_uri

    def _upload_code(self, code, kms_key=None):
        """Uploads a code file or directory specified as a string and returns the S3 URI.

        Args:
            code (str): A file or directory to be uploaded to S3.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            str: The S3 URI of the uploaded file or directory.

        """
        from sagemaker.core.workflow.utilities import _pipeline_config

        if _pipeline_config and _pipeline_config.code_hash:
            desired_s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                _pipeline_config.pipeline_name,
                self._CODE_CONTAINER_INPUT_NAME,
                _pipeline_config.code_hash,
            )
        else:
            desired_s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                self._current_job_name,
                "input",
                self._CODE_CONTAINER_INPUT_NAME,
            )
        return s3.S3Uploader.upload(
            local_path=code,
            desired_s3_uri=desired_s3_uri,
            kms_key=kms_key,
            sagemaker_session=self.sagemaker_session,
        )

    def _convert_code_and_add_to_inputs(self, inputs, s3_uri):
        """Creates a ``ProcessingInput`` object from an S3 URI and adds it to the list of inputs.

        Args:
            inputs (list[sagemaker.core.shapes.ProcessingInput]):
                List of ``ProcessingInput`` objects.
            s3_uri (str): S3 URI of the input to be added to inputs.

        Returns:
            list[sagemaker.core.shapes.ProcessingInput]: A new list of ``ProcessingInput`` objects,
                with the ``ProcessingInput`` object created from ``s3_uri`` appended to the list.

        """

        code_file_input = ProcessingInput(
            input_name=self._CODE_CONTAINER_INPUT_NAME,
            s3_input=ProcessingS3Input(
                s3_uri=s3_uri,
                local_path=str(
                    pathlib.PurePosixPath(
                        self._CODE_CONTAINER_BASE_PATH, self._CODE_CONTAINER_INPUT_NAME
                    )
                ),
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
        )
        return (inputs or []) + [code_file_input]

    def _set_entrypoint(self, command, user_script_name):
        """Sets the entrypoint based on the user's script and corresponding executable.

        Args:
            user_script_name (str): A filename with an extension.
        """
        user_script_location = str(
            pathlib.PurePosixPath(
                self._CODE_CONTAINER_BASE_PATH,
                self._CODE_CONTAINER_INPUT_NAME,
                user_script_name,
            )
        )
        self.entrypoint = command + [user_script_location]


class FrameworkProcessor(ScriptProcessor):
    """Handles Amazon SageMaker processing tasks using ModelTrainer for code packaging."""

    framework_entrypoint_command = ["/bin/bash"]

    def __init__(
        self,
        image_uri: Union[str, PipelineVariable],
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        command: Optional[List[str]] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        code_location: Optional[str] = None,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[Tags] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """Initializes a ``FrameworkProcessor`` instance.

        The ``FrameworkProcessor`` handles Amazon SageMaker Processing tasks using
        ModelTrainer for code packaging instead of Framework estimators.

        Args:
            image_uri (str or PipelineVariable): The URI of the Docker image to use for the
                processing jobs.
            role (str or PipelineVariable): An AWS IAM role name or ARN. Amazon SageMaker
                Processing uses this role to access AWS resources, such as data stored
                in Amazon S3.
            instance_count (int or PipelineVariable): The number of instances to run a
                processing job with.
            instance_type (str or PipelineVariable): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            command ([str]): The command to run, along with any command-line flags
                to *precede* the ```code script```. Example: ["python3", "-v"]. If not
                provided, ["python"] will be chosen (default: None).
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing volume
                (default: None).
            output_kms_key (str or PipelineVariable): The KMS key ID for processing job outputs
                (default: None).
            code_location (str): The S3 prefix URI where custom code will be
                uploaded (default: None). The code file uploaded to S3 is
                'code_location/job-name/source/sourcedir.tar.gz'. If not specified, the
                default ``code location`` is 's3://{sagemaker-default-bucket}'
            max_runtime_in_seconds (int or PipelineVariable): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                processing image name and current timestamp (default: None).
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain (default: None).
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables to
                be passed to the processing jobs (default: None).
            tags (Optional[Tags]): Tags to be passed to the processing job (default: None).
                For more, see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets (default: None).
        """
        if not command:
            command = ["python"]

        super().__init__(
            role=role,
            image_uri=image_uri,
            command=command,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=format_tags(tags),
            network_config=network_config,
        )

        # This subclass uses the "code" input for actual payload and the ScriptProcessor parent's
        # functionality for uploading just a small entrypoint script to invoke it.
        self._CODE_CONTAINER_INPUT_NAME = "entrypoint"

        self.code_location = (
            code_location[:-1] if (code_location and code_location.endswith("/")) else code_location
        )

    def _package_code(
        self,
        entry_point,
        source_dir,
        requirements,
        job_name,
        kms_key,
    ):
        """Package and upload code to S3."""
        import tarfile
        import tempfile

        # If source_dir is not provided, use the directory containing entry_point
        if source_dir is None:
            if os.path.isabs(entry_point):
                source_dir = os.path.dirname(entry_point)
            else:
                source_dir = os.path.dirname(os.path.abspath(entry_point))

        # Resolve source_dir to absolute path
        if not os.path.isabs(source_dir):
            source_dir = os.path.abspath(source_dir)

        if not os.path.exists(source_dir):
            raise ValueError(f"source_dir does not exist: {source_dir}")

        # Create tar.gz with source_dir contents
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                # Add all files from source_dir to the root of the tar
                for item in os.listdir(source_dir):
                    item_path = os.path.join(source_dir, item)
                    tar.add(item_path, arcname=item)

            # Upload to S3
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix or "",
                job_name,
                "source",
                "sourcedir.tar.gz",
            )

            # Upload the tar file directly to S3
            s3.S3Uploader.upload_string_as_file_body(
                body=open(tmp.name, "rb").read(),
                desired_s3_uri=s3_uri,
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )

            os.unlink(tmp.name)
            return s3_uri

    @runnable_by_pipeline
    def run(
        self,
        code: str,
        source_dir: Optional[str] = None,
        requirements: Optional[str] = None,
        inputs: Optional[List[ProcessingInput]] = None,
        outputs: Optional[List["ProcessingOutput"]] = None,
        arguments: Optional[List[Union[str, PipelineVariable]]] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
    ):
        """Runs a processing job.

        Args:
            code (str): This can be an S3 URI or a local path to a file with the
                framework script to run.
            source_dir (str): Path (absolute, relative or an S3 URI) to a directory
                with any other processing source code dependencies aside from the entry
                point file (default: None).
            requirements (str): Path to a requirements.txt file relative to source_dir
                (default: None).
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            arguments (list[str] or list[PipelineVariable]): A list of string arguments
                to be passed to a processing job (default: None).
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the base job name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        Returns:
            None or pipeline step arguments in case the Processor instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        """
        s3_runproc_sh, inputs, job_name = self._pack_and_upload_code(
            code,
            source_dir,
            requirements,
            job_name,
            inputs,
            kms_key,
        )

        # Submit a processing job.
        return super().run(
            code=s3_runproc_sh,
            inputs=inputs,
            outputs=outputs,
            arguments=arguments,
            wait=wait,
            logs=logs,
            job_name=job_name,
            experiment_config=experiment_config,
            kms_key=kms_key,
        )

    def _pack_and_upload_code(
        self,
        code,
        source_dir,
        requirements,
        job_name,
        inputs,
        kms_key=None,
    ):
        """Pack local code bundle and upload to Amazon S3."""
        if code.startswith("s3://"):
            return code, inputs, job_name

        if job_name is None:
            job_name = self._generate_current_job_name(job_name)

        # Package and upload code
        s3_payload = self._package_code(
            entry_point=code,
            source_dir=source_dir,
            requirements=requirements,
            job_name=job_name,
            kms_key=kms_key,
        )

        inputs = self._patch_inputs_with_payload(inputs, s3_payload)

        entrypoint_s3_uri = s3_payload.replace("sourcedir.tar.gz", "runproc.sh")

        script = os.path.basename(code)
        evaluated_kms_key = kms_key if kms_key else self.output_kms_key
        s3_runproc_sh = self._create_and_upload_runproc(
            script, evaluated_kms_key, entrypoint_s3_uri
        )

        return s3_runproc_sh, inputs, job_name

    def _patch_inputs_with_payload(self, inputs, s3_payload) -> List[ProcessingInput]:
        """Add payload sourcedir.tar.gz to processing input."""
        if inputs is None:
            inputs = []

        # make a shallow copy of user inputs
        patched_inputs = copy(inputs)

        # Extract the directory path from the s3_payload (remove the filename)
        s3_code_dir = s3_payload.rsplit("/", 1)[0] + "/"

        patched_inputs.append(
            ProcessingInput(
                input_name="code",
                s3_input=ProcessingS3Input(
                    s3_uri=s3_code_dir,
                    local_path="/opt/ml/processing/input/code/",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                ),
            )
        )
        return patched_inputs

    def _set_entrypoint(self, command, user_script_name):
        """Framework processor override for setting processing job entrypoint."""
        user_script_location = str(
            pathlib.PurePosixPath(
                self._CODE_CONTAINER_BASE_PATH, self._CODE_CONTAINER_INPUT_NAME, user_script_name
            )
        )
        self.entrypoint = self.framework_entrypoint_command + [user_script_location]

    def _create_and_upload_runproc(self, user_script, kms_key, entrypoint_s3_uri):
        """Create runproc shell script and upload to S3 bucket."""
        from sagemaker.core.workflow.utilities import _pipeline_config, hash_object

        if _pipeline_config and _pipeline_config.pipeline_name:
            runproc_file_str = self._generate_framework_script(user_script)
            runproc_file_hash = hash_object(runproc_file_str)
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                _pipeline_config.pipeline_name,
                "code",
                runproc_file_hash,
                "runproc.sh",
            )
            s3_runproc_sh = s3.S3Uploader.upload_string_as_file_body(
                runproc_file_str,
                desired_s3_uri=s3_uri,
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        else:
            s3_runproc_sh = s3.S3Uploader.upload_string_as_file_body(
                self._generate_framework_script(user_script),
                desired_s3_uri=entrypoint_s3_uri,
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )

        return s3_runproc_sh

    def _generate_framework_script(self, user_script: str) -> str:
        """Generate the framework entrypoint file (as text) for a processing job."""
        return dedent(
            """\
            #!/bin/bash
            
            # Exit on any error. SageMaker uses error code to mark failed job.
            set -e

            cd /opt/ml/processing/input/code/
            
            # Debug: List files before extraction
            echo "Files in /opt/ml/processing/input/code/ before extraction:"
            ls -la
            
            # Extract source code
            if [ -f sourcedir.tar.gz ]; then
                tar -xzf sourcedir.tar.gz
                echo "Files after extraction:"
                ls -la
            else
                echo "ERROR: sourcedir.tar.gz not found!"
                exit 1
            fi

            if [[ -f 'requirements.txt' ]]; then
                # Some py3 containers has typing, which may breaks pip install
                pip uninstall --yes typing

                pip install -r requirements.txt
            fi

            {entry_point_command} {entry_point} "$@"
        """
        ).format(
            entry_point_command=" ".join(self.command),
            entry_point=user_script,
        )


class FeatureStoreOutput(ApiObject):
    """Configuration for processing job outputs in Amazon SageMaker Feature Store."""

    feature_group_name: Optional[str] = None


def _processing_input_to_request_dict(processing_input):
    """Convert ProcessingInput to request dictionary format."""
    app_managed = getattr(processing_input, "app_managed", False)
    request_dict = {
        "InputName": processing_input.input_name,
        "AppManaged": app_managed if app_managed is not None else False,
    }

    if processing_input.s3_input:
        request_dict["S3Input"] = {
            "S3Uri": processing_input.s3_input.s3_uri,
            "LocalPath": processing_input.s3_input.local_path,
            "S3DataType": processing_input.s3_input.s3_data_type or "S3Prefix",
            "S3InputMode": processing_input.s3_input.s3_input_mode or "File",
            "S3DataDistributionType": processing_input.s3_input.s3_data_distribution_type
            or "FullyReplicated",
            "S3CompressionType": processing_input.s3_input.s3_compression_type or "None",
        }

    return request_dict


def _processing_output_to_request_dict(processing_output):
    """Convert ProcessingOutput to request dictionary format."""
    app_managed = getattr(processing_output, "app_managed", False)
    request_dict = {
        "OutputName": processing_output.output_name,
        "AppManaged": app_managed if app_managed is not None else False,
    }

    if processing_output.s3_output:
        request_dict["S3Output"] = {
            "S3Uri": processing_output.s3_output.s3_uri,
            "LocalPath": processing_output.s3_output.local_path,
            "S3UploadMode": processing_output.s3_output.s3_upload_mode,
        }

    return request_dict


def _get_process_request(
    inputs,
    output_config,
    job_name,
    resources,
    stopping_condition,
    app_specification,
    environment,
    network_config,
    role_arn,
    tags,
    experiment_config=None,
):
    """Constructs a request compatible for an Amazon SageMaker processing job.

    Args:
        inputs ([dict]): List of up to 10 ProcessingInput dictionaries.
        output_config (dict): A config dictionary, which contains a list of up
            to 10 ProcessingOutput dictionaries, as well as an optional KMS key ID.
        job_name (str): The name of the processing job. The name must be unique
            within an AWS Region in an AWS account. Names should have minimum
            length of 1 and maximum length of 63 characters.
        resources (dict): Encapsulates the resources, including ML instances
            and storage, to use for the processing job.
        stopping_condition (dict[str,int]): Specifies a limit to how long
            the processing job can run, in seconds.
        app_specification (dict[str,str]): Configures the processing job to
            run the given image. Details are in the processing container
            specification.
        environment (dict): Environment variables to start the processing
            container with.
        network_config (dict): Specifies networking options, such as network
            traffic encryption between processing containers, whether to allow
            inbound and outbound network calls to and from processing containers,
            and VPC subnets and security groups to use for VPC-enabled processing
            jobs.
        role_arn (str): The Amazon Resource Name (ARN) of an IAM role that
            Amazon SageMaker can assume to perform tasks on your behalf.
        tags ([dict[str,str]]): A list of dictionaries containing key-value
            pairs.
        experiment_config (dict[str, str]): Experiment management configuration.
            Optionally, the dict can contain three keys:
            'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
            The behavior of setting these keys is as follows:
            * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
            automatically created and the job's Trial Component associated with the Trial.
            * If `TrialName` is supplied and the Trial already exists the job's Trial Component
            will be associated with the Trial.
            * If both `ExperimentName` and `TrialName` are not supplied the trial component
            will be unassociated.
            * `TrialComponentDisplayName` is used for display in Studio.

    Returns:
        Dict: a processing job request dict
    """
    process_request = {
        "ProcessingJobName": job_name,
        "ProcessingResources": resources,
        "AppSpecification": app_specification,
        "RoleArn": role_arn,
    }

    if inputs:
        process_request["ProcessingInputs"] = inputs

    if output_config["Outputs"]:
        process_request["ProcessingOutputConfig"] = output_config

    if environment is not None:
        process_request["Environment"] = environment

    if network_config is not None:
        process_request["NetworkConfig"] = network_config

    if stopping_condition is not None:
        process_request["StoppingCondition"] = stopping_condition

    if tags is not None:
        process_request["Tags"] = tags

    if experiment_config:
        process_request["ExperimentConfig"] = experiment_config

    return process_request


def logs_for_processing_job(sagemaker_session, job_name, wait=False, poll=10):
    """Display logs for a given processing job, optionally tailing them until the is complete.

    Args:
        job_name (str): Name of the processing job to display the logs for.
        wait (bool): Whether to keep looking for new log entries until the job completes
            (default: False).
        poll (int): The interval in seconds between polling for new log entries and job
            completion (default: 5).

    Raises:
        ValueError: If the processing job fails.
    """

    description = _wait_until(
        lambda: ProcessingJob.get(
            processing_job_name=job_name, session=sagemaker_session.boto_session
        )
        .refresh()
        .__dict__,
        poll,
    )

    instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
        sagemaker_session.boto_session, description, job="Processing"
    )

    state = _get_initial_job_state(description, "ProcessingJobStatus", wait)

    # The loop below implements a state machine that alternates between checking the job status
    # and reading whatever is available in the logs at this point. Note, that if we were
    # called with wait == False, we never check the job status.
    #
    # If wait == TRUE and job is not completed, the initial state is TAILING
    # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is
    # complete).
    #
    # The state table:
    #
    # STATE               ACTIONS                        CONDITION             NEW STATE
    # ----------------    ----------------               -----------------     ----------------
    # TAILING             Read logs, Pause, Get status   Job complete          JOB_COMPLETE
    #                                                    Else                  TAILING
    # JOB_COMPLETE        Read logs, Pause               Any                   COMPLETE
    # COMPLETE            Read logs, Exit                                      N/A
    #
    # Notes:
    # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to
    #   Cloudwatch after the job was marked complete.
    last_describe_job_call = time.time()
    while True:
        _flush_log_streams(
            stream_names,
            instance_count,
            client,
            log_group,
            job_name,
            positions,
            dot,
            color_wrap,
        )
        if state == LogState.COMPLETE:
            break

        time.sleep(poll)

        if state == LogState.JOB_COMPLETE:
            state = LogState.COMPLETE
        elif time.time() - last_describe_job_call >= 30:
            description = (
                ProcessingJob.get(
                    processing_job_name=job_name, session=sagemaker_session.boto_session
                )
                .refresh()
                .__dict__
            )
            last_describe_job_call = time.time()

            status = description["ProcessingJobStatus"]

            if status in ("Completed", "Failed", "Stopped"):
                print()
                state = LogState.JOB_COMPLETE

    if wait:
        _check_job_status(job_name, description, "ProcessingJobStatus")
        if dot:
            print()
