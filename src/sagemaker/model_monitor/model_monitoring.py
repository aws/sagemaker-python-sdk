# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This module contains code related to Amazon SageMaker Model Monitoring. These classes
assist with suggesting baselines and creating monitoring schedules for data captured
by SageMaker Endpoints.
"""
from __future__ import print_function, absolute_import

import copy
import json
import os
import pathlib
import logging
import uuid

from six import string_types
from six.moves.urllib.parse import urlparse
from botocore.exceptions import ClientError

from sagemaker import image_uris, s3
from sagemaker.exceptions import UnexpectedStatusException
from sagemaker.model_monitor.monitoring_files import Constraints, ConstraintViolations, Statistics
from sagemaker.network import NetworkConfig
from sagemaker.processing import Processor, ProcessingInput, ProcessingJob, ProcessingOutput
from sagemaker.session import Session
from sagemaker.utils import name_from_base, retries

DEFAULT_REPOSITORY_NAME = "sagemaker-model-monitor-analyzer"

STATISTICS_JSON_DEFAULT_FILE_NAME = "statistics.json"
CONSTRAINTS_JSON_DEFAULT_FILE_NAME = "constraints.json"
CONSTRAINT_VIOLATIONS_JSON_DEFAULT_FILE_NAME = "constraint_violations.json"

_CONTAINER_BASE_PATH = "/opt/ml/processing"
_CONTAINER_INPUT_PATH = "input"
_CONTAINER_ENDPOINT_INPUT_PATH = "endpoint"
_BASELINE_DATASET_INPUT_NAME = "baseline_dataset_input"
_RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME = "record_preprocessor_script_input"
_POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME = "post_analytics_processor_script_input"
_CONTAINER_OUTPUT_PATH = "output"
_DEFAULT_OUTPUT_NAME = "monitoring_output"
_MODEL_MONITOR_S3_PATH = "model-monitor"
_BASELINING_S3_PATH = "baselining"
_MONITORING_S3_PATH = "monitoring"
_RESULTS_S3_PATH = "results"
_INPUT_S3_PATH = "input"

_SUGGESTION_JOB_BASE_NAME = "baseline-suggestion-job"
_MONITORING_SCHEDULE_BASE_NAME = "monitoring-schedule"

_DATASET_SOURCE_PATH_ENV_NAME = "dataset_source"
_DATASET_FORMAT_ENV_NAME = "dataset_format"
_OUTPUT_PATH_ENV_NAME = "output_path"
_RECORD_PREPROCESSOR_SCRIPT_ENV_NAME = "record_preprocessor_script"
_POST_ANALYTICS_PROCESSOR_SCRIPT_ENV_NAME = "post_analytics_processor_script"
_PUBLISH_CLOUDWATCH_METRICS_ENV_NAME = "publish_cloudwatch_metrics"

_LOGGER = logging.getLogger(__name__)

framework_name = "model-monitor"


class ModelMonitor(object):
    """Sets up Amazon SageMaker Monitoring Schedules and baseline suggestions. Use this class when
    you want to provide your own container image containing the code you'd like to run, in order
    to produce your own statistics and constraint validation files.
    For a more guided experience, consider using the DefaultModelMonitor class instead.
    """

    def __init__(
        self,
        role,
        image_uri,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        entrypoint=None,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initializes a ``Monitor`` instance. The Monitor handles baselining datasets and
        creating Amazon SageMaker Monitoring Schedules to monitor SageMaker endpoints.

        Args:
            role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
            image_uri (str): The uri of the image to use for the jobs started by
                the Monitor.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            entrypoint ([str]): The entrypoint for the job.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            base_job_name (str): Prefix for the job name. If not specified,
                a default name is generated based on the training image name and
                current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            env (dict): Environment variables to be passed to the job.
            tags ([dict]): List of tags to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.

        """
        self.role = role
        self.image_uri = image_uri
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.entrypoint = entrypoint
        self.volume_size_in_gb = volume_size_in_gb
        self.volume_kms_key = volume_kms_key
        self.output_kms_key = output_kms_key
        self.max_runtime_in_seconds = max_runtime_in_seconds
        self.base_job_name = base_job_name
        self.sagemaker_session = sagemaker_session or Session()
        self.env = env
        self.tags = tags
        self.network_config = network_config

        self.baselining_jobs = []
        self.latest_baselining_job = None
        self.arguments = None
        self.latest_baselining_job_name = None
        self.monitoring_schedule_name = None

    def run_baseline(
        self, baseline_inputs, output, arguments=None, wait=True, logs=True, job_name=None
    ):
        """Run a processing job meant to baseline your dataset.

        Args:
            baseline_inputs ([sagemaker.processing.ProcessingInput]): Input files for the processing
                job. These must be provided as ProcessingInput objects.
            output (sagemaker.processing.ProcessingOutput): Destination of the constraint_violations
                and statistics json files.
            arguments ([str]): A list of string arguments to be passed to a processing job.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.

        """
        self.latest_baselining_job_name = self._generate_baselining_job_name(job_name=job_name)
        self.arguments = arguments
        normalized_baseline_inputs = self._normalize_baseline_inputs(
            baseline_inputs=baseline_inputs
        )
        normalized_output = self._normalize_processing_output(output=output)

        baselining_processor = Processor(
            role=self.role,
            image_uri=self.image_uri,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            entrypoint=self.entrypoint,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            output_kms_key=self.output_kms_key,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            base_job_name=self.base_job_name,
            sagemaker_session=self.sagemaker_session,
            env=self.env,
            tags=self.tags,
            network_config=self.network_config,
        )

        baselining_processor.run(
            inputs=normalized_baseline_inputs,
            outputs=[normalized_output],
            arguments=self.arguments,
            wait=wait,
            logs=logs,
            job_name=self.latest_baselining_job_name,
        )

        self.latest_baselining_job = BaseliningJob.from_processing_job(
            processing_job=baselining_processor.latest_job
        )
        self.baselining_jobs.append(self.latest_baselining_job)

    def create_monitoring_schedule(
        self,
        endpoint_input,
        output,
        statistics=None,
        constraints=None,
        monitor_schedule_name=None,
        schedule_cron_expression=None,
    ):
        """Creates a monitoring schedule to monitor an Amazon SageMaker Endpoint.

        If constraints and statistics are provided, or if they are able to be retrieved from a
        previous baselining job associated with this monitor, those will be used.
        If constraints and statistics cannot be automatically retrieved, baseline_inputs will be
        required in order to kick off a baselining job.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            output (sagemaker.model_monitor.MonitoringOutput): The output of the monitoring
                schedule.
            statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
                constraints, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
                JSON file.
            constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
                statistics, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
                JSON file.
            monitor_schedule_name (str): Schedule name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job runs at. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions. Default: Daily.

        """
        if self.monitoring_schedule_name is not None:
            message = (
                "It seems that this object was already used to create an Amazon Model "
                "Monitoring Schedule. To create another, first delete the existing one "
                "using my_monitor.delete_monitoring_schedule()."
            )
            print(message)
            raise ValueError(message)

        self.monitoring_schedule_name = self._generate_monitoring_schedule_name(
            schedule_name=monitor_schedule_name
        )

        normalized_endpoint_input = self._normalize_endpoint_input(endpoint_input=endpoint_input)

        normalized_monitoring_output = self._normalize_monitoring_output(output=output)

        statistics_object, constraints_object = self._get_baseline_files(
            statistics=statistics, constraints=constraints, sagemaker_session=self.sagemaker_session
        )

        statistics_s3_uri = None
        if statistics_object is not None:
            statistics_s3_uri = statistics_object.file_s3_uri

        constraints_s3_uri = None
        if constraints_object is not None:
            constraints_s3_uri = constraints_object.file_s3_uri

        monitoring_output_config = {
            "MonitoringOutputs": [normalized_monitoring_output._to_request_dict()]
        }

        if self.output_kms_key is not None:
            monitoring_output_config["KmsKeyId"] = self.output_kms_key

        self.monitoring_schedule_name = (
            monitor_schedule_name
            or self._generate_monitoring_schedule_name(schedule_name=monitor_schedule_name)
        )

        network_config_dict = None
        if self.network_config is not None:
            network_config_dict = self.network_config._to_request_dict()
            self._validate_network_config(network_config_dict)

        self.sagemaker_session.create_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name,
            schedule_expression=schedule_cron_expression,
            statistics_s3_uri=statistics_s3_uri,
            constraints_s3_uri=constraints_s3_uri,
            monitoring_inputs=[normalized_endpoint_input._to_request_dict()],
            monitoring_output_config=monitoring_output_config,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            image_uri=self.image_uri,
            entrypoint=self.entrypoint,
            arguments=self.arguments,
            record_preprocessor_source_uri=None,
            post_analytics_processor_source_uri=None,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            environment=self.env,
            network_config=network_config_dict,
            role_arn=self.sagemaker_session.expand_role(self.role),
            tags=self.tags,
        )

    def update_monitoring_schedule(
        self,
        endpoint_input=None,
        output=None,
        statistics=None,
        constraints=None,
        schedule_cron_expression=None,
        instance_count=None,
        instance_type=None,
        entrypoint=None,
        volume_size_in_gb=None,
        volume_kms_key=None,
        output_kms_key=None,
        arguments=None,
        max_runtime_in_seconds=None,
        env=None,
        network_config=None,
        role=None,
        image_uri=None,
    ):
        """Updates the existing monitoring schedule.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            output (sagemaker.model_monitor.MonitoringOutput): The output of the monitoring
                schedule.
            statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
                constraints, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
                JSON file.
            constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
                statistics, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
                JSON file.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job runs at.  See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            entrypoint (str): The entrypoint for the job.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            arguments ([str]): A list of string arguments to be passed to a processing job.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            env (dict): Environment variables to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
            role (str): An AWS IAM role name or ARN. The Amazon SageMaker jobs use this role.
            image_uri (str): The uri of the image to use for the jobs started by
                the Monitor.

        """
        monitoring_inputs = None
        if endpoint_input is not None:
            monitoring_inputs = [
                self._normalize_endpoint_input(endpoint_input=endpoint_input)._to_request_dict()
            ]

        monitoring_output_config = None
        if output is not None:
            normalized_monitoring_output = self._normalize_monitoring_output(output=output)
            monitoring_output_config = {
                "MonitoringOutputs": [normalized_monitoring_output._to_request_dict()]
            }

        statistics_object, constraints_object = self._get_baseline_files(
            statistics=statistics, constraints=constraints, sagemaker_session=self.sagemaker_session
        )

        statistics_s3_uri = None
        if statistics_object is not None:
            statistics_s3_uri = statistics_object.file_s3_uri

        constraints_s3_uri = None
        if constraints_object is not None:
            constraints_s3_uri = constraints_object.file_s3_uri

        if instance_type is not None:
            self.instance_type = instance_type

        if instance_count is not None:
            self.instance_count = instance_count

        if entrypoint is not None:
            self.entrypoint = entrypoint

        if volume_size_in_gb is not None:
            self.volume_size_in_gb = volume_size_in_gb

        if volume_kms_key is not None:
            self.volume_kms_key = volume_kms_key

        if output_kms_key is not None:
            self.output_kms_key = output_kms_key
            monitoring_output_config["KmsKeyId"] = self.output_kms_key

        if arguments is not None:
            self.arguments = arguments

        if max_runtime_in_seconds is not None:
            self.max_runtime_in_seconds = max_runtime_in_seconds

        if env is not None:
            self.env = env

        if network_config is not None:
            self.network_config = network_config

        if role is not None:
            self.role = role

        if image_uri is not None:
            self.image_uri = image_uri

        network_config_dict = None
        if self.network_config is not None:
            network_config_dict = self.network_config._to_request_dict()
            self._validate_network_config(network_config_dict)

        self.sagemaker_session.update_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name,
            schedule_expression=schedule_cron_expression,
            statistics_s3_uri=statistics_s3_uri,
            constraints_s3_uri=constraints_s3_uri,
            monitoring_inputs=monitoring_inputs,
            monitoring_output_config=monitoring_output_config,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            image_uri=image_uri,
            entrypoint=entrypoint,
            arguments=arguments,
            max_runtime_in_seconds=max_runtime_in_seconds,
            environment=env,
            network_config=network_config_dict,
            role_arn=self.sagemaker_session.expand_role(self.role),
        )

        self._wait_for_schedule_changes_to_apply()

    def start_monitoring_schedule(self):
        """Starts the monitoring schedule."""
        self.sagemaker_session.start_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name
        )

        self._wait_for_schedule_changes_to_apply()

    def stop_monitoring_schedule(self):
        """Stops the monitoring schedule."""
        self.sagemaker_session.stop_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name
        )

        self._wait_for_schedule_changes_to_apply()

    def delete_monitoring_schedule(self):
        """Deletes the monitoring schedule."""
        self.sagemaker_session.delete_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name
        )
        self.monitoring_schedule_name = None

    def baseline_statistics(self, file_name=STATISTICS_JSON_DEFAULT_FILE_NAME):
        """Returns a Statistics object representing the statistics json file generated by the
        latest baselining job.

        Args:
            file_name (str): The name of the .json statistics file

        Returns:
            sagemaker.model_monitor.Statistics: The Statistics object representing the file that
                was generated by the job.

        """
        return self.latest_baselining_job.baseline_statistics(
            file_name=file_name, kms_key=self.output_kms_key
        )

    def suggested_constraints(self, file_name=CONSTRAINTS_JSON_DEFAULT_FILE_NAME):
        """Returns a Statistics object representing the constraints json file generated by the
        latest baselining job

        Args:
            file_name (str): The name of the .json constraints file

        Returns:
            sagemaker.model_monitor.Constraints: The Constraints object representing the file that
                was generated by the job.

        """
        return self.latest_baselining_job.suggested_constraints(
            file_name=file_name, kms_key=self.output_kms_key
        )

    def latest_monitoring_statistics(self, file_name=STATISTICS_JSON_DEFAULT_FILE_NAME):
        """Returns the sagemaker.model_monitor.Statistics generated by the latest monitoring
        execution.

        Args:
            file_name (str): The name of the statistics file to be retrieved. Only override if
                generating a custom file name.

        Returns:
            sagemaker.model_monitoring.Statistics: The Statistics object representing the file
                generated by the latest monitoring execution.

        """
        executions = self.list_executions()
        if len(executions) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return None

        latest_monitoring_execution = executions[-1]
        return latest_monitoring_execution.statistics(file_name=file_name)

    def latest_monitoring_constraint_violations(
        self, file_name=CONSTRAINT_VIOLATIONS_JSON_DEFAULT_FILE_NAME
    ):
        """Returns the sagemaker.model_monitor.ConstraintViolations generated by the latest
        monitoring execution.

        Args:
            file_name (str): The name of the constraint violdations file to be retrieved. Only
                override if generating a custom file name.

        Returns:
            sagemaker.model_monitoring.ConstraintViolations: The ConstraintViolations object
                representing the file generated by the latest monitoring execution.

        """
        executions = self.list_executions()
        if len(executions) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return None

        latest_monitoring_execution = executions[-1]
        return latest_monitoring_execution.constraint_violations(file_name=file_name)

    def describe_latest_baselining_job(self):
        """Describe the latest baselining job kicked off by the suggest workflow."""
        if self.latest_baselining_job is None:
            raise ValueError("No suggestion jobs were kicked off.")
        return self.latest_baselining_job.describe()

    def describe_schedule(self):
        """Describes the schedule that this object represents.

        Returns:
            dict: A dictionary response with the monitoring schedule description.

        """
        return self.sagemaker_session.describe_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name
        )

    def list_executions(self):
        """Get the list of the latest monitoring executions in descending order of "ScheduledTime".
        Statistics or violations can be called following this example:
        Example:
            >>> my_executions = my_monitor.list_executions()
            >>> second_to_last_execution_statistics = my_executions[-1].statistics()
            >>> second_to_last_execution_violations = my_executions[-1].constraint_violations()

        Returns:
            [sagemaker.model_monitor.MonitoringExecution]: List of MonitoringExecutions in
                descending order of "ScheduledTime".

        """
        monitoring_executions_dict = self.sagemaker_session.list_monitoring_executions(
            monitoring_schedule_name=self.monitoring_schedule_name
        )

        if len(monitoring_executions_dict["MonitoringExecutionSummaries"]) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return []

        processing_job_arns = [
            execution_dict["ProcessingJobArn"]
            for execution_dict in monitoring_executions_dict["MonitoringExecutionSummaries"]
            if execution_dict.get("ProcessingJobArn") is not None
        ]
        monitoring_executions = [
            MonitoringExecution.from_processing_arn(
                sagemaker_session=self.sagemaker_session, processing_job_arn=processing_job_arn
            )
            for processing_job_arn in processing_job_arns
        ]
        monitoring_executions.reverse()

        return monitoring_executions

    @classmethod
    def attach(cls, monitor_schedule_name, sagemaker_session=None):
        """Sets this object's schedule name to point to the Amazon Sagemaker Monitoring Schedule
        name provided. This allows subsequent describe_schedule or list_executions calls to point
        to the given schedule.

        Args:
            monitor_schedule_name (str): The name of the schedule to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.

        """
        sagemaker_session = sagemaker_session or Session()
        schedule_desc = sagemaker_session.describe_monitoring_schedule(
            monitoring_schedule_name=monitor_schedule_name
        )

        monitoring_job_definition = schedule_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ]
        role = monitoring_job_definition["RoleArn"]
        image_uri = monitoring_job_definition["MonitoringAppSpecification"].get("ImageUri")
        cluster_config = monitoring_job_definition["MonitoringResources"]["ClusterConfig"]
        instance_count = cluster_config.get("InstanceCount")
        instance_type = cluster_config["InstanceType"]
        volume_size_in_gb = cluster_config["VolumeSizeInGB"]
        volume_kms_key = cluster_config.get("VolumeKmsKeyId")
        entrypoint = monitoring_job_definition["MonitoringAppSpecification"].get(
            "ContainerEntrypoint"
        )
        output_kms_key = monitoring_job_definition["MonitoringOutputConfig"].get("KmsKeyId")
        network_config_dict = monitoring_job_definition.get("NetworkConfig")

        max_runtime_in_seconds = None
        stopping_condition = monitoring_job_definition.get("StoppingCondition")
        if stopping_condition:
            max_runtime_in_seconds = stopping_condition.get("MaxRuntimeInSeconds")

        env = monitoring_job_definition.get("Environment", None)

        vpc_config = None
        if network_config_dict:
            vpc_config = network_config_dict.get("VpcConfig")

        security_group_ids = None
        if vpc_config is not None:
            security_group_ids = vpc_config["SecurityGroupIds"]

        subnets = None
        if vpc_config is not None:
            subnets = vpc_config["Subnets"]

        network_config = None
        if network_config_dict:
            network_config = NetworkConfig(
                enable_network_isolation=network_config_dict["EnableNetworkIsolation"],
                security_group_ids=security_group_ids,
                subnets=subnets,
            )

        tags = sagemaker_session.list_tags(resource_arn=schedule_desc["MonitoringScheduleArn"])

        attached_monitor = cls(
            role=role,
            image_uri=image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            entrypoint=entrypoint,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )
        attached_monitor.monitoring_schedule_name = monitor_schedule_name
        return attached_monitor

    def _generate_baselining_job_name(self, job_name=None):
        """Generate the job name before running a suggestion processing job.

        Args:
            job_name (str): Name of the suggestion processing job to be created. If not
                specified, one is generated using the base name given to the
                constructor, if applicable.

        Returns:
            str: The supplied or generated job name.

        """
        if job_name is not None:
            return job_name

        if self.base_job_name:
            base_name = self.base_job_name
        else:
            base_name = _SUGGESTION_JOB_BASE_NAME

        return name_from_base(base=base_name)

    def _generate_monitoring_schedule_name(self, schedule_name=None):
        """Generate the monitoring schedule name.

        Args:
            schedule_name (str): Name of the monitoring schedule to be created. If not
                specified, one is generated using the base name given to the
                constructor, if applicable.

        Returns:
            str: The supplied or generated job name.

        """
        if schedule_name is not None:
            return schedule_name

        if self.base_job_name:
            base_name = self.base_job_name
        else:
            base_name = _MONITORING_SCHEDULE_BASE_NAME

        return name_from_base(base=base_name)

    @staticmethod
    def _get_baseline_files(statistics, constraints, sagemaker_session=None):
        """Populates baseline values if possible.

        Args:
            statistics (sagemaker.model_monitor.Statistics or str): The statistics object or str.
                If none, this method will attempt to retrieve a previously baselined constraints
                object.
            constraints (sagemaker.model_monitor.Constraints or str): The constraints object or str.
                If none, this method will attempt to retrieve a previously baselined constraints
                object.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, one
                is created using the default AWS configuration chain.

        Returns:
            sagemaker.model_monitor.Statistics, sagemaker.model_monitor.Constraints: The Statistics
                and Constraints objects that were provided or created by the latest
                baselining job. If none were found, returns None.

        """
        if statistics is not None and isinstance(statistics, string_types):
            statistics = Statistics.from_s3_uri(
                statistics_file_s3_uri=statistics, sagemaker_session=sagemaker_session
            )
        if constraints is not None and isinstance(constraints, string_types):
            constraints = Constraints.from_s3_uri(
                constraints_file_s3_uri=constraints, sagemaker_session=sagemaker_session
            )

        return statistics, constraints

    def _normalize_endpoint_input(self, endpoint_input):
        """Ensure that the input is an EndpointInput object.

        Args:
            endpoint_input ([str or sagemaker.processing.EndpointInput]): An endpoint input
                to be normalized. Can be either a string or a EndpointInput object.

        Returns:
            sagemaker.processing.EndpointInput: The normalized EndpointInput object.

        """
        # If the input is a string, turn it into an EndpointInput object.
        if isinstance(endpoint_input, string_types):
            endpoint_input = EndpointInput(
                endpoint_name=endpoint_input,
                destination=str(
                    pathlib.PurePosixPath(
                        _CONTAINER_BASE_PATH, _CONTAINER_INPUT_PATH, _CONTAINER_ENDPOINT_INPUT_PATH
                    )
                ),
            )

        return endpoint_input

    def _normalize_baseline_inputs(self, baseline_inputs=None):
        """Ensure that all the ProcessingInput objects have names and S3 uris.

        Args:
            baseline_inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput
                objects to be normalized.

        Returns:
            [sagemaker.processing.ProcessingInput]: The list of normalized
                ProcessingInput objects.

        """
        # Initialize a list of normalized ProcessingInput objects.
        normalized_inputs = []
        if baseline_inputs is not None:
            # Iterate through the provided list of inputs.
            for count, file_input in enumerate(baseline_inputs, 1):
                if not isinstance(file_input, ProcessingInput):
                    raise TypeError("Your inputs must be provided as ProcessingInput objects.")
                # Generate a name for the ProcessingInput if it doesn't have one.
                if file_input.input_name is None:
                    file_input.input_name = "input-{}".format(count)
                # If the source is a local path, upload it to S3
                # and save the S3 uri in the ProcessingInput source.
                parse_result = urlparse(file_input.source)
                if parse_result.scheme != "s3":
                    s3_uri = s3.s3_path_join(
                        "s3://",
                        self.sagemaker_session.default_bucket(),
                        self.latest_baselining_job_name,
                        file_input.input_name,
                    )
                    s3.S3Uploader.upload(
                        local_path=file_input.source,
                        desired_s3_uri=s3_uri,
                        sagemaker_session=self.sagemaker_session,
                    )
                    file_input.source = s3_uri
                normalized_inputs.append(file_input)
        return normalized_inputs

    def _normalize_processing_output(self, output=None):
        """Ensure that the output is a ProcessingOutput object.

        Args:
            output (str or sagemaker.processing.ProcessingOutput): An output to be normalized.

        Returns:
            sagemaker.processing.ProcessingOutput: The normalized ProcessingOutput object.

        """
        # If the output is a string, turn it into a ProcessingOutput object.
        if isinstance(output, string_types):
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.latest_baselining_job_name,
                "output",
            )
            output = ProcessingOutput(
                source=output, destination=s3_uri, output_name=_DEFAULT_OUTPUT_NAME
            )

        return output

    def _normalize_monitoring_output(self, output=None):
        """Ensure that output has the correct fields.

        Args:
            output (sagemaker.processing.MonitoringOutput): An output to be normalized.

        Returns:
            sagemaker.processing.MonitoringOutput: The normalized MonitoringOutput object.

        """
        # If the output is a string, turn it into a ProcessingOutput object.
        if output.destination is None:
            output.destination = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.monitoring_schedule_name,
                "output",
            )

        return output

    def _s3_uri_from_local_path(self, path):
        """If path is local, uploads to S3 and returns S3 uri. Otherwise returns S3 uri as-is.

        Args:
            path (str): Path to file. This can be a local path or an S3 path.

        Returns:
            str: S3 uri to file.

        """
        parse_result = urlparse(path)
        if parse_result.scheme != "s3":
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                _MODEL_MONITOR_S3_PATH,
                _MONITORING_S3_PATH,
                self.monitoring_schedule_name,
                _INPUT_S3_PATH,
                str(uuid.uuid4()),
            )
            s3.S3Uploader.upload(
                local_path=path, desired_s3_uri=s3_uri, sagemaker_session=self.sagemaker_session
            )
            path = s3.s3_path_join(s3_uri, os.path.basename(path))
        return path

    def _wait_for_schedule_changes_to_apply(self):
        """Waits for the schedule associated with this monitor to no longer be in the 'Pending'
        state.

        """
        for _ in retries(
            max_retry_count=36,  # 36*5 = 3min
            exception_message_prefix="Waiting for schedule to leave 'Pending' status",
            seconds_to_sleep=5,
        ):
            schedule_desc = self.describe_schedule()
            if schedule_desc["MonitoringScheduleStatus"] != "Pending":
                break

    def _validate_network_config(self, network_config_dict):
        """Validates that EnableInterContainerTrafficEncryption is not set in the provided
        NetworkConfig request dictionary.

        Args:
            network_config_dict (dict): NetworkConfig request dictionary.
                Contains parameters from :class:`~sagemaker.network.NetworkConfig` object
                that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.

        """
        if "EnableInterContainerTrafficEncryption" in network_config_dict:
            message = (
                "EnableInterContainerTrafficEncryption is not supported in Model Monitor. "
                "Please ensure that encrypt_inter_container_traffic=None "
                "when creating your NetworkConfig object. "
                "Current encrypt_inter_container_traffic value: {}".format(
                    self.network_config.encrypt_inter_container_traffic
                )
            )
            _LOGGER.info(message)
            raise ValueError(message)


class DefaultModelMonitor(ModelMonitor):
    """Sets up Amazon SageMaker Monitoring Schedules and baseline suggestions. Use this class when
    you want to utilize Amazon SageMaker Monitoring's plug-and-play solution that only requires
    your dataset and optional pre/postprocessing scripts.
    For a more customized experience, consider using the ModelMonitor class instead.
    """

    def __init__(
        self,
        role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initializes a ``Monitor`` instance. The Monitor handles baselining datasets and
        creating Amazon SageMaker Monitoring Schedules to monitor SageMaker endpoints.

        Args:
            role (str): An AWS IAM role name or ARN. The Amazon SageMaker jobs use this role.
            instance_count (int): The number of instances to run the jobs with.
            instance_type (str): Type of EC2 instance to use for the job, for example,
                'ml.m5.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            base_job_name (str): Prefix for the job name. If not specified,
                a default name is generated based on the training image name and
                current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            env (dict): Environment variables to be passed to the job.
            tags ([dict]): List of tags to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.

        """
        session = sagemaker_session or Session()
        super(DefaultModelMonitor, self).__init__(
            role=role,
            image_uri=DefaultModelMonitor._get_default_image_uri(session.boto_session.region_name),
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )

    def suggest_baseline(
        self,
        baseline_dataset,
        dataset_format,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        wait=True,
        logs=True,
        job_name=None,
    ):
        """Suggest baselines for use with Amazon SageMaker Model Monitoring Schedules.

        Args:
            baseline_dataset (str): The path to the baseline_dataset file. This can be a local path
                or an S3 uri.
            dataset_format (dict): The format of the baseline_dataset.
            record_preprocessor_script (str): The path to the record preprocessor script. This can
                be a local path or an S3 uri.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): Desired S3 destination Destination of the constraint_violations
                and statistics json files.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.

        Returns:
            sagemaker.processing.ProcessingJob: The ProcessingJob object representing the
                baselining job.

        """
        self.latest_baselining_job_name = self._generate_baselining_job_name(job_name=job_name)

        normalized_baseline_dataset_input = self._upload_and_convert_to_processing_input(
            source=baseline_dataset,
            destination=str(
                pathlib.PurePosixPath(
                    _CONTAINER_BASE_PATH, _CONTAINER_INPUT_PATH, _BASELINE_DATASET_INPUT_NAME
                )
            ),
            name=_BASELINE_DATASET_INPUT_NAME,
        )

        # Unlike other input, dataset must be a directory for the Monitoring image.
        baseline_dataset_container_path = normalized_baseline_dataset_input.destination

        normalized_record_preprocessor_script_input = self._upload_and_convert_to_processing_input(
            source=record_preprocessor_script,
            destination=str(
                pathlib.PurePosixPath(
                    _CONTAINER_BASE_PATH,
                    _CONTAINER_INPUT_PATH,
                    _RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME,
                )
            ),
            name=_RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME,
        )

        record_preprocessor_script_container_path = None
        if normalized_record_preprocessor_script_input is not None:
            record_preprocessor_script_container_path = str(
                pathlib.PurePosixPath(
                    normalized_record_preprocessor_script_input.destination,
                    os.path.basename(record_preprocessor_script),
                )
            )

        normalized_post_processor_script_input = self._upload_and_convert_to_processing_input(
            source=post_analytics_processor_script,
            destination=str(
                pathlib.PurePosixPath(
                    _CONTAINER_BASE_PATH,
                    _CONTAINER_INPUT_PATH,
                    _POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME,
                )
            ),
            name=_POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME,
        )

        post_processor_script_container_path = None
        if normalized_post_processor_script_input is not None:
            post_processor_script_container_path = str(
                pathlib.PurePosixPath(
                    normalized_post_processor_script_input.destination,
                    os.path.basename(post_analytics_processor_script),
                )
            )

        normalized_baseline_output = self._normalize_baseline_output(output_s3_uri=output_s3_uri)

        normalized_env = self._generate_env_map(
            env=self.env,
            dataset_format=dataset_format,
            output_path=normalized_baseline_output.source,
            enable_cloudwatch_metrics=False,  # Only supported for monitoring schedules
            dataset_source_container_path=baseline_dataset_container_path,
            record_preprocessor_script_container_path=record_preprocessor_script_container_path,
            post_processor_script_container_path=post_processor_script_container_path,
        )

        baselining_processor = Processor(
            role=self.role,
            image_uri=self.image_uri,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            entrypoint=self.entrypoint,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            output_kms_key=self.output_kms_key,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            base_job_name=self.base_job_name,
            sagemaker_session=self.sagemaker_session,
            env=normalized_env,
            tags=self.tags,
            network_config=self.network_config,
        )

        baseline_job_inputs_with_nones = [
            normalized_baseline_dataset_input,
            normalized_record_preprocessor_script_input,
            normalized_post_processor_script_input,
        ]

        baseline_job_inputs = [
            baseline_job_input
            for baseline_job_input in baseline_job_inputs_with_nones
            if baseline_job_input is not None
        ]

        baselining_processor.run(
            inputs=baseline_job_inputs,
            outputs=[normalized_baseline_output],
            arguments=self.arguments,
            wait=wait,
            logs=logs,
            job_name=self.latest_baselining_job_name,
        )

        self.latest_baselining_job = BaseliningJob.from_processing_job(
            processing_job=baselining_processor.latest_job
        )
        self.baselining_jobs.append(self.latest_baselining_job)
        return baselining_processor.latest_job

    def create_monitoring_schedule(
        self,
        endpoint_input,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        constraints=None,
        statistics=None,
        monitor_schedule_name=None,
        schedule_cron_expression=None,
        enable_cloudwatch_metrics=True,
    ):
        """Creates a monitoring schedule to monitor an Amazon SageMaker Endpoint.

        If constraints and statistics are provided, or if they are able to be retrieved from a
        previous baselining job associated with this monitor, those will be used.
        If constraints and statistics cannot be automatically retrieved, baseline_inputs will be
        required in order to kick off a baselining job.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            record_preprocessor_script (str): The path to the record preprocessor script. This can
                be a local path or an S3 uri.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): Desired S3 destination of the constraint_violations and
                statistics json files.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
                statistics, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an s3_uri pointing to a constraints
                JSON file.
            statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
                constraints, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an s3_uri pointing to a constraints
                JSON file.
            monitor_schedule_name (str): Schedule name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions. Default: Daily.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.

        """
        if self.monitoring_schedule_name is not None:
            message = (
                "It seems that this object was already used to create an Amazon Model "
                "Monitoring Schedule. To create another, first delete the existing one "
                "using my_monitor.delete_monitoring_schedule()."
            )
            print(message)
            raise ValueError(message)

        self.monitoring_schedule_name = self._generate_monitoring_schedule_name(
            schedule_name=monitor_schedule_name
        )

        print()
        print("Creating Monitoring Schedule with name: {}".format(self.monitoring_schedule_name))

        normalized_endpoint_input = self._normalize_endpoint_input(endpoint_input=endpoint_input)

        record_preprocessor_script_s3_uri = None
        if record_preprocessor_script is not None:
            record_preprocessor_script_s3_uri = self._s3_uri_from_local_path(
                path=record_preprocessor_script
            )

        post_analytics_processor_script_s3_uri = None
        if post_analytics_processor_script is not None:
            post_analytics_processor_script_s3_uri = self._s3_uri_from_local_path(
                path=post_analytics_processor_script
            )

        normalized_monitoring_output = self._normalize_monitoring_output(
            output_s3_uri=output_s3_uri
        )

        statistics_object, constraints_object = self._get_baseline_files(
            statistics=statistics, constraints=constraints, sagemaker_session=self.sagemaker_session
        )

        constraints_s3_uri = None
        if constraints_object is not None:
            constraints_s3_uri = constraints_object.file_s3_uri

        statistics_s3_uri = None
        if statistics_object is not None:
            statistics_s3_uri = statistics_object.file_s3_uri

        normalized_env = self._generate_env_map(
            env=self.env, enable_cloudwatch_metrics=enable_cloudwatch_metrics
        )

        monitoring_output_config = {
            "MonitoringOutputs": [normalized_monitoring_output._to_request_dict()]
        }

        if self.output_kms_key is not None:
            monitoring_output_config["KmsKeyId"] = self.output_kms_key

        network_config_dict = None
        if self.network_config is not None:
            network_config_dict = self.network_config._to_request_dict()
            super(DefaultModelMonitor, self)._validate_network_config(network_config_dict)

        self.sagemaker_session.create_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name,
            schedule_expression=schedule_cron_expression,
            constraints_s3_uri=constraints_s3_uri,
            statistics_s3_uri=statistics_s3_uri,
            monitoring_inputs=[normalized_endpoint_input._to_request_dict()],
            monitoring_output_config=monitoring_output_config,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            image_uri=self.image_uri,
            entrypoint=self.entrypoint,
            arguments=self.arguments,
            record_preprocessor_source_uri=record_preprocessor_script_s3_uri,
            post_analytics_processor_source_uri=post_analytics_processor_script_s3_uri,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            environment=normalized_env,
            network_config=network_config_dict,
            role_arn=self.sagemaker_session.expand_role(self.role),
            tags=self.tags,
        )

    def update_monitoring_schedule(
        self,
        endpoint_input=None,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        statistics=None,
        constraints=None,
        schedule_cron_expression=None,
        instance_count=None,
        instance_type=None,
        volume_size_in_gb=None,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        env=None,
        network_config=None,
        enable_cloudwatch_metrics=None,
        role=None,
    ):
        """Updates the existing monitoring schedule.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            record_preprocessor_script (str): The path to the record preprocessor script. This can
                be a local path or an S3 uri.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): Desired S3 destination of the constraint_violations and
                statistics json files.
            statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
                constraints, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
                JSON file.
            constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
                statistics, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
                JSON file.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job runs at. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            env (dict): Environment variables to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.
            role (str): An AWS IAM role name or ARN. The Amazon SageMaker jobs use this role.

        """
        monitoring_inputs = None
        if endpoint_input is not None:
            monitoring_inputs = [self._normalize_endpoint_input(endpoint_input)._to_request_dict()]

        record_preprocessor_script_s3_uri = None
        if record_preprocessor_script is not None:
            record_preprocessor_script_s3_uri = self._s3_uri_from_local_path(
                path=record_preprocessor_script
            )

        post_analytics_processor_script_s3_uri = None
        if post_analytics_processor_script is not None:
            post_analytics_processor_script_s3_uri = self._s3_uri_from_local_path(
                path=post_analytics_processor_script
            )

        monitoring_output_config = None
        output_path = None
        if output_s3_uri is not None:
            normalized_monitoring_output = self._normalize_monitoring_output(
                output_s3_uri=output_s3_uri
            )
            monitoring_output_config = {
                "MonitoringOutputs": [normalized_monitoring_output._to_request_dict()]
            }
            output_path = normalized_monitoring_output.source

        if env is not None:
            self.env = env

        normalized_env = self._generate_env_map(
            env=env, output_path=output_path, enable_cloudwatch_metrics=enable_cloudwatch_metrics
        )

        statistics_object, constraints_object = self._get_baseline_files(
            statistics=statistics, constraints=constraints, sagemaker_session=self.sagemaker_session
        )

        statistics_s3_uri = None
        if statistics_object is not None:
            statistics_s3_uri = statistics_object.file_s3_uri

        constraints_s3_uri = None
        if constraints_object is not None:
            constraints_s3_uri = constraints_object.file_s3_uri

        if instance_type is not None:
            self.instance_type = instance_type

        if instance_count is not None:
            self.instance_count = instance_count

        if volume_size_in_gb is not None:
            self.volume_size_in_gb = volume_size_in_gb

        if volume_kms_key is not None:
            self.volume_kms_key = volume_kms_key

        if output_kms_key is not None:
            self.output_kms_key = output_kms_key
            monitoring_output_config["KmsKeyId"] = self.output_kms_key

        if max_runtime_in_seconds is not None:
            self.max_runtime_in_seconds = max_runtime_in_seconds

        if network_config is not None:
            self.network_config = network_config

        network_config_dict = None
        if self.network_config is not None:
            network_config_dict = self.network_config._to_request_dict()
            super(DefaultModelMonitor, self)._validate_network_config(network_config_dict)

        if role is not None:
            self.role = role

        self.sagemaker_session.update_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name,
            schedule_expression=schedule_cron_expression,
            constraints_s3_uri=constraints_s3_uri,
            statistics_s3_uri=statistics_s3_uri,
            monitoring_inputs=monitoring_inputs,
            monitoring_output_config=monitoring_output_config,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            record_preprocessor_source_uri=record_preprocessor_script_s3_uri,
            post_analytics_processor_source_uri=post_analytics_processor_script_s3_uri,
            max_runtime_in_seconds=max_runtime_in_seconds,
            environment=normalized_env,
            network_config=network_config_dict,
            role_arn=self.sagemaker_session.expand_role(self.role),
        )

        self._wait_for_schedule_changes_to_apply()

    def run_baseline(self):
        """Not implemented.

        '.run_baseline()' is only allowed for ModelMonitor objects. Please use
        `suggest_baseline` for DefaultModelMonitor objects, instead.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "'.run_baseline()' is only allowed for ModelMonitor objects. "
            "Please use suggest_baseline for DefaultModelMonitor objects, instead."
        )

    @classmethod
    def attach(cls, monitor_schedule_name, sagemaker_session=None):
        """Sets this object's schedule name to the name provided.

        This allows subsequent describe_schedule or list_executions calls to point
        to the given schedule.

        Args:
            monitor_schedule_name (str): The name of the schedule to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
        """
        sagemaker_session = sagemaker_session or Session()
        schedule_desc = sagemaker_session.describe_monitoring_schedule(
            monitoring_schedule_name=monitor_schedule_name
        )

        role = schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["RoleArn"]
        instance_count = schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringResources"
        ]["ClusterConfig"]["InstanceCount"]
        instance_type = schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringResources"
        ]["ClusterConfig"]["InstanceType"]
        volume_size_in_gb = schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringResources"
        ]["ClusterConfig"]["VolumeSizeInGB"]
        volume_kms_key = schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringResources"
        ]["ClusterConfig"].get("VolumeKmsKeyId")
        output_kms_key = schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringOutputConfig"
        ].get("KmsKeyId")

        max_runtime_in_seconds = None
        if schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"].get(
            "StoppingCondition"
        ):
            max_runtime_in_seconds = schedule_desc["MonitoringScheduleConfig"][
                "MonitoringJobDefinition"
            ]["StoppingCondition"].get("MaxRuntimeInSeconds")

        env = schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["Environment"]

        network_config_dict = schedule_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ].get("NetworkConfig")

        vpc_config = None
        if (
            schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"].get(
                "NetworkConfig"
            )
            is not None
        ):
            vpc_config = schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "NetworkConfig"
            ].get("VpcConfig")

        security_group_ids = None
        if vpc_config is not None:
            security_group_ids = vpc_config["SecurityGroupIds"]

        subnets = None
        if vpc_config is not None:
            subnets = vpc_config["Subnets"]

        network_config = None
        if network_config_dict:
            network_config = NetworkConfig(
                enable_network_isolation=network_config_dict["EnableNetworkIsolation"],
                security_group_ids=security_group_ids,
                subnets=subnets,
            )

        tags = sagemaker_session.list_tags(resource_arn=schedule_desc["MonitoringScheduleArn"])

        attached_monitor = cls(
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )
        attached_monitor.monitoring_schedule_name = monitor_schedule_name
        return attached_monitor

    def latest_monitoring_statistics(self):
        """Returns the sagemaker.model_monitor.Statistics.

        These are the statistics generated by the latest monitoring execution.

        Returns:
            sagemaker.model_monitoring.Statistics: The Statistics object representing the file
                generated by the latest monitoring execution.

        """
        executions = self.list_executions()
        if len(executions) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return None

        latest_monitoring_execution = executions[-1]

        try:
            return latest_monitoring_execution.statistics()
        except ClientError:
            status = latest_monitoring_execution.describe()["ProcessingJobStatus"]
            print(
                "Unable to retrieve statistics as job is in status '{}'. Latest statistics only "
                "available for completed executions.".format(status)
            )

    def latest_monitoring_constraint_violations(self):
        """Returns the sagemaker.model_monitor.ConstraintViolations generated by the latest
        monitoring execution.

        Returns:
            sagemaker.model_monitoring.ConstraintViolations: The ConstraintViolations object
                representing the file generated by the latest monitoring execution.

        """
        executions = self.list_executions()
        if len(executions) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return None

        latest_monitoring_execution = executions[-1]
        try:
            return latest_monitoring_execution.constraint_violations()
        except ClientError:
            status = latest_monitoring_execution.describe()["ProcessingJobStatus"]
            print(
                "Unable to retrieve constraint violations as job is in status '{}'. Latest "
                "violations only available for completed executions.".format(status)
            )

    def _normalize_baseline_output(self, output_s3_uri=None):
        """Ensure that the output is a ProcessingOutput object.

        Args:
            output_s3_uri (str): The output S3 uri to deposit the baseline files in.

        Returns:
            sagemaker.processing.ProcessingOutput: The normalized ProcessingOutput object.

        """
        s3_uri = output_s3_uri or s3.s3_path_join(
            "s3://",
            self.sagemaker_session.default_bucket(),
            _MODEL_MONITOR_S3_PATH,
            _BASELINING_S3_PATH,
            self.latest_baselining_job_name,
            _RESULTS_S3_PATH,
        )
        return ProcessingOutput(
            source=str(pathlib.PurePosixPath(_CONTAINER_BASE_PATH, _CONTAINER_OUTPUT_PATH)),
            destination=s3_uri,
            output_name=_DEFAULT_OUTPUT_NAME,
        )

    def _normalize_monitoring_output(self, output_s3_uri=None):
        """Ensure that the output is a MonitoringOutput object.

        Args:
            output_s3_uri (str): The output S3 uri to deposit the monitoring evaluation files in.

        Returns:
            sagemaker.model_monitor.MonitoringOutput: The normalized MonitoringOutput object.

        """
        s3_uri = output_s3_uri or s3.s3_path_join(
            "s3://",
            self.sagemaker_session.default_bucket(),
            _MODEL_MONITOR_S3_PATH,
            _MONITORING_S3_PATH,
            self.monitoring_schedule_name,
            _RESULTS_S3_PATH,
        )
        output = MonitoringOutput(
            source=str(pathlib.PurePosixPath(_CONTAINER_BASE_PATH, _CONTAINER_OUTPUT_PATH)),
            destination=s3_uri,
        )

        return output

    @staticmethod
    def _generate_env_map(
        env,
        output_path=None,
        enable_cloudwatch_metrics=None,
        record_preprocessor_script_container_path=None,
        post_processor_script_container_path=None,
        dataset_format=None,
        dataset_source_container_path=None,
    ):
        """Generate a list of environment variables from first-class parameters.

        Args:
            dataset_format (dict): The format of the baseline_dataset.
            output_path (str): Local path to the output.
            record_preprocessor_script_container_path (str): The path to the record preprocessor
                script.
            post_processor_script_container_path (str): The path to the post analytics processor
                script.
            dataset_source_container_path (str): The path to the dataset source.

        Returns:
            dict: Dictionary of environment keys and values.

        """
        cloudwatch_env_map = {True: "Enabled", False: "Disabled"}

        if env is not None:
            env = copy.deepcopy(env)
        env = env or {}

        if output_path is not None:
            env[_OUTPUT_PATH_ENV_NAME] = output_path

        if enable_cloudwatch_metrics is not None:
            env[_PUBLISH_CLOUDWATCH_METRICS_ENV_NAME] = cloudwatch_env_map[
                enable_cloudwatch_metrics
            ]

        if dataset_format is not None:
            env[_DATASET_FORMAT_ENV_NAME] = json.dumps(dataset_format)

        if record_preprocessor_script_container_path is not None:
            env[_RECORD_PREPROCESSOR_SCRIPT_ENV_NAME] = record_preprocessor_script_container_path

        if post_processor_script_container_path is not None:
            env[_POST_ANALYTICS_PROCESSOR_SCRIPT_ENV_NAME] = post_processor_script_container_path

        if dataset_source_container_path is not None:
            env[_DATASET_SOURCE_PATH_ENV_NAME] = dataset_source_container_path

        return env

    def _upload_and_convert_to_processing_input(self, source, destination, name):
        """Generates a ProcessingInput object from a source. Source can be a local path or an S3
        uri.

        Args:
            source (str): The source of the data. This can be a local path or an S3 uri.
            destination (str): The desired container path for the data to be downloaded to.
            name (str): The name of the ProcessingInput.

        Returns:
            sagemaker.processing.ProcessingInput: The ProcessingInput object.

        """
        if source is None:
            return None

        parse_result = urlparse(url=source)

        if parse_result.scheme != "s3":
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                _MODEL_MONITOR_S3_PATH,
                _BASELINING_S3_PATH,
                self.latest_baselining_job_name,
                _INPUT_S3_PATH,
                name,
            )
            s3.S3Uploader.upload(
                local_path=source, desired_s3_uri=s3_uri, sagemaker_session=self.sagemaker_session
            )
            source = s3_uri

        return ProcessingInput(source=source, destination=destination, input_name=name)

    @staticmethod
    def _get_default_image_uri(region):
        """Returns the Default Model Monitoring image uri based on the region.

        Args:
            region (str): The AWS region.

        Returns:
            str: The Default Model Monitoring image uri based on the region.
        """
        return image_uris.retrieve(framework=framework_name, region=region)


class BaseliningJob(ProcessingJob):
    """Provides functionality to retrieve baseline-specific files output from baselining job."""

    def __init__(self, sagemaker_session, job_name, inputs, outputs, output_kms_key=None):
        """Initializes a Baselining job that tracks a baselining job kicked off by the suggest
        workflow.

        Args:
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            job_name (str): Name of the Amazon SageMaker Model Monitoring Baselining Job.
            inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput objects.
            outputs ([sagemaker.processing.ProcessingOutput]): A list of ProcessingOutput objects.
            output_kms_key (str): The output kms key associated with the job. Defaults to None
                if not provided.

        """
        self.inputs = inputs
        self.outputs = outputs
        super(BaseliningJob, self).__init__(
            sagemaker_session=sagemaker_session,
            job_name=job_name,
            inputs=inputs,
            outputs=outputs,
            output_kms_key=output_kms_key,
        )

    @classmethod
    def from_processing_job(cls, processing_job):
        """Initializes a Baselining job from a processing job.

        Args:
            processing_job (sagemaker.processing.ProcessingJob): The ProcessingJob used for
                baselining instance.

        Returns:
            sagemaker.processing.BaseliningJob: The instance of ProcessingJob created
                using the current job name.

        """
        return cls(
            processing_job.sagemaker_session,
            processing_job.job_name,
            processing_job.inputs,
            processing_job.outputs,
            processing_job.output_kms_key,
        )

    def baseline_statistics(self, file_name=STATISTICS_JSON_DEFAULT_FILE_NAME, kms_key=None):
        """Returns a sagemaker.model_monitor.Statistics object representing the statistics
        JSON file generated by this baselining job.

        Args:
            file_name (str): The name of the json-formatted statistics file
            kms_key (str): The kms key to use when retrieving the file.

        Returns:
            sagemaker.model_monitor.Statistics: The Statistics object representing the file that
                was generated by the job.

        Raises:
            UnexpectedStatusException: This is thrown if the job is not in a 'Complete' state.

        """
        try:
            baselining_job_output_s3_path = self.outputs[0].destination
            return Statistics.from_s3_uri(
                statistics_file_s3_uri=s3.s3_path_join(baselining_job_output_s3_path, file_name),
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        except ClientError as client_error:
            if client_error.response["Error"]["Code"] == "NoSuchKey":
                status = self.sagemaker_session.describe_processing_job(job_name=self.job_name)[
                    "ProcessingJobStatus"
                ]
                if status != "Completed":
                    raise UnexpectedStatusException(
                        message="The underlying job is not in 'Completed' state. You may only "
                        "retrieve files for a job that has completed successfully.",
                        allowed_statuses="Completed",
                        actual_status=status,
                    )
            else:
                raise client_error

    def suggested_constraints(self, file_name=CONSTRAINTS_JSON_DEFAULT_FILE_NAME, kms_key=None):
        """Returns a sagemaker.model_monitor.Constraints object representing the constraints
        JSON file generated by this baselining job.

        Args:
            file_name (str): The name of the json-formatted constraints file
            kms_key (str): The kms key to use when retrieving the file.

        Returns:
            sagemaker.model_monitor.Constraints: The Constraints object representing the file that
                was generated by the job.

        Raises:
            UnexpectedStatusException: This is thrown if the job is not in a 'Complete' state.

        """
        try:
            baselining_job_output_s3_path = self.outputs[0].destination
            return Constraints.from_s3_uri(
                constraints_file_s3_uri=s3.s3_path_join(baselining_job_output_s3_path, file_name),
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        except ClientError as client_error:
            if client_error.response["Error"]["Code"] == "NoSuchKey":
                status = self.sagemaker_session.describe_processing_job(job_name=self.job_name)[
                    "ProcessingJobStatus"
                ]
                if status != "Completed":
                    raise UnexpectedStatusException(
                        message="The underlying job is not in 'Completed' state. You may only "
                        "retrieve files for a job that has completed successfully.",
                        allowed_statuses="Completed",
                        actual_status=status,
                    )
            else:
                raise client_error


class MonitoringExecution(ProcessingJob):
    """Provides functionality to retrieve monitoring-specific files output from monitoring
    executions
    """

    def __init__(self, sagemaker_session, job_name, inputs, output, output_kms_key=None):
        """Initializes a MonitoringExecution job that tracks a monitoring execution kicked off by
        an Amazon SageMaker Model Monitoring Schedule.

        Args:
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            job_name (str): The name of the monitoring execution job.
            output (sagemaker.Processing.ProcessingOutput): The output associated with the
                monitoring execution.
            output_kms_key (str): The output kms key associated with the job. Defaults to None
                if not provided.

        """
        self.output = output
        super(MonitoringExecution, self).__init__(
            sagemaker_session=sagemaker_session,
            job_name=job_name,
            inputs=inputs,
            outputs=[output],
            output_kms_key=output_kms_key,
        )

    @classmethod
    def from_processing_arn(cls, sagemaker_session, processing_job_arn):
        """Initializes a Baselining job from a processing arn.

        Args:
            processing_job_arn (str): ARN of the processing job to create a MonitoringExecution
            out of.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.

        Returns:
            sagemaker.processing.BaseliningJob: The instance of ProcessingJob created
                using the current job name.

        """
        processing_job_name = processing_job_arn.split(":")[5][
            len("processing-job/") :
        ]  # This is necessary while the API only vends an arn.
        job_desc = sagemaker_session.describe_processing_job(job_name=processing_job_name)

        return cls(
            sagemaker_session=sagemaker_session,
            job_name=processing_job_name,
            inputs=[
                ProcessingInput(
                    source=processing_input["S3Input"]["S3Uri"],
                    destination=processing_input["S3Input"]["LocalPath"],
                    input_name=processing_input["InputName"],
                    s3_data_type=processing_input["S3Input"].get("S3DataType"),
                    s3_input_mode=processing_input["S3Input"].get("S3InputMode"),
                    s3_data_distribution_type=processing_input["S3Input"].get(
                        "S3DataDistributionType"
                    ),
                    s3_compression_type=processing_input["S3Input"].get("S3CompressionType"),
                )
                for processing_input in job_desc["ProcessingInputs"]
            ],
            output=ProcessingOutput(
                source=job_desc["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["LocalPath"],
                destination=job_desc["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
                output_name=job_desc["ProcessingOutputConfig"]["Outputs"][0]["OutputName"],
            ),
            output_kms_key=job_desc["ProcessingOutputConfig"].get("KmsKeyId"),
        )

    def statistics(self, file_name=STATISTICS_JSON_DEFAULT_FILE_NAME, kms_key=None):
        """Returns a sagemaker.model_monitor.Statistics object representing the statistics
        JSON file generated by this monitoring execution.

        Args:
            file_name (str): The name of the json-formatted statistics file
            kms_key (str): The kms key to use when retrieving the file.

        Returns:
            sagemaker.model_monitor.Statistics: The Statistics object representing the file that
                was generated by the execution.

        Raises:
            UnexpectedStatusException: This is thrown if the job is not in a 'Complete' state.

        """
        try:
            baselining_job_output_s3_path = self.outputs[0].destination
            return Statistics.from_s3_uri(
                statistics_file_s3_uri=s3.s3_path_join(baselining_job_output_s3_path, file_name),
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        except ClientError as client_error:
            if client_error.response["Error"]["Code"] == "NoSuchKey":
                status = self.sagemaker_session.describe_processing_job(job_name=self.job_name)[
                    "ProcessingJobStatus"
                ]
                if status != "Completed":
                    raise UnexpectedStatusException(
                        message="The underlying job is not in 'Completed' state. You may only "
                        "retrieve files for a job that has completed successfully.",
                        allowed_statuses="Completed",
                        actual_status=status,
                    )
            else:
                raise client_error

    def constraint_violations(
        self, file_name=CONSTRAINT_VIOLATIONS_JSON_DEFAULT_FILE_NAME, kms_key=None
    ):
        """Returns a sagemaker.model_monitor.ConstraintViolations object representing the
        constraint violations JSON file generated by this monitoring execution.

        Args:
            file_name (str): The name of the json-formatted constraint violations file.
            kms_key (str): The kms key to use when retrieving the file.

        Returns:
            sagemaker.model_monitor.ConstraintViolations: The ConstraintViolations object
                representing the file that was generated by the monitoring execution.

        Raises:
            UnexpectedStatusException: This is thrown if the job is not in a 'Complete' state.

        """
        try:
            baselining_job_output_s3_path = self.outputs[0].destination
            return ConstraintViolations.from_s3_uri(
                constraint_violations_file_s3_uri=s3.s3_path_join(
                    baselining_job_output_s3_path, file_name
                ),
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        except ClientError as client_error:
            if client_error.response["Error"]["Code"] == "NoSuchKey":
                status = self.sagemaker_session.describe_processing_job(job_name=self.job_name)[
                    "ProcessingJobStatus"
                ]
                if status != "Completed":
                    raise UnexpectedStatusException(
                        message="The underlying job is not in 'Completed' state. You may only "
                        "retrieve files for a job that has completed successfully.",
                        allowed_statuses="Completed",
                        actual_status=status,
                    )
            else:
                raise client_error


class EndpointInput(object):
    """Accepts parameters that specify an endpoint input for monitoring execution.

    It also provides a method to turn those parameters into a dictionary.
    """

    def __init__(
        self,
        endpoint_name,
        destination,
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated",
    ):
        """Initialize an ``EndpointInput`` instance. EndpointInput accepts parameters
        that specify an endpoint input for a monitoring job and provides a method
        to turn those parameters into a dictionary.

        Args:
            endpoint_name (str): The name of the endpoint.
            destination (str): The destination of the input.
            s3_input_mode (str): The S3 input mode. Can be one of: "File", "Pipe. Default: "File".
            s3_data_distribution_type (str): The S3 Data Distribution Type. Can be one of:
                "FullyReplicated", "ShardedByS3Key"

        """
        self.endpoint_name = endpoint_name
        self.destination = destination
        self.s3_input_mode = s3_input_mode
        self.s3_data_distribution_type = s3_data_distribution_type

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        endpoint_input_request = {
            "EndpointInput": {
                "EndpointName": self.endpoint_name,
                "LocalPath": self.destination,
                "S3InputMode": self.s3_input_mode,
                "S3DataDistributionType": self.s3_data_distribution_type,
            }
        }

        return endpoint_input_request


class MonitoringOutput(object):
    """Accepts parameters that specify an S3 output for a monitoring job.

    It also provides a method to turn those parameters into a dictionary.
    """

    def __init__(self, source, destination=None, s3_upload_mode="Continuous"):
        """Initialize a ``MonitoringOutput`` instance. MonitoringOutput accepts parameters that
        specify an S3 output for a monitoring job and provides a method to turn
        those parameters into a dictionary.

        Args:
            source (str): The source for the output.
            destination (str): The destination of the output. Optional.
                Default: s3://<default-session-bucket/schedule_name/output
            s3_upload_mode (str): The S3 upload mode.

        """
        self.source = source
        self.destination = destination
        self.s3_upload_mode = s3_upload_mode

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class.

        Returns:
            dict: The request dictionary.

        """
        s3_output_request = {
            "S3Output": {
                "S3Uri": self.destination,
                "LocalPath": self.source,
                "S3UploadMode": self.s3_upload_mode,
            }
        }

        return s3_output_request
