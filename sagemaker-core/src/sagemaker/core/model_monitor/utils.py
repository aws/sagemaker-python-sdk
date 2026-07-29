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
"""Placeholder docstring"""
from __future__ import absolute_import, annotations, print_function

import json
from typing import Dict, Optional

from sagemaker.core._studio import _append_project_tags
from sagemaker.core.config.config_schema import (
    MONITORING_JOB_ENVIRONMENT_PATH,
    MONITORING_JOB_ROLE_ARN_PATH,
    MONITORING_JOB_VOLUME_KMS_KEY_ID_PATH,
    MONITORING_JOB_NETWORK_CONFIG_PATH,
    MONITORING_JOB_OUTPUT_KMS_KEY_ID_PATH,
    MONITORING_SCHEDULE,
    MONITORING_SCHEDULE_INTER_CONTAINER_ENCRYPTION_PATH,
    KMS_KEY_ID,
    SAGEMAKER,
    TAGS,
)
from sagemaker.core.common_utils import (
    resolve_value_from_config,
    resolve_nested_dict_value_from_config,
    update_nested_dictionary_with_values_from_config,
    format_tags,
)
from sagemaker.core.config.config_utils import _append_sagemaker_config_tags
import logging

# Setting LOGGER for backward compatibility, in case users import it...
logger = LOGGER = logging.getLogger("sagemaker")

MODEL_MONITOR_ONE_TIME_SCHEDULE = "NOW"


def boto_create_monitoring_schedule(
    sagemaker_session,
    monitoring_schedule_name,
    schedule_expression,
    statistics_s3_uri,
    constraints_s3_uri,
    monitoring_inputs,
    monitoring_output_config,
    instance_count,
    instance_type,
    volume_size_in_gb,
    volume_kms_key=None,
    image_uri=None,
    entrypoint=None,
    arguments=None,
    record_preprocessor_source_uri=None,
    post_analytics_processor_source_uri=None,
    max_runtime_in_seconds=None,
    environment=None,
    network_config=None,
    role_arn=None,
    tags=None,
    data_analysis_start_time=None,
    data_analysis_end_time=None,
):
    """Create an Amazon SageMaker monitoring schedule.

    Args:
        monitoring_schedule_name (str): The name of the monitoring schedule. The name must be
            unique within an AWS Region in an AWS account. Names should have a minimum length
            of 1 and a maximum length of 63 characters.
        schedule_expression (str): The cron expression that dictates the monitoring execution
            schedule.
        statistics_s3_uri (str): The S3 uri of the statistics file to use.
        constraints_s3_uri (str): The S3 uri of the constraints file to use.
        monitoring_inputs ([dict]): List of MonitoringInput dictionaries.
        monitoring_output_config (dict): A config dictionary, which contains a list of
            MonitoringOutput dictionaries, as well as an optional KMS key ID.
        instance_count (int): The number of instances to run.
        instance_type (str): The type of instance to run.
        volume_size_in_gb (int): Size of the volume in GB.
        volume_kms_key (str): KMS key to use when encrypting the volume.
        image_uri (str): The image uri to use for monitoring executions.
        entrypoint (str): The entrypoint to the monitoring execution image.
        arguments (str): The arguments to pass to the monitoring execution image.
        record_preprocessor_source_uri (str or None): The S3 uri that points to the script that
            pre-processes the dataset (only applicable to first-party images).
        post_analytics_processor_source_uri (str or None): The S3 uri that points to the script
            that post-processes the dataset (only applicable to first-party images).
        max_runtime_in_seconds (int): Specifies a limit to how long
            the processing job can run, in seconds.
        environment (dict): Environment variables to start the monitoring execution
            container with.
        network_config (dict): Specifies networking options, such as network
            traffic encryption between processing containers, whether to allow
            inbound and outbound network calls to and from processing containers,
            and VPC subnets and security groups to use for VPC-enabled processing
            jobs.
        role_arn (str): The Amazon Resource Name (ARN) of an IAM role that
            Amazon SageMaker can assume to perform tasks on your behalf.
        tags (Optional[Tags]): A list of dictionaries containing key-value
            pairs.
        data_analysis_start_time (str): Start time for the data analysis window
            for the one time monitoring schedule (NOW), e.g. "-PT1H"
        data_analysis_end_time (str): End time for the data analysis window
            for the one time monitoring schedule (NOW), e.g. "-PT1H"
    """
    role_arn = resolve_value_from_config(
        role_arn, MONITORING_JOB_ROLE_ARN_PATH, sagemaker_session=sagemaker_session
    )
    volume_kms_key = resolve_value_from_config(
        volume_kms_key, MONITORING_JOB_VOLUME_KMS_KEY_ID_PATH, sagemaker_session=sagemaker_session
    )
    inferred_network_config_from_config = update_nested_dictionary_with_values_from_config(
        network_config, MONITORING_JOB_NETWORK_CONFIG_PATH, sagemaker_session=sagemaker_session
    )
    environment = resolve_value_from_config(
        direct_input=environment,
        config_path=MONITORING_JOB_ENVIRONMENT_PATH,
        default_value=None,
        sagemaker_session=sagemaker_session,
    )
    monitoring_schedule_request = {
        "MonitoringScheduleName": monitoring_schedule_name,
        "MonitoringScheduleConfig": {
            "MonitoringJobDefinition": {
                "Environment": environment,
                "MonitoringInputs": monitoring_inputs,
                "MonitoringResources": {
                    "ClusterConfig": {
                        "InstanceCount": instance_count,
                        "InstanceType": instance_type,
                        "VolumeSizeInGB": volume_size_in_gb,
                    }
                },
                "MonitoringAppSpecification": {"ImageUri": image_uri},
                "RoleArn": role_arn,
            }
        },
    }

    if schedule_expression is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"] = {
            "ScheduleExpression": schedule_expression,
        }
        if data_analysis_start_time is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"][
                "DataAnalysisStartTime"
            ] = data_analysis_start_time

        if data_analysis_end_time is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"][
                "DataAnalysisEndTime"
            ] = data_analysis_end_time

    if monitoring_output_config is not None:
        kms_key_from_config = resolve_value_from_config(
            config_path=MONITORING_JOB_OUTPUT_KMS_KEY_ID_PATH, sagemaker_session=sagemaker_session
        )
        if KMS_KEY_ID not in monitoring_output_config and kms_key_from_config:
            monitoring_output_config[KMS_KEY_ID] = kms_key_from_config
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringOutputConfig"
        ] = monitoring_output_config

    if statistics_s3_uri is not None or constraints_s3_uri is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "BaselineConfig"
        ] = {}

    if statistics_s3_uri is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "BaselineConfig"
        ]["StatisticsResource"] = {"S3Uri": statistics_s3_uri}

    if constraints_s3_uri is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "BaselineConfig"
        ]["ConstraintsResource"] = {"S3Uri": constraints_s3_uri}

    if record_preprocessor_source_uri is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["RecordPreprocessorSourceUri"] = record_preprocessor_source_uri

    if post_analytics_processor_source_uri is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["PostAnalyticsProcessorSourceUri"] = post_analytics_processor_source_uri

    if entrypoint is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["ContainerEntrypoint"] = entrypoint

    if arguments is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["ContainerArguments"] = arguments

    if volume_kms_key is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringResources"
        ]["ClusterConfig"]["VolumeKmsKeyId"] = volume_kms_key

    if max_runtime_in_seconds is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "StoppingCondition"
        ] = {"MaxRuntimeInSeconds": max_runtime_in_seconds}

    if environment is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "Environment"
        ] = environment

    if inferred_network_config_from_config is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "NetworkConfig"
        ] = inferred_network_config_from_config

    tags = _append_project_tags(format_tags(tags))
    tags = _append_sagemaker_config_tags(
        sagemaker_session, tags, "{}.{}.{}".format(SAGEMAKER, MONITORING_SCHEDULE, TAGS)
    )

    if tags is not None:
        monitoring_schedule_request["Tags"] = tags

    logger.info("Creating monitoring schedule name %s.", monitoring_schedule_name)
    logger.debug(
        "monitoring_schedule_request= %s", json.dumps(monitoring_schedule_request, indent=4)
    )
    sagemaker_session.sagemaker_client.create_monitoring_schedule(**monitoring_schedule_request)


def boto_update_monitoring_schedule(
    sagemaker_session,
    monitoring_schedule_name,
    schedule_expression=None,
    statistics_s3_uri=None,
    constraints_s3_uri=None,
    monitoring_inputs=None,
    monitoring_output_config=None,
    instance_count=None,
    instance_type=None,
    volume_size_in_gb=None,
    volume_kms_key=None,
    image_uri=None,
    entrypoint=None,
    arguments=None,
    record_preprocessor_source_uri=None,
    post_analytics_processor_source_uri=None,
    max_runtime_in_seconds=None,
    environment=None,
    network_config=None,
    role_arn=None,
    data_analysis_start_time=None,
    data_analysis_end_time=None,
):
    """Update an Amazon SageMaker monitoring schedule.

    Args:
        monitoring_schedule_name (str): The name of the monitoring schedule. The name must be
            unique within an AWS Region in an AWS account. Names should have a minimum length
            of 1 and a maximum length of 63 characters.
        schedule_expression (str): The cron expression that dictates the monitoring execution
            schedule.
        statistics_s3_uri (str): The S3 uri of the statistics file to use.
        constraints_s3_uri (str): The S3 uri of the constraints file to use.
        monitoring_inputs ([dict]): List of MonitoringInput dictionaries.
        monitoring_output_config (dict): A config dictionary, which contains a list of
            MonitoringOutput dictionaries, as well as an optional KMS key ID.
        instance_count (int): The number of instances to run.
        instance_type (str): The type of instance to run.
        volume_size_in_gb (int): Size of the volume in GB.
        volume_kms_key (str): KMS key to use when encrypting the volume.
        image_uri (str): The image uri to use for monitoring executions.
        entrypoint (str): The entrypoint to the monitoring execution image.
        arguments (str): The arguments to pass to the monitoring execution image.
        record_preprocessor_source_uri (str or None): The S3 uri that points to the script that
            pre-processes the dataset (only applicable to first-party images).
        post_analytics_processor_source_uri (str or None): The S3 uri that points to the script
            that post-processes the dataset (only applicable to first-party images).
        max_runtime_in_seconds (int): Specifies a limit to how long
            the processing job can run, in seconds.
        environment (dict): Environment variables to start the monitoring execution
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
        data_analysis_start_time (str): Start time for the data analysis window
            for the one time monitoring schedule (NOW), e.g. "-PT1H"
        data_analysis_end_time (str): End time for the data analysis window
            for the one time monitoring schedule (NOW), e.g. "-PT1H"
    """
    existing_desc = sagemaker_session.sagemaker_client.describe_monitoring_schedule(
        MonitoringScheduleName=monitoring_schedule_name
    )

    existing_schedule_config = None
    existing_data_analysis_start_time = None
    existing_data_analysis_end_time = None

    if (
        existing_desc.get("MonitoringScheduleConfig") is not None
        and existing_desc["MonitoringScheduleConfig"].get("ScheduleConfig") is not None
        and existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"]
        is not None
    ):
        existing_schedule_config = existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"][
            "ScheduleExpression"
        ]
        if (
            existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"].get("DataAnalysisStartTime")
            is not None
        ):
            existing_data_analysis_start_time = existing_desc["MonitoringScheduleConfig"][
                "ScheduleConfig"
            ]["DataAnalysisStartTime"]
        if (
            existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"].get("DataAnalysisEndTime")
            is not None
        ):
            existing_data_analysis_end_time = existing_desc["MonitoringScheduleConfig"][
                "ScheduleConfig"
            ]["DataAnalysisEndTime"]

    request_schedule_expression = schedule_expression or existing_schedule_config
    request_data_analysis_start_time = data_analysis_start_time or existing_data_analysis_start_time
    request_data_analysis_end_time = data_analysis_end_time or existing_data_analysis_end_time

    if request_schedule_expression == MODEL_MONITOR_ONE_TIME_SCHEDULE and (
        request_data_analysis_start_time is None or request_data_analysis_end_time is None
    ):
        message = (
            "Both data_analysis_start_time and data_analysis_end_time are required "
            "for one time monitoring schedule "
        )
        LOGGER.error(message)
        raise ValueError(message)

    request_monitoring_inputs = (
        monitoring_inputs
        or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["MonitoringInputs"]
    )
    request_instance_count = (
        instance_count
        or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringResources"
        ]["ClusterConfig"]["InstanceCount"]
    )
    request_instance_type = (
        instance_type
        or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringResources"
        ]["ClusterConfig"]["InstanceType"]
    )
    request_volume_size_in_gb = (
        volume_size_in_gb
        or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringResources"
        ]["ClusterConfig"]["VolumeSizeInGB"]
    )
    request_image_uri = (
        image_uri
        or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["ImageUri"]
    )
    request_role_arn = (
        role_arn or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["RoleArn"]
    )

    monitoring_schedule_request = {
        "MonitoringScheduleName": monitoring_schedule_name,
        "MonitoringScheduleConfig": {
            "MonitoringJobDefinition": {
                "MonitoringInputs": request_monitoring_inputs,
                "MonitoringResources": {
                    "ClusterConfig": {
                        "InstanceCount": request_instance_count,
                        "InstanceType": request_instance_type,
                        "VolumeSizeInGB": request_volume_size_in_gb,
                    }
                },
                "MonitoringAppSpecification": {"ImageUri": request_image_uri},
                "RoleArn": request_role_arn,
            }
        },
    }

    if existing_schedule_config is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"] = {
            "ScheduleExpression": request_schedule_expression,
        }

        if request_data_analysis_start_time is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"][
                "DataAnalysisStartTime"
            ] = request_data_analysis_start_time

        if request_data_analysis_end_time is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"][
                "DataAnalysisEndTime"
            ] = request_data_analysis_end_time

    existing_monitoring_output_config = existing_desc["MonitoringScheduleConfig"][
        "MonitoringJobDefinition"
    ].get("MonitoringOutputConfig")
    if monitoring_output_config is not None or existing_monitoring_output_config is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringOutputConfig"
        ] = (monitoring_output_config or existing_monitoring_output_config)

    existing_statistics_s3_uri = None
    existing_constraints_s3_uri = None
    if (
        existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"].get("BaselineConfig")
        is not None
    ):
        if (
            existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "BaselineConfig"
            ].get("StatisticsResource")
            is not None
        ):
            existing_statistics_s3_uri = existing_desc["MonitoringScheduleConfig"][
                "MonitoringJobDefinition"
            ]["BaselineConfig"]["StatisticsResource"]["S3Uri"]

        if (
            existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "BaselineConfig"
            ].get("ConstraintsResource")
            is not None
        ):
            existing_statistics_s3_uri = existing_desc["MonitoringScheduleConfig"][
                "MonitoringJobDefinition"
            ]["BaselineConfig"]["ConstraintsResource"]["S3Uri"]

    if (
        statistics_s3_uri is not None
        or constraints_s3_uri is not None
        or existing_statistics_s3_uri is not None
        or existing_constraints_s3_uri is not None
    ):
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "BaselineConfig"
        ] = {}

    if statistics_s3_uri is not None or existing_statistics_s3_uri is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "BaselineConfig"
        ]["StatisticsResource"] = {"S3Uri": statistics_s3_uri or existing_statistics_s3_uri}

    if constraints_s3_uri is not None or existing_constraints_s3_uri is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "BaselineConfig"
        ]["ConstraintsResource"] = {"S3Uri": constraints_s3_uri or existing_constraints_s3_uri}

    existing_record_preprocessor_source_uri = existing_desc["MonitoringScheduleConfig"][
        "MonitoringJobDefinition"
    ]["MonitoringAppSpecification"].get("RecordPreprocessorSourceUri")
    if (
        record_preprocessor_source_uri is not None
        or existing_record_preprocessor_source_uri is not None
    ):
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["RecordPreprocessorSourceUri"] = (
            record_preprocessor_source_uri or existing_record_preprocessor_source_uri
        )

    existing_post_analytics_processor_source_uri = existing_desc["MonitoringScheduleConfig"][
        "MonitoringJobDefinition"
    ]["MonitoringAppSpecification"].get("PostAnalyticsProcessorSourceUri")
    if (
        post_analytics_processor_source_uri is not None
        or existing_post_analytics_processor_source_uri is not None
    ):
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["PostAnalyticsProcessorSourceUri"] = (
            post_analytics_processor_source_uri or existing_post_analytics_processor_source_uri
        )

    existing_entrypoint = existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
        "MonitoringAppSpecification"
    ].get("ContainerEntrypoint")
    if entrypoint is not None or existing_entrypoint is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["ContainerEntrypoint"] = (entrypoint or existing_entrypoint)

    existing_arguments = existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
        "MonitoringAppSpecification"
    ].get("ContainerArguments")
    if arguments is not None or existing_arguments is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["ContainerArguments"] = (arguments or existing_arguments)

    existing_volume_kms_key = existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
        "MonitoringResources"
    ]["ClusterConfig"].get("VolumeKmsKeyId")

    if volume_kms_key is not None or existing_volume_kms_key is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringResources"
        ]["ClusterConfig"]["VolumeKmsKeyId"] = (volume_kms_key or existing_volume_kms_key)

    existing_max_runtime_in_seconds = None
    if existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"].get(
        "StoppingCondition"
    ):
        existing_max_runtime_in_seconds = existing_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ]["StoppingCondition"].get("MaxRuntimeInSeconds")

    if max_runtime_in_seconds is not None or existing_max_runtime_in_seconds is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "StoppingCondition"
        ] = {"MaxRuntimeInSeconds": max_runtime_in_seconds or existing_max_runtime_in_seconds}

    existing_environment = existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"].get(
        "Environment"
    )
    if environment is not None or existing_environment is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "Environment"
        ] = (environment or existing_environment)

    existing_network_config = existing_desc["MonitoringScheduleConfig"][
        "MonitoringJobDefinition"
    ].get("NetworkConfig")

    _network_config = network_config or existing_network_config
    _network_config = resolve_nested_dict_value_from_config(
        _network_config,
        ["EnableInterContainerTrafficEncryption"],
        MONITORING_SCHEDULE_INTER_CONTAINER_ENCRYPTION_PATH,
        sagemaker_session=sagemaker_session,
    )
    if _network_config is not None:
        monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "NetworkConfig"
        ] = _network_config

    logger.info("Updating monitoring schedule with name: %s .", monitoring_schedule_name)
    logger.debug(
        "monitoring_schedule_request= %s", json.dumps(monitoring_schedule_request, indent=4)
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule(**monitoring_schedule_request)


def boto_start_monitoring_schedule(sagemaker_session, monitoring_schedule_name):
    """Starts a monitoring schedule.

    Args:
        monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
            Schedule to start.
    """
    logger.info("Starting Monitoring Schedule with name: %s", monitoring_schedule_name)
    sagemaker_session.sagemaker_client.start_monitoring_schedule(
        MonitoringScheduleName=monitoring_schedule_name
    )


def boto_stop_monitoring_schedule(sagemaker_session, monitoring_schedule_name):
    """Stops a monitoring schedule.

    Args:
        monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
            Schedule to stop.
    """
    logger.info("Stopping Monitoring Schedule with name: %s", monitoring_schedule_name)
    sagemaker_session.sagemaker_client.stop_monitoring_schedule(
        MonitoringScheduleName=monitoring_schedule_name
    )


def boto_delete_monitoring_schedule(sagemaker_session, monitoring_schedule_name):
    """Deletes a monitoring schedule.

    Args:
        monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
            Schedule to delete.
    """
    logger.info("Deleting Monitoring Schedule with name: %s", monitoring_schedule_name)
    sagemaker_session.sagemaker_client.delete_monitoring_schedule(
        MonitoringScheduleName=monitoring_schedule_name
    )


def boto_describe_monitoring_schedule(sagemaker_session, monitoring_schedule_name):
    """Calls the DescribeMonitoringSchedule API for given name and returns the response.

    Args:
        monitoring_schedule_name (str): The name of the processing job to describe.

    Returns:
        dict: A dictionary response with the processing job description.
    """
    return sagemaker_session.sagemaker_client.describe_monitoring_schedule(
        MonitoringScheduleName=monitoring_schedule_name
    )


def boto_list_monitoring_executions(
    sagemaker_session,
    monitoring_schedule_name,
    sort_by="ScheduledTime",
    sort_order="Descending",
    max_results=100,
):
    """Lists the monitoring executions associated with the given monitoring_schedule_name.

    Args:
        monitoring_schedule_name (str): The monitoring_schedule_name for which to retrieve the
            monitoring executions.
        sort_by (str): The field to sort by. Can be one of: "CreationTime", "ScheduledTime",
            "Status". Default: "ScheduledTime".
        sort_order (str): The sort order. Can be one of: "Ascending", "Descending".
            Default: "Descending".
        max_results (int): The maximum number of results to return. Must be between 1 and 100.

    Returns:
        dict: Dictionary of monitoring schedule executions.
    """
    response = sagemaker_session.sagemaker_client.list_monitoring_executions(
        MonitoringScheduleName=monitoring_schedule_name,
        SortBy=sort_by,
        SortOrder=sort_order,
        MaxResults=max_results,
    )
    return response


def boto_list_monitoring_schedules(
    sagemaker_session,
    endpoint_name=None,
    sort_by="CreationTime",
    sort_order="Descending",
    max_results=100,
):
    """Lists the monitoring executions associated with the given monitoring_schedule_name.

    Args:
        endpoint_name (str): The name of the endpoint to filter on. If not provided, does not
            filter on it. Default: None.
        sort_by (str): The field to sort by. Can be one of: "Name", "CreationTime", "Status".
            Default: "CreationTime".
        sort_order (str): The sort order. Can be one of: "Ascending", "Descending".
            Default: "Descending".
        max_results (int): The maximum number of results to return. Must be between 1 and 100.

    Returns:
        dict: Dictionary of monitoring schedule executions.
    """
    if endpoint_name is not None:
        response = sagemaker_session.sagemaker_client.list_monitoring_schedules(
            EndpointName=endpoint_name,
            SortBy=sort_by,
            SortOrder=sort_order,
            MaxResults=max_results,
        )
    else:
        response = sagemaker_session.sagemaker_client.list_monitoring_schedules(
            SortBy=sort_by, SortOrder=sort_order, MaxResults=max_results
        )

    return response


def boto_update_monitoring_alert(
    sagemaker_session,
    monitoring_schedule_name: str,
    monitoring_alert_name: str,
    data_points_to_alert: int,
    evaluation_period: int,
):
    """Update the monitoring alerts associated with the given schedule_name and alert_name

    Args:
        monitoring_schedule_name (str): The name of the monitoring schedule to update.
        monitoring_alert_name (str): The name of the monitoring alert to update.
        data_points_to_alert (int):  The data point to alert.
        evaluation_period (int): The period to evaluate the alert status.

    Returns:
        dict: A dict represents the update alert response.
    """
    return sagemaker_session.sagemaker_client.update_monitoring_alert(
        MonitoringScheduleName=monitoring_schedule_name,
        MonitoringAlertName=monitoring_alert_name,
        DatapointsToAlert=data_points_to_alert,
        EvaluationPeriod=evaluation_period,
    )


def boto_list_monitoring_alerts(
    sagemaker_session,
    monitoring_schedule_name: str,
    next_token: Optional[str] = None,
    max_results: Optional[int] = 10,
) -> Dict:
    """Lists the monitoring alerts associated with the given monitoring_schedule_name.

    Args:
        monitoring_schedule_name (str): The name of the monitoring schedule to filter on.
            If not provided, does not filter on it.
        next_token (Optional[str]):  The pagination token. Default: None
        max_results (Optional[int]): The maximum number of results to return.
            Must be between 1 and 100. Default: 10

    Returns:
        dict: list of monitoring alerts.
    """
    params = {
        "MonitoringScheduleName": monitoring_schedule_name,
        "MaxResults": max_results,
    }
    if next_token:
        params.update({"NextToken": next_token})

    return sagemaker_session.sagemaker_client.list_monitoring_alerts(**params)


def boto_list_monitoring_alert_history(
    sagemaker_session,
    monitoring_schedule_name: Optional[str] = None,
    monitoring_alert_name: Optional[str] = None,
    sort_by: Optional[str] = "CreationTime",
    sort_order: Optional[str] = "Descending",
    next_token: Optional[str] = None,
    max_results: Optional[int] = 10,
    creation_time_before: Optional[str] = None,
    creation_time_after: Optional[str] = None,
    status_equals: Optional[str] = None,
) -> Dict:
    """Lists the alert history associated with the given schedule_name and alert_name.

    Args:
        monitoring_schedule_name (Optional[str]): The name of the monitoring_schedule_name
            to filter on. If not provided, does not filter on it. Default: None.
        monitoring_alert_name (Optional[str]): The name of the monitoring_alert_name
            to filter on. If not provided, does not filter on it. Default: None.
        sort_by (Optional[str]): sort_by (str): The field to sort by.
            Can be one of: "Name", "CreationTime" Default: "CreationTime".
        sort_order (Optional[str]): The sort order. Can be one of: "Ascending", "Descending".
            Default: "Descending".
        next_token (Optional[str]):  The pagination token. Default: None
        max_results (Optional[int]): The maximum number of results to return.
            Must be between 1 and 100. Default: 10.
        creation_time_before (Optional[str]): A filter to filter alert history before a time
        creation_time_after (Optional[str]): A filter to filter alert history after a time
            Default: None.
        status_equals (Optional[str]): A filter to filter alert history by status
            Default: None.

    Returns:
        dict: list of monitoring alert history.
    """
    params = {
        "MonitoringScheduleName": monitoring_schedule_name,
        "SortBy": sort_by,
        "SortOrder": sort_order,
        "MaxResults": max_results,
    }
    if monitoring_alert_name:
        params.update({"MonitoringAlertName": monitoring_alert_name})
    if creation_time_before:
        params.update({"CreationTimeBefore": creation_time_before})
    if creation_time_after:
        params.update({"CreationTimeAfter": creation_time_after})
    if status_equals:
        params.update({"StatusEquals": status_equals})
    if next_token:
        params.update({"NextToken": next_token})

    return sagemaker_session.sagemaker_client.list_monitoring_alert_history(**params)
