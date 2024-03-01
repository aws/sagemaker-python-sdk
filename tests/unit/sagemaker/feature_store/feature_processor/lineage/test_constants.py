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
"""Contains constants of feature processor to be used for unit tests."""
from __future__ import absolute_import

import datetime
from typing import List, Sequence, Union

from botocore.exceptions import ClientError
from mock import Mock
from pyspark.sql import DataFrame

from sagemaker import Session
from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
    ParquetDataSource,
    BaseDataSource,
)
from sagemaker.feature_store.feature_processor.lineage._feature_group_contexts import (
    FeatureGroupContexts,
)
from sagemaker.feature_store.feature_processor.lineage._pipeline_schedule import (
    PipelineSchedule,
)
from sagemaker.feature_store.feature_processor.lineage._pipeline_trigger import PipelineTrigger
from sagemaker.feature_store.feature_processor.lineage._transformation_code import (
    TransformationCode,
)
from sagemaker.lineage._api_types import ContextSource
from sagemaker.lineage.artifact import Artifact, ArtifactSource, ArtifactSummary
from sagemaker.lineage.context import Context

PIPELINE_NAME = "test-pipeline-01"
PIPELINE_ARN = "arn:aws:sagemaker:us-west-2:12345789012:pipeline/test-pipeline-01"
CREATION_TIME = "123123123"
LAST_UPDATE_TIME = "234234234"
SAGEMAKER_SESSION_MOCK = Mock(Session)
CONTEXT_MOCK_01 = Mock(Context)
CONTEXT_MOCK_02 = Mock(Context)


class MockDataSource(BaseDataSource):

    data_source_unique_id = "test_source_unique_id"
    data_source_name = "test_source_name"

    def read_data(self, spark, params) -> DataFrame:
        return None


FEATURE_GROUP_DATA_SOURCE: List[FeatureGroupDataSource] = [
    FeatureGroupDataSource(
        name="feature-group-01",
    ),
    FeatureGroupDataSource(
        name="feature-group-02",
    ),
]

FEATURE_GROUP_INPUT: List[FeatureGroupContexts] = [
    FeatureGroupContexts(
        name="feature-group-01",
        pipeline_context_arn="feature-group-01-pipeline-context-arn",
        pipeline_version_context_arn="feature-group-01-pipeline-version-context-arn",
    ),
    FeatureGroupContexts(
        name="feature-group-02",
        pipeline_context_arn="feature-group-02-pipeline-context-arn",
        pipeline_version_context_arn="feature-group-02-pipeline-version-context-arn",
    ),
]

RAW_DATA_INPUT: Sequence[Union[CSVDataSource, ParquetDataSource, BaseDataSource]] = [
    CSVDataSource(s3_uri="raw-data-uri-01"),
    CSVDataSource(s3_uri="raw-data-uri-02"),
    ParquetDataSource(s3_uri="raw-data-uri-03"),
    MockDataSource(),
]

RAW_DATA_INPUT_ARTIFACTS: List[Artifact] = [
    Artifact(artifact_arn="artifact-01-arn"),
    Artifact(artifact_arn="artifact-02-arn"),
    Artifact(artifact_arn="artifact-03-arn"),
    Artifact(artifact_arn="artifact-04-arn"),
]

PIPELINE_SCHEDULE = PipelineSchedule(
    schedule_name="schedule-name",
    schedule_arn="schedule-arn",
    schedule_expression="schedule-expression",
    pipeline_name="pipeline-name",
    state="state",
    start_date="123123123",
)

PIPELINE_SCHEDULE_2 = PipelineSchedule(
    schedule_name="schedule-name-2",
    schedule_arn="schedule-arn",
    schedule_expression="schedule-expression-2",
    pipeline_name="pipeline-name",
    state="state-2",
    start_date="234234234",
)

PIPELINE_TRIGGER = PipelineTrigger(
    trigger_name="trigger-name",
    trigger_arn="trigger-arn",
    pipeline_name="pipeline-name",
    event_pattern="event-pattern",
    state="Enabled",
)

PIPELINE_TRIGGER_2 = PipelineTrigger(
    trigger_name="trigger-name-2",
    trigger_arn="trigger-arn",
    pipeline_name="pipeline-name",
    event_pattern="event-pattern-2",
    state="Enabled",
)

PIPELINE_TRIGGER_ARTIFACT: Artifact = Artifact(
    artifact_arn="arn:aws:sagemaker:us-west-2:789975069016:artifact/7be06af3274fd01d1c18c96f97141f32",
    artifact_name="sm-fs-fe-trigger-trigger-name",
    artifact_type="PipelineTrigger",
    source={"source_uri": "trigger-arn"},
    properties=dict(
        pipeline_name=PIPELINE_TRIGGER.pipeline_name,
        event_pattern=PIPELINE_TRIGGER.event_pattern,
        state=PIPELINE_TRIGGER.state,
    ),
)

PIPELINE_TRIGGER_ARTIFACT_SUMMARY: ArtifactSummary = ArtifactSummary(
    artifact_arn="arn:aws:sagemaker:us-west-2:789975069016:artifact/7be06af3274fd01d1c18c96f97141f32",
    artifact_name="sm-fs-fe-trigger-trigger-name",
    source=ArtifactSource(
        source_uri="trigger-arn",
    ),
    artifact_type="PipelineTrigger",
    creation_time=datetime.datetime(2023, 4, 27, 21, 4, 17, 926000),
)

ARTIFACT_RESULT: Artifact = Artifact(
    artifact_arn="arn:aws:sagemaker:us-west-2:789975069016:artifact/7be06af3274fd01d1c18c96f97141f32",
    artifact_name="sm-fs-fe-raw-data",
    source={
        "source_uri": "s3://sagemaker-us-west-2-789975069016/transform-2023-04-28-21-50-14-616/"
        "transform-2023-04-28-21-50-14-616/output/model.tar.gz"
    },
    artifact_type="DataSet",
    creation_time=datetime.datetime(2023, 4, 28, 21, 53, 47, 912000),
)

SCHEDULE_ARTIFACT_RESULT: Artifact = Artifact(
    artifact_arn="arn:aws:sagemaker:us-west-2:789975069016:artifact/7be06af3274fd01d1c18c96f97141f32",
    artifact_name="sm-fs-fe-raw-data",
    source={
        "source_uri": "s3://sagemaker-us-west-2-789975069016/transform-2023-04-28-21-50-14-616/"
        "transform-2023-04-28-21-50-14-616/output/model.tar.gz"
    },
    properties=dict(
        pipeline_name=PIPELINE_SCHEDULE.pipeline_name,
        schedule_expression=PIPELINE_SCHEDULE.schedule_expression,
        state=PIPELINE_SCHEDULE.state,
        start_date=PIPELINE_SCHEDULE.start_date,
    ),
    artifact_type="DataSet",
    creation_time=datetime.datetime(2023, 4, 28, 21, 53, 47, 912000),
)

ARTIFACT_SUMMARY: ArtifactSummary = ArtifactSummary(
    artifact_arn="arn:aws:sagemaker:us-west-2:789975069016:artifact/7be06af3274fd01d1c18c96f97141f32",
    artifact_name="sm-fs-fe-raw-data",
    source=ArtifactSource(
        source_uri="s3://sagemaker-us-west-2-789975069016/transform-2023-04-28-21-50-14-616/"
        "transform-2023-04-28-21-50-14-616/output/model.tar.gz",
        source_types=[],
    ),
    artifact_type="DataSet",
    creation_time=datetime.datetime(2023, 4, 27, 21, 4, 17, 926000),
)

TRANSFORMATION_CODE_ARTIFACT_1 = Artifact(
    artifact_arn="ts-artifact-01-arn",
    artifact_name="sm-fs-fe-transformation-code",
    source={
        "source_uri": "s3://sagemaker-us-west-2-789975069016/transform-2023-04-28-21-50-14-616/"
        "transform-2023-04-28-21-50-14-616/output/model.tar.gz",
        "source_types": [{"source_id_type": "Custom", "value": "1684369626"}],
    },
    properties={
        "name": "test-name",
        "author": "test-author",
        "inclusive_start_date": "1684369626",
        "state": "Active",
    },
)

TRANSFORMATION_CODE_ARTIFACT_2 = Artifact(
    artifact_arn="ts-artifact-02-arn",
    artifact_name="sm-fs-fe-transformation-code",
    source={
        "source_uri": "s3://sagemaker-us-west-2-789975069016/transform-2023-04-28-21-50-14-616/"
        "transform-2023-04-28-21-50-14-616/output/model.tar.gz/2",
        "source_types": [{"source_id_type": "Custom", "value": "1684369626"}],
    },
    properties={
        "name": "test-name",
        "author": "test-author",
        "inclusive_start_date": "1684369626",
        "state": "Active",
    },
)

INACTIVE_TRANSFORMATION_CODE_ARTIFACT_1 = Artifact(
    artifact_arn="ts-artifact-02-arn",
    artifact_name="sm-fs-fe-transformation-code",
    source={
        "source_uri": "s3://sagemaker-us-west-2-789975069016/transform-2023-04-28-21-50-14-616/"
        "transform-2023-04-28-21-50-14-616/output/model.tar.gz/2",
        "source_types": [{"source_id_type": "Custom", "value": "1684369307"}],
    },
    Properties={
        "name": "test-name",
        "author": "test-author",
        "exclusive_end_date": "1684369626",
        "inclusive_start_date": "1684369307",
        "state": "Inactive",
    },
)

VALIDATION_EXCEPTION = ClientError(
    {"Error": {"Code": "ValidationException", "Message": "AssociationAlreadyExists"}},
    "Operation",
)

RESOURCE_NOT_FOUND_EXCEPTION = ClientError(
    {"Error": {"Code": "ResourceNotFound", "Message": "ResourceDoesNotExists"}},
    "Operation",
)

NON_VALIDATION_EXCEPTION = ClientError(
    {"Error": {"Code": "NonValidationException", "Message": "NonValidationError"}},
    "Operation",
)

FEATURE_GROUP_NAME = "feature-group-name-01"
FEATURE_GROUP = {
    "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:789975069016:feature-group/feature-group-name-01",
    "FeatureGroupName": "feature-group-name-01",
    "RecordIdentifierFeatureName": "model_year_status",
    "EventTimeFeatureName": "ingest_time",
    "FeatureDefinitions": [
        {"FeatureName": "model_year_status", "FeatureType": "String"},
        {"FeatureName": "avg_mileage", "FeatureType": "String"},
        {"FeatureName": "max_mileage", "FeatureType": "String"},
        {"FeatureName": "avg_price", "FeatureType": "String"},
        {"FeatureName": "max_price", "FeatureType": "String"},
        {"FeatureName": "avg_msrp", "FeatureType": "String"},
        {"FeatureName": "max_msrp", "FeatureType": "String"},
        {"FeatureName": "ingest_time", "FeatureType": "Fractional"},
    ],
    "CreationTime": datetime.datetime(2023, 4, 27, 21, 4, 17, 926000),
    "OnlineStoreConfig": {"EnableOnlineStore": True},
    "OfflineStoreConfig": {
        "S3StorageConfig": {
            "S3Uri": "s3://sagemaker-us-west-2-789975069016/"
            "feature-store/feature-processor/"
            "suryans-v2/offline-store",
            "ResolvedOutputS3Uri": "s3://sagemaker-us-west-2-"
            "789975069016/feature-store/"
            "feature-processor/suryans-v2/"
            "offline-store/789975069016/"
            "sagemaker/us-west-2/"
            "offline-store/"
            "feature-group-name-01-"
            "1682629457/data",
        },
        "DisableGlueTableCreation": False,
        "DataCatalogConfig": {
            "TableName": "feature-group-name-01_1682629457",
            "Catalog": "AwsDataCatalog",
            "Database": "sagemaker_featurestore",
        },
    },
    "RoleArn": "arn:aws:iam::789975069016:role/service-role/AmazonSageMaker-ExecutionRole-20230421T100744",
    "FeatureGroupStatus": "Created",
    "OnlineStoreTotalSizeBytes": 0,
    "ResponseMetadata": {
        "RequestId": "8f139791-345d-4388-8d6d-40420495a3c4",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "8f139791-345d-4388-8d6d-40420495a3c4",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "1608",
            "date": "Mon, 01 May 2023 21:42:59 GMT",
        },
        "RetryAttempts": 0,
    },
}

PIPELINE = {
    "PipelineArn": "arn:aws:sagemaker:us-west-2:597217924798:pipeline/test-pipeline-26",
    "PipelineName": "test-pipeline-26",
    "PipelineDisplayName": "test-pipeline-26",
    "PipelineDefinition": '{"Version": "2020-12-01", "Metadata": {}, '
    '"Parameters": [{"Name": "scheduled-time", "Type": "String"}], '
    '"PipelineExperimentConfig": {"ExperimentName": {"Get": "Execution.PipelineName"}, '
    '"TrialName": {"Get": "Execution.PipelineExecutionId"}}, '
    '"Steps": [{"Name": "test-pipeline-26-training-step", "Type": '
    '"Training", "Arguments": {"AlgorithmSpecification": {"TrainingInputMode": '
    '"File", "TrainingImage": "153931337802.dkr.ecr.us-west-2.amazonaws.com/'
    'sagemaker-spark-processing:3.2-cpu-py39-v1.1", "ContainerEntrypoint": '
    '["/bin/bash", "/opt/ml/input/data/sagemaker_remote_function_bootstrap/'
    'job_driver.sh", "--files", "s3://bugbash-schema-update/temp.sh", '
    '"/opt/ml/input/data/sagemaker_remote_function_bootstrap/spark_app.py"], '
    '"ContainerArguments": ["--s3_base_uri", '
    '"s3://bugbash-schema-update-suryans/test-pipeline-26", '
    '"--region", "us-west-2", "--client_python_version", "3.9"]}, '
    '"OutputDataConfig": {"S3OutputPath": '
    '"s3://bugbash-schema-update-suryans/test-pipeline-26"}, '
    '"StoppingCondition": {"MaxRuntimeInSeconds": 86400}, "ResourceConfig": '
    '{"VolumeSizeInGB": 30, "InstanceCount": 1, "InstanceType": "ml.m5.xlarge"}, '
    '"RoleArn": "arn:aws:iam::597217924798:role/Admin", "InputDataConfig": '
    '[{"DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": '
    '"s3://bugbash-schema-update-suryans/test-pipeline-26/'
    'sagemaker_remote_function_bootstrap", "S3DataDistributionType": '
    '"FullyReplicated"}}, "ChannelName": "sagemaker_remote_function_bootstrap"}, '
    '{"DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": '
    '"s3://bugbash-schema-update/sagemaker-2.142.1.dev0-py2.py3-none-any.whl", '
    '"S3DataDistributionType": "FullyReplicated"}}, "ChannelName": '
    '"sagemaker_wheel_file"}], "Environment": {"AWS_DEFAULT_REGION": "us-west-2"}, '
    '"DebugHookConfig": {"S3OutputPath": '
    '"s3://bugbash-schema-update-suryans/test-pipeline-26", '
    '"CollectionConfigurations": []},'
    ' "ProfilerConfig": {"S3OutputPath": '
    '"s3://bugbash-schema-update-suryans/test-pipeline-26", '
    '"DisableProfiler": false}, "RetryStrategy": {"MaximumRetryAttempts": 1}}}]}',
    "RoleArn": "arn:aws:iam::597217924798:role/Admin",
    "PipelineStatus": "Active",
    "CreationTime": datetime.datetime(2023, 4, 27, 9, 46, 35, 686000),
    "LastModifiedTime": datetime.datetime(2023, 4, 27, 20, 27, 36, 648000),
    "CreatedBy": {},
    "LastModifiedBy": {},
    "ResponseMetadata": {
        "RequestId": "2075bc1c-1b34-4fe5-b7d8-7cfdf784a7d9",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "2075bc1c-1b34-4fe5-b7d8-7cfdf784a7d9",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "2555",
            "date": "Thu, 04 May 2023 00:28:35 GMT",
        },
        "RetryAttempts": 0,
    },
}

PIPELINE_CONTEXT: Context = Context(
    context_arn=f"{PIPELINE_NAME}-context-arn",
    context_name=f"sm-fs-fe-{PIPELINE_NAME}-{CREATION_TIME}-fep",
    context_type="FeatureEngineeringPipeline",
    source=ContextSource(source_uri=PIPELINE_ARN, source_types=[]),
    properties={
        "PipelineName": PIPELINE_NAME,
        "PipelineCreationTime": CREATION_TIME,
        "LastUpdateTime": LAST_UPDATE_TIME,
    },
)

PIPELINE_VERSION_CONTEXT: Context = Context(
    context_arn=f"{PIPELINE_NAME}-version-context-arn",
    context_name=f"sm-fs-fe-{PIPELINE_NAME}-{LAST_UPDATE_TIME}-fep-ver",
    context_type=f"FeatureEngineeringPipelineVersion-{PIPELINE_NAME}",
    source=ContextSource(source_uri=PIPELINE_ARN, source_types=LAST_UPDATE_TIME),
    properties={"PipelineName": PIPELINE_NAME, "LastUpdateTime": LAST_UPDATE_TIME},
)

TRANSFORMATION_CODE_INPUT_1: TransformationCode = TransformationCode(
    s3_uri="s3://sagemaker-us-west-2-789975069016/transform-2023-04-28-21-50-14-616/"
    "transform-2023-04-28-21-50-14-616/output/model.tar.gz",
    author="test-author",
    name="test-name",
)

TRANSFORMATION_CODE_INPUT_2: TransformationCode = TransformationCode(
    s3_uri="s3://sagemaker-us-west-2-789975069016/transform-2023-04-28-21-50-14-616/"
    "transform-2023-04-28-21-50-14-616/output/model.tar.gz/2",
    author="test-author",
    name="test-name",
)
