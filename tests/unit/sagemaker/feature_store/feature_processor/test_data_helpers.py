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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import datetime
import json

from dateutil.tz import tzlocal
from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
)
from sagemaker.feature_store.feature_processor._enums import FeatureProcessorMode
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)

INPUT_S3_URI = "s3://bucket/prefix/"
INPUT_FEATURE_GROUP_NAME = "input-fg"
INPUT_FEATURE_GROUP_ARN = "arn:aws:sagemaker:us-west-2:12345789012:feature-group/input-fg"
INPUT_FEATURE_GROUP_S3_URI = "s3://bucket/input-fg/"
INPUT_FEATURE_GROUP_RESOLVED_OUTPUT_S3_URI = (
    "s3://bucket/input-fg/feature-store/12345789012/"
    "sagemaker/us-west-2/offline-store/input-fg-12345/data"
)

FEATURE_GROUP_DATA_SOURCE = FeatureGroupDataSource(name=INPUT_FEATURE_GROUP_ARN)
S3_DATA_SOURCE = CSVDataSource(s3_uri=INPUT_S3_URI)
FEATURE_PROCESSOR_INPUTS = [FEATURE_GROUP_DATA_SOURCE, S3_DATA_SOURCE]
OUTPUT_FEATURE_GROUP_ARN = "arn:aws:sagemaker:us-west-2:12345789012:feature-group/output-fg"

FEATURE_GROUP_SYSTEM_PARAMS = {
    "feature_group_name": "input-fg",
    "online_store_enabled": True,
    "offline_store_enabled": False,
    "offline_store_resolved_s3_uri": None,
}
SYSTEM_PARAMS = {"system": {"scheduled_time": "2023-03-25T02:01:26Z"}}
USER_INPUT_PARAMS = {
    "some-key": "some-value",
    "some-other-key": {"some-key": "some-value"},
}

DATA_SOURCE_UNIQUE_ID_TOO_LONG = """
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
"""

DESCRIBE_FEATURE_GROUP_RESPONSE = {
    "FeatureGroupArn": INPUT_FEATURE_GROUP_ARN,
    "FeatureGroupName": INPUT_FEATURE_GROUP_NAME,
    "RecordIdentifierFeatureName": "id",
    "EventTimeFeatureName": "ingest_time",
    "FeatureDefinitions": [
        {"FeatureName": "id", "FeatureType": "String"},
        {"FeatureName": "model", "FeatureType": "String"},
        {"FeatureName": "model_year", "FeatureType": "String"},
        {"FeatureName": "status", "FeatureType": "String"},
        {"FeatureName": "mileage", "FeatureType": "String"},
        {"FeatureName": "price", "FeatureType": "String"},
        {"FeatureName": "msrp", "FeatureType": "String"},
        {"FeatureName": "ingest_time", "FeatureType": "Fractional"},
    ],
    "CreationTime": datetime.datetime(2023, 3, 29, 19, 15, 47, 20000, tzinfo=tzlocal()),
    "OnlineStoreConfig": {"EnableOnlineStore": True},
    "OfflineStoreConfig": {
        "S3StorageConfig": {
            "S3Uri": INPUT_FEATURE_GROUP_S3_URI,
            "ResolvedOutputS3Uri": INPUT_FEATURE_GROUP_RESOLVED_OUTPUT_S3_URI,
        },
        "DisableGlueTableCreation": False,
        "DataCatalogConfig": {
            "TableName": "input_fg_1680142547",
            "Catalog": "AwsDataCatalog",
            "Database": "sagemaker_featurestore",
        },
    },
    "RoleArn": "arn:aws:iam::12345789012:role/role-name",
    "FeatureGroupStatus": "Created",
    "OnlineStoreTotalSizeBytes": 12345,
    "ResponseMetadata": {
        "RequestId": "d36d3647-1632-4f4e-9f7c-2a4e38e4c6f8",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "d36d3647-1632-4f4e-9f7c-2a4e38e4c6f8",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "1311",
            "date": "Fri, 31 Mar 2023 01:05:49 GMT",
        },
        "RetryAttempts": 0,
    },
}

PIPELINE = {
    "PipelineArn": "some_pipeline_arn",
    "RoleArn": "some_execution_role_arn",
    "CreationTime": datetime.datetime(2023, 3, 29, 19, 15, 47, 20000, tzinfo=tzlocal()),
    "PipelineDefinition": json.dumps(
        {
            "Steps": [
                {
                    "RetryPolicies": [
                        {
                            "BackoffRate": 2.0,
                            "IntervalSeconds": 1,
                            "MaxAttempts": 5,
                            "ExceptionType": ["Step.SERVICE_FAULT", "Step.THROTTLING"],
                        },
                        {
                            "BackoffRate": 2.0,
                            "IntervalSeconds": 1,
                            "MaxAttempts": 5,
                            "ExceptionType": [
                                "SageMaker.JOB_INTERNAL_ERROR",
                                "SageMaker.CAPACITY_ERROR",
                                "SageMaker.RESOURCE_LIMIT",
                            ],
                        },
                    ]
                }
            ]
        }
    ),
}


def create_fp_config(
    inputs=None,
    output=OUTPUT_FEATURE_GROUP_ARN,
    mode=FeatureProcessorMode.PYSPARK,
    target_stores=None,
    enable_ingestion=True,
    parameters=None,
    spark_config=None,
):
    """Helper method to create a FeatureProcessorConfig with fewer arguments."""

    return FeatureProcessorConfig.create(
        inputs=inputs or FEATURE_PROCESSOR_INPUTS,
        output=output,
        mode=mode,
        target_stores=target_stores,
        enable_ingestion=enable_ingestion,
        parameters=parameters,
        spark_config=spark_config,
    )
