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

import glob
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict
from datetime import datetime
from pyspark.sql import DataFrame
import pytz

import pytest
import pandas as pd
import numpy as np
import json
import attr
from boto3 import client

from tests.integ import DATA_DIR
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.s3 import S3Uploader
from urllib.parse import urlparse
from sagemaker.core.remote_function import remote
from sagemaker.core.remote_function.spark_config import SparkConfig
from sagemaker.mlops.feature_store import (
    FeatureGroup,
    FeatureDefinition,
    FeatureTypeEnum,
    OnlineStoreConfig,
    OfflineStoreConfig,
    S3StorageConfig,
)
from sagemaker.mlops.feature_store.feature_utils import create_athena_query
from sagemaker.mlops.feature_store.feature_processor import (
    feature_processor,
    CSVDataSource,
    PySparkDataSource,
    FeatureProcessorPipelineEvents,
    FeatureProcessorPipelineExecutionStatus,
)
from sagemaker.mlops.feature_store.feature_processor.feature_scheduler import (
    to_pipeline,
    describe,
    execute,
    schedule,
    delete_schedule,
    put_trigger,
    enable_trigger,
    disable_trigger,
    delete_trigger,
)
from sagemaker.mlops.workflow.pipeline import Pipeline

CAR_SALES_FG_FEATURE_DEFINITIONS = [
    FeatureDefinition(feature_name="id", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="model", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="model_year", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="status", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="mileage", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="price", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="msrp", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="ingest_time", feature_type=FeatureTypeEnum.FRACTIONAL.value),
]
CAR_SALES_FG_RECORD_IDENTIFIER_NAME = "id"
CAR_SALES_FG_EVENT_TIME_FEATURE_NAME = "ingest_time"

AGG_CAR_SALES_FG_FEATURE_DEFINITIONS = [
    FeatureDefinition(feature_name="model_year_status", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="avg_mileage", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="max_mileage", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="avg_price", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="max_price", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="avg_msrp", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="max_msrp", feature_type=FeatureTypeEnum.STRING.value),
    FeatureDefinition(feature_name="ingest_time", feature_type=FeatureTypeEnum.FRACTIONAL.value),
]
AGG_CAR_SALES_FG_RECORD_IDENTIFIER_NAME = "model_year_status"
AGG_CAR_SALES_FG_EVENT_TIME_FEATURE_NAME = "ingest_time"

BUCKET_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "FeatureStoreOfflineStoreS3BucketPolicy",
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": ["s3:PutObject", "s3:PutObjectAcl"],
            "Resource": "arn:aws:s3:::{bucket_name}-{region_name}/*",
            "Condition": {"StringEquals": {"s3:x-amz-acl": "bucket-owner-full-control"}},
        },
        {
            "Sid": "FeatureStoreOfflineStoreS3BucketPolicy",
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "s3:GetBucketAcl",
            "Resource": "arn:aws:s3:::{bucket_name}-{region_name}",
        },
    ],
}

_FEATURE_PROCESSOR_DIR = os.path.join(DATA_DIR, "feature_store/feature_processor")

SCHEDULE_EXPRESSION_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"  # 2023-01-01T07:00:00


@pytest.fixture(scope="module")
def sagemaker_session():
    return Session()


@pytest.fixture(scope="module")
def pre_execution_commands(sagemaker_session):
    return get_pre_execution_commands(sagemaker_session=sagemaker_session)


@pytest.fixture(scope="module")
def dependencies_path():
    return os.path.join(_FEATURE_PROCESSOR_DIR, "requirements.txt")


@pytest.mark.slow_test
def test_feature_processor_transform_online_only_store_ingestion(
    sagemaker_session,
):
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OnlineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        # this calls spark 3.3 which requires java 11
        transform() 

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 26

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        assert dataset.empty
    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )


@pytest.mark.slow_test
def test_feature_processor_transform_with_customized_data_source(
    sagemaker_session,
):
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()

    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @attr.s
        class TestCSVDataSource(PySparkDataSource):

            s3_uri = attr.ib()
            data_source_name = "TestCSVDataSource"
            data_source_unique_id = "s3_uri"

            def read_data(self, spark, params) -> DataFrame:
                s3a_uri = self.s3_uri.replace("s3://", "s3a://")
                return spark.read.csv(s3a_uri, header=True, inferSchema=False)

        @feature_processor(
            inputs=[TestCSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OnlineStore"],
            spark_config={
                "spark.hadoop.fs.s3a.aws.credentials.provider": ",".join(
                    [
                        "com.amazonaws.auth.ContainerCredentialsProvider",
                        "com.amazonaws.auth.profile.ProfileCredentialsProvider",
                        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                    ]
                )
            },
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        transform()

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 26

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        assert dataset.empty
    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )


@pytest.mark.slow_test
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_feature_processor_transform_offline_only_store_ingestion(
    sagemaker_session,
):
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OfflineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        transform()

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 0

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        dataset = dataset.drop(
            columns=["ingest_time", "write_time", "api_invocation_time", "is_deleted"]
        )

        expected = get_expected_dataframe()
        dataset_sorted = dataset.sort_values(by="id").reset_index(drop=True)
        expected_sorted = expected.sort_values(by="id").reset_index(drop=True)
        assert dataset_sorted.equals(expected_sorted)
    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )


@pytest.mark.slow_test
@pytest.mark.skipif(
    not sys.version.startswith("3.9"),
    reason="Only allow this test to run with py39",
)
def test_feature_processor_transform_offline_only_store_ingestion_run_with_remote(
    sagemaker_session,
    pre_execution_commands,
    dependencies_path,
):
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @remote(
            pre_execution_commands=pre_execution_commands,
            dependencies=dependencies_path,
            spark_config=SparkConfig(),
            instance_type="ml.m5.xlarge",
        )
        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OfflineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        transform()

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 0

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        dataset = dataset.drop(
            columns=["ingest_time", "write_time", "api_invocation_time", "is_deleted"]
        )

        expected = get_expected_dataframe()
        dataset_sorted = dataset.sort_values(by="id").reset_index(drop=True)
        expected_sorted = expected.sort_values(by="id").reset_index(drop=True)
        assert dataset_sorted.equals(expected_sorted)
    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )


@pytest.mark.slow_test
@pytest.mark.skipif(
    not sys.version.startswith("3.9"),
    reason="Only allow this test to run with py39",
)
def test_to_pipeline_and_execute(
    sagemaker_session,
    pre_execution_commands,
    dependencies_path,
):
    pipeline_name = "pipeline-name-01"
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @remote(
            pre_execution_commands=pre_execution_commands,
            dependencies=dependencies_path,
            spark_config=SparkConfig(),
            instance_type="ml.m5.xlarge",
        )
        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OfflineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        _wait_for_feature_group_lineage_contexts(
            car_data_feature_group_name, sagemaker_session
        )

        pipeline_arn = to_pipeline(
            pipeline_name=pipeline_name,
            step=transform,
            role_arn=get_execution_role(sagemaker_session),
            max_retries=2,
            tags=[("integ_test_tag_key_1", "integ_test_tag_key_2")],
            sagemaker_session=sagemaker_session,
        )
        _sagemaker_client = get_sagemaker_client(sagemaker_session=sagemaker_session)

        assert pipeline_arn is not None

        tags = _sagemaker_client.list_tags(ResourceArn=pipeline_arn)["Tags"]

        tag_keys = [tag["Key"] for tag in tags]
        assert "integ_test_tag_key_1" in tag_keys

        pipeline_description = Pipeline(name=pipeline_name).describe()
        assert pipeline_arn == pipeline_description["PipelineArn"]
        assert get_execution_role(sagemaker_session) == pipeline_description["RoleArn"]

        pipeline_definition = json.loads(pipeline_description["PipelineDefinition"])
        assert len(pipeline_definition["Steps"]) == 1
        for retry_policy in pipeline_definition["Steps"][0]["RetryPolicies"]:
            assert retry_policy["MaxAttempts"] == 2

        pipeline_execution_arn = execute(
            pipeline_name=pipeline_name, sagemaker_session=sagemaker_session
        )

        status = _wait_for_pipeline_execution_to_reach_terminal_state(
            pipeline_execution_arn=pipeline_execution_arn,
            sagemaker_client=_sagemaker_client,
        )
        assert status == "Succeeded"

    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )
        # cleanup_pipeline(pipeline_name="pipeline-name-01", sagemaker_session=sagemaker_session)


@pytest.mark.slow_test
@pytest.mark.skipif(
    not sys.version.startswith("3.9"),
    reason="Only allow this test to run with py39",
)
def test_schedule_and_event_trigger(
    sagemaker_session,
    pre_execution_commands,
    dependencies_path,
):
    pipeline_name = "pipeline-name-01"
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @remote(
            pre_execution_commands=pre_execution_commands,
            dependencies=dependencies_path,
            spark_config=SparkConfig(),
            instance_type="ml.m5.xlarge",
        )
        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OfflineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        _wait_for_feature_group_lineage_contexts(
            car_data_feature_group_name, sagemaker_session
        )

        pipeline_arn = to_pipeline(
            pipeline_name=pipeline_name,
            step=transform,
            role_arn=get_execution_role(sagemaker_session),
            max_retries=2,
            sagemaker_session=sagemaker_session,
        )

        assert pipeline_arn is not None

        pipeline_description = Pipeline(name=pipeline_name).describe()
        assert pipeline_arn == pipeline_description["PipelineArn"]
        assert get_execution_role(sagemaker_session) == pipeline_description["RoleArn"]

        pipeline_definition = json.loads(pipeline_description["PipelineDefinition"])
        assert len(pipeline_definition["Steps"]) == 1
        for retry_policy in pipeline_definition["Steps"][0]["RetryPolicies"]:
            assert retry_policy["MaxAttempts"] == 2
        now = datetime.now(tz=pytz.utc)
        schedule_expression = f"at({now.strftime(SCHEDULE_EXPRESSION_TIMESTAMP_FORMAT)})"
        schedule(
            pipeline_name=pipeline_name,
            schedule_expression=schedule_expression,
            start_date=now,
            sagemaker_session=sagemaker_session,
        )
        time.sleep(60)
        executions = sagemaker_session.sagemaker_client.list_pipeline_executions(
            PipelineName=pipeline_name
        )
        pipeline_execution_arn = executions["PipelineExecutionSummaries"][0]["PipelineExecutionArn"]

        status = _wait_for_pipeline_execution_to_reach_terminal_state(
            pipeline_execution_arn=pipeline_execution_arn,
            sagemaker_client=get_sagemaker_client(sagemaker_session=sagemaker_session),
        )
        assert status == "Succeeded"

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 0

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        dataset = dataset.drop(
            columns=["ingest_time", "write_time", "api_invocation_time", "is_deleted"]
        )

        # assert dataset.equals(get_expected_dataframe())

        put_trigger(
            source_pipeline_events=[
                FeatureProcessorPipelineEvents(
                    pipeline_name=pipeline_name,
                    pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.FAILED],
                )
            ],
            target_pipeline=pipeline_name,
        )

        assert "trigger" in describe(
            pipeline_name=pipeline_name, sagemaker_session=sagemaker_session
        )
        assert describe(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)[
            "event_pattern"
        ] == json.dumps(
            {
                "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
                "source": ["aws.sagemaker"],
                "detail": {
                    "currentPipelineExecutionStatus": ["Failed"],
                    "pipelineArn": [pipeline_arn],
                },
            }
        )
        enable_trigger(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
        assert (
            describe(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)[
                "trigger_state"
            ]
            == "ENABLED"
        )
        disable_trigger(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
        assert (
            describe(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)[
                "trigger_state"
            ]
            == "DISABLED"
        )

        delete_schedule(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
        assert "schedule_arn" not in describe(
            pipeline_name=pipeline_name, sagemaker_session=sagemaker_session
        )
        delete_trigger(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
        assert "trigger" not in describe(
            pipeline_name=pipeline_name, sagemaker_session=sagemaker_session
        )

    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )


def get_car_data_feature_group_name():
    return f"car-data-{int(time.time() * 10 ** 7)}"


def get_car_data_aggregated_feature_group_name():
    return f"car-data-aggregated-{int(time.time() * 10 ** 7)}"


def get_offline_store_s3_uri(sagemaker_session):
    region_name = sagemaker_session.boto_region_name
    bucket = f"sagemaker-test-featurestore-{region_name}-{sagemaker_session.account_id()}"
    sagemaker_session._create_s3_bucket_if_it_does_not_exist(bucket, region_name)
    s3 = sagemaker_session.boto_session.client("s3", region_name=region_name)
    BUCKET_POLICY["Statement"][0]["Resource"] = f"arn:aws:s3:::{bucket}/*"
    BUCKET_POLICY["Statement"][1]["Resource"] = f"arn:aws:s3:::{bucket}"
    s3.put_bucket_policy(
        Bucket=f"{bucket}",
        Policy=json.dumps(BUCKET_POLICY),
    )
    return f"s3://{bucket}"


def get_raw_car_data_s3_uri(sagemaker_session) -> str:
    uri = "s3://{}/{}/input/data/{}".format(
        sagemaker_session.default_bucket(),
        "feature-processor-test",
        "csv-data",
    )
    raw_car_data_s3_uri = S3Uploader.upload(
        os.path.join(_FEATURE_PROCESSOR_DIR, "car-data.csv"),
        uri,
        sagemaker_session=sagemaker_session,
    )
    print(f"Upload complete: {raw_car_data_s3_uri}")
    return raw_car_data_s3_uri


def get_wheel_file_s3_uri(sagemaker_session):
    """Upload all SDK wheels to S3 and return (s3_prefix, wheel_basenames).

    Returns:
        tuple: (s3_prefix, [sagemaker_whl, core_whl, mlops_whl]) where each
        element is the basename of the corresponding wheel file.
    """
    s3_prefix = "s3://{}/{}/wheel-file".format(
        sagemaker_session.default_bucket(), "feature-processor-test"
    )
    sources = _generate_and_move_sagemaker_sdk_tar()
    for source in sources:
        print(source)
        S3Uploader.upload(source, s3_prefix, sagemaker_session=sagemaker_session)
    wheel_names = [os.path.basename(s) for s in sources]
    return s3_prefix, wheel_names


def get_pre_execution_commands(sagemaker_session):
    """Build SDK wheels, upload to S3, and return pre-execution install commands."""
    s3_prefix, wheel_names = get_wheel_file_s3_uri(sagemaker_session=sagemaker_session)
    sagemaker_whl, core_whl, mlops_whl = wheel_names
    print(f'{sagemaker_whl=}, {core_whl=}, {mlops_whl}')
    return [
        f"aws s3 cp {s3_prefix}/ /tmp/packages/ --recursive",
        "pip3 install 'setuptools<75'",
        f"pip3 install --no-build-isolation '/tmp/packages/{sagemaker_whl}[feature-processor]' 'numpy<2.0.0' 'ml_dtypes<=0.4.1' 'setuptools<75' || true",
        f"pip3 install --no-deps --force-reinstall /tmp/packages/{sagemaker_whl}",
        f"pip3 install --no-deps --force-reinstall /tmp/packages/{core_whl} /tmp/packages/{mlops_whl}",
    ]


def create_feature_groups(
    sagemaker_session,
    car_data_feature_group_name,
    car_data_aggregated_feature_group_name,
    offline_store_s3_uri,
) -> Dict:
    # Create Feature Group -  Car sale records.
    car_sales_fg = None
    agg_car_sales_fg = None
    try:
        car_sales_fg = FeatureGroup.create(
            feature_group_name=car_data_feature_group_name,
            record_identifier_feature_name=CAR_SALES_FG_RECORD_IDENTIFIER_NAME,
            event_time_feature_name=CAR_SALES_FG_EVENT_TIME_FEATURE_NAME,
            feature_definitions=CAR_SALES_FG_FEATURE_DEFINITIONS,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"{offline_store_s3_uri}/car-data"),
            ),
            online_store_config=OnlineStoreConfig(enable_online_store=True),
            role_arn=get_execution_role(sagemaker_session),
            session=sagemaker_session.boto_session,
        )
        print(f"Created feature group {car_sales_fg.feature_group_name}")
    except Exception as e:
        if "ResourceInUse" in str(e):
            print("Feature Group already exists")
            car_sales_fg = FeatureGroup.get(
                feature_group_name=car_data_feature_group_name,
                session=sagemaker_session.boto_session,
            )
        else:
            raise e

    # Create Feature Group -  Aggregated car sales records.
    try:
        agg_car_sales_fg = FeatureGroup.create(
            feature_group_name=car_data_aggregated_feature_group_name,
            record_identifier_feature_name=AGG_CAR_SALES_FG_RECORD_IDENTIFIER_NAME,
            event_time_feature_name=AGG_CAR_SALES_FG_EVENT_TIME_FEATURE_NAME,
            feature_definitions=AGG_CAR_SALES_FG_FEATURE_DEFINITIONS,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(
                    s3_uri=f"{offline_store_s3_uri}/car-data-aggregated"
                ),
            ),
            online_store_config=OnlineStoreConfig(enable_online_store=True),
            role_arn=get_execution_role(sagemaker_session),
            session=sagemaker_session.boto_session,
        )
        print(f"Created feature group {agg_car_sales_fg.feature_group_name}")
        print("Sleeping for a bit, to let Feature Groups get ready.")
    except Exception as e:
        if "ResourceInUse" in str(e):
            print("Feature Group already exists")
            agg_car_sales_fg = FeatureGroup.get(
                feature_group_name=car_data_aggregated_feature_group_name,
                session=sagemaker_session.boto_session,
            )
        else:
            raise e

    _wait_for_feature_group_create(car_sales_fg)
    _wait_for_feature_group_create(agg_car_sales_fg)

    return dict(
        car_data_arn=car_sales_fg.feature_group_arn,
        car_data_feature_group=car_sales_fg,
        car_data_aggregated_arn=agg_car_sales_fg.feature_group_arn,
        car_data_aggregated_feature_group=agg_car_sales_fg,
    )


def get_expected_dataframe():
    expected_dataframe = pd.read_csv(os.path.join(_FEATURE_PROCESSOR_DIR, "car-data.csv"))
    expected_dataframe["Model"].replace("^\d\d\d\d\s", "", regex=True, inplace=True)  # noqa: W605
    expected_dataframe["Mileage"].replace("(,)|(mi\.)", "", regex=True, inplace=True)  # noqa: W605
    expected_dataframe["Mileage"].replace("Not available", np.nan, inplace=True)
    expected_dataframe["Price"].replace("\$", "", regex=True, inplace=True)  # noqa: W605
    expected_dataframe["Price"].replace(",", "", regex=True, inplace=True)
    expected_dataframe["MSRP"].replace(
        "(^MSRP\s\\$)|(,)", "", regex=True, inplace=True  # noqa: W605
    )
    expected_dataframe["MSRP"].replace("Not specified", np.nan, inplace=True)
    expected_dataframe["MSRP"].replace(
        "\\$\d+[a-zA-Z\s]+", np.nan, regex=True, inplace=True  # noqa: W605
    )
    expected_dataframe["Mileage"] = expected_dataframe["Mileage"].astype(float)
    expected_dataframe["Price"] = expected_dataframe["Price"].astype(float)
    expected_dataframe.rename(
        columns={
            "Id": "id",
            "Model": "model",
            "Year": "model_year",
            "Status": "status",
            "Mileage": "mileage",
            "Price": "price",
            "MSRP": "msrp",
        },
        inplace=True,
    )
    return expected_dataframe


def _wait_for_feature_group_create(feature_group: FeatureGroup):
    feature_group.wait_for_status(target_status="Created", poll=5)


def _wait_for_pipeline_execution_to_stop(pipeline_execution_arn: str, sagemaker_client: client):
    status = sagemaker_client.describe_pipeline_execution(
        PipelineExecutionArn=pipeline_execution_arn
    )["PipelineExecutionStatus"]
    if status != "Stopping" and status != "Stopped":
        raise RuntimeError(
            f"Pipeline execution Arn: {pipeline_execution_arn} "
            f"status is not in Stopping or Stopped mode, instead is in {status} mode."
        )
    while status == "Stopping":
        print("Waiting for Pipeline Execution to Stop")
        time.sleep(5)
        status = sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=pipeline_execution_arn
        )["PipelineExecutionStatus"]
    if status != "Stopped":
        raise RuntimeError(f"Failed to Stop pipeline execution {pipeline_execution_arn}")
    logging.info(f"pipeline execution {pipeline_execution_arn} successfully Stopped.")


def _wait_for_pipeline_execution_to_reach_terminal_state(
    pipeline_execution_arn: str, sagemaker_client: client
) -> str:
    status = sagemaker_client.describe_pipeline_execution(
        PipelineExecutionArn=pipeline_execution_arn
    )["PipelineExecutionStatus"]
    while status == "Stopping" or status == "Executing":
        print("Waiting for Pipeline Execution to reach terminal state")
        time.sleep(60)
        status = sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=pipeline_execution_arn
        )["PipelineExecutionStatus"]
    logging.info(
        f"pipeline execution {pipeline_execution_arn} successfully reached a terminal state {status}."
    )
    return status


def cleanup_feature_group(feature_group: FeatureGroup, sagemaker_session: Session):
    try:
        feature_group.delete()
        print(f"{feature_group.feature_group_name} is deleted.")
    except sagemaker_session.sagemaker_client.exceptions.ResourceNotFound:
        print(f"{feature_group.feature_group_name} not found.")
        pass
    except Exception as e:
        raise RuntimeError(
            f"Failed to delete feature group with name {feature_group.feature_group_name}", e
        )


def cleanup_pipeline(pipeline_name: str, sagemaker_session: Session):
    try:
        pipeline = Pipeline(name=pipeline_name, sagemaker_session=sagemaker_session)

        sagemaker_client = get_sagemaker_client(sagemaker_session=sagemaker_session)
        executions = sagemaker_client.list_pipeline_executions(PipelineName=pipeline_name)
        for execution in executions["PipelineExecutionSummaries"]:
            if execution["PipelineExecutionStatus"] == "Executing":
                logging.info(f'Stopping pipeline execution: {execution["PipelineExecutionArn"]}')
                sagemaker_client.stop_pipeline_execution(
                    PipelineExecutionArn=execution["PipelineExecutionArn"]
                )
                _wait_for_pipeline_execution_to_stop(
                    pipeline_execution_arn=execution["PipelineExecutionArn"],
                    sagemaker_client=sagemaker_client,
                )
            if execution["PipelineExecutionStatus"] == "Stopping":
                _wait_for_pipeline_execution_to_stop(
                    pipeline_execution_arn=execution["PipelineExecutionArn"],
                    sagemaker_client=sagemaker_client,
                )
        pipeline.delete()
        logging.info(f"{pipeline_name} is deleted.")
    except sagemaker_session.sagemaker_client.exceptions.ResourceNotFound:
        print(f"{pipeline_name} not found.")
        pass
    except sagemaker_session.sagemaker_client.exceptions.ClientError as ce:
        if ce.response["Error"]["Code"] == "ValidationException":
            if (
                "Pipelines with running executions cannot be deleted."
                in ce.response["Error"]["Message"]
            ):
                cleanup_pipeline(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
                pass
        raise RuntimeError(f"Failed to delete Pipeline with name {pipeline_name}", ce)
    except Exception as e:
        raise RuntimeError(f"Failed to delete Pipeline with name {pipeline_name}", e)


def cleanup_offline_store(feature_group: FeatureGroup, sagemaker_session: Session):
    feature_group_name = feature_group.feature_group_name
    try:
        feature_group.refresh()
        s3_uri = feature_group.offline_store_config.s3_storage_config.resolved_output_s3_uri
        parsed_uri = urlparse(s3_uri)
        bucket_name, prefix = parsed_uri.netloc, parsed_uri.path
        prefix = prefix.strip("/")
        prefix = prefix[:-5] if prefix.endswith("/data") else prefix
        region_name = sagemaker_session.boto_region_name
        s3_client = sagemaker_session.boto_session.client(
            service_name="s3", region_name=region_name
        )
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        files_in_folder = response["Contents"]
        files_to_delete = []
        for f in files_in_folder:
            files_to_delete.append({"Key": f["Key"]})
        s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": files_to_delete})
    except sagemaker_session.sagemaker_client.exceptions.ResourceNotFound:
        print(f"{feature_group.feature_group_name} not found.")
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to delete data for feature_group {feature_group_name}", e)


def get_sagemaker_client(sagemaker_session=Session) -> client:
    region_name = sagemaker_session.boto_session.region_name
    return sagemaker_session.boto_session.client(service_name="sagemaker", region_name=region_name)


def _generate_and_move_sagemaker_sdk_tar():
    """Build all three SDK wheel files and return their paths."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
    dist_dir = os.path.join(repo_root, "dist")

    # Build wheels for all three sub-packages into the shared dist/ directory
    build_dirs = [
        repo_root,
        os.path.join(repo_root, "sagemaker-core"),
        os.path.join(repo_root, "sagemaker-mlops"),
    ]
    for build_dir in build_dirs:
        subprocess.run(
            f"python -m build --wheel --outdir {dist_dir}",
            shell=True,
            cwd=build_dir,
            check=True,
        )

    # Locate the three expected wheels by prefix pattern
    wheel_patterns = [
        "sagemaker-[0-9]*.whl",
        "sagemaker_core-*.whl",
        "sagemaker_mlops-*.whl",
    ]
    paths = []
    for pattern in wheel_patterns:
        matches = glob.glob(os.path.join(dist_dir, pattern))
        if not matches:
            raise FileNotFoundError(
                f"No wheel found matching {pattern} in {dist_dir}"
            )
        paths.append(matches[0])
    return paths


def _wait_for_feature_group_lineage_contexts(
    feature_group_name, sagemaker_session, max_attempts=12, delay=10
):
    """Wait for the lineage contexts to be created for a feature group.

    A third-party service asynchronously creates lineage contexts after a feature group
    is created. This helper polls until they exist or the timeout is reached.
    """
    from sagemaker.mlops.feature_store.feature_processor.feature_scheduler import (
        _validate_fg_lineage_resources,
    )

    for attempt in range(max_attempts):
        try:
            _validate_fg_lineage_resources(feature_group_name, sagemaker_session)
            logging.info(
                "Lineage contexts ready for %s after %d seconds.",
                feature_group_name,
                attempt * delay,
            )
            return
        except (ValueError, Exception):
            logging.info(
                "Waiting for lineage contexts for %s (attempt %d/%d)...",
                feature_group_name,
                attempt + 1,
                max_attempts,
            )
            time.sleep(delay)

    raise TimeoutError(
        f"Lineage contexts for feature group {feature_group_name} were not created "
        f"after {max_attempts * delay} seconds."
    )
