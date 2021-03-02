# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
import time
from contextlib import contextmanager

import boto3
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.inputs import FeatureValue
from sagemaker.session import get_execution_role, Session
from tests.integ.timeout import timeout

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


@pytest.fixture(scope="module")
def region_name(feature_store_session):
    return feature_store_session.boto_session.region_name


@pytest.fixture(scope="module")
def role(feature_store_session):
    return get_execution_role(feature_store_session)


# TODO-reinvent-2020: remove use of specified region and this fixture
@pytest.fixture(scope="module")
def feature_store_session():
    boto_session = boto3.Session(region_name="us-east-2")

    sagemaker_client = boto_session.client("sagemaker")
    featurestore_runtime_client = boto_session.client("sagemaker-featurestore-runtime")

    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime_client,
    )


@pytest.fixture
def feature_group_name():
    return f"my-feature-group-{int(time.time() * 10**7)}"


@pytest.fixture
def offline_store_s3_uri(feature_store_session, region_name):
    bucket = f"sagemaker-test-featurestore-{region_name}-{feature_store_session.account_id()}"
    feature_store_session._create_s3_bucket_if_it_does_not_exist(bucket, region_name)
    s3 = feature_store_session.boto_session.client("s3", region_name=region_name)
    BUCKET_POLICY["Statement"][0]["Resource"] = f"arn:aws:s3:::{bucket}/*"
    BUCKET_POLICY["Statement"][1]["Resource"] = f"arn:aws:s3:::{bucket}"
    s3.put_bucket_policy(
        Bucket=f"{bucket}",
        Policy=json.dumps(BUCKET_POLICY),
    )
    return f"s3://{bucket}"


@pytest.fixture
def pandas_data_frame():
    df = pd.DataFrame(
        {
            "feature1": pd.Series(np.arange(10.0), dtype="float64"),
            "feature2": pd.Series(np.arange(10), dtype="int64"),
            "feature3": pd.Series(["2020-10-30T03:43:21Z"] * 10, dtype="string"),
            "feature4": pd.Series(np.arange(5.0), dtype="float64"),  # contains nan
        }
    )
    return df


@pytest.fixture
def pandas_data_frame_without_string():
    df = pd.DataFrame(
        {
            "feature1": pd.Series(np.arange(10), dtype="int64"),
            "feature2": pd.Series([3141592.6535897] * 10, dtype="float64"),
        }
    )
    return df


@pytest.fixture
def record():
    return [
        FeatureValue(feature_name="feature1", value_as_string="10.0"),
        FeatureValue(feature_name="feature2", value_as_string="10"),
        FeatureValue(feature_name="feature3", value_as_string="2020-10-30T03:43:21Z"),
    ]


@pytest.fixture
def create_table_ddl():
    return (
        "CREATE EXTERNAL TABLE IF NOT EXISTS sagemaker_featurestore.{feature_group_name} (\n"
        "  feature1 FLOAT\n"
        "  feature2 INT\n"
        "  feature3 STRING\n"
        "  feature4 FLOAT\n"
        "  write_time TIMESTAMP\n"
        "  event_time TIMESTAMP\n"
        "  is_deleted BOOLEAN\n"
        ")\n"
        "ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'\n"
        "  STORED AS\n"
        "  INPUTFORMAT 'parquet.hive.DeprecatedParquetInputFormat'\n"
        "  OUTPUTFORMAT 'parquet.hive.DeprecatedParquetOutputFormat'\n"
        "LOCATION 's3://sagemaker-test-featurestore-{region}-{account}"
        "/{account}/sagemaker/us-east-2/offline-store/{feature_group_name}'"
    )


def test_create_feature_store_online_only(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        output = feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)

    assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_create_feature_store(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
    record,
    create_table_ddl,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        output = feature_group.create(
            s3_uri=offline_store_s3_uri,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)

        # Ingest data
        feature_group.put_record(record=record)
        ingestion_manager = feature_group.ingest(
            data_frame=pandas_data_frame, max_workers=3, wait=False
        )
        ingestion_manager.wait()

        # Query the integrated Glue table.
        athena_query = feature_group.athena_query()
        df = DataFrame()
        with timeout(minutes=10):
            while df.shape[0] < 11:
                athena_query.run(
                    query_string=f'SELECT * FROM "{athena_query.table_name}"',
                    output_location=f"{offline_store_s3_uri}/query_results",
                )
                athena_query.wait()
                assert "SUCCEEDED" == athena_query.get_query_execution().get("QueryExecution").get(
                    "Status"
                ).get("State")
                df = athena_query.as_dataframe()
                print(f"Found {df.shape[0]} records.")
                time.sleep(60)

        assert df.shape[0] == 11
        nans = pd.isna(df.loc[df["feature1"].isin([5, 6, 7, 8, 9])]["feature4"])
        for is_na in nans.items():
            assert is_na
        assert (
            create_table_ddl.format(
                feature_group_name=feature_group_name,
                region=feature_store_session.boto_session.region_name,
                account=feature_store_session.account_id(),
            )
            == feature_group.as_hive_ddl()
        )
    assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_ingest_without_string_feature(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame_without_string,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame_without_string)

    with cleanup_feature_group(feature_group):
        output = feature_group.create(
            s3_uri=offline_store_s3_uri,
            record_identifier_name="feature1",
            event_time_feature_name="feature2",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)

        ingestion_manager = feature_group.ingest(
            data_frame=pandas_data_frame_without_string, max_workers=3, wait=False
        )
        ingestion_manager.wait()

    assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def _wait_for_feature_group_create(feature_group: FeatureGroup):
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group Creation")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        print(feature_group.describe())
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")


@contextmanager
def cleanup_feature_group(feature_group: FeatureGroup):
    try:
        yield
    finally:
        try:
            feature_group.delete()
        except Exception:
            pass
