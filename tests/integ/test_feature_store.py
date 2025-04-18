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

import datetime
import json
import time
import dateutil.parser as date_parser
from contextlib import contextmanager

import boto3
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from sagemaker.feature_store.feature_utils import get_feature_group_as_dataframe
from sagemaker.feature_store.feature_definition import (
    FractionalFeatureDefinition,
    StringFeatureDefinition,
    ListCollectionType,
)
from sagemaker.feature_store.feature_group import FeatureGroup, IngestionError
from sagemaker.feature_store.feature_store import FeatureStore
from sagemaker.feature_store.inputs import (
    FeatureValue,
    FeatureParameter,
    TableFormatEnum,
    Filter,
    ResourceEnum,
    Identifier,
    DeletionModeEnum,
    TtlDuration,
    OnlineStoreConfigUpdate,
    OnlineStoreStorageTypeEnum,
    ThroughputConfig,
    ThroughputModeEnum,
    ThroughputConfigUpdate,
    TargetStoreEnum,
)
from sagemaker.feature_store.dataset_builder import (
    JoinTypeEnum,
)
from sagemaker.session import get_execution_role, Session
from tests.integ.timeout import timeout
from urllib.parse import urlparse

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
def base_name():
    return f"my-base-{int(time.time() * 10**7)}"


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
def get_record_results_for_data_frame():
    return {
        "0.0": [
            {"FeatureName": "feature1", "ValueAsString": "0.0"},
            {"FeatureName": "feature2", "ValueAsString": "0"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "0.0"},
        ],
        "1.0": [
            {"FeatureName": "feature1", "ValueAsString": "1.0"},
            {"FeatureName": "feature2", "ValueAsString": "1"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "1.0"},
        ],
        "2.0": [
            {"FeatureName": "feature1", "ValueAsString": "2.0"},
            {"FeatureName": "feature2", "ValueAsString": "2"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "2.0"},
        ],
        "3.0": [
            {"FeatureName": "feature1", "ValueAsString": "3.0"},
            {"FeatureName": "feature2", "ValueAsString": "3"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "3.0"},
        ],
        "4.0": [
            {"FeatureName": "feature1", "ValueAsString": "4.0"},
            {"FeatureName": "feature2", "ValueAsString": "4"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "4.0"},
        ],
        "5.0": [
            {"FeatureName": "feature1", "ValueAsString": "5.0"},
            {"FeatureName": "feature2", "ValueAsString": "5"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "6.0": [
            {"FeatureName": "feature1", "ValueAsString": "6.0"},
            {"FeatureName": "feature2", "ValueAsString": "6"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "7.0": [
            {"FeatureName": "feature1", "ValueAsString": "7.0"},
            {"FeatureName": "feature2", "ValueAsString": "7"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "8.0": [
            {"FeatureName": "feature1", "ValueAsString": "8.0"},
            {"FeatureName": "feature2", "ValueAsString": "8"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "9.0": [
            {"FeatureName": "feature1", "ValueAsString": "9.0"},
            {"FeatureName": "feature2", "ValueAsString": "9"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
    }


@pytest.fixture
def pandas_data_frame_with_collection_type():
    df = pd.DataFrame(
        {
            "feature1": pd.Series(np.arange(10.0), dtype="float64"),
            "feature2": pd.Series(np.arange(10), dtype="int64"),
            "feature3": pd.Series(["2020-10-30T03:43:21Z"] * 10, dtype="string"),
            "feature4": pd.Series(np.arange(5.0), dtype="float64"),  # contains nan
            "feature5": pd.Series(
                [["a", "abc"], ["b", "c"], ["c", "f"], ["d"], []], dtype="object"
            ),
            "feature6": pd.Series([[1, 2], [1, 2, 3], [1, 5], [1], []], dtype="object"),
            "feature7": pd.Series(
                [[1.1, 2.3], [1.4, 2.5, 3.2, 25], [1.0, 5.3], [1.2], []], dtype="object"
            ),
            "feature8": pd.Series([[1, 2], [1, 2, 3], [1, 5], [1], [], []], dtype="object"),
            "feature9": pd.Series(
                [[1.1, 2.3], [1.4, 25, 3.2], [1.0, 3, 4], [1.2], []], dtype="object"
            ),
            "feature10": pd.Series(
                [["a", "abc"], ["b", "c"], ["c", "None"], ["d"], []], dtype="object"
            ),
        }
    )
    return df


@pytest.fixture
def get_record_results_for_data_frame_with_collection_type():
    return {
        "0.0": [
            {"FeatureName": "feature1", "ValueAsString": "0.0"},
            {"FeatureName": "feature2", "ValueAsString": "0"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "0.0"},
            {"FeatureName": "feature5", "ValueAsStringList": ["a", "abc"]},
            {"FeatureName": "feature6", "ValueAsStringList": ["1", "2"]},
            {"FeatureName": "feature7", "ValueAsStringList": ["1.1", "2.3"]},
            {"FeatureName": "feature8", "ValueAsStringList": ["1", "2"]},
            {"FeatureName": "feature9", "ValueAsStringList": ["1.1", "2.3"]},
            {"FeatureName": "feature10", "ValueAsStringList": ["a", "abc"]},
        ],
        "1.0": [
            {"FeatureName": "feature1", "ValueAsString": "1.0"},
            {"FeatureName": "feature2", "ValueAsString": "1"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "1.0"},
            {"FeatureName": "feature5", "ValueAsStringList": ["b", "c"]},
            {"FeatureName": "feature6", "ValueAsStringList": ["1", "2", "3"]},
            {"FeatureName": "feature7", "ValueAsStringList": ["1.4", "2.5", "3.2", "25"]},
            {"FeatureName": "feature8", "ValueAsStringList": ["1", "2", "3"]},
            {"FeatureName": "feature9", "ValueAsStringList": ["1.4", "25", "3.2"]},
            {"FeatureName": "feature10", "ValueAsStringList": ["b", "c"]},
        ],
        "2.0": [
            {"FeatureName": "feature1", "ValueAsString": "2.0"},
            {"FeatureName": "feature2", "ValueAsString": "2"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "2.0"},
            {"FeatureName": "feature5", "ValueAsStringList": ["c", "f"]},
            {"FeatureName": "feature6", "ValueAsStringList": ["1", "5"]},
            {"FeatureName": "feature7", "ValueAsStringList": ["1.0", "5.3"]},
            {"FeatureName": "feature8", "ValueAsStringList": ["1", "5"]},
            {"FeatureName": "feature9", "ValueAsStringList": ["1.0", "3", "4"]},
            {"FeatureName": "feature10", "ValueAsStringList": ["c", "None"]},
        ],
        "3.0": [
            {"FeatureName": "feature1", "ValueAsString": "3.0"},
            {"FeatureName": "feature2", "ValueAsString": "3"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "3.0"},
            {"FeatureName": "feature5", "ValueAsStringList": ["d"]},
            {"FeatureName": "feature6", "ValueAsStringList": ["1"]},
            {"FeatureName": "feature7", "ValueAsStringList": ["1.2"]},
            {"FeatureName": "feature8", "ValueAsStringList": ["1"]},
            {"FeatureName": "feature9", "ValueAsStringList": ["1.2"]},
            {"FeatureName": "feature10", "ValueAsStringList": ["d"]},
        ],
        "4.0": [
            {"FeatureName": "feature1", "ValueAsString": "4.0"},
            {"FeatureName": "feature2", "ValueAsString": "4"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "4.0"},
        ],
        "5.0": [
            {"FeatureName": "feature1", "ValueAsString": "5.0"},
            {"FeatureName": "feature2", "ValueAsString": "5"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "6.0": [
            {"FeatureName": "feature1", "ValueAsString": "6.0"},
            {"FeatureName": "feature2", "ValueAsString": "6"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "7.0": [
            {"FeatureName": "feature1", "ValueAsString": "7.0"},
            {"FeatureName": "feature2", "ValueAsString": "7"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "8.0": [
            {"FeatureName": "feature1", "ValueAsString": "8.0"},
            {"FeatureName": "feature2", "ValueAsString": "8"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "9.0": [
            {"FeatureName": "feature1", "ValueAsString": "9.0"},
            {"FeatureName": "feature2", "ValueAsString": "9"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
    }


@pytest.fixture
def get_record_results_for_data_frame_without_collection_type():
    return {
        "0.0": [
            {"FeatureName": "feature1", "ValueAsString": "0.0"},
            {"FeatureName": "feature2", "ValueAsString": "0"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "0.0"},
            {"FeatureName": "feature5", "ValueAsString": "['a', 'abc']"},
            {"FeatureName": "feature6", "ValueAsString": "[1, 2]"},
            {"FeatureName": "feature7", "ValueAsString": "[1.1, 2.3]"},
            {"FeatureName": "feature8", "ValueAsString": "[1, 2]"},
            {"FeatureName": "feature9", "ValueAsString": "[1.1, 2.3]"},
            {"FeatureName": "feature10", "ValueAsString": "['a', 'abc']"},
        ],
        "1.0": [
            {"FeatureName": "feature1", "ValueAsString": "1.0"},
            {"FeatureName": "feature2", "ValueAsString": "1"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "1.0"},
            {"FeatureName": "feature5", "ValueAsString": "['b', 'c']"},
            {"FeatureName": "feature6", "ValueAsString": "[1, 2, 3]"},
            {"FeatureName": "feature7", "ValueAsString": "[1.4, 2.5, 3.2, 25]"},
            {"FeatureName": "feature8", "ValueAsString": "[1, 2, 3]"},
            {"FeatureName": "feature9", "ValueAsString": "[1.4, 25, 3.2]"},
            {"FeatureName": "feature10", "ValueAsString": "['b', 'c']"},
        ],
        "2.0": [
            {"FeatureName": "feature1", "ValueAsString": "2.0"},
            {"FeatureName": "feature2", "ValueAsString": "2"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "2.0"},
            {"FeatureName": "feature5", "ValueAsString": "['c', 'f']"},
            {"FeatureName": "feature6", "ValueAsString": "[1, 5]"},
            {"FeatureName": "feature7", "ValueAsString": "[1.0, 5.3]"},
            {"FeatureName": "feature8", "ValueAsString": "[1, 5]"},
            {"FeatureName": "feature9", "ValueAsString": "[1.0, 3, 4]"},
            {"FeatureName": "feature10", "ValueAsString": "['c', 'None']"},
        ],
        "3.0": [
            {"FeatureName": "feature1", "ValueAsString": "3.0"},
            {"FeatureName": "feature2", "ValueAsString": "3"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "3.0"},
            {"FeatureName": "feature5", "ValueAsString": "['d']"},
            {"FeatureName": "feature6", "ValueAsString": "[1]"},
            {"FeatureName": "feature7", "ValueAsString": "[1.2]"},
            {"FeatureName": "feature8", "ValueAsString": "[1]"},
            {"FeatureName": "feature9", "ValueAsString": "[1.2]"},
            {"FeatureName": "feature10", "ValueAsString": "['d']"},
        ],
        "4.0": [
            {"FeatureName": "feature1", "ValueAsString": "4.0"},
            {"FeatureName": "feature2", "ValueAsString": "4"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "4.0"},
        ],
        "5.0": [
            {"FeatureName": "feature1", "ValueAsString": "5.0"},
            {"FeatureName": "feature2", "ValueAsString": "5"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "6.0": [
            {"FeatureName": "feature1", "ValueAsString": "6.0"},
            {"FeatureName": "feature2", "ValueAsString": "6"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "7.0": [
            {"FeatureName": "feature1", "ValueAsString": "7.0"},
            {"FeatureName": "feature2", "ValueAsString": "7"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "8.0": [
            {"FeatureName": "feature1", "ValueAsString": "8.0"},
            {"FeatureName": "feature2", "ValueAsString": "8"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
        "9.0": [
            {"FeatureName": "feature1", "ValueAsString": "9.0"},
            {"FeatureName": "feature2", "ValueAsString": "9"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
        ],
    }


@pytest.fixture
def base_dataframe():
    base_data = [
        [1, 187512346.0, 123, 128],
        [2, 187512347.0, 168, 258],
        [3, 187512348.0, 125, 184],
        [1, 187512349.0, 195, 206],
    ]
    return pd.DataFrame(
        base_data, columns=["base_id", "base_time", "base_feature_1", "base_feature_2"]
    )


@pytest.fixture
def feature_group_dataframe():
    feature_group_data = [
        [1, 187512246.0, 456, 325],
        [2, 187512247.0, 729, 693],
        [3, 187512348.0, 129, 901],
        [1, 187512449.0, 289, 286],
    ]
    return pd.DataFrame(
        feature_group_data, columns=["fg_id", "fg_time", "fg_feature_1", "fg_feature_2"]
    )


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
def historic_record():
    return [
        FeatureValue(feature_name="feature1", value_as_string="10.0"),
        FeatureValue(feature_name="feature2", value_as_string="7"),
        FeatureValue(feature_name="feature3", value_as_string="2020-10-29T03:43:21Z"),
    ]


@pytest.fixture
def record():
    return [
        FeatureValue(feature_name="feature1", value_as_string="10.0"),
        FeatureValue(feature_name="feature2", value_as_string="10"),
        FeatureValue(feature_name="feature3", value_as_string="2020-10-30T03:43:21Z"),
    ]


@pytest.fixture
def collection_type_record():
    return [
        FeatureValue(feature_name="feature1", value_as_string="10.0"),
        FeatureValue(feature_name="feature2", value_as_string="10"),
        FeatureValue(feature_name="feature3", value_as_string="2020-10-30T03:43:21Z"),
        FeatureValue(feature_name="feature4", value_as_string_list=["val1", "val2"]),
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
        "LOCATION '{resolved_output_s3_uri}'"
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
            tags=[{"Key": "key1", "Value": "value1"}],
        )
        _wait_for_feature_group_create(feature_group)
        assert feature_group.list_tags() == [{"Key": "key1", "Value": "value1"}]
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

        resolved_output_s3_uri = (
            feature_group.describe()
            .get("OfflineStoreConfig")
            .get("S3StorageConfig")
            .get("ResolvedOutputS3Uri")
        )
        # Ingest data
        feature_group.put_record(record=record)
        ingestion_manager = feature_group.ingest(
            data_frame=pandas_data_frame, max_workers=3, wait=False
        )
        ingestion_manager.wait()
        assert 0 == len(ingestion_manager.failed_rows)

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
                resolved_output_s3_uri=resolved_output_s3_uri,
            )
            == feature_group.as_hive_ddl()
        )
    assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_create_feature_store_ingest_with_offline_target_stores(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
    record,
    create_table_ddl,
    get_record_results_for_data_frame,
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

        resolved_output_s3_uri = (
            feature_group.describe()
            .get("OfflineStoreConfig")
            .get("S3StorageConfig")
            .get("ResolvedOutputS3Uri")
        )
        # Ingest data
        feature_group.put_record(record=record)
        ingestion_manager = feature_group.ingest(
            data_frame=pandas_data_frame,
            target_stores=[TargetStoreEnum.OFFLINE_STORE],
            max_workers=3,
            max_processes=2,
            wait=False,
        )
        ingestion_manager.wait()
        assert 0 == len(ingestion_manager.failed_rows)

        for index, value in pandas_data_frame["feature1"].items():
            assert feature_group.get_record(record_identifier_value_as_string=str(value)) is None

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
                resolved_output_s3_uri=resolved_output_s3_uri,
            )
            == feature_group.as_hive_ddl()
        )
        assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_create_feature_store_ingest_with_online_offline_target_stores(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
    record,
    create_table_ddl,
    get_record_results_for_data_frame,
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

        resolved_output_s3_uri = (
            feature_group.describe()
            .get("OfflineStoreConfig")
            .get("S3StorageConfig")
            .get("ResolvedOutputS3Uri")
        )
        # Ingest data
        feature_group.put_record(record=record)
        ingestion_manager = feature_group.ingest(
            data_frame=pandas_data_frame,
            target_stores=[TargetStoreEnum.ONLINE_STORE, TargetStoreEnum.OFFLINE_STORE],
            max_workers=3,
            max_processes=2,
            wait=False,
        )
        ingestion_manager.wait()
        assert 0 == len(ingestion_manager.failed_rows)

        for index, value in pandas_data_frame["feature1"].items():
            assert (
                feature_group.get_record(record_identifier_value_as_string=str(value))
                == get_record_results_for_data_frame[str(value)]
            )

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
                resolved_output_s3_uri=resolved_output_s3_uri,
            )
            == feature_group.as_hive_ddl()
        )
        assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_create_feature_store_ingest_with_online_target_stores(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
    record,
    create_table_ddl,
    get_record_results_for_data_frame,
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
        ingestion_manager = feature_group.ingest(
            data_frame=pandas_data_frame,
            target_stores=[TargetStoreEnum.ONLINE_STORE],
            max_workers=3,
            max_processes=2,
            wait=False,
        )
        ingestion_manager.wait()
        assert 0 == len(ingestion_manager.failed_rows)

        for index, value in pandas_data_frame["feature1"].items():
            assert (
                feature_group.get_record(record_identifier_value_as_string=str(value))
                == get_record_results_for_data_frame[str(value)]
            )

        feature_group.put_record(
            record=[
                FeatureValue(feature_name="feature1", value_as_string="100.0"),
                FeatureValue(feature_name="feature2", value_as_string="100"),
                FeatureValue(feature_name="feature3", value_as_string="2020-10-30T03:43:21Z"),
                FeatureValue(feature_name="feature4", value_as_string="100.0"),
            ],
            target_stores=[TargetStoreEnum.OFFLINE_STORE],
        )
        assert feature_group.get_record(record_identifier_value_as_string="100.0") is None

        # Query the integrated Glue table.
        athena_query = feature_group.athena_query()
        df = DataFrame()
        with timeout(minutes=10):
            while df.shape[0] < 1:
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

        assert df.shape[0] == 1
        assert df.loc[0, "feature1"] == 100.0
        assert df.loc[0, "feature2"] == 100
        assert df.loc[0, "feature3"] == "2020-10-30T03:43:21Z"
        assert df.loc[0, "feature4"] == 100.0
        assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_put_record_with_target_stores(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
    record,
    create_table_ddl,
    get_record_results_for_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)
    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=offline_store_s3_uri,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)
        feature_group.put_record(
            record=[
                FeatureValue(feature_name="feature1", value_as_string="100.0"),
                FeatureValue(feature_name="feature2", value_as_string="100"),
                FeatureValue(feature_name="feature3", value_as_string="2020-10-30T03:43:21Z"),
                FeatureValue(feature_name="feature4", value_as_string="100.0"),
            ],
            target_stores=[TargetStoreEnum.OFFLINE_STORE],
        )
        assert feature_group.get_record(record_identifier_value_as_string="100.0") is None

        feature_group.put_record(
            record=[
                FeatureValue(feature_name="feature1", value_as_string="100.0"),
                FeatureValue(feature_name="feature2", value_as_string="100"),
                FeatureValue(feature_name="feature3", value_as_string="2020-10-30T03:43:21Z"),
                FeatureValue(feature_name="feature4", value_as_string="100.0"),
            ],
            target_stores=[TargetStoreEnum.ONLINE_STORE],
        )
        assert feature_group.get_record(record_identifier_value_as_string="100.0") == [
            {"FeatureName": "feature1", "ValueAsString": "100.0"},
            {"FeatureName": "feature2", "ValueAsString": "100"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "100.0"},
        ]


def test_create_feature_group_iceberg_table_format(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=offline_store_s3_uri,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            table_format=TableFormatEnum.ICEBERG,
        )
        _wait_for_feature_group_create(feature_group)

        table_format = feature_group.describe().get("OfflineStoreConfig").get("TableFormat")
        assert table_format == "Iceberg"


def test_create_feature_group_glue_table_format(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=offline_store_s3_uri,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            table_format=TableFormatEnum.GLUE,
        )
        _wait_for_feature_group_create(feature_group)

        table_format = feature_group.describe().get("OfflineStoreConfig").get("TableFormat")
        assert table_format == "Glue"


def test_create_feature_group_in_memory_storage_type(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            online_store_storage_type=OnlineStoreStorageTypeEnum.IN_MEMORY,
        )
        _wait_for_feature_group_create(feature_group)

        storage_type = feature_group.describe().get("OnlineStoreConfig").get("StorageType")
        assert storage_type == "InMemory"


def test_create_feature_group_standard_storage_type(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            online_store_storage_type=OnlineStoreStorageTypeEnum.STANDARD,
        )
        _wait_for_feature_group_create(feature_group)

        storage_type = feature_group.describe().get("OnlineStoreConfig").get("StorageType")
        assert storage_type == "Standard"


def test_throughput_create_as_provisioned_and_update_to_ondemand(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)
    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            throughput_config=ThroughputConfig(ThroughputModeEnum.PROVISIONED, 4000, 4000),
        )
        _wait_for_feature_group_create(feature_group)

        tp_config = feature_group.describe().get("ThroughputConfig")
        mode = tp_config.get("ThroughputMode")
        rcu = tp_config.get("ProvisionedReadCapacityUnits")
        wcu = tp_config.get("ProvisionedWriteCapacityUnits")
        assert mode == ThroughputModeEnum.PROVISIONED.value
        assert rcu == 4000
        assert wcu == 4000

        feature_group.update(throughput_config=ThroughputConfigUpdate(ThroughputModeEnum.ON_DEMAND))
        _wait_for_feature_group_update(feature_group)

        tp_config = feature_group.describe().get("ThroughputConfig")
        mode = tp_config.get("ThroughputMode")
        assert mode == ThroughputModeEnum.ON_DEMAND.value


def test_throughput_create_as_ondemand_and_update_to_provisioned(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)
    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            throughput_config=ThroughputConfig(ThroughputModeEnum.ON_DEMAND),
        )
        _wait_for_feature_group_create(feature_group)

        tp_config = feature_group.describe().get("ThroughputConfig")
        mode = tp_config.get("ThroughputMode")
        assert mode == ThroughputModeEnum.ON_DEMAND.value

        feature_group.update(
            throughput_config=ThroughputConfigUpdate(ThroughputModeEnum.PROVISIONED, 100, 200)
        )
        _wait_for_feature_group_update(feature_group)

        tp_config = feature_group.describe().get("ThroughputConfig")
        mode = tp_config.get("ThroughputMode")
        rcu = tp_config.get("ProvisionedReadCapacityUnits")
        wcu = tp_config.get("ProvisionedWriteCapacityUnits")
        assert mode == ThroughputModeEnum.PROVISIONED.value
        assert rcu == 100
        assert wcu == 200


def test_ttl_duration(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
    record,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    record_identifier_value_as_string = record[0].value_as_string

    ttl_duration = TtlDuration(unit="Seconds", value=30)

    with cleanup_feature_group(feature_group):
        output = feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            ttl_duration=ttl_duration,
        )
        _wait_for_feature_group_create(feature_group)

        feature_store = FeatureStore(sagemaker_session=feature_store_session)

        describe = feature_group.describe()

        assert ttl_duration.unit == (
            describe.get("OnlineStoreConfig").get("TtlDuration").get("Unit")
        )

        assert ttl_duration.value == (
            describe.get("OnlineStoreConfig").get("TtlDuration").get("Value")
        )

        record[2].value_as_string = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .split("+")[0]
            + "Z"
        )

        feature_group.put_record(record=record)

        sample_record = feature_store_session.get_record(
            feature_group_name=feature_group_name,
            record_identifier_value_as_string=record_identifier_value_as_string,
            feature_names=None,
            expiration_time_response="Enabled",
        )
        assert sample_record.get("Record") is not None
        _verify_expires_at(sample_record.get("ExpiresAt"), record[2].value_as_string, ttl_duration)

        batch_records = feature_store.batch_get_record(
            identifiers=[
                Identifier(
                    feature_group_name=feature_group_name,
                    record_identifiers_value_as_string=[record_identifier_value_as_string],
                )
            ],
            expiration_time_response="Enabled",
        ).get("Records")
        assert len(batch_records) == 1
        assert batch_records[0].get("Record") == sample_record.get("Record")
        assert batch_records[0].get("ExpiresAt") == sample_record.get("ExpiresAt")

        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string
        )

        assert sample_record.get("Record") == retrieved_record

        sample_record = feature_store_session.get_record(
            feature_group_name=feature_group_name,
            record_identifier_value_as_string=record_identifier_value_as_string,
            feature_names=None,
        )

        assert sample_record.get("Record") is not None
        assert sample_record.get("ExpiresAt") is None

        batch_records = feature_store.batch_get_record(
            identifiers=[
                Identifier(
                    feature_group_name=feature_group_name,
                    record_identifiers_value_as_string=[record_identifier_value_as_string],
                )
            ],
            expiration_time_response="Disabled",
        ).get("Records")

        assert len(batch_records) == 1
        assert batch_records[0].get("Record") == sample_record.get("Record")
        assert batch_records[0].get("ExpiresAt") is None

        time.sleep(35)

        sample_record = feature_store_session.get_record(
            feature_group_name=feature_group_name,
            record_identifier_value_as_string=record_identifier_value_as_string,
            feature_names=None,
        )
        assert sample_record.get("Record") is None

        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string
        )
        assert retrieved_record is None

        batch_records = feature_store.batch_get_record(
            identifiers=[
                Identifier(
                    feature_group_name=feature_group_name,
                    record_identifiers_value_as_string=[record_identifier_value_as_string],
                )
            ],
            expiration_time_response="Enabled",
        )["Records"]
        assert len(batch_records) == 0

        # customer record level ttl
        ttl_duration = TtlDuration(unit="Seconds", value=15)

        record[2].value_as_string = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .split("+")[0]
            + "Z"
        )

        feature_group.put_record(record=record, ttl_duration=ttl_duration)

        sample_record = feature_store_session.get_record(
            feature_group_name=feature_group_name,
            record_identifier_value_as_string=record_identifier_value_as_string,
            feature_names=None,
            expiration_time_response="Enabled",
        )
        assert sample_record.get("Record") is not None
        _verify_expires_at(sample_record.get("ExpiresAt"), record[2].value_as_string, ttl_duration)

        time.sleep(15)

        sample_record = feature_store_session.get_record(
            feature_group_name=feature_group_name,
            record_identifier_value_as_string=record_identifier_value_as_string,
            feature_names=None,
        )
        assert sample_record.get("Record") is None

        # update default ttl for feature group
        ttl_duration = TtlDuration(unit="Seconds", value=40)

        online_store_config = OnlineStoreConfigUpdate(ttl_duration=ttl_duration)
        feature_group.update(online_store_config=online_store_config)
        # cache feature group for 5min, wait for new ttl to take effect
        time.sleep(300)

        describe = feature_group.describe()

        assert ttl_duration.unit == (
            describe.get("OnlineStoreConfig").get("TtlDuration").get("Unit")
        )

        assert ttl_duration.value == (
            describe.get("OnlineStoreConfig").get("TtlDuration").get("Value")
        )

        record[2].value_as_string = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .split("+")[0]
            + "Z"
        )

        feature_group.put_record(record=record)

        sample_record = feature_store_session.get_record(
            feature_group_name=feature_group_name,
            record_identifier_value_as_string=record_identifier_value_as_string,
            feature_names=None,
            expiration_time_response="Enabled",
        )
        assert sample_record.get("Record") is not None
        _verify_expires_at(sample_record.get("ExpiresAt"), record[2].value_as_string, ttl_duration)

        time.sleep(40)

        sample_record = feature_store_session.get_record(
            feature_group_name=feature_group_name,
            record_identifier_value_as_string=record_identifier_value_as_string,
            feature_names=None,
        )
        assert sample_record.get("Record") is None

    assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_get_and_batch_get_record(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
    record,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    record_identifier_value_as_string = record[0].value_as_string
    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)
        # Ingest data
        feature_group.put_record(record=record)
        # Retrieve data
        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
        )
        record_names = list(map(lambda r: r.feature_name, record))
        assert len(retrieved_record) == len(record_names)
        for feature in retrieved_record:
            assert feature["FeatureName"] in record_names
        removed_feature_name = record_names.pop()
        # Retrieve data
        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
            feature_names=record_names,
        )
        assert len(retrieved_record) == len(record_names)
        for feature in retrieved_record:
            assert feature["FeatureName"] in record_names
            assert feature["FeatureName"] is not removed_feature_name
        # Retrieve data
        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string="1.0",
        )
        assert retrieved_record is None

        # Retrieve data using batch_get_record
        feature_store = FeatureStore(sagemaker_session=feature_store_session)
        records = feature_store.batch_get_record(
            identifiers=[
                Identifier(
                    feature_group_name=feature_group_name,
                    record_identifiers_value_as_string=[record_identifier_value_as_string],
                    feature_names=record_names,
                )
            ]
        )["Records"]
        assert records[0]["FeatureGroupName"] == feature_group_name
        assert records[0]["RecordIdentifierValueAsString"] == record_identifier_value_as_string
        assert len(records[0]["Record"]) == len(record_names)
        for feature in records[0]["Record"]:
            assert feature["FeatureName"] in record_names
            assert feature["FeatureName"] is not removed_feature_name


def test_put_and_get_collection_type_record(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
    collection_type_record,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_definition_with_collection = [
        StringFeatureDefinition(feature_name="feature1"),
        StringFeatureDefinition(feature_name="feature2"),
        StringFeatureDefinition(feature_name="feature3"),
        StringFeatureDefinition(feature_name="feature4", collection_type=ListCollectionType()),
    ]
    feature_group.feature_definitions = feature_definition_with_collection
    record_identifier_value_as_string = collection_type_record[0].value_as_string
    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            online_store_storage_type=OnlineStoreStorageTypeEnum.IN_MEMORY,
        )
        _wait_for_feature_group_create(feature_group)
        # Ingest data
        feature_group.put_record(record=collection_type_record)
        # Retrieve data
        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
        )

        assert retrieved_record is not None
        record_names = list(map(lambda r: r.feature_name, collection_type_record))
        assert len(retrieved_record) == len(record_names)

        retrieved_feature_map = {}
        for feature in retrieved_record:
            assert feature["FeatureName"] in record_names
            retrieved_feature_map[feature["FeatureName"]] = (
                feature.get("ValueAsStringList")
                if feature.get("ValueAsString") is None
                else feature.get("ValueAsString")
            )

        assert collection_type_record[0].value_as_string == retrieved_feature_map.get("feature1")
        assert collection_type_record[1].value_as_string == retrieved_feature_map.get("feature2")
        assert collection_type_record[2].value_as_string == retrieved_feature_map.get("feature3")
        assert collection_type_record[3].value_as_string_list == retrieved_feature_map.get(
            "feature4"
        )


def test_soft_delete_record(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
    record,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    record_identifier_value_as_string = record[0].value_as_string
    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)
        # Ingest data
        feature_group.put_record(record=record)
        # Retrieve data
        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
        )
        assert retrieved_record is not None
        # Delete data
        feature_group.delete_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
            event_time=datetime.datetime.now().replace(microsecond=0).isoformat() + "Z",
        )
        # Retrieve data
        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
        )
        assert retrieved_record is None


def test_hard_delete_record(
    feature_store_session,
    role,
    feature_group_name,
    pandas_data_frame,
    historic_record,
    record,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    record_identifier_value_as_string = record[0].value_as_string
    historic_record_identifier_value_as_string = historic_record[0].value_as_string
    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)
        # Ingest data
        feature_group.put_record(record=record)
        # Retrieve data
        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
        )
        assert retrieved_record is not None
        # Delete data
        feature_group.delete_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
            event_time=datetime.datetime.now().replace(microsecond=0).isoformat() + "Z",
            deletion_mode=DeletionModeEnum.HARD_DELETE,
        )
        # Retrieve data
        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
        )
        assert retrieved_record is None
        # Ingest data
        feature_group.put_record(historic_record)
        # Retrieve data
        retrieved_record = feature_group.get_record(
            record_identifier_value_as_string=historic_record_identifier_value_as_string,
        )
        assert retrieved_record is not None


def test_update_feature_group(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=offline_store_s3_uri,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)

        new_feature_name = "new_feature"
        new_features = [FractionalFeatureDefinition(feature_name=new_feature_name)]
        feature_group.update(new_features)
        _wait_for_feature_group_update(feature_group)
        feature_definitions = feature_group.describe().get("FeatureDefinitions")
        assert any([True for elem in feature_definitions if new_feature_name in elem.values()])


def test_list_feature_groups(feature_store_session, role, feature_group_name, pandas_data_frame):
    feature_store = FeatureStore(sagemaker_session=feature_store_session)
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)
        output = feature_store.list_feature_groups(name_contains=feature_group_name)

    assert output["FeatureGroupSummaries"][0]["FeatureGroupName"] == feature_group_name


def test_feature_metadata(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
):
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=offline_store_s3_uri,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)

        parameter_additions = [
            FeatureParameter(key="key1", value="value1"),
            FeatureParameter(key="key2", value="value2"),
        ]
        description = "test description"
        feature_name = "feature1"
        feature_group.update_feature_metadata(
            feature_name=feature_name,
            description=description,
            parameter_additions=parameter_additions,
        )
        describe_feature_metadata = feature_group.describe_feature_metadata(
            feature_name=feature_name
        )
        print(describe_feature_metadata)
        assert description == describe_feature_metadata.get("Description")
        assert 2 == len(describe_feature_metadata.get("Parameters"))
        assert [
            {"Key": "key1", "Value": "value1"},
            {"Key": "key2", "Value": "value2"},
        ] == feature_group.list_parameters_for_feature_metadata(feature_name=feature_name)

        parameter_removals = ["key1"]
        feature_group.update_feature_metadata(
            feature_name=feature_name, parameter_removals=parameter_removals
        )
        describe_feature_metadata = feature_group.describe_feature_metadata(
            feature_name=feature_name
        )
        assert description == describe_feature_metadata.get("Description")
        assert 1 == len(describe_feature_metadata.get("Parameters"))
        assert [
            {"Key": "key2", "Value": "value2"}
        ] == feature_group.list_parameters_for_feature_metadata(feature_name=feature_name)


@pytest.mark.skip(reason="Failing test. Fix is pending.")
def test_search(feature_store_session, role, feature_group_name, pandas_data_frame):
    feature_store = FeatureStore(sagemaker_session=feature_store_session)
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(data_frame=pandas_data_frame)

    with cleanup_feature_group(feature_group):
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
        )
        _wait_for_feature_group_create(feature_group)
        output = feature_store.search(
            resource=ResourceEnum.FEATURE_GROUP,
            filters=[Filter(name="FeatureGroupName", value=feature_group_name)],
        )
    print(output)
    assert output["Results"][0]["FeatureGroup"]["FeatureGroupName"] == feature_group_name


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


def test_ingest_multi_process(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
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

        feature_group.ingest(
            data_frame=pandas_data_frame, max_workers=3, max_processes=2, wait=True
        )

    assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_create_dataset_with_feature_group_base(
    feature_store_session,
    region_name,
    role,
    base_name,
    feature_group_name,
    offline_store_s3_uri,
    base_dataframe,
    feature_group_dataframe,
):
    base = FeatureGroup(name=base_name, sagemaker_session=feature_store_session)
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    with cleanup_feature_group(base), cleanup_feature_group(feature_group):
        _create_feature_group_and_ingest_data(
            base, base_dataframe, offline_store_s3_uri, "base_id", "base_time", role
        )
        _create_feature_group_and_ingest_data(
            feature_group,
            feature_group_dataframe,
            offline_store_s3_uri,
            "fg_id",
            "fg_time",
            role,
        )
        base_table_name = _get_athena_table_name_after_data_replication(
            feature_store_session, base, offline_store_s3_uri
        )
        feature_group_table_name = _get_athena_table_name_after_data_replication(
            feature_store_session, feature_group, offline_store_s3_uri
        )

        with (
            timeout(minutes=10)
            and cleanup_offline_store(base, feature_store_session)
            and cleanup_offline_store(feature_group, feature_store_session)
        ):
            feature_store = FeatureStore(sagemaker_session=feature_store_session)
            df, query_string = (
                feature_store.create_dataset(base=base, output_path=offline_store_s3_uri)
                .with_number_of_recent_records_by_record_identifier(4)
                .with_feature_group(feature_group)
                .to_dataframe()
            )
            sorted_df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
            merged_df = base_dataframe.merge(
                feature_group_dataframe, left_on="base_id", right_on="fg_id"
            )

            expect_df = merged_df.sort_values(by=list(merged_df.columns)).reset_index(drop=True)

            expect_df.rename(
                columns={
                    "fg_id": "fg_id.1",
                    "fg_time": "fg_time.1",
                    "fg_feature_1": "fg_feature_1.1",
                    "fg_feature_2": "fg_feature_2.1",
                },
                inplace=True,
            )

            assert sorted_df.equals(expect_df)
            assert (
                query_string.strip()
                == "WITH fg_base AS (WITH table_base AS (\n"
                + "SELECT *\n"
                + "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + 'PARTITION BY origin_base."base_id", origin_base."base_time"\n'
                + 'ORDER BY origin_base."api_invocation_time" DESC, origin_base."write_time" DESC\n'
                + ") AS dedup_row_base\n"
                + f'FROM "sagemaker_featurestore"."{base_table_name}" origin_base\n'
                + ")\n"
                + "WHERE dedup_row_base = 1\n"
                + "),\n"
                + "deleted_base AS (\n"
                + "SELECT *\n"
                + "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + 'PARTITION BY origin_base."base_id"\n'
                + 'ORDER BY origin_base."base_time" DESC,'
                ' origin_base."api_invocation_time" DESC,'
                ' origin_base."write_time" DESC\n'
                + ") AS deleted_row_base\n"
                + f'FROM "sagemaker_featurestore"."{base_table_name}" origin_base\n'
                + "WHERE is_deleted\n"
                + ")\n"
                + "WHERE deleted_row_base = 1\n"
                + ")\n"
                + 'SELECT table_base."base_id", table_base."base_time",'
                ' table_base."base_feature_1", table_base."base_feature_2"\n'
                + "FROM (\n"
                + 'SELECT table_base."base_id", table_base."base_time",'
                ' table_base."base_feature_1", table_base."base_feature_2",'
                ' table_base."write_time"\n'
                + "FROM table_base\n"
                + "LEFT JOIN deleted_base\n"
                + 'ON table_base."base_id" = deleted_base."base_id"\n'
                + 'WHERE deleted_base."base_id" IS NULL\n'
                + "UNION ALL\n"
                + 'SELECT table_base."base_id", table_base."base_time",'
                ' table_base."base_feature_1", table_base."base_feature_2",'
                ' table_base."write_time"\n'
                + "FROM deleted_base\n"
                + "JOIN table_base\n"
                + 'ON table_base."base_id" = deleted_base."base_id"\n'
                + "AND (\n"
                + 'table_base."base_time" > deleted_base."base_time"\n'
                + 'OR (table_base."base_time" = deleted_base."base_time" AND'
                ' table_base."api_invocation_time" >'
                ' deleted_base."api_invocation_time")\n'
                + 'OR (table_base."base_time" = deleted_base."base_time" AND'
                ' table_base."api_invocation_time" ='
                ' deleted_base."api_invocation_time" AND'
                ' table_base."write_time" > deleted_base."write_time")\n'
                + ")\n"
                + ") AS table_base\n"
                + "),\n"
                + "fg_0 AS (WITH table_0 AS (\n"
                + "SELECT *\n"
                + "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + 'PARTITION BY origin_0."fg_id", origin_0."fg_time"\n'
                + 'ORDER BY origin_0."api_invocation_time" DESC, origin_0."write_time" DESC\n'
                + ") AS dedup_row_0\n"
                + f'FROM "sagemaker_featurestore"."{feature_group_table_name}" origin_0\n'
                + ")\n"
                + "WHERE dedup_row_0 = 1\n"
                + "),\n"
                + "deleted_0 AS (\n"
                + "SELECT *\n"
                + "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + 'PARTITION BY origin_0."fg_id"\n'
                + 'ORDER BY origin_0."fg_time" DESC, origin_0."api_invocation_time" DESC,'
                ' origin_0."write_time" DESC\n'
                + ") AS deleted_row_0\n"
                + f'FROM "sagemaker_featurestore"."{feature_group_table_name}" origin_0\n'
                + "WHERE is_deleted\n"
                + ")\n"
                + "WHERE deleted_row_0 = 1\n"
                + ")\n"
                + 'SELECT table_0."fg_id", table_0."fg_time", table_0."fg_feature_1",'
                ' table_0."fg_feature_2"\n'
                + "FROM (\n"
                + 'SELECT table_0."fg_id", table_0."fg_time",'
                ' table_0."fg_feature_1", table_0."fg_feature_2",'
                ' table_0."write_time"\n'
                + "FROM table_0\n"
                + "LEFT JOIN deleted_0\n"
                + 'ON table_0."fg_id" = deleted_0."fg_id"\n'
                + 'WHERE deleted_0."fg_id" IS NULL\n'
                + "UNION ALL\n"
                + 'SELECT table_0."fg_id", table_0."fg_time",'
                ' table_0."fg_feature_1", table_0."fg_feature_2",'
                ' table_0."write_time"\n'
                + "FROM deleted_0\n"
                + "JOIN table_0\n"
                + 'ON table_0."fg_id" = deleted_0."fg_id"\n'
                + "AND (\n"
                + 'table_0."fg_time" > deleted_0."fg_time"\n'
                + 'OR (table_0."fg_time" = deleted_0."fg_time" AND'
                ' table_0."api_invocation_time" >'
                ' deleted_0."api_invocation_time")\n'
                + 'OR (table_0."fg_time" = deleted_0."fg_time" AND'
                ' table_0."api_invocation_time" ='
                ' deleted_0."api_invocation_time" AND table_0."write_time" >'
                ' deleted_0."write_time")\n'
                + ")\n"
                + ") AS table_0\n"
                + ")\n"
                + "SELECT base_id, base_time, base_feature_1, base_feature_2,"
                ' "fg_id.1", "fg_time.1", "fg_feature_1.1",'
                ' "fg_feature_2.1"\n' + "FROM (\n" + "SELECT fg_base.base_id, fg_base.base_time,"
                " fg_base.base_feature_1, fg_base.base_feature_2,"
                ' fg_0."fg_id" as "fg_id.1", fg_0."fg_time" as "fg_time.1",'
                ' fg_0."fg_feature_1" as "fg_feature_1.1",'
                ' fg_0."fg_feature_2" as "fg_feature_2.1", row_number()'
                " OVER (\n"
                + 'PARTITION BY fg_base."base_id"\n'
                + 'ORDER BY fg_base."base_time" DESC, fg_0."fg_time" DESC\n'
                + ") AS row_recent\n"
                + "FROM fg_base\n"
                + "JOIN fg_0\n"
                + 'ON fg_base."base_id" = fg_0."fg_id"\n'
                + ")\n"
                + "WHERE row_recent <= 4"
            )


def test_create_dataset_with_feature_group_base_with_additional_params(
    feature_store_session,
    region_name,
    role,
    base_name,
    feature_group_name,
    offline_store_s3_uri,
    base_dataframe,
    feature_group_dataframe,
):
    base = FeatureGroup(name=base_name, sagemaker_session=feature_store_session)
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    with cleanup_feature_group(base), cleanup_feature_group(feature_group):
        _create_feature_group_and_ingest_data(
            base, base_dataframe, offline_store_s3_uri, "base_id", "base_time", role
        )
        _create_feature_group_and_ingest_data(
            feature_group,
            feature_group_dataframe,
            offline_store_s3_uri,
            "fg_id",
            "fg_time",
            role,
        )
        base_table_name = _get_athena_table_name_after_data_replication(
            feature_store_session, base, offline_store_s3_uri
        )
        feature_group_table_name = _get_athena_table_name_after_data_replication(
            feature_store_session, feature_group, offline_store_s3_uri
        )

        with (
            timeout(minutes=10)
            and cleanup_offline_store(base, feature_store_session)
            and cleanup_offline_store(feature_group, feature_store_session)
        ):
            feature_store = FeatureStore(sagemaker_session=feature_store_session)
            df, query_string = (
                feature_store.create_dataset(base=base, output_path=offline_store_s3_uri)
                .with_number_of_recent_records_by_record_identifier(4)
                .with_feature_group(
                    feature_group,
                    target_feature_name_in_base="base_time",
                    feature_name_in_target="fg_time",
                    join_type=JoinTypeEnum.FULL_JOIN,
                )
                .to_dataframe()
            )
            sorted_df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
            merged_df = base_dataframe.merge(
                feature_group_dataframe,
                left_on="base_time",
                right_on="fg_time",
                how="outer",
            )

            expect_df = merged_df.sort_values(by=list(merged_df.columns)).reset_index(drop=True)

            expect_df.rename(
                columns={
                    "fg_id": "fg_id.1",
                    "fg_time": "fg_time.1",
                    "fg_feature_1": "fg_feature_1.1",
                    "fg_feature_2": "fg_feature_2.1",
                },
                inplace=True,
            )

            assert sorted_df.equals(expect_df)
            assert (
                query_string.strip()
                == "WITH fg_base AS (WITH table_base AS (\n"
                + "SELECT *\n"
                + "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + 'PARTITION BY origin_base."base_id", origin_base."base_time"\n'
                + 'ORDER BY origin_base."api_invocation_time" DESC, origin_base."write_time" DESC\n'
                + ") AS dedup_row_base\n"
                + f'FROM "sagemaker_featurestore"."{base_table_name}" origin_base\n'
                + ")\n"
                + "WHERE dedup_row_base = 1\n"
                + "),\n"
                + "deleted_base AS (\n"
                + "SELECT *\n"
                + "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + 'PARTITION BY origin_base."base_id"\n'
                + 'ORDER BY origin_base."base_time" DESC,'
                ' origin_base."api_invocation_time" DESC,'
                ' origin_base."write_time" DESC\n'
                + ") AS deleted_row_base\n"
                + f'FROM "sagemaker_featurestore"."{base_table_name}" origin_base\n'
                + "WHERE is_deleted\n"
                + ")\n"
                + "WHERE deleted_row_base = 1\n"
                + ")\n"
                + 'SELECT table_base."base_id", table_base."base_time",'
                ' table_base."base_feature_1", table_base."base_feature_2"\n'
                + "FROM (\n"
                + 'SELECT table_base."base_id", table_base."base_time",'
                ' table_base."base_feature_1", table_base."base_feature_2",'
                ' table_base."write_time"\n'
                + "FROM table_base\n"
                + "LEFT JOIN deleted_base\n"
                + 'ON table_base."base_id" = deleted_base."base_id"\n'
                + 'WHERE deleted_base."base_id" IS NULL\n'
                + "UNION ALL\n"
                + 'SELECT table_base."base_id", table_base."base_time",'
                ' table_base."base_feature_1", table_base."base_feature_2",'
                ' table_base."write_time"\n'
                + "FROM deleted_base\n"
                + "JOIN table_base\n"
                + 'ON table_base."base_id" = deleted_base."base_id"\n'
                + "AND (\n"
                + 'table_base."base_time" > deleted_base."base_time"\n'
                + 'OR (table_base."base_time" = deleted_base."base_time" AND'
                ' table_base."api_invocation_time" >'
                ' deleted_base."api_invocation_time")\n'
                + 'OR (table_base."base_time" = deleted_base."base_time" AND'
                ' table_base."api_invocation_time" ='
                ' deleted_base."api_invocation_time" AND'
                ' table_base."write_time" > deleted_base."write_time")\n'
                + ")\n"
                + ") AS table_base\n"
                + "),\n"
                + "fg_0 AS (WITH table_0 AS (\n"
                + "SELECT *\n"
                + "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + 'PARTITION BY origin_0."fg_id", origin_0."fg_time"\n'
                + 'ORDER BY origin_0."api_invocation_time" DESC, origin_0."write_time" DESC\n'
                + ") AS dedup_row_0\n"
                + f'FROM "sagemaker_featurestore"."{feature_group_table_name}" origin_0\n'
                + ")\n"
                + "WHERE dedup_row_0 = 1\n"
                + "),\n"
                + "deleted_0 AS (\n"
                + "SELECT *\n"
                + "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + 'PARTITION BY origin_0."fg_id"\n'
                + 'ORDER BY origin_0."fg_time" DESC, origin_0."api_invocation_time" DESC,'
                ' origin_0."write_time" DESC\n'
                + ") AS deleted_row_0\n"
                + f'FROM "sagemaker_featurestore"."{feature_group_table_name}" origin_0\n'
                + "WHERE is_deleted\n"
                + ")\n"
                + "WHERE deleted_row_0 = 1\n"
                + ")\n"
                + 'SELECT table_0."fg_id", table_0."fg_time", table_0."fg_feature_1",'
                ' table_0."fg_feature_2"\n'
                + "FROM (\n"
                + 'SELECT table_0."fg_id", table_0."fg_time",'
                ' table_0."fg_feature_1", table_0."fg_feature_2",'
                ' table_0."write_time"\n'
                + "FROM table_0\n"
                + "LEFT JOIN deleted_0\n"
                + 'ON table_0."fg_id" = deleted_0."fg_id"\n'
                + 'WHERE deleted_0."fg_id" IS NULL\n'
                + "UNION ALL\n"
                + 'SELECT table_0."fg_id", table_0."fg_time",'
                ' table_0."fg_feature_1", table_0."fg_feature_2",'
                ' table_0."write_time"\n'
                + "FROM deleted_0\n"
                + "JOIN table_0\n"
                + 'ON table_0."fg_id" = deleted_0."fg_id"\n'
                + "AND (\n"
                + 'table_0."fg_time" > deleted_0."fg_time"\n'
                + 'OR (table_0."fg_time" = deleted_0."fg_time" AND'
                ' table_0."api_invocation_time" >'
                ' deleted_0."api_invocation_time")\n'
                + 'OR (table_0."fg_time" = deleted_0."fg_time" AND'
                ' table_0."api_invocation_time" ='
                ' deleted_0."api_invocation_time" AND table_0."write_time" >'
                ' deleted_0."write_time")\n'
                + ")\n"
                + ") AS table_0\n"
                + ")\n"
                + "SELECT base_id, base_time, base_feature_1, base_feature_2,"
                ' "fg_id.1", "fg_time.1", "fg_feature_1.1",'
                ' "fg_feature_2.1"\n' + "FROM (\n" + "SELECT fg_base.base_id, fg_base.base_time,"
                " fg_base.base_feature_1, fg_base.base_feature_2,"
                ' fg_0."fg_id" as "fg_id.1", fg_0."fg_time" as "fg_time.1",'
                ' fg_0."fg_feature_1" as "fg_feature_1.1",'
                ' fg_0."fg_feature_2" as "fg_feature_2.1", row_number()'
                " OVER (\n"
                + 'PARTITION BY fg_base."base_id"\n'
                + 'ORDER BY fg_base."base_time" DESC, fg_0."fg_time" DESC\n'
                + ") AS row_recent\n"
                + "FROM fg_base\n"
                + "FULL JOIN fg_0\n"
                + 'ON fg_base."base_time" = fg_0."fg_time"\n'
                + ")\n"
                + "WHERE row_recent <= 4"
            )


def _verify_expires_at(expires_at: str, event_time: str, ttl_duration: TtlDuration):

    if ttl_duration.unit == "Seconds":
        ttl_duration_secs = ttl_duration.value
    elif ttl_duration.unit == "Minutes":
        ttl_duration_secs = ttl_duration.value * 60
    elif ttl_duration.unit == "Hours":
        ttl_duration_secs = ttl_duration.value * 60 * 60
    elif ttl_duration.unit == "Days":
        ttl_duration_secs = ttl_duration.value * 60 * 60 * 24
    elif ttl_duration.unit == "Weeks":
        ttl_duration_secs = ttl_duration.value * 60 * 60 * 24 * 7

    expected_ttl_secs = date_parser.parse(event_time).timestamp() + ttl_duration_secs
    actual_ttl_secs = date_parser.parse(expires_at).timestamp()

    assert actual_ttl_secs == expected_ttl_secs


def _create_feature_group_and_ingest_data(
    feature_group: FeatureGroup,
    dataframe: DataFrame,
    offline_store_s3_uri: str,
    record_identifier_name: str,
    event_time_name: str,
    role: str,
):
    feature_group.load_feature_definitions(data_frame=dataframe)
    feature_group.create(
        s3_uri=offline_store_s3_uri,
        record_identifier_name=record_identifier_name,
        event_time_feature_name=event_time_name,
        role_arn=role,
        enable_online_store=True,
    )
    _wait_for_feature_group_create(feature_group)

    ingestion_manager = feature_group.ingest(data_frame=dataframe, max_workers=3, wait=False)
    ingestion_manager.wait()
    assert 0 == len(ingestion_manager.failed_rows)


def _get_athena_table_name_after_data_replication(
    feature_store_session, feature_group: FeatureGroup, offline_store_s3_uri
):
    feature_group_metadata = feature_group.describe()
    resolved_output_s3_uri = (
        feature_group_metadata.get("OfflineStoreConfig", None)
        .get("S3StorageConfig", None)
        .get("ResolvedOutputS3Uri", None)
    )
    s3_prefix = resolved_output_s3_uri.replace(f"{offline_store_s3_uri}/", "")
    region_name = feature_store_session.boto_session.region_name
    s3_client = feature_store_session.boto_session.client(
        service_name="s3", region_name=region_name
    )
    while True:
        objects_in_bucket = s3_client.list_objects(
            Bucket=offline_store_s3_uri.replace("s3://", ""), Prefix=s3_prefix
        )
        if "Contents" in objects_in_bucket and len(objects_in_bucket["Contents"]) > 1:
            break
        else:
            print(f"Waiting for {feature_group.name} data in offline store...")
            time.sleep(60)
    print(f"{feature_group.name} data available.")
    return (
        feature_group_metadata.get("OfflineStoreConfig", None)
        .get("DataCatalogConfig", None)
        .get("TableName", None)
    )


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


def _wait_for_feature_group_update(feature_group: FeatureGroup):
    status = feature_group.describe().get("LastUpdateStatus").get("Status")
    while status == "InProgress":
        print("Waiting for Feature Group Update")
        time.sleep(5)
        status = feature_group.describe().get("LastUpdateStatus").get("Status")
    if status != "Successful":
        print(feature_group.describe())
        raise RuntimeError(f"Failed to update feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully updated.")


def test_get_feature_group_with_region(
    feature_store_session,
    region_name,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
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

        feature_group.ingest(
            data_frame=pandas_data_frame, max_workers=3, max_processes=2, wait=True
        )

        dataset = get_feature_group_as_dataframe(
            feature_group_name=feature_group_name,
            region=str(region_name),
            event_time_feature_name="feature3",
            latest_ingestion=True,
            athena_bucket=f"{offline_store_s3_uri}/query",
            verbose=False,
        )

        assert isinstance(dataset, DataFrame)
    assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_get_feature_group_with_session(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame,
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

        feature_group.ingest(
            data_frame=pandas_data_frame, max_workers=3, max_processes=2, wait=True
        )

        dataset = get_feature_group_as_dataframe(
            feature_group_name=feature_group_name,
            session=feature_store_session,
            event_time_feature_name="feature3",
            latest_ingestion=True,
            athena_bucket=f"{offline_store_s3_uri}/query",
            verbose=False,
            low_memory=False,
        )  # Using kwargs to pass a parameter to pandas.read_csv

        assert isinstance(dataset, DataFrame)
    assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")


def test_ingest_in_memory_multi_process_with_collection_types(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame_with_collection_type,
    get_record_results_for_data_frame_with_collection_type,
):
    feature_group = FeatureGroup(feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(
        data_frame=pandas_data_frame_with_collection_type,
        online_storage_type=OnlineStoreStorageTypeEnum.IN_MEMORY,
    )

    with cleanup_feature_group(feature_group):
        output = feature_group.create(
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            online_store_storage_type=OnlineStoreStorageTypeEnum.IN_MEMORY,
            s3_uri=False,
        )
        _wait_for_feature_group_create(feature_group)

        ingestion_manager = feature_group.ingest(
            data_frame=pandas_data_frame_with_collection_type,
            max_workers=3,
            max_processes=2,
            wait=True,
        )
        ingestion_manager.wait()
        assert 0 == len(ingestion_manager.failed_rows)
        assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")
        for index, value in pandas_data_frame_with_collection_type["feature1"].items():
            assert (
                feature_group.get_record(record_identifier_value_as_string=str(value))
                == get_record_results_for_data_frame_with_collection_type[str(value)]
            )

        new_row_data = [
            10.0,
            10,
            "2020-10-30T03:43:21Z",
            5.0,
            ["a", "b"],
            [1, 2, None],
            [3.0, 4.0],
            [1, 2],
            [3.0, 4.0],
            ["a", "b"],
        ]
        pandas_data_frame_with_collection_type.loc[len(pandas_data_frame_with_collection_type)] = (
            new_row_data
        )
        with pytest.raises(IngestionError):
            feature_group.ingest(
                data_frame=pandas_data_frame_with_collection_type,
                max_workers=1,
                max_processes=1,
                wait=True,
            )


def test_ingest_in_memory_single_process_with_collection_types(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame_with_collection_type,
    get_record_results_for_data_frame_with_collection_type,
):
    feature_group = FeatureGroup(feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(
        data_frame=pandas_data_frame_with_collection_type,
        online_storage_type=OnlineStoreStorageTypeEnum.IN_MEMORY,
    )

    with cleanup_feature_group(feature_group):
        output = feature_group.create(
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            online_store_storage_type=OnlineStoreStorageTypeEnum.IN_MEMORY,
            s3_uri=False,
        )
        _wait_for_feature_group_create(feature_group)

        ingestion_manager = feature_group.ingest(
            data_frame=pandas_data_frame_with_collection_type,
            max_workers=1,
            max_processes=1,
            wait=True,
        )
        assert 0 == len(ingestion_manager.failed_rows)
        assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")
        for index, value in pandas_data_frame_with_collection_type["feature1"].items():
            assert (
                feature_group.get_record(record_identifier_value_as_string=str(value))
                == get_record_results_for_data_frame_with_collection_type[str(value)]
            )

        new_row_data = [
            10.0,
            10,
            "2020-10-30T03:43:21Z",
            5.0,
            ["a", "b"],
            [1, 2, None],
            [3.0, 4.0],
            [1, 2],
            [3.0, 4.0],
            ["a", "b"],
        ]
        pandas_data_frame_with_collection_type.loc[len(pandas_data_frame_with_collection_type)] = (
            new_row_data
        )
        with pytest.raises(IngestionError):
            feature_group.ingest(
                data_frame=pandas_data_frame_with_collection_type,
                max_workers=1,
                max_processes=1,
                wait=True,
            )


def test_ingest_standard_multi_process_with_collection_types(
    feature_store_session,
    role,
    feature_group_name,
    offline_store_s3_uri,
    pandas_data_frame_with_collection_type,
    get_record_results_for_data_frame_without_collection_type,
):
    feature_group = FeatureGroup(feature_group_name, sagemaker_session=feature_store_session)
    feature_group.load_feature_definitions(
        data_frame=pandas_data_frame_with_collection_type,
        online_storage_type=OnlineStoreStorageTypeEnum.STANDARD,
    )

    with cleanup_feature_group(feature_group):
        output = feature_group.create(
            record_identifier_name="feature1",
            event_time_feature_name="feature3",
            role_arn=role,
            enable_online_store=True,
            online_store_storage_type=OnlineStoreStorageTypeEnum.STANDARD,
            s3_uri=False,
        )
        _wait_for_feature_group_create(feature_group)

        new_row_data = [
            10.0,
            10,
            "2020-10-30T03:43:21Z",
            5.0,
            ["a", "b"],
            [1, 2, None],
            [3.0, 4.0],
            [1, 2],
            [3.0, 4.0],
            ["a", "b"],
        ]
        pandas_data_frame_with_collection_type.loc[len(pandas_data_frame_with_collection_type)] = (
            new_row_data
        )

        ingestion_manager = feature_group.ingest(
            data_frame=pandas_data_frame_with_collection_type,
            max_workers=3,
            max_processes=2,
            wait=True,
        )
        ingestion_manager.wait()
        assert 0 == len(ingestion_manager.failed_rows)
        get_record_results_for_data_frame_without_collection_type["10.0"] = [
            {"FeatureName": "feature1", "ValueAsString": "10.0"},
            {"FeatureName": "feature2", "ValueAsString": "10"},
            {"FeatureName": "feature3", "ValueAsString": "2020-10-30T03:43:21Z"},
            {"FeatureName": "feature4", "ValueAsString": "5.0"},
            {"FeatureName": "feature5", "ValueAsString": "['a', 'b']"},
            {"FeatureName": "feature6", "ValueAsString": "[1, 2, None]"},
            {"FeatureName": "feature7", "ValueAsString": "[3.0, 4.0]"},
            {"FeatureName": "feature8", "ValueAsString": "[1, 2]"},
            {"FeatureName": "feature9", "ValueAsString": "[3.0, 4.0]"},
            {"FeatureName": "feature10", "ValueAsString": "['a', 'b']"},
        ]
        assert output["FeatureGroupArn"].endswith(f"feature-group/{feature_group_name}")
        for index, value in pandas_data_frame_with_collection_type["feature1"].items():
            assert (
                feature_group.get_record(record_identifier_value_as_string=str(value))
                == get_record_results_for_data_frame_without_collection_type[str(value)]
            )


@contextmanager
def cleanup_feature_group(feature_group: FeatureGroup):
    try:
        yield
    finally:
        try:
            feature_group.delete()
            print(f"{feature_group.name} is deleted")
        except Exception:
            raise RuntimeError(f"Failed to delete feature group with name {feature_group.name}")


@contextmanager
def cleanup_offline_store(feature_group: FeatureGroup, feature_store_session: Session):
    try:
        yield
    finally:
        feature_group_metadata = feature_group.describe()
        feature_group_name = feature_group_metadata["FeatureGroupName"]
        try:
            s3_uri = feature_group_metadata["OfflineStoreConfig"]["S3StorageConfig"][
                "ResolvedOutputS3Uri"
            ]
            parsed_uri = urlparse(s3_uri)
            bucket_name, prefix = parsed_uri.netloc, parsed_uri.path
            prefix = prefix.strip("/")
            prefix = prefix[:-5] if prefix.endswith("/data") else prefix
            region_name = feature_store_session.boto_session.region_name
            s3_client = feature_store_session.boto_session.client(
                service_name="s3", region_name=region_name
            )
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            files_in_folder = response["Contents"]
            files_to_delete = []
            for f in files_in_folder:
                files_to_delete.append({"Key": f["Key"]})
            s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": files_to_delete})
        except Exception as e:
            raise RuntimeError(f"Failed to delete data for feature_group {feature_group_name}", e)
