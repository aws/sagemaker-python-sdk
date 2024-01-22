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


import pandas as pd
import pytest
from mock import Mock, patch, MagicMock
from botocore.exceptions import ProfileNotFound

from sagemaker.feature_store.feature_definition import (
    FractionalFeatureDefinition,
    IntegralFeatureDefinition,
    StringFeatureDefinition,
    FeatureTypeEnum,
    VectorCollectionType,
    SetCollectionType,
    ListCollectionType,
)
from sagemaker.feature_store.feature_group import (
    FeatureGroup,
    IngestionManagerPandas,
    AthenaQuery,
    IngestionError,
)
from sagemaker.feature_store.inputs import (
    FeatureParameter,
    DeletionModeEnum,
    TtlDuration,
    OnlineStoreConfigUpdate,
    OnlineStoreStorageTypeEnum,
    ThroughputModeEnum,
    ThroughputConfig,
    ThroughputConfigUpdate,
)

from tests.unit import SAGEMAKER_CONFIG_FEATURE_GROUP


class PicklableMock(Mock):
    def __reduce__(self):
        return (Mock, ())


@pytest.fixture
def role_arn():
    return "arn:role"


@pytest.fixture
def s3_uri():
    return "s3://some/uri"


@pytest.fixture
def sagemaker_session_mock():
    sagemaker_session_mock = Mock()
    sagemaker_session_mock.sagemaker_config = {}
    return sagemaker_session_mock


@pytest.fixture
def fs_runtime_client_config_mock():
    return PicklableMock()


@pytest.fixture
def feature_group_dummy_definitions():
    return [
        FractionalFeatureDefinition(feature_name="feature1"),
        IntegralFeatureDefinition(feature_name="feature2"),
        StringFeatureDefinition(feature_name="feature3"),
    ]


@pytest.fixture
def create_table_ddl():
    return (
        "CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table_name} (\n"
        "  feature1 FLOAT\n"
        "  feature2 INT\n"
        "  feature3 STRING\n"
        "  write_time TIMESTAMP\n"
        "  event_time TIMESTAMP\n"
        "  is_deleted BOOLEAN\n"
        ")\n"
        "ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'\n"
        "  STORED AS\n"
        "  INPUTFORMAT 'parquet.hive.DeprecatedParquetInputFormat'\n"
        "  OUTPUTFORMAT 'parquet.hive.DeprecatedParquetOutputFormat'\n"
        "LOCATION 's3://resolved_output_s3_uri'"
    )


def test_feature_group_create_without_role(
    sagemaker_session_mock, feature_group_dummy_definitions, s3_uri
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    with pytest.raises(ValueError):
        feature_group.create(
            s3_uri=s3_uri,
            record_identifier_name="feature1",
            event_time_feature_name="feature2",
            enable_online_store=True,
        )


def test_feature_store_create_with_config_injection(
    sagemaker_session, role_arn, feature_group_dummy_definitions, s3_uri
):

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_FEATURE_GROUP
    sagemaker_session.create_feature_group = Mock()

    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session)
    feature_group.feature_definitions = feature_group_dummy_definitions
    feature_group.create(
        s3_uri=s3_uri,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        enable_online_store=True,
    )
    expected_offline_store_kms_key_id = SAGEMAKER_CONFIG_FEATURE_GROUP["SageMaker"]["FeatureGroup"][
        "OfflineStoreConfig"
    ]["S3StorageConfig"]["KmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_FEATURE_GROUP["SageMaker"]["FeatureGroup"]["RoleArn"]
    expected_online_store_kms_key_id = SAGEMAKER_CONFIG_FEATURE_GROUP["SageMaker"]["FeatureGroup"][
        "OnlineStoreConfig"
    ]["SecurityConfig"]["KmsKeyId"]
    sagemaker_session.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        role_arn=expected_role_arn,
        description=None,
        tags=None,
        online_store_config={
            "EnableOnlineStore": True,
            "SecurityConfig": {"KmsKeyId": expected_online_store_kms_key_id},
        },
        offline_store_config={
            "DisableGlueTableCreation": False,
            "S3StorageConfig": {
                "S3Uri": s3_uri,
                "KmsKeyId": expected_offline_store_kms_key_id,
            },
        },
    )


def test_feature_store_create(
    sagemaker_session_mock, role_arn, feature_group_dummy_definitions, s3_uri
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    feature_group.create(
        s3_uri=s3_uri,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn=role_arn,
        enable_online_store=True,
    )
    sagemaker_session_mock.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        role_arn=role_arn,
        description=None,
        tags=None,
        online_store_config={"EnableOnlineStore": True},
        offline_store_config={
            "DisableGlueTableCreation": False,
            "S3StorageConfig": {"S3Uri": s3_uri},
        },
    )


def test_feature_store_create_with_ttl_duration(
    sagemaker_session_mock, role_arn, feature_group_dummy_definitions, s3_uri
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    ttl_duration = TtlDuration(unit="Minutes", value=123)
    feature_group.create(
        s3_uri=s3_uri,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn=role_arn,
        enable_online_store=True,
        ttl_duration=ttl_duration,
    )
    sagemaker_session_mock.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        role_arn=role_arn,
        description=None,
        tags=None,
        online_store_config={
            "EnableOnlineStore": True,
            "TtlDuration": ttl_duration.to_dict(),
        },
        offline_store_config={
            "DisableGlueTableCreation": False,
            "S3StorageConfig": {"S3Uri": s3_uri},
        },
    )


def test_feature_store_create_online_only(
    sagemaker_session_mock, role_arn, feature_group_dummy_definitions
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    feature_group.create(
        s3_uri=False,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn=role_arn,
        enable_online_store=True,
    )
    sagemaker_session_mock.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        role_arn=role_arn,
        description=None,
        tags=None,
        online_store_config={"EnableOnlineStore": True},
    )


def test_feature_store_create_online_only_with_in_memory(
    sagemaker_session_mock, role_arn, feature_group_dummy_definitions
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    feature_group.create(
        s3_uri=False,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn=role_arn,
        enable_online_store=True,
        online_store_storage_type=OnlineStoreStorageTypeEnum.IN_MEMORY,
    )
    sagemaker_session_mock.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        role_arn=role_arn,
        description=None,
        tags=None,
        online_store_config={"EnableOnlineStore": True, "StorageType": "InMemory"},
    )


def test_feature_store_create_with_in_memory_collection_types(
    sagemaker_session_mock, role_arn, feature_group_dummy_definitions
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_definition_with_collection = [
        FractionalFeatureDefinition(feature_name="feature1"),
        IntegralFeatureDefinition(feature_name="feature2"),
        StringFeatureDefinition(feature_name="feature3"),
        FractionalFeatureDefinition(
            feature_name="feature4",
            collection_type=VectorCollectionType(dimension=2000),
        ),
        IntegralFeatureDefinition(feature_name="feature5", collection_type=SetCollectionType()),
        StringFeatureDefinition(feature_name="feature6", collection_type=ListCollectionType()),
    ]
    feature_group.feature_definitions = feature_definition_with_collection

    feature_group.create(
        s3_uri=False,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn=role_arn,
        enable_online_store=True,
        online_store_storage_type=OnlineStoreStorageTypeEnum.IN_MEMORY,
    )
    sagemaker_session_mock.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_definition_with_collection],
        role_arn=role_arn,
        description=None,
        tags=None,
        online_store_config={"EnableOnlineStore": True, "StorageType": "InMemory"},
    )


def test_feature_store_create_in_provisioned_throughput_mode(
    sagemaker_session_mock, role_arn, feature_group_dummy_definitions
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    feature_group.create(
        s3_uri=False,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn=role_arn,
        enable_online_store=True,
        throughput_config=ThroughputConfig(ThroughputModeEnum.PROVISIONED, 1000, 2000),
    )
    sagemaker_session_mock.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        role_arn=role_arn,
        description=None,
        tags=None,
        online_store_config={"EnableOnlineStore": True},
        throughput_config={
            "ThroughputMode": "Provisioned",
            "ProvisionedReadCapacityUnits": 1000,
            "ProvisionedWriteCapacityUnits": 2000,
        },
    )


def test_feature_store_create_in_ondemand_throughput_mode(
    sagemaker_session_mock, role_arn, feature_group_dummy_definitions
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    feature_group.create(
        s3_uri=False,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn=role_arn,
        enable_online_store=True,
        throughput_config=ThroughputConfig(ThroughputModeEnum.ON_DEMAND),
    )

    sagemaker_session_mock.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        role_arn=role_arn,
        description=None,
        tags=None,
        online_store_config={"EnableOnlineStore": True},
        throughput_config={"ThroughputMode": "OnDemand"},
    )


def test_feature_store_delete(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.delete()
    sagemaker_session_mock.delete_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup"
    )


def test_feature_store_describe(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.describe()
    sagemaker_session_mock.describe_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup", next_token=None
    )


def test_feature_store_update(sagemaker_session_mock, feature_group_dummy_definitions):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.update(feature_group_dummy_definitions)
    sagemaker_session_mock.update_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        feature_additions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        throughput_config=None,
        online_store_config=None,
    )


def test_feature_store_throughput_update_to_provisioned(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.update(
        throughput_config=ThroughputConfigUpdate(ThroughputModeEnum.PROVISIONED, 999, 777)
    )
    sagemaker_session_mock.update_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        feature_additions=None,
        throughput_config={
            "ThroughputMode": "Provisioned",
            "ProvisionedReadCapacityUnits": 999,
            "ProvisionedWriteCapacityUnits": 777,
        },
        online_store_config=None,
    )


def test_feature_store_throughput_update_to_ondemand(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.update(throughput_config=ThroughputConfigUpdate(ThroughputModeEnum.ON_DEMAND))
    sagemaker_session_mock.update_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        feature_additions=None,
        throughput_config={"ThroughputMode": "OnDemand"},
        online_store_config=None,
    )


def test_feature_store_update_with_ttl_duration(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    online_store_config = OnlineStoreConfigUpdate(
        ttl_duration=TtlDuration(unit="Minutes", value=123)
    )
    feature_group.update(online_store_config=online_store_config)
    sagemaker_session_mock.update_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        feature_additions=None,
        online_store_config=online_store_config.to_dict(),
        throughput_config=None,
    )


def test_feature_metadata_update(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)

    parameter_additions = [FeatureParameter(key="key1", value="value1")]
    parameter_removals = ["key2"]

    feature_group.update_feature_metadata(
        feature_name="Feature1",
        description="TestDescription",
        parameter_additions=parameter_additions,
        parameter_removals=parameter_removals,
    )
    sagemaker_session_mock.update_feature_metadata.assert_called_with(
        feature_group_name="MyFeatureGroup",
        feature_name="Feature1",
        description="TestDescription",
        parameter_additions=[pa.to_dict() for pa in parameter_additions],
        parameter_removals=parameter_removals,
    )
    feature_group.update_feature_metadata(feature_name="Feature1", description="TestDescription")
    sagemaker_session_mock.update_feature_metadata.assert_called_with(
        feature_group_name="MyFeatureGroup",
        feature_name="Feature1",
        description="TestDescription",
        parameter_additions=[],
        parameter_removals=[],
    )


def test_feature_metadata_describe(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.describe_feature_metadata(feature_name="Feature1")
    sagemaker_session_mock.describe_feature_metadata.assert_called_with(
        feature_group_name="MyFeatureGroup", feature_name="Feature1"
    )


def test_get_record(sagemaker_session_mock):
    feature_group_name = "MyFeatureGroup"
    feature_names = ["MyFeature1", "MyFeature2"]
    record_identifier_value_as_string = "1.0"
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session_mock)
    feature_group.get_record(
        record_identifier_value_as_string=record_identifier_value_as_string,
        feature_names=feature_names,
    )
    sagemaker_session_mock.get_record.assert_called_with(
        record_identifier_value_as_string=record_identifier_value_as_string,
        feature_group_name=feature_group_name,
        feature_names=feature_names,
    )


def test_put_record(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.put_record(record=[])
    sagemaker_session_mock.put_record.assert_called_with(
        feature_group_name="MyFeatureGroup", record=[]
    )


def test_put_record_ttl_duration(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    ttl_duration = TtlDuration(unit="Minutes", value=123)
    feature_group.put_record(record=[], ttl_duration=ttl_duration)
    sagemaker_session_mock.put_record.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record=[],
        ttl_duration=ttl_duration.to_dict(),
    )


def test_delete_record(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    record_identifier_value_as_string = "1.0"
    event_time = "2022-09-14"
    feature_group.delete_record(
        record_identifier_value_as_string=record_identifier_value_as_string,
        event_time=event_time,
    )
    sagemaker_session_mock.delete_record.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_value_as_string=record_identifier_value_as_string,
        event_time=event_time,
        deletion_mode=DeletionModeEnum.SOFT_DELETE.value,
    )


def test_soft_delete_record(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    record_identifier_value_as_string = "1.0"
    event_time = "2022-09-14"
    feature_group.delete_record(
        record_identifier_value_as_string=record_identifier_value_as_string,
        event_time=event_time,
        deletion_mode=DeletionModeEnum.SOFT_DELETE,
    )
    sagemaker_session_mock.delete_record.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_value_as_string=record_identifier_value_as_string,
        event_time=event_time,
        deletion_mode=DeletionModeEnum.SOFT_DELETE.value,
    )


def test_hard_delete_record(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    record_identifier_value_as_string = "1.0"
    event_time = "2022-09-14"
    feature_group.delete_record(
        record_identifier_value_as_string=record_identifier_value_as_string,
        event_time=event_time,
        deletion_mode=DeletionModeEnum.HARD_DELETE,
    )
    sagemaker_session_mock.delete_record.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_value_as_string=record_identifier_value_as_string,
        event_time=event_time,
        deletion_mode=DeletionModeEnum.HARD_DELETE.value,
    )


def test_load_feature_definition(sagemaker_session_mock):
    feature_group = FeatureGroup(name="SomeGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame(
        {
            "float": pd.Series([2.0], dtype="float64"),
            "int": pd.Series([2], dtype="int64"),
            "string": pd.Series(["f1"], dtype="string"),
        }
    )
    feature_definitions = feature_group.load_feature_definitions(data_frame=df)
    names = [fd.feature_name for fd in feature_definitions]
    types = [fd.feature_type for fd in feature_definitions]
    assert names == ["float", "int", "string"]
    assert types == [
        FeatureTypeEnum.FRACTIONAL,
        FeatureTypeEnum.INTEGRAL,
        FeatureTypeEnum.STRING,
    ]


def test_load_feature_definition_unsupported_types(sagemaker_session_mock):
    feature_group = FeatureGroup(name="FailedGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame(
        {
            "float": pd.Series([2.0], dtype="float64"),
            "int": pd.Series([2], dtype="int64"),
            "bool": pd.Series([True], dtype="bool"),
        }
    )
    with pytest.raises(ValueError) as error:
        feature_group.load_feature_definitions(data_frame=df)
    assert "Failed to infer Feature type based on dtype bool for column bool." in str(error)


def test_list_tags(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    sagemaker_session_mock.describe_feature_group.return_value = {"FeatureGroupArn": "test-arn"}
    feature_group.list_tags()
    sagemaker_session_mock.list_tags.assert_called_with(resource_arn="test-arn")


def test_list_parameters_for_feature_metadata(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    test_feature_metadata = {"Parameters": [{"Key": "k", "Value": "y"}]}
    sagemaker_session_mock.describe_feature_metadata.return_value = test_feature_metadata
    assert feature_group.list_parameters_for_feature_metadata(feature_name="feature") == [
        {"Key": "k", "Value": "y"}
    ]
    sagemaker_session_mock.describe_feature_metadata.assert_called_with(
        feature_group_name="MyFeatureGroup", feature_name="feature"
    )


def test_ingest_zero_processes():
    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    df = Mock()
    with pytest.raises(RuntimeError) as error:
        feature_group.ingest(data_frame=df, max_workers=1, max_processes=0)

    assert "max_processes must be greater than 0." in str(error)


def test_ingest_zero_workers():
    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    df = Mock()
    with pytest.raises(RuntimeError) as error:
        feature_group.ingest(data_frame=df, max_workers=0, max_processes=1)

    assert "max_workers must be greater than 0." in str(error)


@patch("sagemaker.feature_store.feature_group.IngestionManagerPandas")
def test_ingest(ingestion_manager_init, sagemaker_session_mock, fs_runtime_client_config_mock):
    sagemaker_session_mock.sagemaker_featurestore_runtime_client.meta.config = (
        fs_runtime_client_config_mock
    )

    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame(dict((f"float{i}", pd.Series([2.0], dtype="float64")) for i in range(300)))

    mock_ingestion_manager_instance = Mock()
    ingestion_manager_init.return_value = mock_ingestion_manager_instance
    feature_group.ingest(data_frame=df, max_workers=10)

    ingestion_manager_init.assert_called_once_with(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        sagemaker_fs_runtime_client_config=fs_runtime_client_config_mock,
        max_workers=10,
        max_processes=1,
        profile_name=sagemaker_session_mock.boto_session.profile_name,
    )
    mock_ingestion_manager_instance.run.assert_called_once_with(
        data_frame=df, wait=True, timeout=None
    )


@patch("sagemaker.feature_store.feature_group.IngestionManagerPandas")
def test_ingest_default(ingestion_manager_init, sagemaker_session_mock):
    sagemaker_session_mock.sagemaker_featurestore_runtime_client.meta.config = (
        fs_runtime_client_config_mock
    )
    sagemaker_session_mock.boto_session.profile_name = "default"

    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame(dict((f"float{i}", pd.Series([2.0], dtype="float64")) for i in range(300)))

    mock_ingestion_manager_instance = Mock()
    ingestion_manager_init.return_value = mock_ingestion_manager_instance
    feature_group.ingest(data_frame=df)

    ingestion_manager_init.assert_called_once_with(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        sagemaker_fs_runtime_client_config=fs_runtime_client_config_mock,
        max_workers=1,
        max_processes=1,
        profile_name=None,
    )
    mock_ingestion_manager_instance.run.assert_called_once_with(
        data_frame=df, wait=True, timeout=None
    )


@patch("sagemaker.feature_store.feature_group.IngestionManagerPandas")
def test_ingest_with_profile_name(
    ingestion_manager_init, sagemaker_session_mock, fs_runtime_client_config_mock
):
    sagemaker_session_mock.sagemaker_featurestore_runtime_client.meta.config = (
        fs_runtime_client_config_mock
    )

    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame(dict((f"float{i}", pd.Series([2.0], dtype="float64")) for i in range(300)))

    mock_ingestion_manager_instance = Mock()
    ingestion_manager_init.return_value = mock_ingestion_manager_instance
    feature_group.ingest(data_frame=df, max_workers=10, profile_name="profile_name")

    ingestion_manager_init.assert_called_once_with(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        sagemaker_fs_runtime_client_config=fs_runtime_client_config_mock,
        max_workers=10,
        max_processes=1,
        profile_name="profile_name",
    )
    mock_ingestion_manager_instance.run.assert_called_once_with(
        data_frame=df, wait=True, timeout=None
    )


def test_as_hive_ddl_with_default_values(
    create_table_ddl, feature_group_dummy_definitions, sagemaker_session_mock
):
    sagemaker_session_mock.describe_feature_group.return_value = {
        "OfflineStoreConfig": {
            "S3StorageConfig": {
                "S3Uri": "s3://some-bucket",
                "ResolvedOutputS3Uri": "s3://resolved_output_s3_uri",
            }
        }
    }
    sagemaker_session_mock.account_id.return_value = "1234"
    sagemaker_session_mock.boto_session.region_name = "us-west-2"

    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    assert (
        create_table_ddl.format(
            database="sagemaker_featurestore",
            table_name="MyGroup",
            account="1234",
            region="us-west-2",
            feature_group_name="MyGroup",
        )
        == feature_group.as_hive_ddl()
    )


def test_as_hive_ddl(create_table_ddl, feature_group_dummy_definitions, sagemaker_session_mock):
    sagemaker_session_mock.describe_feature_group.return_value = {
        "OfflineStoreConfig": {
            "S3StorageConfig": {
                "S3Uri": "s3://some-bucket",
                "ResolvedOutputS3Uri": "s3://resolved_output_s3_uri",
            }
        }
    }
    sagemaker_session_mock.account_id.return_value = "1234"
    sagemaker_session_mock.boto_session.region_name = "us-west-2"

    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    assert create_table_ddl.format(
        database="MyDatabase",
        table_name="MyTable",
        account="1234",
        region="us-west-2",
        feature_group_name="MyGroup",
    ) == feature_group.as_hive_ddl(database="MyDatabase", table_name="MyTable")


@patch(
    "sagemaker.feature_store.feature_group.IngestionManagerPandas._run_multi_process",
    MagicMock(),
)
def test_ingestion_manager_run_success():
    df = pd.DataFrame({"float": pd.Series([2.0], dtype="float64")})
    manager = IngestionManagerPandas(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        sagemaker_fs_runtime_client_config=fs_runtime_client_config_mock,
        max_workers=10,
    )
    manager.run(df)

    manager._run_multi_process.assert_called_once_with(data_frame=df, wait=True, timeout=None)


@patch(
    "sagemaker.feature_store.feature_group.IngestionManagerPandas._run_multi_threaded",
    PicklableMock(return_value=[]),
)
def test_ingestion_manager_run_multi_process_with_multi_thread_success(
    fs_runtime_client_config_mock,
):
    df = pd.DataFrame({"float": pd.Series([2.0], dtype="float64")})
    manager = IngestionManagerPandas(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        sagemaker_fs_runtime_client_config=fs_runtime_client_config_mock,
        max_workers=2,
        max_processes=2,
    )
    manager.run(df)


@patch(
    "sagemaker.feature_store.feature_group.IngestionManagerPandas._ingest_single_batch",
    MagicMock(return_value=[1]),
)
def test_ingestion_manager_run_failure():
    df = pd.DataFrame({"float": pd.Series([2.0], dtype="float64")})
    manager = IngestionManagerPandas(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        sagemaker_fs_runtime_client_config=fs_runtime_client_config_mock,
        max_workers=2,
    )

    with pytest.raises(IngestionError) as error:
        manager.run(df)

    assert "Failed to ingest some data into FeatureGroup MyGroup" in str(error)
    assert error.value.failed_rows == [1, 1]
    assert manager.failed_rows == [1, 1]


@patch(
    "sagemaker.feature_store.feature_group.IngestionManagerPandas._ingest_single_batch",
    MagicMock(side_effect=ProfileNotFound(profile="non_exist")),
)
def test_ingestion_manager_with_profile_name_run_failure():
    df = pd.DataFrame({"float": pd.Series([2.0], dtype="float64")})
    manager = IngestionManagerPandas(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        sagemaker_fs_runtime_client_config=fs_runtime_client_config_mock,
        max_workers=1,
        profile_name="non_exist",
    )

    try:
        manager.run(df)
    except Exception as e:
        assert "The config profile (non_exist) could not be found" in str(e)


@patch(
    "sagemaker.feature_store.feature_group.IngestionManagerPandas._ingest_single_batch",
    PicklableMock(return_value=[1]),
)
def test_ingestion_manager_run_multi_process_failure():
    df = pd.DataFrame({"float": pd.Series([2.0], dtype="float64")})
    manager = IngestionManagerPandas(
        feature_group_name="MyGroup",
        sagemaker_session=None,
        sagemaker_fs_runtime_client_config=None,
        max_workers=2,
        max_processes=2,
    )

    with pytest.raises(IngestionError) as error:
        manager.run(df)

    assert "Failed to ingest some data into FeatureGroup MyGroup" in str(error)
    assert error.value.failed_rows == [1, 1, 1, 1]
    assert manager.failed_rows == [1, 1, 1, 1]


@pytest.fixture
def query(sagemaker_session_mock):
    return AthenaQuery(
        catalog="catalog",
        database="database",
        table_name="table_name",
        sagemaker_session=sagemaker_session_mock,
    )


def test_athena_query_run(sagemaker_session_mock, query):
    sagemaker_session_mock.start_query_execution.return_value = {"QueryExecutionId": "query_id"}
    query.run(
        query_string="query",
        output_location="s3://some-bucket/some-path",
        workgroup="workgroup",
    )
    sagemaker_session_mock.start_query_execution.assert_called_with(
        catalog="catalog",
        database="database",
        query_string="query",
        output_location="s3://some-bucket/some-path",
        kms_key=None,
        workgroup="workgroup",
    )
    assert "some-bucket" == query._result_bucket
    assert "some-path" == query._result_file_prefix
    assert "query_id" == query._current_query_execution_id


def test_athena_query_wait(sagemaker_session_mock, query):
    query._current_query_execution_id = "query_id"
    query.wait()
    sagemaker_session_mock.wait_for_athena_query.assert_called_with(query_execution_id="query_id")


def test_athena_query_get_query_execution(sagemaker_session_mock, query):
    query._current_query_execution_id = "query_id"
    query.get_query_execution()
    sagemaker_session_mock.get_query_execution.assert_called_with(query_execution_id="query_id")


@patch("tempfile.gettempdir", Mock(return_value="tmp"))
@patch("pandas.read_csv")
def test_athena_query_as_dataframe(read_csv, sagemaker_session_mock, query):
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
    }
    query._current_query_execution_id = "query_id"
    query._result_bucket = "bucket"
    query._result_file_prefix = "prefix"
    query.as_dataframe()
    sagemaker_session_mock.download_athena_query_result.assert_called_with(
        bucket="bucket",
        prefix="prefix",
        query_execution_id="query_id",
        filename="tmp/query_id.csv",
    )
    read_csv.assert_called_with(filepath_or_buffer="tmp/query_id.csv", delimiter=",")


@patch("tempfile.gettempdir", Mock(return_value="tmp"))
def test_athena_query_as_dataframe_query_failed(sagemaker_session_mock, query):
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "FAILED"}}
    }
    query._current_query_execution_id = "query_id"
    with pytest.raises(RuntimeError) as error:
        query.as_dataframe()
    assert "Failed to execute query query_id" in str(error)


@patch("tempfile.gettempdir", Mock(return_value="tmp"))
def test_athena_query_as_dataframe_query_queued(sagemaker_session_mock, query):
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "QUEUED"}}
    }
    query._current_query_execution_id = "query_id"
    with pytest.raises(RuntimeError) as error:
        query.as_dataframe()
    assert "Current query query_id is still being executed" in str(error)


@patch("tempfile.gettempdir", Mock(return_value="tmp"))
def test_athena_query_as_dataframe_query_running(sagemaker_session_mock, query):
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "RUNNING"}}
    }
    query._current_query_execution_id = "query_id"
    with pytest.raises(RuntimeError) as error:
        query.as_dataframe()
    assert "Current query query_id is still being executed" in str(error)
