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

from sagemaker.feature_store.inputs import (
    OnlineStoreSecurityConfig,
    OnlineStoreConfig,
    S3StorageConfig,
    DataCatalogConfig,
    OfflineStoreConfig,
    FeatureParameter,
    TableFormatEnum,
    OnlineStoreStorageTypeEnum,
    Filter,
    FilterOperatorEnum,
    Identifier,
    FeatureValue,
)


def ordered(obj):
    """Helper function for dict comparison"""
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def test_online_store_security_config():
    config = OnlineStoreSecurityConfig(kms_key_id="kms")
    assert ordered(config.to_dict()) == ordered({"KmsKeyId": "kms"})


def test_online_store_config():
    config = OnlineStoreConfig(enable_online_store=True)
    assert ordered(config.to_dict()) == ordered({"EnableOnlineStore": True})

    config_with_kms = OnlineStoreConfig(
        enable_online_store=True,
        online_store_security_config=OnlineStoreSecurityConfig(kms_key_id="kms"),
    )
    assert ordered(config_with_kms.to_dict()) == ordered(
        {
            "EnableOnlineStore": True,
            "SecurityConfig": {
                "KmsKeyId": "kms",
            },
        }
    )

    config_with_inmemory = OnlineStoreConfig(
        enable_online_store=True,
        storage_type=OnlineStoreStorageTypeEnum.IN_MEMORY,
    )
    assert ordered(config_with_inmemory.to_dict()) == ordered(
        {
            "EnableOnlineStore": True,
            "StorageType": "InMemory",
        }
    )

    config_with_kms_standard = OnlineStoreConfig(
        enable_online_store=True,
        online_store_security_config=OnlineStoreSecurityConfig(kms_key_id="kms"),
        storage_type=OnlineStoreStorageTypeEnum.STANDARD,
    )
    assert ordered(config_with_kms_standard.to_dict()) == ordered(
        {
            "EnableOnlineStore": True,
            "SecurityConfig": {
                "KmsKeyId": "kms",
            },
            "StorageType": "Standard",
        }
    )


def test_s3_store_config():
    config = S3StorageConfig(s3_uri="uri", kms_key_id="kms")
    assert ordered(config.to_dict()) == ordered({"S3Uri": "uri", "KmsKeyId": "kms"})


def test_data_catalog_config():
    config = DataCatalogConfig(
        table_name="table",
        catalog="catalog",
        database="database",
    )
    assert ordered(config.to_dict()) == ordered(
        {
            "TableName": "table",
            "Catalog": "catalog",
            "Database": "database",
        }
    )


def test_offline_data_store_config():
    config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri="uri"))
    assert ordered(config.to_dict()) == ordered(
        {
            "S3StorageConfig": {"S3Uri": "uri"},
            "DisableGlueTableCreation": False,
        }
    )


def test_offline_data_store_config_with_glue_table_format():
    config = OfflineStoreConfig(
        s3_storage_config=S3StorageConfig(s3_uri="uri"),
        table_format=TableFormatEnum.GLUE,
    )
    assert ordered(config.to_dict()) == ordered(
        {
            "S3StorageConfig": {"S3Uri": "uri"},
            "DisableGlueTableCreation": False,
            "TableFormat": "Glue",
        }
    )


def test_offline_data_store_config_with_iceberg_table_format():
    config = OfflineStoreConfig(
        s3_storage_config=S3StorageConfig(s3_uri="uri"),
        table_format=TableFormatEnum.ICEBERG,
    )
    assert ordered(config.to_dict()) == ordered(
        {
            "S3StorageConfig": {"S3Uri": "uri"},
            "DisableGlueTableCreation": False,
            "TableFormat": "Iceberg",
        }
    )


def test_feature_metadata():
    config = FeatureParameter(key="key", value="value")
    assert ordered(config.to_dict()) == ordered({"Key": "key", "Value": "value"})


def test_filter():
    filter = Filter(name="name", value="value", operator=FilterOperatorEnum.CONTAINS)
    assert ordered(filter.to_dict()) == ordered(
        {
            "Name": "name",
            "Value": "value",
            "Operator": "Contains",
        }
    )


def test_filter_with_none_operator():
    filter = Filter(name="name", value="value", operator=None)
    assert ordered(filter.to_dict()) == ordered(
        {
            "Name": "name",
            "Value": "value",
        }
    )


def test_identifier():
    identifier = Identifier(
        feature_group_name="name",
        record_identifiers_value_as_string=["record_identifier"],
        feature_names=["feature_1"],
    )

    assert ordered(identifier.to_dict()) == ordered(
        {
            "FeatureGroupName": "name",
            "RecordIdentifiersValueAsString": ["record_identifier"],
            "FeatureNames": ["feature_1"],
        }
    )


def test_identifier_with_none_feature_names():
    identifier = Identifier(
        feature_group_name="name",
        record_identifiers_value_as_string=["record_identifier"],
        feature_names=None,
    )

    assert ordered(identifier.to_dict()) == ordered(
        {
            "FeatureGroupName": "name",
            "RecordIdentifiersValueAsString": ["record_identifier"],
        }
    )


def test_feature_value():
    value = FeatureValue(
        feature_name="feature1",
        value_as_string="value1",
    )

    assert ordered(value.to_dict()) == ordered(
        {
            "FeatureName": "feature1",
            "ValueAsString": "value1",
        }
    )

    collection_value = FeatureValue(
        feature_name="feature2",
        value_as_string_list=["value1", "value2"],
    )

    assert ordered(collection_value.to_dict()) == ordered(
        {
            "FeatureName": "feature2",
            "ValueAsStringList": ["value1", "value2"],
        }
    )
