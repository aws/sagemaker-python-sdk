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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from sagemaker.feature_store.inputs import (
    OnlineStoreSecurityConfig,
    OnlineStoreConfig,
    S3StorageConfig,
    DataCatalogConfig,
    OfflineStoreConfig,
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
