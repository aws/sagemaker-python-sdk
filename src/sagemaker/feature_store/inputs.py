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
"""The input configs for FeatureStore.

A feature store serves as the single source of truth to store, retrieve,
remove, track, share, discover, and control access to features.

You can configure two types of feature stores, an online features store
and an offline feature store.

The online features store is a low latency, high availability cache for a
feature group that enables real-time lookup of records. Only the latest record is stored.

The offline feature store use when low (sub-second) latency reads are not needed.
This is the case when you want to store and serve features for exploration, model training,
and batch inference. The offline store uses your Amazon Simple Storage Service (Amazon S3)
bucket for storage. A prefixing scheme based on event time is used to store your data in Amazon S3.
"""
from __future__ import absolute_import

import abc
from typing import Dict, Any

import attr


class Config(abc.ABC):
    """Base config object for FeatureStore.

    Configs must implement the to_dict method.
    """

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Get the dictionary from attributes.

        Returns:
            dict contains the attributes.
        """

    @classmethod
    def construct_dict(cls, **kwargs) -> Dict[str, Any]:
        """Construct the dictionary based on the args.

        args:
            kwargs: args to be used to construct the dict.

        Returns:
            dict represents the given kwargs.
        """
        result = dict()
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, Config):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result


@attr.s
class OnlineStoreSecurityConfig(Config):
    """OnlineStoreSecurityConfig for FeatureStore.

    Attributes:
        kms_key_id (str): KMS key id.
    """

    kms_key_id: str = attr.ib(factory=str)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes."""
        return Config.construct_dict(KmsKeyId=self.kms_key_id)


@attr.s
class OnlineStoreConfig(Config):
    """OnlineStoreConfig for FeatureStore.

    Attributes:
        enable_online_store (bool): whether to enable the online store.
        online_store_security_config (OnlineStoreSecurityConfig): configuration of security setting.
    """

    enable_online_store: bool = attr.ib(default=True)
    online_store_security_config: OnlineStoreSecurityConfig = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            EnableOnlineStore=self.enable_online_store,
            SecurityConfig=self.online_store_security_config,
        )


@attr.s
class S3StorageConfig(Config):
    """S3StorageConfig for FeatureStore.

    Attributes:
        s3_uri (str): S3 URI.
        kms_key_id (str): KMS key id.
    """

    s3_uri: str = attr.ib()
    kms_key_id: str = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes provided.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            S3Uri=self.s3_uri,
            KmsKeyId=self.kms_key_id,
        )


@attr.s
class DataCatalogConfig(Config):
    """DataCatalogConfig for FeatureStore.

    Attributes:
        table_name (str): name of the table.
        catalog (str): name of the catalog.
        database (str): name of the database.
    """

    table_name: str = attr.ib(factory=str)
    catalog: str = attr.ib(factory=str)
    database: str = attr.ib(factory=str)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes provided.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            TableName=self.table_name,
            Catalog=self.catalog,
            Database=self.database,
        )


@attr.s
class OfflineStoreConfig(Config):
    """OfflineStoreConfig for FeatureStore.

    Attributes:
        s3_storage_config (S3StorageConfig): configuration of S3 storage.
        disable_glue_table_creation (bool): whether to disable the Glue table creation.
        data_catalog_config (DataCatalogConfig): configuration of the data catalog.
    """

    s3_storage_config: S3StorageConfig = attr.ib()
    disable_glue_table_creation: bool = attr.ib(default=False)
    data_catalog_config: DataCatalogConfig = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            DisableGlueTableCreation=self.disable_glue_table_creation,
            S3StorageConfig=self.s3_storage_config,
            DataCatalogConfig=self.data_catalog_config,
        )


@attr.s
class FeatureValue(Config):
    """FeatureValue for FeatureStore.

    Attributes:
        feature_name (str): name of the Feature.
        value_as_string (str): value of the Feature in string form.
    """

    feature_name: str = attr.ib(default=None)
    value_as_string: str = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes provided.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            FeatureName=self.feature_name,
            ValueAsString=self.value_as_string,
        )
