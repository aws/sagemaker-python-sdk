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
"""Contains classes that loads user specified input sources (e.g. Feature Groups, S3 URIs, etc)."""
from __future__ import absolute_import

import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

import attr
from pyspark.sql import DataFrame

from sagemaker import Session
from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
    ParquetDataSource,
    IcebergTableDataSource,
)
from sagemaker.feature_store.feature_processor._spark_factory import SparkSessionFactory

T = TypeVar("T")

logger = logging.getLogger("sagemaker")


class InputLoader(Generic[T], ABC):
    """Loads the contents of a Feature Group's offline store or contents at an S3 URI."""

    @abstractmethod
    def load_from_feature_group(self, feature_group_data_source: FeatureGroupDataSource) -> T:
        """Load the data from a Feature Group's offline store.

        Args:
            feature_group_data_source (FeatureGroupDataSource): the feature group source.

        Returns:
            T: The contents of the offline store as an instance of type T.
        """

    @abstractmethod
    def load_from_s3(self, s3_data_source: Union[CSVDataSource, ParquetDataSource]) -> T:
        """Load the contents from an S3 based data source.

        Args:
            s3_data_source (Union[CSVDataSource, ParquetDataSource]): a data source that is based
                in S3.

        Returns:
            T: The contents stored at the data source as an instance of type T.
        """


@attr.s
class SparkDataFrameInputLoader(InputLoader[DataFrame]):
    """InputLoader that reads data in as a Spark DataFrame."""

    spark_session_factory: SparkSessionFactory = attr.ib()
    sagemaker_session: Optional[Session] = attr.ib(default=None)
    _supported_table_format = ["Iceberg", "Glue", None]

    def load_from_feature_group(
        self, feature_group_data_source: FeatureGroupDataSource
    ) -> DataFrame:
        """Load the contents of a Feature Group's offline store as a DataFrame.

        Args:
            feature_group_data_source (FeatureGroupDataSource): the Feature Group source.

        Raises:
            ValueError: If the Feature Group does not have an Offline Store.
            ValueError: If the Feature Group's Table Type is not supported by the feature_processor.

        Returns:
            DataFrame: A Spark DataFrame containing the contents of the Feature Group's
                offline store.
        """
        sagemaker_session: Session = self.sagemaker_session or Session()

        feature_group_name = feature_group_data_source.name
        feature_group = sagemaker_session.describe_feature_group(
            self._parse_name_from_arn(feature_group_name)
        )
        logger.debug(
            "Called describe_feature_group with %s and received: %s",
            feature_group_name,
            feature_group,
        )

        if "OfflineStoreConfig" not in feature_group:
            raise ValueError(
                f"Input Feature Groups must have an enabled Offline Store."
                f" Feature Group: {feature_group_name} does not have an Offline Store enabled."
            )

        offline_store_uri = feature_group["OfflineStoreConfig"]["S3StorageConfig"][
            "ResolvedOutputS3Uri"
        ]

        table_format = feature_group["OfflineStoreConfig"].get("TableFormat", None)

        if table_format not in self._supported_table_format:
            raise ValueError(
                f"Feature group with table format {table_format} is not supported. "
                f"The table format should be one of {self._supported_table_format}."
            )

        if table_format == "Iceberg":
            data_catalog_config = feature_group["OfflineStoreConfig"]["DataCatalogConfig"]
            return self.load_from_iceberg_table(
                IcebergTableDataSource(
                    offline_store_uri,
                    data_catalog_config["Catalog"],
                    data_catalog_config["Database"],
                    data_catalog_config["TableName"],
                )
            )

        return self.load_from_s3(ParquetDataSource(offline_store_uri))

    def load_from_s3(self, s3_data_source: Union[CSVDataSource, ParquetDataSource]) -> DataFrame:
        """Load the contents from an S3 based data source as a DataFrame.

        Args:
            s3_data_source (Union[CSVDataSource, ParquetDataSource]):
                A data source that is based in S3.

        Raises:
            ValueError: If an invalid DataSource is provided.

        Returns:
            DataFrame: Contents of the data loaded from S3.
        """
        spark_session = self.spark_session_factory.spark_session
        s3a_uri = s3_data_source.s3_uri.replace("s3://", "s3a://")

        if isinstance(s3_data_source, CSVDataSource):
            # TODO: Accept `schema` parameter. (Inferring schema requires a pass through every row)
            logger.info("Loading data from %s.", s3a_uri)
            return spark_session.read.csv(
                s3a_uri,
                header=s3_data_source.csv_header,
                inferSchema=s3_data_source.csv_infer_schema,
            )

        if isinstance(s3_data_source, ParquetDataSource):
            logger.info("Loading data from %s.", s3a_uri)
            return spark_session.read.parquet(s3a_uri)

        raise ValueError("An invalid data source was provided.")

    def load_from_iceberg_table(
        self, iceberg_table_data_source: IcebergTableDataSource
    ) -> DataFrame:
        """Load the contents from an Iceberg table as a DataFrame.

        Args:
            iceberg_table_data_source (IcebergTableDataSource): An Iceberg Table source.

        Returns:
            DataFrame: Contents of the Iceberg Table as a Spark DataFrame.
        """
        catalog = iceberg_table_data_source.catalog.lower()
        database = iceberg_table_data_source.database.lower()
        table = iceberg_table_data_source.table.lower()

        spark_session = self.spark_session_factory.get_spark_session_with_iceberg_config(
            iceberg_table_data_source.warehouse_s3_uri, catalog
        )

        return spark_session.table(f"{catalog}.{database}.{table}")

    def _parse_name_from_arn(self, fg_uri: str) -> str:
        """Parse a Feature Group's name from an arn.

        Args:
            fg_uri (str): a string identifier of the Feature Group.

        Returns:
            str: the name of the feature group.
        """
        if fg_uri.startswith("arn:aws:sagemaker:"):
            return fg_uri.split("/")[-1]

        return fg_uri
