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
"""The FeatureGroup entity for FeatureStore.

A feature group is a logical grouping of features, defined in the Feature Store,
to describe records. A feature group definition is composed of a list of feature definitions,
a record identifier name, and configurations for its online and offline store.
Create feature group, describe feature group, update feature groups, delete feature group and
list feature groups APIs can be used to manage feature groups.
"""

from __future__ import absolute_import

import logging
import math
import os
import tempfile
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, List, Dict, Any, Union
from urllib.parse import urlparse

import attr
import pandas as pd
from pandas import DataFrame

from sagemaker import Session
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)
from sagemaker.feature_store.inputs import (
    OnlineStoreConfig,
    OnlineStoreSecurityConfig,
    S3StorageConfig,
    OfflineStoreConfig,
    DataCatalogConfig,
    FeatureValue,
)

logger = logging.getLogger(__name__)


@attr.s
class AthenaQuery:
    """Class to manage querying of feature store data with AWS Athena.

    This class instantiates a AthenaQuery object that is used to retrieve data from feature store
    via standard SQL queries.

    Attributes:
        catalog (str): name of the data catalog.
        database (str): name of the database.
        table_name (str): name of the table.
        sagemaker_session (Session): instance of the Session class to perform boto calls.
    """

    catalog: str = attr.ib()
    database: str = attr.ib()
    table_name: str = attr.ib()
    sagemaker_session: Session = attr.ib()
    _current_query_execution_id: str = attr.ib(init=False, default=None)
    _result_bucket: str = attr.ib(init=False, default=None)
    _result_file_prefix: str = attr.ib(init=False, default=None)

    def run(self, query_string: str, output_location: str, kms_key: str = None) -> str:
        """Execute a SQL query given a query string, output location and kms key.

        This method executes the SQL query using Athena and outputs the results to output_location
        and returns the execution id of the query.

        Args:
            query_string: SQL query string.
            output_location: S3 URI of the query result.
            kms_key: KMS key id. If set, will be used to encrypt the query result file.

        Returns:
            Execution id of the query.
        """
        response = self.sagemaker_session.start_query_execution(
            catalog=self.catalog,
            database=self.database,
            query_string=query_string,
            output_location=output_location,
            kms_key=kms_key,
        )
        self._current_query_execution_id = response["QueryExecutionId"]
        parse_result = urlparse(output_location, allow_fragments=False)
        self._result_bucket = parse_result.netloc
        self._result_file_prefix = parse_result.path.strip("/")
        return self._current_query_execution_id

    def wait(self):
        """Wait for the current query to finish."""
        self.sagemaker_session.wait_for_athena_query(
            query_execution_id=self._current_query_execution_id
        )

    def get_query_execution(self) -> Dict[str, Any]:
        """Get execution status of the current query.

        Returns:
            Response dict from Athena.
        """
        return self.sagemaker_session.get_query_execution(
            query_execution_id=self._current_query_execution_id
        )

    def as_dataframe(self) -> DataFrame:
        """Download the result of the current query and load it into a DataFrame.

        Returns:
            A pandas DataFrame contains the query result.
        """
        query_state = self.get_query_execution().get("QueryExecution").get("Status").get("State")
        if query_state != "SUCCEEDED":
            if query_state in ("QUEUED", "RUNNING"):
                raise RuntimeError(
                    f"Current query {self._current_query_execution_id} is still being executed."
                )
            raise RuntimeError(f"Failed to execute query {self._current_query_execution_id}")

        output_filename = os.path.join(
            tempfile.gettempdir(), f"{self._current_query_execution_id}.csv"
        )
        self.sagemaker_session.download_athena_query_result(
            bucket=self._result_bucket,
            prefix=self._result_file_prefix,
            query_execution_id=self._current_query_execution_id,
            filename=output_filename,
        )
        return pd.read_csv(output_filename, delimiter=",")


@attr.s
class IngestionManagerPandas:
    """Class to manage the multi-threaded data ingestion process.

    This class will manage the data ingestion process which is multi-threaded.

    Attributes:
        feature_group_name (str): name of the Feature Group.
        sagemaker_session (Session): instance of the Session class to perform boto calls.
        data_frame (DataFrame): pandas DataFrame to be ingested to the given feature group.
        max_workers (int): number of threads to create.
    """

    feature_group_name: str = attr.ib()
    sagemaker_session: Session = attr.ib()
    data_frame: DataFrame = attr.ib()
    max_workers: int = attr.ib(default=1)
    _futures: Dict[Any, Any] = attr.ib(init=False, factory=dict)

    @staticmethod
    def _ingest_single_batch(
        data_frame: DataFrame,
        feature_group_name: str,
        sagemaker_session: Session,
        start_index: int,
        end_index: int,
    ):
        """Ingest a single batch of DataFrame rows into FeatureStore.

        Args:
            data_frame (DataFrame): source DataFrame to be ingested.
            feature_group_name (str): name of the Feature Group.
            sagemaker_session (Session): session instance to perform boto calls.
            start_index (int): starting position to ingest in this batch.
            end_index (int): ending position to ingest in this batch.
        """
        logger.info("Started ingesting index %d to %d", start_index, end_index)
        for row in data_frame[start_index:end_index].itertuples(index=False):
            record = [
                FeatureValue(
                    feature_name=data_frame.columns[index], value_as_string=str(row[index])
                )
                for index in range(len(row))
            ]
            sagemaker_session.put_record(
                feature_group_name=feature_group_name, record=[value.to_dict() for value in record]
            )

    def wait(self, timeout=None):
        """Wait for the ingestion process to finish.

        Args:
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
                if timeout is reached.
        """
        failed = False
        for future in as_completed(self._futures, timeout=timeout):
            start, end = self._futures[future]
            try:
                future.result()
            except Exception as e:  # pylint: disable=broad-except
                failed = True
                logger.error("Failed to ingest row %d to %d: %s", start, end, e)
            else:
                logger.info("Successfully ingested row %d to %d", start, end)

        if failed:
            raise RuntimeError(
                f"Failed to ingest some data into FeatureGroup {self.feature_group_name}"
            )

    def run(self, wait=True, timeout=None):
        """Start the ingestion process.

        Args:
            wait (bool): whether to wait for the ingestion to finish or not.
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
            if timeout is reached.
        """
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        batch_size = math.ceil(self.data_frame.shape[0] / self.max_workers)

        futures = {}
        for i in range(self.max_workers):
            start_index = min(i * batch_size, self.data_frame.shape[0])
            end_index = min(i * batch_size + batch_size, self.data_frame.shape[0])
            futures[
                executor.submit(
                    self._ingest_single_batch,
                    feature_group_name=self.feature_group_name,
                    sagemaker_session=self.sagemaker_session,
                    data_frame=self.data_frame,
                    start_index=start_index,
                    end_index=end_index,
                )
            ] = (start_index, end_index)

        self._futures = futures
        if wait:
            self.wait(timeout=timeout)
        executor.shutdown(wait=False)


@attr.s
class FeatureGroup:
    """FeatureGroup definition.

    This class instantiates a FeatureGroup object that comprises of a name for the FeatureGroup,
    session instance, and a list of feature definition objects i.e., FeatureDefinition.

    Attributes:
        name (str): name of the FeatureGroup instance.
        sagemaker_session (Session): session instance to perform boto calls.
        feature_definitions (Sequence[FeatureDefinition]): list of FeatureDefinitions.
    """

    name: str = attr.ib(factory=str)
    sagemaker_session: Session = attr.ib(default=Session)
    feature_definitions: Sequence[FeatureDefinition] = attr.ib(factory=list)

    _INTEGER_TYPES = [
        "int_",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]
    _FLOAT_TYPES = ["float_", "float16", "float32", "float64"]
    _DTYPE_TO_FEATURE_DEFINITION_CLS_MAP: Dict[str, FeatureTypeEnum] = {
        type: FeatureTypeEnum.INTEGRAL for type in _INTEGER_TYPES
    }
    _DTYPE_TO_FEATURE_DEFINITION_CLS_MAP.update(
        {type: FeatureTypeEnum.FRACTIONAL for type in _FLOAT_TYPES}
    )
    _DTYPE_TO_FEATURE_DEFINITION_CLS_MAP["string"] = FeatureTypeEnum.STRING

    _FEATURE_TYPE_TO_DDL_DATA_TYPE_MAP = {
        FeatureTypeEnum.INTEGRAL.value: "INT",
        FeatureTypeEnum.FRACTIONAL.value: "FLOAT",
        FeatureTypeEnum.STRING.value: "STRING",
    }

    def create(
        self,
        s3_uri: Union[str, bool],
        record_identifier_name: str,
        event_time_feature_name: str,
        role_arn: str,
        online_store_kms_key_id: str = None,
        enable_online_store: bool = False,
        offline_store_kms_key_id: str = None,
        disable_glue_table_creation: bool = False,
        data_catalog_config: DataCatalogConfig = None,
        description: str = None,
        tags: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a SageMaker FeatureStore FeatureGroup.

        Args:
            s3_uri (Union[str, bool]): S3 URI of the offline store, set to
                ``False`` to disable offline store.
            record_identifier_name (str): name of the record identifier feature.
            event_time_feature_name (str): name of the event time feature.
            role_arn (str): ARN of the role used to call CreateFeatureGroup.
            online_store_kms_key_id (str): KMS key id for online store.
            enable_online_store (bool): whether to enable online store or not.
            offline_store_kms_key_id (str): KMS key id for offline store.
            disable_glue_table_creation (bool): whether to turn off Glue table creation no not.
            data_catalog_config (DataCatalogConfig): configuration for Metadata store.
            description (str): description of the FeatureGroup.
            tags (List[Dict[str, str]]): list of tags for labeling a FeatureGroup.

        Returns:
            Response dict from service.
        """
        create_feature_store_args = dict(
            feature_group_name=self.name,
            record_identifier_name=record_identifier_name,
            event_time_feature_name=event_time_feature_name,
            feature_definitions=[
                feature_definition.to_dict() for feature_definition in self.feature_definitions
            ],
            role_arn=role_arn,
            description=description,
            tags=tags,
        )

        # online store configuration
        if enable_online_store:
            online_store_config = OnlineStoreConfig(enable_online_store=enable_online_store)
            if online_store_kms_key_id is not None:
                online_store_config.online_store_security_config = OnlineStoreSecurityConfig(
                    kms_key_id=online_store_kms_key_id
                )
            create_feature_store_args.update({"online_store_config": online_store_config.to_dict()})

        # offline store configuration
        if s3_uri:
            s3_storage_config = S3StorageConfig(s3_uri=s3_uri)
            if offline_store_kms_key_id:
                s3_storage_config.kms_key_id = offline_store_kms_key_id
            offline_store_config = OfflineStoreConfig(
                s3_storage_config=s3_storage_config,
                disable_glue_table_creation=disable_glue_table_creation,
                data_catalog_config=data_catalog_config,
            )
            create_feature_store_args.update(
                {"offline_store_config": offline_store_config.to_dict()}
            )

        return self.sagemaker_session.create_feature_group(**create_feature_store_args)

    def delete(self):
        """Delete a FeatureGroup."""
        self.sagemaker_session.delete_feature_group(feature_group_name=self.name)

    def describe(self, next_token: str = None) -> Dict[str, Any]:
        """Describe a FeatureGroup.

        Args:
            next_token (str): next_token to get next page of features.

        Returns:
            Response dict from the service.
        """
        return self.sagemaker_session.describe_feature_group(
            feature_group_name=self.name, next_token=next_token
        )

    def load_feature_definitions(
        self,
        data_frame: DataFrame,
    ) -> Sequence[FeatureDefinition]:
        """Load feature definitions from a Pandas DataFrame.

        Column name is used as feature name. Feature type is inferred from the dtype
        of the column. Dtype int_, int8, int16, int32, int64, uint8, uint16, uint32
        and uint64 are mapped to Integral feature type. Dtype float_, float16, float32
        and float64 are mapped to Fractional feature type. string dtype is mapped to
        String feature type.

        No feature definitions will be loaded if the given data_frame contains
        unsupported dtypes.

        Args:
            data_frame (DataFrame):

        Returns:
            list of FeatureDefinition
        """
        feature_definitions = []
        for column in data_frame:
            feature_type = self._DTYPE_TO_FEATURE_DEFINITION_CLS_MAP.get(
                str(data_frame[column].dtype), None
            )
            if feature_type:
                feature_definitions.append(
                    FeatureDefinition(feature_name=column, feature_type=feature_type)
                )
            else:
                raise ValueError(
                    f"Failed to infer Feature type based on dtype {data_frame[column].dtype} "
                    f"for column {column}."
                )
        self.feature_definitions = feature_definitions
        return self.feature_definitions

    def put_record(self, record: Sequence[FeatureValue]):
        """Put a single record in the FeatureGroup.

        Args:
            record (Sequence[FeatureValue]): a list contains feature values.
        """
        return self.sagemaker_session.put_record(
            feature_group_name=self.name, record=[value.to_dict() for value in record]
        )

    def ingest(
        self,
        data_frame: DataFrame,
        max_workers: int = 1,
        wait: bool = True,
        timeout: Union[int, float] = None,
    ) -> IngestionManagerPandas:
        """Ingest the content of a pandas DataFrame to feature store.

        ``max_worker`` number of thread will be created to work on different partitions of
        the ``data_frame`` in parallel.

        Args:
            data_frame (DataFrame): data_frame to be ingested to feature store.
            max_workers (int): number of threads to be created.
            wait (bool): whether to wait for the ingestion to finish or not.
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
                if timeout is reached.

        Returns:
            An instance of IngestionManagerPandas.
        """
        manager = IngestionManagerPandas(
            feature_group_name=self.name,
            sagemaker_session=self.sagemaker_session,
            data_frame=data_frame,
            max_workers=max_workers,
        )
        manager.run(wait=wait, timeout=timeout)
        return manager

    def athena_query(self) -> AthenaQuery:
        """Create an AthenaQuery instance.

        Returns:
            An instance of AthenaQuery initialized with data catalog configurations.
        """
        response = self.describe()
        data_catalog_config = response.get("OfflineStoreConfig").get("DataCatalogConfig", None)
        disable_glue = data_catalog_config.get("DisableGlueTableCreation", False)
        if data_catalog_config:
            query = AthenaQuery(
                catalog=data_catalog_config["Catalog"] if disable_glue else "AwsDataCatalog",
                database=data_catalog_config["Database"],
                table_name=data_catalog_config["TableName"],
                sagemaker_session=self.sagemaker_session,
            )
            return query
        raise RuntimeError("No metastore is configured with this feature group.")

    def as_hive_ddl(self, database: str = "sagemaker_featurestore", table_name: str = None) -> str:
        """Generate Hive DDL commands to define or change structure of tables or databases in Hive.

        Schema of the table is generated based on the feature definitions. Columns are named
        after feature name and data-type are inferred based on feature type. Integral feature
        type is mapped to INT data-type. Fractional feature type is mapped to FLOAT data-type.
        String feature type is mapped to STRING data-type.

        Args:
            database: name of the database. If not set "sagemaker_featurestore" will be used.
            table_name: name of the table. If not set the name of this feature group will be
                used.

        Returns:
            Generated create table DDL string.
        """
        if not table_name:
            table_name = self.name

        s3_uri = self.describe().get("OfflineStoreConfig").get("S3StorageConfig").get("S3Uri")
        offline_store_s3_uri = os.path.join(
            s3_uri,
            self.sagemaker_session.account_id(),
            "sagemaker",
            self.sagemaker_session.boto_session.region_name,
            "offline-store",
            self.name,
        )

        ddl = f"CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table_name} (\n"
        for definition in self.feature_definitions:
            ddl += (
                f"  {definition.feature_name} "
                f"{self._FEATURE_TYPE_TO_DDL_DATA_TYPE_MAP.get(definition.feature_type.value)}\n"
            )
        ddl += "  write_time TIMESTAMP\n"
        ddl += "  event_time TIMESTAMP\n"
        ddl += "  is_deleted BOOLEAN\n"
        ddl += ")\n"
        ddl += (
            "ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'\n"
            "  STORED AS\n"
            "  INPUTFORMAT 'parquet.hive.DeprecatedParquetInputFormat'\n"
            "  OUTPUTFORMAT 'parquet.hive.DeprecatedParquetOutputFormat'\n"
            f"LOCATION '{offline_store_s3_uri}'"
        )
        return ddl
