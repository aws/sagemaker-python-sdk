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
"""Dataset Builder

A Dataset Builder is a builder class for generating a dataset by providing conditions.
"""
from __future__ import absolute_import

import datetime
from typing import Any, Dict, List, Sequence, Union

import attr
import pandas as pd

from sagemaker import Session, s3, utils
from sagemaker.feature_store.feature_group import FeatureGroup


@attr.s
class FeatureGroupToBeMerged:
    """FeatureGroup metadata which will be used for SQL join.

    This class instantiates a FeatureGroupToBeMerged object that comprises a list of feature names,
    a list of feature names which will be included in SQL query, a database, an Athena table name,
    a feature name of record identifier, a feature name of event time identifier and a feature name
    of base which is the target join key.

    Attributes:
        features (List[str]): A list of strings representing feature names of this FeatureGroup.
        included_feature_names (Sequence[str]): A list of strings representing features to be
            included in the output.
        database (str): A string representing the database.
        table_name (str): A string representing the Athena table name of this FeatureGroup.
        record_dentifier_feature_name (str): A string representing the record identifier feature.
        event_time_identifier_feature_name (str): A string representing the event time identifier
            feature.
        target_feature_name_in_base (str): A string representing the feature name in base which will
            be used as target join key (default: None).
    """

    features: List[str] = attr.ib()
    included_feature_names: Sequence[str] = attr.ib()
    database: str = attr.ib()
    table_name: str = attr.ib()
    record_identifier_feature_name: str = attr.ib()
    event_time_identifier_feature_name: str = attr.ib()
    target_feature_name_in_base: str = attr.ib(default=None)


@attr.s
class DatasetBuilder:
    """DatasetBuilder definition.

    This class instantiates a DatasetBuilder object that comprises a base, a list of feature names,
    an output path and a KMS key ID.

    Attributes:
        _sagemaker_session (Session): Session instance to perform boto calls.
        _base (Union[FeatureGroup, DataFrame]): A base which can be either a FeatureGroup or a
            pandas.DataFrame and will be used to merge other FeatureGroups and generate a Dataset.
        _output_path (str): An S3 URI which stores the output .csv file.
        _record_identifier_feature_name (str): A string representing the record identifier feature
            if base is a DataFrame (default: None).
        _event_time_identifier_feature_name (str): A string representing the event time identifier
            feature if base is a DataFrame (default: None).
        _included_feature_names (Sequence[str]): A list of strings representing features to be
            included in the output (default: None).
        _kms_key_id (str): An KMS key id. If set, will be used to encrypt the result file
            (default: None).
        _point_in_time_accurate_join (bool): A boolean representing whether using point in time join
            or not (default: False).
        _include_duplicated_records (bool): A boolean representing whether including duplicated
            records or not (default: False).
        _include_deleted_records (bool): A boolean representing whether including deleted records or
            not (default: False).
        _number_of_recent_records (int): An int that how many records will be returned for each
            record identifier (default: 1).
        _number_of_records (int): An int that how many records will be returned (default: None).
        _write_time_ending_timestamp (datetime.datetime): A datetime that all records' write time in
            dataset will be before it (default: None).
        _event_time_starting_timestamp (datetime.datetime): A datetime that all records' event time
            in dataset will be after it (default: None).
        _event_time_ending_timestamp (datetime.datetime): A datetime that all records' event time in
            dataset will be before it (default: None).
        _feature_groups_to_be_merged (List[FeatureGroupToBeMerged]): A list of
            FeatureGroupToBeMerged which will be joined to base (default: []).
    """

    _sagemaker_session: Session = attr.ib()
    _base: Union[FeatureGroup, pd.DataFrame] = attr.ib()
    _output_path: str = attr.ib()
    _record_identifier_feature_name: str = attr.ib(default=None)
    _event_time_identifier_feature_name: str = attr.ib(default=None)
    _included_feature_names: Sequence[str] = attr.ib(default=None)
    _kms_key_id: str = attr.ib(default=None)

    _point_in_time_accurate_join: bool = attr.ib(init=False, default=False)
    _include_duplicated_records: bool = attr.ib(init=False, default=False)
    _include_deleted_records: bool = attr.ib(init=False, default=False)
    _number_of_recent_records: int = attr.ib(init=False, default=1)
    _number_of_records: int = attr.ib(init=False, default=None)
    _write_time_ending_timestamp: datetime.datetime = attr.ib(init=False, default=None)
    _event_time_starting_timestamp: datetime.datetime = attr.ib(init=False, default=None)
    _event_time_ending_timestamp: datetime.datetime = attr.ib(init=False, default=None)
    _feature_groups_to_be_merged: List[FeatureGroupToBeMerged] = attr.ib(init=False, default=[])

    def with_feature_group(
        self,
        feature_group: FeatureGroup,
        target_feature_name_in_base: str = None,
        included_feature_names: Sequence[str] = None,
    ):
        """Join FeatureGroup with base.

        Args:
            feature_group (FeatureGroup): A FeatureGroup which will be joined to base.
            target_feature_name_in_base (str): A string representing the feature name in base which
                will be used as target join key (default: None).
            included_feature_names (Sequence[str]): A list of strings representing features to be
                included in the output (default: None).
        Returns:
            This DatasetBuilder object.
        """
        # TODO: handle pagination and input feature validation
        # TODO: potential refactor with FeatureGroup base
        feature_group_metadata = feature_group.describe()
        data_catalog_config = feature_group_metadata.get("OfflineStoreConfig", None).get(
            "DataCatalogConfig", None
        )
        if not data_catalog_config:
            raise RuntimeError(
                f"No metastore is configured with FeatureGroup {feature_group.name}."
            )

        record_identifier_feature_name = feature_group_metadata.get(
            "RecordIdentifierFeatureName", None
        )
        event_time_identifier_feature_name = feature_group_metadata.get(
            "EventTimeFeatureName", None
        )
        # TODO: back fill feature definitions due to UpdateFG
        table_name = data_catalog_config.get("TableName", None)
        database = data_catalog_config.get("Database", None)
        features = [feature.feature_name for feature in feature_group.feature_definitions]
        if not target_feature_name_in_base:
            target_feature_name_in_base = self._record_identifier_feature_name
        self._feature_groups_to_be_merged.append(
            FeatureGroupToBeMerged(
                features,
                included_feature_names,
                database,
                table_name,
                record_identifier_feature_name,
                event_time_identifier_feature_name,
                target_feature_name_in_base,
            )
        )
        return self

    def point_in_time_accurate_join(self):
        """Set join type as point in time accurate join.

        Returns:
            This DatasetBuilder object.
        """
        self._point_in_time_accurate_join = True
        return self

    def include_duplicated_records(self):
        """Include duplicated records in dataset.

        Returns:
            This DatasetBuilder object.
        """
        self._include_duplicated_records = True
        return self

    def include_deleted_records(self):
        """Include deleted records in dataset.

        Returns:
            This DatasetBuilder object.
        """
        self._include_deleted_records = True
        return self

    def with_number_of_recent_records_by_record_identifier(self, number_of_recent_records: int):
        """Set number_of_recent_records field with provided input.

        Args:
            number_of_recent_records (int): An int that how many recent records will be returned for
                each record identifier.
        Returns:
            This DatasetBuilder object.
        """
        self._number_of_recent_records = number_of_recent_records
        return self

    def with_number_of_records_from_query_results(self, number_of_records: int):
        """Set number_of_records field with provided input.

        Args:
            number_of_records (int): An int that how many records will be returned.
        Returns:
            This DatasetBuilder object.
        """
        self._number_of_records = number_of_records
        return self

    def as_of(self, timestamp: datetime.datetime):
        """Set write_time_ending_timestamp field with provided input.

        Args:
            timestamp (datetime.datetime): A datetime that all records' write time in dataset will
                be before it.
        Returns:
            This DatasetBuilder object.
        """
        self._write_time_ending_timestamp = timestamp
        return self

    def with_event_time_range(
        self,
        starting_timestamp: datetime.datetime = None,
        ending_timestamp: datetime.datetime = None,
    ):
        """Set event_time_starting_timestamp and event_time_ending_timestamp with provided inputs.

        Args:
            starting_timestamp (datetime.datetime): A datetime that all records' event time in
                dataset will be after it (default: None).
            ending_timestamp (datetime.datetime): A datetime that all records' event time in dataset
                will be before it (default: None).
        Returns:
            This DatasetBuilder object.
        """
        self._event_time_starting_timestamp = starting_timestamp
        self._event_time_ending_timestamp = ending_timestamp
        return self

    def to_csv(self):
        """Get query string and result in .csv format

        Returns:
            The S3 path of the .csv file.
            The query string executed.
        """
        if isinstance(self._base, pd.DataFrame):
            temp_id = utils.unique_name_from_base("dataframe-base")
            local_filename = f"{temp_id}.csv"
            desired_s3_folder = f"{self._output_path}/{temp_id}"
            self._base.to_csv(local_filename, index=False, header=False)
            s3.S3Uploader.upload(
                local_path=local_filename,
                desired_s3_uri=desired_s3_folder,
                sagemaker_session=self._sagemaker_session,
                kms_key=self._kms_key_id,
            )
            temp_table_name = f"dataframe_{temp_id}"
            self._create_temp_table(temp_table_name, desired_s3_folder)
            base_features = list(self._base.columns)
            query_string = self._construct_query_string(
                temp_table_name,
                "sagemaker_featurestore",
                base_features,
            )
            query_result = self._run_query(query_string, "AwsDataCatalog", "sagemaker_featurestore")
            # TODO: cleanup local file and temp table
            return query_result.get("QueryExecution", None).get("ResultConfiguration", None).get(
                "OutputLocation", None
            ), query_result.get("QueryExecution", None).get("Query", None)
        if isinstance(self._base, FeatureGroup):
            # TODO: handle pagination and input feature validation
            base_feature_group = self._base.describe()
            data_catalog_config = base_feature_group.get("OfflineStoreConfig", None).get(
                "DataCatalogConfig", None
            )
            if not data_catalog_config:
                raise RuntimeError("No metastore is configured with the base FeatureGroup.")
            disable_glue = base_feature_group.get("DisableGlueTableCreation", False)
            self._record_identifier_feature_name = base_feature_group.get(
                "RecordIdentifierFeatureName", None
            )
            self._event_time_identifier_feature_name = base_feature_group.get(
                "EventTimeFeatureName", None
            )
            base_features = [
                feature.get("FeatureName", None)
                for feature in base_feature_group.get("FeatureDefinitions", None)
            ]

            query_string = self._construct_query_string(
                data_catalog_config.get("TableName", None),
                data_catalog_config.get("Database", None),
                base_features,
            )
            query_result = self._run_query(
                query_string,
                data_catalog_config.get("Catalog", None) if disable_glue else "AwsDataCatalog",
                data_catalog_config.get("Database", None),
            )
            return query_result.get("QueryExecution", None).get("ResultConfiguration", None).get(
                "OutputLocation", None
            ), query_result.get("QueryExecution", None).get("Query", None)
        raise ValueError("Base must be either a FeatureGroup or a DataFrame.")

    def _construct_where_query_string(self, suffix: str, event_time_identifier_feature_name: str):
        """Internal method for constructing SQL WHERE query string by parameters.

        Args:
            suffix (str): A temp identifier of the FeatureGroup.
            event_time_identifier_feature_name (str): A string representing the event time
                identifier feature.
        Returns:
            The WHERE query string.
        """
        where_conditions = []
        if not self._include_deleted_records:
            where_conditions.append("NOT is_deleted")
        if self._write_time_ending_timestamp:
            where_conditions.append(
                f'table_{suffix}."write_time" <= {self._write_time_ending_timestamp}'
            )
        if self._event_time_starting_timestamp:
            where_conditions.append(
                f'table_{suffix}."{event_time_identifier_feature_name}" >= '
                + str(self._event_time_starting_timestamp)
            )
        if self._event_time_ending_timestamp:
            where_conditions.append(
                f'table_{suffix}."{event_time_identifier_feature_name}" <= '
                + str(self._event_time_ending_timestamp)
            )
        if len(where_conditions) == 0:
            return ""
        return "WHERE " + "\nAND ".join(where_conditions)

    def _construct_table_query(self, feature_group: FeatureGroupToBeMerged, suffix: str):
        """Internal method for constructing SQL query string by parameters.

        Args:
            feature_group (FeatureGroupToBeMerged): A FeatureGroupToBeMerged object which has the
                FeatureGroup metadata.
            suffix (str): A temp identifier of the FeatureGroup.
        Returns:
            The query string.
        """
        included_features = ", ".join(
            [
                f'table_{suffix}."{include_feature_name}"'
                for include_feature_name in feature_group.included_feature_names
            ]
        )
        query_string = f"SELECT {included_features}\n"
        if self._include_duplicated_records:
            query_string += (
                f'FROM "{feature_group.database}"."{feature_group.table_name}" table_{suffix}\n'
            )
        else:
            features = feature_group.features
            features.remove(feature_group.event_time_identifier_feature_name)
            dedup_features = ", ".join([f'dedup_{suffix}."{feature}"' for feature in features])
            query_string += (
                "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + f"PARTITION BY {dedup_features}\n"
                + f'ORDER BY dedup_{suffix}."{feature_group.event_time_identifier_feature_name}" '
                + f'DESC, dedup_{suffix}."api_invocation_time" DESC, '
                + f'dedup_{suffix}."write_time" DESC\n'
                + f") AS row_{suffix}\n"
                + f'FROM "{feature_group.database}"."{feature_group.table_name}" dedup_{suffix}\n'
                + f") AS table_{suffix}\n"
                + f"WHERE row_{suffix} = 1\n"
            )
        return query_string + self._construct_where_query_string(
            suffix, feature_group.event_time_identifier_feature_name
        )

    def _construct_query_string(
        self, base_table_name: str, database: str, base_features: list
    ) -> str:
        """Internal method for constructing SQL query string by parameters.

        Args:
            base_table_name (str): The Athena table name of base FeatureGroup or pandas.DataFrame.
            database (str): The Athena database of the base table.
            base_features (list): The list of features of the base table.
        Returns:
            The query string.
        """
        base = FeatureGroupToBeMerged(
            base_features,
            self._included_feature_names,
            database,
            base_table_name,
            self._record_identifier_feature_name,
            self._event_time_identifier_feature_name,
        )
        base_table_query_string = self._construct_table_query(base, "base")
        query_string = f"WITH fg_base AS ({base_table_query_string})"
        if len(self._feature_groups_to_be_merged) > 0:
            with_subquery_string = "".join(
                [
                    f",\nfg_{i} AS ({self._construct_table_query(feature_group, str(i))})"
                    for i, feature_group in enumerate(self._feature_groups_to_be_merged)
                ]
            )
            query_string += with_subquery_string
        query_string += "SELECT *\nFROM fg_base"
        if len(self._feature_groups_to_be_merged) > 0:
            join_subquery_string = "".join(
                [
                    self._construct_join_condition(feature_group, str(i))
                    for i, feature_group in enumerate(self._feature_groups_to_be_merged)
                ]
            )
            query_string += join_subquery_string
        return query_string

    def _construct_join_condition(self, feature_group: FeatureGroupToBeMerged, suffix: str):
        """Internal method for constructing SQL JOIN query string by parameters.

        Args:
            feature_group (FeatureGroupToBeMerged): A FeatureGroupToBeMerged object which has the
                FeatureGroup metadata.
            suffix (str): A temp identifier of the FeatureGroup.
        Returns:
            The JOIN query string.
        """
        join_condition_string = (
            f"\nJOIN fg_{suffix}\n"
            + f'ON fg_base."{feature_group.target_feature_name_in_base}" = '
            + f'fg_{suffix}."{feature_group.record_identifier_feature_name}"'
        )
        if self._point_in_time_accurate_join:
            join_condition_string += (
                f'\nAND fg_base."{self._event_time_identifier_feature_name}" >= '
                + f'fg_{suffix}."{feature_group.event_time_identifier_feature_name}"'
            )
        return join_condition_string

    def _create_temp_table(self, temp_table_name: str, desired_s3_folder: str):
        """Internal method for creating a temp Athena table for the base pandas.Dataframe.

        Args:
            temp_table_name (str): The Athena table name of base pandas.DataFrame.
            desired_s3_folder (str): The S3 URI of the folder of the data.
        """
        columns_string = ", ".join(
            [self._construct_athena_table_column_string(column) for column in self._base.columns]
        )
        serde_properties = '"separatorChar" = ",", "quoteChar" = "`", "escapeChar" = "\\\\"'
        query_string = (
            f"CREATE EXTERNAL TABLE {temp_table_name} ({columns_string}) "
            + "ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' "
            + f"WITH SERDEPROPERTIES ({serde_properties}) "
            + f"LOCATION '{desired_s3_folder}';"
        )
        self._run_query(query_string, "AwsDataCatalog", "sagemaker_featurestore")

    def _construct_athena_table_column_string(self, column: str) -> str:
        """Internal method for constructing string of Athena column.

        Args:
            column (str): The column name from pandas.Dataframe.
        Returns:
            The Athena column string.

        Raises:
            RuntimeError: The type of pandas.Dataframe column is not support yet.
        """
        dataframe_type = self._base[column].dtypes
        if dataframe_type == "object":
            column_type = "STRING"
        elif dataframe_type == "int64":
            column_type = "INT"
        elif dataframe_type == "float64":
            column_type = "DOUBLE"
        elif dataframe_type == "bool":
            column_type = "BOOLEAN"
        elif dataframe_type == "datetime64":
            column_type = "TIMESTAMP"
        else:
            raise RuntimeError(f"The dataframe type {dataframe_type} is not supported yet.")
        return f"{column} {column_type}"

    def _run_query(self, query_string: str, catalog: str, database: str) -> Dict[str, Any]:
        """Internal method for execute Athena query, wait for query finish and get query result.

        Args:
            query_string (str): The SQL query statements to be executed.
            catalog (str): The name of the data catalog used in the query execution.
            database (str): The name of the database used in the query execution.
        Returns:
            The query result.

        Raises:
            RuntimeError: Athena query failed.
        """
        query = self._sagemaker_session.start_query_execution(
            catalog=catalog,
            database=database,
            query_string=query_string,
            output_location=self._output_path,
            kms_key=self._kms_key_id,
        )
        query_id = query.get("QueryExecutionId", None)
        self._sagemaker_session.wait_for_athena_query(query_execution_id=query_id)
        query_result = self._sagemaker_session.get_query_execution(query_execution_id=query_id)
        query_state = (
            query_result.get("QueryExecution", None).get("Status", None).get("State", None)
        )
        if query_state != "SUCCEEDED":
            raise RuntimeError(f"Failed to execute query {query_id}.")
        return query_result
