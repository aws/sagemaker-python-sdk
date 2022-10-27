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
from typing import Sequence, Union

import attr
import pandas as pd

from sagemaker import Session
from sagemaker.feature_store.feature_group import FeatureGroup


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
            if base is a DataFrame.
        _event_time_identifier_feature_name (str): A string representing the event time identifier
            feature if base is a DataFrame.
        _included_feature_names (List[str]): A list of features to be included in the output.
        _kms_key_id (str): An KMS key id. If set, will be used to encrypt the result file.
        _point_in_time_accurate_join (bool): A boolean representing whether using point in time join
            or not.
        _include_duplicated_records (bool): A boolean representing whether including duplicated
            records or not.
        _include_deleted_records (bool): A boolean representing whether including deleted records or
            not.
        _number_of_recent_records (int): An int that how many records will be returned for each
            record identifier.
        _number_of_records (int): An int that how many records will be returned.
        _write_time_ending_timestamp (datetime.datetime): A datetime that all records' write time in
            dataset will be before it.
        _event_time_starting_timestamp (datetime.datetime): A datetime that all records' event time
            in dataset will be after it.
        _event_time_ending_timestamp (datetime.datetime): A datetime that all records' event time in
            dataset will be before it.
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

            query = self._sagemaker_session.start_query_execution(
                catalog=data_catalog_config.get("Catalog", None)
                if disable_glue
                else "AwsDataCatalog",
                database=data_catalog_config.get("Database", None),
                query_string=self._construct_query_string(
                    data_catalog_config.get("TableName", None),
                    data_catalog_config.get("Database", None),
                    base_features,
                ),
                output_location=self._output_path,
                kms_key=self._kms_key_id,
            )
            query_id = query.get("QueryExecutionId", None)
            self._sagemaker_session.wait_for_athena_query(
                query_execution_id=query_id,
            )
            query_state = (
                self._sagemaker_session.get_query_execution(
                    query_execution_id=query_id,
                )
                .get("QueryExecution", None)
                .get("Status", None)
                .get("State", None)
            )
            if query_state != "SUCCEEDED":
                raise RuntimeError(f"Failed to execute query {query_id}.")

            return query_state.get("QueryExecution", None).get("ResultConfiguration", None).get(
                "OutputLocation", None
            ), query_state.get("QueryExecution", None).get("Query", None)
        raise ValueError("Base must be either a FeatureGroup or a DataFrame.")

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
        included_features = ", ".join(
            [
                f'base."{include_feature_name}"'
                for include_feature_name in self._included_feature_names
            ]
        )
        query_string = f"SELECT {included_features}\n"
        if self._include_duplicated_records:
            query_string += f'FROM "{database}"."{base_table_name}" base\n'
            if not self._include_deleted_records:
                query_string += "WHERE NOT is_deleted\n"
        else:
            base_features.remove(self._event_time_identifier_feature_name)
            dedup_features = ", ".join([f'dedup_base."{feature}"' for feature in base_features])
            query_string += (
                "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + f"PARTITION BY {dedup_features}\n"
                + f'ORDER BY dedup_base."{self._event_time_identifier_feature_name}" '
                + 'DESC, dedup_base."api_invocation_time" DESC, dedup_base."write_time" DESC\n'
                + ") AS row_base\n"
                + f'FROM "{database}"."{base_table_name}" dedup_base\n'
                + ") AS base\n"
                + "WHERE row_base = 1\n"
            )
            if not self._include_deleted_records:
                query_string += "AND NOT is_deleted\n"
        return query_string
