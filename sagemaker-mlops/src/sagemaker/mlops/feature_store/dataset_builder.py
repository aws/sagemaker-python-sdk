# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Dataset Builder for FeatureStore."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union
import datetime

import pandas as pd

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.telemetry import Feature, _telemetry_emitter
from sagemaker.mlops.feature_store import FeatureGroup
from sagemaker.mlops.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.mlops.feature_store.feature_utils import (
    upload_dataframe_to_s3,
    download_csv_from_s3,
    run_athena_query,
)

_DEFAULT_CATALOG = "AwsDataCatalog"
_DEFAULT_DATABASE = "sagemaker_featurestore"

_DTYPE_TO_FEATURE_TYPE = {
    "object": "String", "string": "String",
    "int64": "Integral", "int32": "Integral",
    "float64": "Fractional", "float32": "Fractional",
}

_DTYPE_TO_ATHENA_TYPE = {
    "object": "STRING", "int64": "INT", "float64": "DOUBLE",
    "bool": "BOOLEAN", "datetime64[ns]": "TIMESTAMP",
}


class TableType(Enum):
    FEATURE_GROUP = "FeatureGroup"
    DATA_FRAME = "DataFrame"


class JoinTypeEnum(Enum):
    INNER_JOIN = "JOIN"
    LEFT_JOIN = "LEFT JOIN"
    RIGHT_JOIN = "RIGHT JOIN"
    FULL_JOIN = "FULL JOIN"
    CROSS_JOIN = "CROSS JOIN"


class JoinComparatorEnum(Enum):
    EQUALS = "="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL_TO = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL_TO = "<="
    NOT_EQUAL_TO = "<>"


@dataclass
class FeatureGroupToBeMerged:
    """FeatureGroup metadata which will be used for SQL join.

    This class instantiates a FeatureGroupToBeMerged object that comprises a list of feature names,
    a list of feature names which will be included in SQL query, a database, an Athena table name,
    a feature name of record identifier, a feature name of event time identifier and a feature name
    of base which is the target join key.

    Attributes:
        features (List[str]): A list of strings representing feature names of this FeatureGroup.
        included_feature_names (List[str]): A list of strings representing features to be
            included in the SQL join.
        projected_feature_names (List[str]): A list of strings representing features to be
            included for final projection in output.
        catalog (str): A string representing the catalog.
        database (str): A string representing the database.
        table_name (str): A string representing the Athena table name of this FeatureGroup.
        record_identifier_feature_name (str): A string representing the record identifier feature.
        event_time_identifier_feature (FeatureDefinition): A FeatureDefinition representing the
            event time identifier feature.
        target_feature_name_in_base (str): A string representing the feature name in base which will
            be used as target join key (default: None).
        table_type (TableType): A TableType representing the type of table if it is Feature Group or
            Panda Data Frame (default: None).
        feature_name_in_target (str): A string representing the feature name in the target feature
            group that will be compared to the target feature in the base feature group.
            If None is provided, the record identifier feature will be used in the
            SQL join. (default: None).
        join_comparator (JoinComparatorEnum): A JoinComparatorEnum representing the comparator
            used when joining the target feature in the base feature group and the feature
            in the target feature group. (default: JoinComparatorEnum.EQUALS).
        join_type (JoinTypeEnum): A JoinTypeEnum representing the type of join between
            the base and target feature groups. (default: JoinTypeEnum.INNER_JOIN).
    """
    features: List[str]
    included_feature_names: List[str]
    projected_feature_names: List[str]
    catalog: str
    database: str
    table_name: str
    record_identifier_feature_name: str
    event_time_identifier_feature: FeatureDefinition
    target_feature_name_in_base: str = None
    table_type: TableType = None
    feature_name_in_target: str = None
    join_comparator: JoinComparatorEnum = JoinComparatorEnum.EQUALS
    join_type: JoinTypeEnum = JoinTypeEnum.INNER_JOIN


def construct_feature_group_to_be_merged(
    target_feature_group: FeatureGroup,
    included_feature_names: List[str],
    target_feature_name_in_base: str = None,
    feature_name_in_target: str = None,
    join_comparator: JoinComparatorEnum = JoinComparatorEnum.EQUALS,
    join_type: JoinTypeEnum = JoinTypeEnum.INNER_JOIN,
) -> FeatureGroupToBeMerged:
    """Construct a FeatureGroupToBeMerged object by provided parameters.

    Args:
        target_feature_group (FeatureGroup): A FeatureGroup object.
        included_feature_names (List[str]): A list of strings representing features to be
            included in the output.
        target_feature_name_in_base (str): A string representing the feature name in base which
            will be used as target join key (default: None).
        feature_name_in_target (str): A string representing the feature name in the target feature
            group that will be compared to the target feature in the base feature group.
            If None is provided, the record identifier feature will be used in the
            SQL join. (default: None).
        join_comparator (JoinComparatorEnum): A JoinComparatorEnum representing the comparator
            used when joining the target feature in the base feature group and the feature
            in the target feature group. (default: JoinComparatorEnum.EQUALS).
        join_type (JoinTypeEnum): A JoinTypeEnum representing the type of join between
            the base and target feature groups. (default: JoinTypeEnum.INNER_JOIN).

    Returns:
        A FeatureGroupToBeMerged object.

    Raises:
        RuntimeError: No metastore is configured with the FeatureGroup.
        ValueError: Invalid feature name(s) in included_feature_names.
    """
    fg = FeatureGroup.get(feature_group_name=target_feature_group.feature_group_name)

    if not fg.offline_store_config or not fg.offline_store_config.data_catalog_config:
        raise RuntimeError(f"No metastore configured for FeatureGroup {fg.feature_group_name}.")

    catalog_config = fg.offline_store_config.data_catalog_config
    disable_glue = getattr(catalog_config, "disable_glue_table_creation", False) or False

    features = [fd.feature_name for fd in fg.feature_definitions]
    record_id = fg.record_identifier_feature_name
    event_time_name = fg.event_time_feature_name
    event_time_type = next(
        (fd.feature_type for fd in fg.feature_definitions if fd.feature_name == event_time_name),
        None
    )

    if feature_name_in_target and feature_name_in_target not in features:
        raise ValueError(f"Feature {feature_name_in_target} not found in {fg.feature_group_name}")

    for feat in included_feature_names or []:
        if feat not in features:
            raise ValueError(f"Feature {feat} not found in {fg.feature_group_name}")

    if not included_feature_names:
        included_feature_names = features.copy()
        projected_feature_names = features.copy()
    else:
        projected_feature_names = included_feature_names.copy()
        if record_id not in included_feature_names:
            included_feature_names.append(record_id)
        if event_time_name not in included_feature_names:
            included_feature_names.append(event_time_name)

    return FeatureGroupToBeMerged(
        features=features,
        included_feature_names=included_feature_names,
        projected_feature_names=projected_feature_names,
        catalog=catalog_config.catalog if disable_glue else _DEFAULT_CATALOG,
        database=catalog_config.database,
        table_name=catalog_config.table_name,
        record_identifier_feature_name=record_id,
        event_time_identifier_feature=FeatureDefinition(
            feature_name=event_time_name, feature_type=FeatureTypeEnum(event_time_type).value
        ),
        target_feature_name_in_base=target_feature_name_in_base,
        table_type=TableType.FEATURE_GROUP,
        feature_name_in_target=feature_name_in_target,
        join_comparator=join_comparator,
        join_type=join_type,
    )


@dataclass
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
        _included_feature_names (List[str]): A list of strings representing features to be
            included in the output. If not set, all features will be included in the output.
            (default: None).
        _kms_key_id (str): A KMS key id. If set, will be used to encrypt the result file
            (default: None).
        _point_in_time_accurate_join (bool): A boolean representing if point-in-time join
            is applied to the resulting dataframe when calling "to_dataframe".
            When set to True, users can retrieve data using "row-level time travel"
            according to the event times provided to the DatasetBuilder. This requires that the
            entity dataframe with event times is submitted as the base in the constructor
            (default: False).
        _include_duplicated_records (bool): A boolean representing whether the resulting dataframe
            when calling "to_dataframe" should include duplicated records (default: False).
        _include_deleted_records (bool): A boolean representing whether the resulting
            dataframe when calling "to_dataframe" should include deleted records (default: False).
        _number_of_recent_records (int): An integer representing how many records will be
            returned for each record identifier (default: 1).
        _number_of_records (int): An integer representing the number of records that should be
            returned in the resulting dataframe when calling "to_dataframe" (default: None).
        _write_time_ending_timestamp (datetime.datetime): A datetime that represents the latest
            write time for a record to be included in the resulting dataset. Records with a
            newer write time will be omitted from the resulting dataset. (default: None).
        _event_time_starting_timestamp (datetime.datetime): A datetime that represents the earliest
            event time for a record to be included in the resulting dataset. Records
            with an older event time will be omitted from the resulting dataset. (default: None).
        _event_time_ending_timestamp (datetime.datetime): A datetime that represents the latest
            event time for a record to be included in the resulting dataset. Records
            with a newer event time will be omitted from the resulting dataset. (default: None).
        _feature_groups_to_be_merged (List[FeatureGroupToBeMerged]): A list of
            FeatureGroupToBeMerged which will be joined to base (default: []).
        _event_time_identifier_feature_type (FeatureTypeEnum): A FeatureTypeEnum representing the
            type of event time identifier feature (default: None).
    """

    _sagemaker_session: Session
    _base: Union[FeatureGroup, pd.DataFrame]
    _output_path: str
    _record_identifier_feature_name: str = None
    _event_time_identifier_feature_name: str = None
    _included_feature_names: List[str] = None
    _kms_key_id: str = None
    _event_time_identifier_feature_type: FeatureTypeEnum = None

    _point_in_time_accurate_join: bool = field(default=False, init=False)
    _include_duplicated_records: bool = field(default=False, init=False)
    _include_deleted_records: bool = field(default=False, init=False)
    _number_of_recent_records: int = field(default=None, init=False)
    _number_of_records: int = field(default=None, init=False)
    _write_time_ending_timestamp: datetime.datetime = field(default=None, init=False)
    _event_time_starting_timestamp: datetime.datetime = field(default=None, init=False)
    _event_time_ending_timestamp: datetime.datetime = field(default=None, init=False)
    _feature_groups_to_be_merged: List[FeatureGroupToBeMerged] = field(default_factory=list, init=False)

    @classmethod
    def create(
        cls,
        base: Union[FeatureGroup, pd.DataFrame],
        output_path: str,
        session: Session,
        record_identifier_feature_name: str = None,
        event_time_identifier_feature_name: str = None,
        included_feature_names: List[str] = None,
        kms_key_id: str = None,
    ) -> "DatasetBuilder":
        """Create a DatasetBuilder for generating a Dataset.

        Args:
            base: A FeatureGroup or DataFrame to use as the base.
            output_path: S3 URI for output.
            session: SageMaker session.
            record_identifier_feature_name: Required if base is DataFrame.
            event_time_identifier_feature_name: Required if base is DataFrame.
            included_feature_names: Features to include in output.
            kms_key_id: KMS key for encryption.

        Returns:
            DatasetBuilder instance.
        """
        if isinstance(base, pd.DataFrame):
            if not record_identifier_feature_name or not event_time_identifier_feature_name:
                raise ValueError(
                    "record_identifier_feature_name and event_time_identifier_feature_name "
                    "are required when base is a DataFrame."
                )
        return cls(
            _sagemaker_session=session,
            _base=base,
            _output_path=output_path,
            _record_identifier_feature_name=record_identifier_feature_name,
            _event_time_identifier_feature_name=event_time_identifier_feature_name,
            _included_feature_names=included_feature_names,
            _kms_key_id=kms_key_id,
        )

    def with_feature_group(
        self,
        feature_group: FeatureGroup,
        target_feature_name_in_base: str = None,
        included_feature_names: List[str] = None,
        feature_name_in_target: str = None,
        join_comparator: JoinComparatorEnum = JoinComparatorEnum.EQUALS,
        join_type: JoinTypeEnum = JoinTypeEnum.INNER_JOIN,
    ) -> "DatasetBuilder":
        """Join FeatureGroup with base.

        Args:
            feature_group (FeatureGroup): A target FeatureGroup which will be joined to base.
            target_feature_name_in_base (str): A string representing the feature name in base which
                will be used as a join key (default: None).
            included_feature_names (List[str]): A list of strings representing features to be
                included in the output (default: None).
            feature_name_in_target (str): A string representing the feature name in the target
                feature group that will be compared to the target feature in the base feature group.
                If None is provided, the record identifier feature will be used in the
                SQL join. (default: None).
            join_comparator (JoinComparatorEnum): A JoinComparatorEnum representing the comparator
                used when joining the target feature in the base feature group and the feature
                in the target feature group. (default: JoinComparatorEnum.EQUALS).
            join_type (JoinTypeEnum): A JoinTypeEnum representing the type of join between
                the base and target feature groups. (default: JoinTypeEnum.INNER_JOIN).

        Returns:
            This DatasetBuilder object.
        """
        self._feature_groups_to_be_merged.append(
            construct_feature_group_to_be_merged(
                feature_group, included_feature_names, target_feature_name_in_base,
                feature_name_in_target, join_comparator, join_type,
            )
        )
        return self

    def point_in_time_accurate_join(self) -> "DatasetBuilder":
        """Enable point-in-time accurate join.

        Returns:
            This DatasetBuilder object.
        """
        self._point_in_time_accurate_join = True
        return self

    def include_duplicated_records(self) -> "DatasetBuilder":
        """Include duplicated records in dataset.

        Returns:
            This DatasetBuilder object.
        """
        self._include_duplicated_records = True
        return self

    def include_deleted_records(self) -> "DatasetBuilder":
        """Include deleted records in dataset.

        Returns:
            This DatasetBuilder object.
        """
        self._include_deleted_records = True
        return self

    def with_number_of_recent_records_by_record_identifier(self, n: int) -> "DatasetBuilder":
        """Set number_of_recent_records field with provided input.

        Args:
            n (int): An int that how many recent records will be returned for
                each record identifier.

        Returns:
            This DatasetBuilder object.
        """
        self._number_of_recent_records = n
        return self

    def with_number_of_records_from_query_results(self, n: int) -> "DatasetBuilder":
        """Set number_of_records field with provided input.

        Args:
            n (int): An int that how many records will be returned.

        Returns:
            This DatasetBuilder object.
        """
        self._number_of_records = n
        return self

    def as_of(self, timestamp: datetime.datetime) -> "DatasetBuilder":
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
    ) -> "DatasetBuilder":
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

    @_telemetry_emitter(Feature.FEATURE_STORE, "DatasetBuilder.to_csv_file")
    def to_csv_file(self) -> tuple[str, str]:
        """Get query string and result in .csv format file.

        Returns:
            tuple: A tuple containing:
                - str: The S3 path of the .csv file
                - str: The query string executed
        
        Note:
            This method returns a tuple (csv_path, query_string).
            To get just the CSV path: csv_path, _ = builder.to_csv_file()
        """
        if isinstance(self._base, pd.DataFrame):
            return self._to_csv_from_dataframe()
        if isinstance(self._base, FeatureGroup):
            return self._to_csv_from_feature_group()
        raise ValueError("Base must be either a FeatureGroup or a DataFrame.")

    @_telemetry_emitter(Feature.FEATURE_STORE, "DatasetBuilder.to_dataframe")
    def to_dataframe(self) -> tuple[pd.DataFrame, str]:
        """Get query string and result in pandas.DataFrame.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: The pandas DataFrame object
                - str: The query string executed
        
        Note:
            This method returns a tuple (dataframe, query_string).
            To get just the DataFrame: df, _ = builder.to_dataframe()
        """
        csv_file, query_string = self.to_csv_file()
        df = download_csv_from_s3(csv_file, self._sagemaker_session, self._kms_key_id)
        if "row_recent" in df.columns:
            df = df.drop("row_recent", axis="columns")
        return df, query_string


    def _to_csv_from_dataframe(self) -> tuple[str, str]:
        s3_folder, temp_table_name = upload_dataframe_to_s3(
            self._base, self._output_path, self._sagemaker_session, self._kms_key_id
        )
        self._create_temp_table(temp_table_name, s3_folder)

        base_features = list(self._base.columns)
        event_time_dtype = str(self._base[self._event_time_identifier_feature_name].dtypes)
        self._event_time_identifier_feature_type = FeatureTypeEnum(
            _DTYPE_TO_FEATURE_TYPE.get(event_time_dtype, "String")
        )

        included = self._included_feature_names or base_features
        fg_to_merge = FeatureGroupToBeMerged(
            features=base_features,
            included_feature_names=included,
            projected_feature_names=included,
            catalog=_DEFAULT_CATALOG,
            database=_DEFAULT_DATABASE,
            table_name=temp_table_name,
            record_identifier_feature_name=self._record_identifier_feature_name,
            event_time_identifier_feature=FeatureDefinition(
                feature_name=self._event_time_identifier_feature_name,
                feature_type=self._event_time_identifier_feature_type,
            ),
            table_type=TableType.DATA_FRAME,
        )

        query_string = self._construct_query_string(fg_to_merge)
        result = self._run_query(query_string, _DEFAULT_CATALOG, _DEFAULT_DATABASE)
        return self._extract_result(result)

    def _to_csv_from_feature_group(self) -> tuple[str, str]:
        base_fg = construct_feature_group_to_be_merged(self._base, self._included_feature_names)
        self._record_identifier_feature_name = base_fg.record_identifier_feature_name
        self._event_time_identifier_feature_name = base_fg.event_time_identifier_feature.feature_name
        self._event_time_identifier_feature_type = base_fg.event_time_identifier_feature.feature_type

        query_string = self._construct_query_string(base_fg)
        result = self._run_query(query_string, base_fg.catalog, base_fg.database)
        return self._extract_result(result)

    def _extract_result(self, query_result: dict) -> tuple[str, str]:
        execution = query_result.get("QueryExecution", {})
        return (
            execution.get("ResultConfiguration", {}).get("OutputLocation"),
            execution.get("Query"),
        )

    def _run_query(self, query_string: str, catalog: str, database: str) -> Dict[str, Any]:
        return run_athena_query(
            session=self._sagemaker_session,
            catalog=catalog,
            database=database,
            query_string=query_string,
            output_location=self._output_path,
            kms_key=self._kms_key_id,
        )

    def _create_temp_table(self, temp_table_name: str, s3_folder: str):
        columns = ", ".join(
            f"{col} {_DTYPE_TO_ATHENA_TYPE.get(str(self._base[col].dtypes), 'STRING')}"
            for col in self._base.columns
        )
        serde = '"separatorChar" = ",", "quoteChar" = "`", "escapeChar" = "\\\\"'
        query = (
            f"CREATE EXTERNAL TABLE {temp_table_name} ({columns}) "
            f"ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' "
            f"WITH SERDEPROPERTIES ({serde}) LOCATION '{s3_folder}';"
        )
        self._run_query(query, _DEFAULT_CATALOG, _DEFAULT_DATABASE)


    def _construct_query_string(self, base: FeatureGroupToBeMerged) -> str:
        base_query = self._construct_table_query(base, "base")
        query = f"WITH fg_base AS ({base_query})"

        for i, fg in enumerate(self._feature_groups_to_be_merged):
            fg_query = self._construct_table_query(fg, str(i))
            query += f",\nfg_{i} AS ({fg_query})"

        selected = ", ".join(f"fg_base.{f}" for f in base.projected_feature_names)
        selected_final = ", ".join(base.projected_feature_names)

        for i, fg in enumerate(self._feature_groups_to_be_merged):
            selected += ", " + ", ".join(
                f'fg_{i}."{f}" as "{f}.{i+1}"' for f in fg.projected_feature_names
            )
            selected_final += ", " + ", ".join(
                f'"{f}.{i+1}"' for f in fg.projected_feature_names
            )

        query += (
            f"\nSELECT {selected_final}\nFROM (\n"
            f"SELECT {selected}, row_number() OVER (\n"
            f'PARTITION BY fg_base."{base.record_identifier_feature_name}"\n'
            f'ORDER BY fg_base."{base.event_time_identifier_feature.feature_name}" DESC'
        )

        join_strings = []
        for i, fg in enumerate(self._feature_groups_to_be_merged):
            if not fg.target_feature_name_in_base:
                fg.target_feature_name_in_base = self._record_identifier_feature_name
            elif fg.target_feature_name_in_base not in base.features:
                raise ValueError(f"Feature {fg.target_feature_name_in_base} not found in base")
            query += f', fg_{i}."{fg.event_time_identifier_feature.feature_name}" DESC'
            join_strings.append(self._construct_join_condition(fg, str(i)))

        recent_where = ""
        if self._number_of_recent_records is not None and self._number_of_recent_records >= 0:
            recent_where = f"WHERE row_recent <= {self._number_of_recent_records}"

        query += f"\n) AS row_recent\nFROM fg_base{''.join(join_strings)}\n)\n{recent_where}"

        if self._number_of_records is not None and self._number_of_records >= 0:
            query += f"\nLIMIT {self._number_of_records}"

        return query

    def _construct_table_query(self, fg: FeatureGroupToBeMerged, suffix: str) -> str:
        included = ", ".join(f'table_{suffix}."{f}"' for f in fg.included_feature_names)
        included_with_write = included
        if fg.table_type is TableType.FEATURE_GROUP:
            included_with_write += f', table_{suffix}."write_time"'

        record_id = fg.record_identifier_feature_name
        event_time = fg.event_time_identifier_feature.feature_name

        if self._include_duplicated_records and self._include_deleted_records:
            return (
                f"SELECT {included}\n"
                f'FROM "{fg.database}"."{fg.table_name}" table_{suffix}\n'
                + self._construct_where_query_string(suffix, fg.event_time_identifier_feature, ["NOT is_deleted"])
            )

        if fg.table_type is TableType.FEATURE_GROUP and self._include_deleted_records:
            rank = f'ORDER BY origin_{suffix}."api_invocation_time" DESC, origin_{suffix}."write_time" DESC\n'
            return (
                f"SELECT {included}\nFROM (\n"
                f"SELECT *, row_number() OVER (\n"
                f'PARTITION BY origin_{suffix}."{record_id}", origin_{suffix}."{event_time}"\n'
                f"{rank}) AS row_{suffix}\n"
                f'FROM "{fg.database}"."{fg.table_name}" origin_{suffix}\n'
                f"WHERE NOT is_deleted) AS table_{suffix}\n"
                + self._construct_where_query_string(suffix, fg.event_time_identifier_feature, [f"row_{suffix} = 1"])
            )

        if fg.table_type is TableType.FEATURE_GROUP:
            dedup = self._construct_dedup_query(fg, suffix)
            deleted = self._construct_deleted_query(fg, suffix)
            rank_cond = (
                f'OR (table_{suffix}."{event_time}" = deleted_{suffix}."{event_time}" '
                f'AND table_{suffix}."api_invocation_time" > deleted_{suffix}."api_invocation_time")\n'
                f'OR (table_{suffix}."{event_time}" = deleted_{suffix}."{event_time}" '
                f'AND table_{suffix}."api_invocation_time" = deleted_{suffix}."api_invocation_time" '
                f'AND table_{suffix}."write_time" > deleted_{suffix}."write_time")\n'
            )

            if self._include_duplicated_records:
                return (
                    f"WITH {deleted}\n"
                    f"SELECT {included}\nFROM (\n"
                    f"SELECT {included_with_write}\n"
                    f'FROM "{fg.database}"."{fg.table_name}" table_{suffix}\n'
                    f"LEFT JOIN deleted_{suffix} ON table_{suffix}.\"{record_id}\" = deleted_{suffix}.\"{record_id}\"\n"
                    f'WHERE deleted_{suffix}."{record_id}" IS NULL\n'
                    f"UNION ALL\n"
                    f"SELECT {included_with_write}\nFROM deleted_{suffix}\n"
                    f'JOIN "{fg.database}"."{fg.table_name}" table_{suffix}\n'
                    f'ON table_{suffix}."{record_id}" = deleted_{suffix}."{record_id}"\n'
                    f'AND (table_{suffix}."{event_time}" > deleted_{suffix}."{event_time}"\n{rank_cond})\n'
                    f") AS table_{suffix}\n"
                    + self._construct_where_query_string(suffix, fg.event_time_identifier_feature, [])
                )

            return (
                f"WITH {dedup},\n{deleted}\n"
                f"SELECT {included}\nFROM (\n"
                f"SELECT {included_with_write}\nFROM table_{suffix}\n"
                f"LEFT JOIN deleted_{suffix} ON table_{suffix}.\"{record_id}\" = deleted_{suffix}.\"{record_id}\"\n"
                f'WHERE deleted_{suffix}."{record_id}" IS NULL\n'
                f"UNION ALL\n"
                f"SELECT {included_with_write}\nFROM deleted_{suffix}\n"
                f"JOIN table_{suffix} ON table_{suffix}.\"{record_id}\" = deleted_{suffix}.\"{record_id}\"\n"
                f'AND (table_{suffix}."{event_time}" > deleted_{suffix}."{event_time}"\n{rank_cond})\n'
                f") AS table_{suffix}\n"
                + self._construct_where_query_string(suffix, fg.event_time_identifier_feature, [])
            )

        dedup = self._construct_dedup_query(fg, suffix)
        return (
            f"WITH {dedup}\n"
            f"SELECT {included}\nFROM (\n"
            f"SELECT {included_with_write}\nFROM table_{suffix}\n"
            f") AS table_{suffix}\n"
            + self._construct_where_query_string(suffix, fg.event_time_identifier_feature, [])
        )

    def _construct_dedup_query(self, fg: FeatureGroupToBeMerged, suffix: str) -> str:
        record_id = fg.record_identifier_feature_name
        event_time = fg.event_time_identifier_feature.feature_name
        rank = ""
        is_fg = fg.table_type is TableType.FEATURE_GROUP

        if is_fg:
            rank = f'ORDER BY origin_{suffix}."api_invocation_time" DESC, origin_{suffix}."write_time" DESC\n'

        where_conds = []
        if is_fg and self._write_time_ending_timestamp:
            where_conds.append(self._construct_write_time_condition(f"origin_{suffix}"))
        where_conds.extend(self._construct_event_time_conditions(f"origin_{suffix}", fg.event_time_identifier_feature))
        where_str = f"WHERE {' AND '.join(where_conds)}\n" if where_conds else ""

        dedup_where = f"WHERE dedup_row_{suffix} = 1\n" if is_fg else ""

        return (
            f"table_{suffix} AS (\n"
            f"SELECT *\nFROM (\n"
            f"SELECT *, row_number() OVER (\n"
            f'PARTITION BY origin_{suffix}."{record_id}", origin_{suffix}."{event_time}"\n'
            f"{rank}) AS dedup_row_{suffix}\n"
            f'FROM "{fg.database}"."{fg.table_name}" origin_{suffix}\n'
            f"{where_str})\n{dedup_where})"
        )

    def _construct_deleted_query(self, fg: FeatureGroupToBeMerged, suffix: str) -> str:
        record_id = fg.record_identifier_feature_name
        event_time = fg.event_time_identifier_feature.feature_name
        rank = f'ORDER BY origin_{suffix}."{event_time}" DESC'

        if fg.table_type is TableType.FEATURE_GROUP:
            rank += f', origin_{suffix}."api_invocation_time" DESC, origin_{suffix}."write_time" DESC\n'

        write_cond = ""
        if fg.table_type is TableType.FEATURE_GROUP and self._write_time_ending_timestamp:
            write_cond = f" AND {self._construct_write_time_condition(f'origin_{suffix}')}\n"

        event_conds = ""
        if self._event_time_starting_timestamp and self._event_time_ending_timestamp:
            conds = self._construct_event_time_conditions(f"origin_{suffix}", fg.event_time_identifier_feature)
            event_conds = "".join(f"AND {c}\n" for c in conds)

        return (
            f"deleted_{suffix} AS (\n"
            f"SELECT *\nFROM (\n"
            f"SELECT *, row_number() OVER (\n"
            f'PARTITION BY origin_{suffix}."{record_id}"\n'
            f"{rank}) AS deleted_row_{suffix}\n"
            f'FROM "{fg.database}"."{fg.table_name}" origin_{suffix}\n'
            f"WHERE is_deleted{write_cond}{event_conds})\n"
            f"WHERE deleted_row_{suffix} = 1\n)"
        )

    def _construct_where_query_string(
        self, suffix: str, event_time_feature: FeatureDefinition, conditions: List[str]
    ) -> str:
        self._validate_options()

        if isinstance(self._base, FeatureGroup) and self._write_time_ending_timestamp:
            conditions.append(self._construct_write_time_condition(f"table_{suffix}"))

        conditions.extend(self._construct_event_time_conditions(f"table_{suffix}", event_time_feature))
        return f"WHERE {' AND '.join(conditions)}" if conditions else ""

    def _validate_options(self):
        is_df_base = isinstance(self._base, pd.DataFrame)
        no_joins = len(self._feature_groups_to_be_merged) == 0

        if self._number_of_recent_records is not None and self._number_of_recent_records < 0:
            raise ValueError("number_of_recent_records must be non-negative.")
        if self._number_of_records is not None and self._number_of_records < 0:
            raise ValueError("number_of_records must be non-negative.")
        if is_df_base and no_joins:
            if self._include_deleted_records:
                raise ValueError("include_deleted_records() only works for FeatureGroup if no join.")
            if self._include_duplicated_records:
                raise ValueError("include_duplicated_records() only works for FeatureGroup if no join.")
            if self._write_time_ending_timestamp:
                raise ValueError("as_of() only works for FeatureGroup if no join.")
        if self._point_in_time_accurate_join and no_joins:
            raise ValueError("point_in_time_accurate_join() requires at least one join.")

    def _construct_event_time_conditions(self, table: str, event_time_feature: FeatureDefinition) -> List[str]:
        cast_fn = "from_iso8601_timestamp" if event_time_feature.feature_type == FeatureTypeEnum.STRING else "from_unixtime"
        conditions = []
        if self._event_time_starting_timestamp:
            conditions.append(
                f'{cast_fn}({table}."{event_time_feature.feature_name}") >= '
                f"from_unixtime({self._event_time_starting_timestamp.timestamp()})"
            )
        if self._event_time_ending_timestamp:
            conditions.append(
                f'{cast_fn}({table}."{event_time_feature.feature_name}") <= '
                f"from_unixtime({self._event_time_ending_timestamp.timestamp()})"
            )
        return conditions

    def _construct_write_time_condition(self, table: str) -> str:
        ts = self._write_time_ending_timestamp.replace(microsecond=0)
        return f'{table}."write_time" <= to_timestamp(\'{ts}\', \'yyyy-mm-dd hh24:mi:ss\')'

    def _construct_join_condition(self, fg: FeatureGroupToBeMerged, suffix: str) -> str:
        target_feature = fg.feature_name_in_target or fg.record_identifier_feature_name
        join = (
            f"\n{fg.join_type.value} fg_{suffix}\n"
            f'ON fg_base."{fg.target_feature_name_in_base}" {fg.join_comparator.value} fg_{suffix}."{target_feature}"'
        )

        if self._point_in_time_accurate_join:
            base_cast = "from_iso8601_timestamp" if self._event_time_identifier_feature_type == FeatureTypeEnum.STRING else "from_unixtime"
            fg_cast = "from_iso8601_timestamp" if fg.event_time_identifier_feature.feature_type == FeatureTypeEnum.STRING else "from_unixtime"
            join += (
                f'\nAND {base_cast}(fg_base."{self._event_time_identifier_feature_name}") >= '
                f'{fg_cast}(fg_{suffix}."{fg.event_time_identifier_feature.feature_name}")'
            )

        return join
