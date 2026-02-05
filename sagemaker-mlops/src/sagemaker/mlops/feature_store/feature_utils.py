# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Utilities for working with FeatureGroups and FeatureStores."""
import logging
import os
import time
from pathlib import Path
import re
from typing import Any, Dict, Sequence, Union

import boto3
import pandas
import pandas as pd
from pandas import DataFrame, Series, read_csv

from sagemaker.mlops.feature_store import FeatureGroup as CoreFeatureGroup, FeatureGroup
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.s3.client import S3Uploader, S3Downloader
from sagemaker.mlops.feature_store.feature_definition import (
    FeatureDefinition,
    FractionalFeatureDefinition,
    IntegralFeatureDefinition,
    ListCollectionType,
    StringFeatureDefinition,
)
from sagemaker.mlops.feature_store.ingestion_manager_pandas import IngestionManagerPandas

from sagemaker.core.utils import unique_name_from_base


logger = logging.getLogger(__name__)

# --- Constants ---

_FEATURE_TYPE_TO_DDL_DATA_TYPE_MAP = {
    "Integral": "INT",
    "Fractional": "FLOAT",
    "String": "STRING",
}

_DTYPE_TO_FEATURE_TYPE_MAP = {
    "object": "String",
    "string": "String",
    "int64": "Integral",
    "float64": "Fractional",
}

_INTEGER_TYPES = {"int_", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
_FLOAT_TYPES = {"float_", "float16", "float32", "float64"}


def _get_athena_client(session: Session):
    """Get Athena client from session."""
    return session.boto_session.client("athena", region_name=session.boto_region_name)


def _get_s3_client(session: Session):
    """Get S3 client from session."""
    return session.boto_session.client("s3", region_name=session.boto_region_name)


def start_query_execution(
    session: Session,
    catalog: str,
    database: str,
    query_string: str,
    output_location: str,
    kms_key: str = None,
    workgroup: str = None,
) -> Dict[str, str]:
    """Start Athena query execution.

    Args:
        session: Session instance for boto calls.
        catalog: Name of the data catalog.
        database: Name of the database.
        query_string: SQL query string.
        output_location: S3 URI for query results.
        kms_key: KMS key for encryption (default: None).
        workgroup: Athena workgroup name (default: None).

    Returns:
        Response dict with QueryExecutionId.
    """
    kwargs = {
        "QueryString": query_string,
        "QueryExecutionContext": {"Catalog": catalog, "Database": database},
        "ResultConfiguration": {"OutputLocation": output_location},
    }
    if kms_key:
        kwargs["ResultConfiguration"]["EncryptionConfiguration"] = {
            "EncryptionOption": "SSE_KMS",
            "KmsKey": kms_key,
        }
    if workgroup:
        kwargs["WorkGroup"] = workgroup
    return _get_athena_client(session).start_query_execution(**kwargs)


def get_query_execution(session: Session, query_execution_id: str) -> Dict[str, Any]:
    """Get execution status of an Athena query.

    Args:
        session: Session instance for boto calls.
        query_execution_id: The query execution ID.

    Returns:
        Response dict from Athena.
    """
    return _get_athena_client(session).get_query_execution(QueryExecutionId=query_execution_id)


def wait_for_athena_query(session: Session, query_execution_id: str, poll: int = 5):
    """Wait for Athena query to finish.

    Args:
        session: Session instance for boto calls.
        query_execution_id: The query execution ID.
        poll: Polling interval in seconds (default: 5).
    """
    while True:
        state = get_query_execution(session, query_execution_id)["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED"):
            logger.info("Query %s %s.", query_execution_id, state.lower())
            break
        logger.info("Query %s is being executed.", query_execution_id)
        time.sleep(poll)


def run_athena_query(
    session: Session,
    catalog: str,
    database: str,
    query_string: str,
    output_location: str,
    kms_key: str = None,
) -> Dict[str, Any]:
    """Execute Athena query, wait for completion, and return result.

    Args:
        session: Session instance for boto calls.
        catalog: Name of the data catalog.
        database: Name of the database.
        query_string: SQL query string.
        output_location: S3 URI for query results.
        kms_key: KMS key for encryption (default: None).

    Returns:
        Query execution result dict.

    Raises:
        RuntimeError: If query fails.
    """
    response = start_query_execution(
        session=session,
        catalog=catalog,
        database=database,
        query_string=query_string,
        output_location=output_location,
        kms_key=kms_key,
    )
    query_id = response["QueryExecutionId"]
    wait_for_athena_query(session, query_id)

    result = get_query_execution(session, query_id)
    if result["QueryExecution"]["Status"]["State"] != "SUCCEEDED":
        raise RuntimeError(f"Athena query {query_id} failed.")
    return result


def download_athena_query_result(
    session: Session,
    bucket: str,
    prefix: str,
    query_execution_id: str,
    filename: str,
):
    """Download query result file from S3.

    Args:
        session: Session instance for boto calls.
        bucket: S3 bucket name.
        prefix: S3 key prefix.
        query_execution_id: The query execution ID.
        filename: Local filename to save to.
    """
    _get_s3_client(session).download_file(
        Bucket=bucket,
        Key=f"{prefix}/{query_execution_id}.csv",
        Filename=filename,
    )


def upload_dataframe_to_s3(
    data_frame: DataFrame,
    output_path: str,
    session: Session,
    kms_key: str = None,
) -> tuple[str, str]:
    """Upload DataFrame to S3 as CSV.

    Args:
        data_frame: DataFrame to upload.
        output_path: S3 URI base path.
        session: Session instance for boto calls.
        kms_key: KMS key for encryption (default: None).

    Returns:
        Tuple of (s3_folder, temp_table_name).
    """

    temp_id = unique_name_from_base("dataframe-base")
    local_file = f"{temp_id}.csv"
    s3_folder = os.path.join(output_path, temp_id)

    data_frame.to_csv(local_file, index=False, header=False)
    S3Uploader.upload(
        local_path=local_file,
        desired_s3_uri=s3_folder,
        sagemaker_session=session,
        kms_key=kms_key,
    )
    os.remove(local_file)

    table_name = f'dataframe_{temp_id.replace("-", "_")}'
    return s3_folder, table_name


def download_csv_from_s3(
    s3_uri: str,
    session: Session,
    kms_key: str = None,
) -> DataFrame:
    """Download CSV from S3 and return as DataFrame.

    Args:
        s3_uri: S3 URI of the CSV file.
        session: Session instance for boto calls.
        kms_key: KMS key for decryption (default: None).

    Returns:
        DataFrame with CSV contents.
    """

    S3Downloader.download(
        s3_uri=s3_uri,
        local_path="./",
        kms_key=kms_key,
        sagemaker_session=session,
    )

    local_file = s3_uri.split("/")[-1]
    df = pd.read_csv(local_file)
    os.remove(local_file)

    metadata_file = f"{local_file}.metadata"
    if os.path.exists(metadata_file):
        os.remove(metadata_file)

    return df


def get_session_from_role(region: str, assume_role: str = None) -> Session:
    """Get a Session from a region and optional IAM role.

    Args:
        region: AWS region name.
        assume_role: IAM role ARN to assume (default: None).

    Returns:
        Session instance.
    """
    boto_session = boto3.Session(region_name=region)

    if assume_role:
        sts = boto_session.client("sts", region_name=region)
        credentials = sts.assume_role(
            RoleArn=assume_role,
            RoleSessionName="SagemakerExecution",
        )["Credentials"]

        boto_session = boto3.Session(
            region_name=region,
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )

    return Session(
        boto_session=boto_session,
        sagemaker_client=boto_session.client("sagemaker"),
        sagemaker_runtime_client=boto_session.client("sagemaker-runtime"),
        sagemaker_featurestore_runtime_client=boto_session.client("sagemaker-featurestore-runtime"),
    )


# --- FeatureDefinition Functions ---

def _is_collection_column(series: Series, sample_size: int = 1000) -> bool:
    """Check if column contains list/set values."""
    sample = series.head(sample_size).dropna()
    return sample.apply(lambda x: isinstance(x, (list, set))).any()


def _generate_feature_definition(
    series: Series,
    online_storage_type: str = None,
) -> FeatureDefinition:
    """Generate a FeatureDefinition from a pandas Series."""
    dtype = str(series.dtype)
    collection_type = None

    if online_storage_type == "InMemory" and _is_collection_column(series):
        collection_type = ListCollectionType()

    if dtype in _INTEGER_TYPES:
        return IntegralFeatureDefinition(series.name, collection_type)
    if dtype in _FLOAT_TYPES:
        return FractionalFeatureDefinition(series.name, collection_type)
    return StringFeatureDefinition(series.name, collection_type)


def load_feature_definitions_from_dataframe(
    data_frame: DataFrame,
    online_storage_type: str = None,
) -> Sequence[FeatureDefinition]:
    """Infer FeatureDefinitions from DataFrame dtypes.

    Column name is used as feature name. Feature type is inferred from the dtype
    of the column. Integer dtypes are mapped to Integral feature type. Float dtypes
    are mapped to Fractional feature type. All other dtypes are mapped to String.

    For IN_MEMORY online_storage_type, collection type columns within DataFrame
    will be inferred as List instead of String.

    Args:
        data_frame: DataFrame to infer features from.
        online_storage_type: "Standard" or "InMemory" (default: None).

    Returns:
        List of FeatureDefinition objects.
    """
    return [
        _generate_feature_definition(data_frame[col], online_storage_type)
        for col in data_frame.columns
    ]


# --- FeatureGroup Functions ---

def create_athena_query(feature_group_name: str, session: Session):
    """Create an AthenaQuery for a FeatureGroup.

    Args:
        feature_group_name: Name of the FeatureGroup.
        session: Session instance for Athena boto calls.

    Returns:
        AthenaQuery initialized with data catalog config.

    Raises:
        RuntimeError: If no metastore is configured.
    """
    from sagemaker.mlops.feature_store.athena_query import AthenaQuery

    fg = CoreFeatureGroup.get(feature_group_name=feature_group_name)

    if not fg.offline_store_config or not fg.offline_store_config.data_catalog_config:
        raise RuntimeError("No metastore is configured with this feature group.")

    catalog_config = fg.offline_store_config.data_catalog_config
    disable_glue = catalog_config.disable_glue_table_creation or False

    return AthenaQuery(
        catalog=catalog_config.catalog if disable_glue else "AwsDataCatalog",
        database=catalog_config.database,
        table_name=catalog_config.table_name,
        sagemaker_session=session,
    )


def as_hive_ddl(
    feature_group_name: str,
    database: str = "sagemaker_featurestore",
    table_name: str = None,
) -> str:
    """Generate Hive DDL for a FeatureGroup's offline store table.

    Schema of the table is generated based on the feature definitions. Columns are named
    after feature name and data-type are inferred based on feature type. Integral feature
    type is mapped to INT data-type. Fractional feature type is mapped to FLOAT data-type.
    String feature type is mapped to STRING data-type.

    Args:
        feature_group_name: Name of the FeatureGroup.
        database: Hive database name (default: "sagemaker_featurestore").
        table_name: Hive table name (default: feature_group_name).

    Returns:
        CREATE EXTERNAL TABLE DDL string.
    """
    fg = CoreFeatureGroup.get(feature_group_name=feature_group_name)
    table_name = table_name or feature_group_name
    resolved_output_s3_uri = fg.offline_store_config.s3_storage_config.resolved_output_s3_uri

    ddl = f"CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table_name} (\n"
    for fd in fg.feature_definitions:
        ddl += f"  {fd.feature_name} {_FEATURE_TYPE_TO_DDL_DATA_TYPE_MAP.get(fd.feature_type)}\n"
    ddl += "  write_time TIMESTAMP\n"
    ddl += "  event_time TIMESTAMP\n"
    ddl += "  is_deleted BOOLEAN\n"
    ddl += ")\n"
    ddl += (
        "ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'\n"
        "  STORED AS\n"
        "  INPUTFORMAT 'parquet.hive.DeprecatedParquetInputFormat'\n"
        "  OUTPUTFORMAT 'parquet.hive.DeprecatedParquetOutputFormat'\n"
        f"LOCATION '{resolved_output_s3_uri}'"
    )
    return ddl


def ingest_dataframe(
    feature_group_name: str,
    data_frame: DataFrame,
    max_workers: int = 1,
    max_processes: int = 1,
    wait: bool = True,
    timeout: Union[int, float] = None,
):
    """Ingest a pandas DataFrame to a FeatureGroup.

    Args:
        feature_group_name: Name of the FeatureGroup.
        data_frame: DataFrame to ingest.
        max_workers: Threads per process (default: 1).
        max_processes: Number of processes (default: 1).
        wait: Wait for ingestion to complete (default: True).
        timeout: Timeout in seconds (default: None).

    Returns:
        IngestionManagerPandas instance.

    Raises:
        ValueError: If max_workers or max_processes <= 0.
    """
    
    if max_processes <= 0:
        raise ValueError("max_processes must be greater than 0.")
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than 0.")

    fg = CoreFeatureGroup.get(feature_group_name=feature_group_name)
    feature_definitions = {fd.feature_name: fd.feature_type for fd in fg.feature_definitions}

    manager = IngestionManagerPandas(
        feature_group_name=feature_group_name,
        feature_definitions=feature_definitions,
        max_workers=max_workers,
        max_processes=max_processes,
    )
    manager.run(data_frame=data_frame, wait=wait, timeout=timeout)
    return manager

def get_feature_group_as_dataframe(
    feature_group_name: str,
    athena_bucket: str,
    query: str = """SELECT * FROM "sagemaker_featurestore"."#{table}"
                    WHERE is_deleted=False """,
    role: str = None,
    region: str = None,
    session=None,
    event_time_feature_name: str = None,
    latest_ingestion: bool = True,
    verbose: bool = True,
    **kwargs,
) -> DataFrame:
    """:class:`sagemaker.feature_store.feature_group.FeatureGroup` as :class:`pandas.DataFrame`

    Examples:
        >>> from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe
        >>>
        >>> region = "eu-west-1"
        >>> fg_data = get_feature_group_as_dataframe(feature_group_name="feature_group",
        >>>                                          athena_bucket="s3://bucket/athena_queries",
        >>>                                          region=region,
        >>>                                          event_time_feature_name="EventTimeId"
        >>>                                          )
        >>>
        >>> type(fg_data)
        <class 'pandas.core.frame.DataFrame'>

    Description:
        Method to run an athena query over a
        :class:`sagemaker.feature_store.feature_group.FeatureGroup` in a Feature Store
        to retrieve its data. It needs the :class:`sagemaker.session.Session` linked to a role
        or the region and/or role used to work with Feature Stores (it uses the module
        `sagemaker.feature_store.feature_utils.get_session_from_role`
        to get the session).

    Args:
        region (str): region of the target Feature Store
        feature_group_name (str): feature store name
        query (str): query to run. By default, it will take the latest ingest with data that
                    wasn't deleted. If latest_ingestion is False it will take all the data
                    in the feature group that wasn't deleted. It needs to use the keyword
                    "#{table}" to refer to the FeatureGroup name. e.g.:
                    'SELECT * FROM "sagemaker_featurestore"."#{table}"'
                    It must not end by ';'.
        athena_bucket (str): Amazon S3 bucket for running the query
        role (str): role to be assumed to extract data from feature store. If not specified
                    the default sagemaker execution role will be used.
        session (str): :class:`sagemaker.session.Session`
                        of SageMaker used to work with the feature store. Optional, with
                        role and region parameters it will infer the session.
        event_time_feature_name (str): eventTimeId feature. Mandatory only if the
                                        latest ingestion is True.
        latest_ingestion (bool): if True it will get the data only from the latest ingestion.
                                 If False it will take whatever is specified in the query, or
                                 if not specify it, it will get all the data that wasn't deleted.
        verbose (bool): if True show messages, if False is silent.
        **kwargs (object): key arguments used for the method pandas.read_csv to be able to
                    have a better tuning on data. For more info read:
                    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    Returns:
        :class:`pandas.DataFrame`: dataset with the data retrieved from feature group
    """

    logger.setLevel(logging.WARNING)
    if verbose:
        logger.setLevel(logging.INFO)

    if latest_ingestion:
        if event_time_feature_name is not None:
            query += str(
                f"AND {event_time_feature_name}=(SELECT "
                f"MAX({event_time_feature_name}) FROM "
                '"sagemaker_featurestore"."#{table}")'
            )
        else:
            exc = Exception(
                "Argument event_time_feature_name must be specified "
                "when using latest_ingestion=True."
            )
            logger.exception(exc)
            raise exc

    query += ";"

    if session is not None:
        sagemaker_session = session
    elif region is not None:
        sagemaker_session = get_session_from_role(region=region, assume_role=role)
    else:
        exc = Exception("Argument Session or role and region must be specified.")
        logger.exception(exc)
        raise exc

    msg = f"Feature Group used: {feature_group_name}"
    logger.info(msg)

    fg = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

    sample_query = fg.athena_query()
    query_string = re.sub(r"#\{(table)\}", sample_query.table_name, query)

    msg = f"Running query:\n\t{sample_query} \n\n\t-> Save on bucket {athena_bucket}\n"
    logger.info(msg)

    sample_query.run(query_string=query_string, output_location=athena_bucket)

    sample_query.wait()

    # run Athena query. The output is loaded to a Pandas dataframe.
    dataset = sample_query.as_dataframe(**kwargs)

    msg = f"Data shape retrieve from {feature_group_name}: {dataset.shape}"
    logger.info(msg)

    return dataset


def prepare_fg_from_dataframe_or_file(
    dataframe_or_path: Union[str, Path, pandas.DataFrame],
    feature_group_name: str,
    role: str = None,
    region: str = None,
    session=None,
    record_id: str = "record_id",
    event_id: str = "data_as_of_date",
    verbose: bool = False,
    **kwargs,
) -> FeatureGroup:
    """Prepares a dataframe to create a :class:`sagemaker.feature_store.feature_group.FeatureGroup`

    Description:
        Function to prepare a :class:`pandas.DataFrame` read from a path to a csv file or pass it
        directly to create a :class:`sagemaker.feature_store.feature_group.FeatureGroup`.
        The path to the file needs proper dtypes, feature names and mandatory features (record_id,
        event_id).
        It needs the :class:`sagemaker.session.Session` linked to a role
        or the region and/or role used to work with Feature Stores (it uses the module
        `sagemaker.feature_store.feature_utils.get_session_from_role`
        to get the session).
        If record_id or event_id are not specified it will create ones
        by default with the names 'record_id' and 'data_as_of_date'.

    Args:
        feature_group_name (str): feature group name
        dataframe_or_path (str, Path, pandas.DataFrame) : pandas.DataFrame or path to the data
        verbose (bool)           : True for displaying messages, False for silent method.
        record_id (str, 'record_id'): (Optional) Feature identifier of the rows. If specified each
                                    value of that feature has to be unique. If not specified or
                                    record_id='record_id', then it will create a new feature from
                                    the index of the pandas.DataFrame.
        event_id (str)           : (Optional) Feature with the time of the creation of data rows.
                                    If not specified it will create one with the current time
                                    called `data_as_of_date`
        role (str)               : role used to get the session.
        region (str)             : region used to get the session.
        session (str): session of SageMaker used to work with the feature store
        **kwargs (object): key arguments used for the method pandas.read_csv to be able to
                    have a better tuning on data. For more info read:
                    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

    Returns:
        :class:`sagemaker.feature_store.feature_group.FeatureGroup`:
            FG prepared with all the methods and definitions properly defined
    """

    logger.setLevel(logging.WARNING)
    if verbose:
        logger.setLevel(logging.INFO)

    if isinstance(dataframe_or_path, DataFrame):
        data = dataframe_or_path
    elif isinstance(dataframe_or_path, str):
        kwargs.pop("filepath_or_buffer", None)
        data = read_csv(filepath_or_buffer=dataframe_or_path, **kwargs)
    else:
        exc = Exception(
            str(
                f"Invalid type {type(dataframe_or_path)} for "
                "argument dataframe_or_path. \nParameter must be"
                " of type pandas.DataFrame or string"
            )
        )
        logger.exception(exc)
        raise exc

    # Formatting cols
    data = _format_column_names(data=data)
    data = _cast_object_to_string(data_frame=data)

    if record_id == "record_id" and record_id not in data.columns:
        data[record_id] = data.index

    lg_uniq = len(data[record_id].unique())
    lg_id = len(data[record_id])

    if lg_id != lg_uniq:
        exc = Exception(
            str(
                f"Record identifier {record_id} have {abs(lg_id - lg_uniq)} "
                "duplicated rows. \nRecord identifier must be unique"
                " in each row."
            )
        )
        logger.exception(exc)
        raise exc

    if event_id not in data.columns:
        import time

        current_time_sec = int(round(time.time()))
        data[event_id] = Series([current_time_sec] * lg_id, dtype="float64")

    if session is not None:
        sagemaker_session = session
    elif role is not None and region is not None:
        sagemaker_session = get_session_from_role(region=region)
    else:
        exc = Exception("Argument Session or role and region must be specified.")
        logger.exception(exc)
        raise exc

    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

    feature_group.load_feature_definitions(data_frame=data)

    return feature_group


def _format_column_names(data: pandas.DataFrame) -> pandas.DataFrame:
    """Formats the column names for :class:`sagemaker.feature_store.feature_group.FeatureGroup`

    Description:
        Module to format correctly the name of the columns of a DataFrame
        to later generate the features names of a Feature Group

    Args:
        data (:class:`pandas.DataFrame`): dataframe used

    Returns:
        :class:`pandas.DataFrame`
    """
    data.rename(columns=lambda x: x.replace(" ", "_").replace(".", "").lower()[:62], inplace=True)
    return data

def _cast_object_to_string(data_frame: pandas.DataFrame) -> pandas.DataFrame:
    """Cast properly pandas object types to strings

    Description:
        Method to convert 'object' and 'O' column dtypes of a pandas.DataFrame to
        a valid string type recognized by Feature Groups.

    Args:
        data_frame: dataframe used
    Returns:
        pandas.DataFrame
    """
    for label in data_frame.select_dtypes(["object", "O"]).columns.tolist():
        data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame