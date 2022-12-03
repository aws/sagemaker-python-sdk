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
"""Utilities for working with FeatureGroups and FeatureStores."""
from __future__ import absolute_import

import re
import logging

from typing import Union
from pathlib import Path

import pandas
import boto3
from pandas import DataFrame, Series, read_csv

from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

logger = logging.getLogger(__name__)


def _get_session_from_role(role: str, region: str):
    """Method use to get the sagemaker session from a role and a region.

    Helpful in case it's invoke from a session with a role without permission it can assume
    another role temporarily to perform certain taks.

    Args:
        role: role name
        region: region name

    Returns:

    """
    boto_session = boto3.Session(region_name=region)

    sts = boto_session.client(
        "sts", region_name=region, endpoint_url="https://sts.eu-west-1.amazonaws.com"
    )

    metadata = sts.assume_role(RoleArn=role, RoleSessionName="SagemakerExecution")

    access_key_id = metadata["Credentials"]["AccessKeyId"]
    secret_access_key = metadata["Credentials"]["SecretAccessKey"]
    session_token = metadata["Credentials"]["SessionToken"]

    boto_session = boto3.session.Session(
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token,
    )

    # Sessions
    sagemaker_client = boto_session.client("sagemaker")
    sagemaker_runtime = boto_session.client("sagemaker-runtime")
    runtime_client = boto_session.client(service_name="sagemaker-featurestore-runtime")
    sagemaker_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=sagemaker_runtime,
        sagemaker_featurestore_runtime_client=runtime_client,
    )

    return sagemaker_session


def get_feature_group_as_dataframe(
    feature_group_name: str,
    athena_bucket: str,
    query: str = str(
        "SELECT * FROM " '"sagemaker_featurestore"."#{table}" ' "WHERE is_deleted=False"
    ),
    role: str = None,
    region: str = None,
    session=None,
    event_time_feature_name: str = None,
    latest_ingestion: bool = True,
    verbose: bool = True,
    **pandas_read_csv_kwargs,
) -> DataFrame:
    """Get a feature group as a pandas.DataFrame

    Description:
        Method to run an athena query over a Feature Group in a Feature Store
        to retrieve its data.It needs the sagemaker.Session linked to a role
        or the role and region used to work Feature Stores.Returns a dataframe
        with the data.

    Args:
        region (str): region of the target feature store
        feature_group_name (str): feature store name
        query (str): query to run. By default, it will take the latest ingest with data that
                    wasn't deleted. If latest_ingestion is False it will take all the data
                    in the feature group that wasn't deleted. It needs to use the keyword
                    "#{table}" to refer to the table. e.g.:
                        'SELECT * FROM "sagemaker_featurestore"."#{table}"'
        athena_bucket (str): S3 bucket for running the query
        role (str): role of the account used to extract data from feature store
        session (str): session of SageMaker used to work with the feature store
        event_time_feature_name (str): eventTimeId feature. Mandatory only if the
                                        latest ingestion is True
        latest_ingestion (bool): if True it will get the data only from the latest ingestion.
                                 If False it will take whatever is specified in the query, or
                                 if not specify it, it will get all the data that wasn't deleted.
        verbose (bool): if True show messages, if False is silent.

    Returns:
        dataset (pandas.DataFrame): dataset with the data retrieved from feature group
    """

    logger.setLevel(logging.WARNING)
    if verbose:
        logger.setLevel(logging.INFO)

    if latest_ingestion:
        if event_time_feature_name is not None:
            query += str(
                f"AND {event_time_feature_name}=(SELECT "
                f"MAX({event_time_feature_name}) FROM "
                + f'"sagemaker_featurestore"."{feature_group_name}")'
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
    elif role is not None and region is not None:
        sagemaker_session = _get_session_from_role(role=role, region=region)
    else:
        exc = Exception("Argument Session or role and region must be specified.")
        logger.exception(exc)
        raise exc

    logger.info(f"Feature Group used: {feature_group_name}")

    fg = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

    sample_query = fg.athena_query()
    query_string = re.sub(r"#\{(table)\}", sample_query.table_name, query)

    logger.info(
        f"Running query:\n\t{sample_query} \n\n\t-> Save on bucket {athena_bucket}\n"
    )

    sample_query.run(query_string=query_string, output_location=athena_bucket)

    sample_query.wait()

    # run Athena query. The output is loaded to a Pandas dataframe.
    dataset = sample_query.as_dataframe(**pandas_read_csv_kwargs)

    logger.info(f"Data shape retrieve from {feature_group_name}: {dataset.shape}")

    return dataset


def _format_column_names(data: pandas.DataFrame) -> pandas.DataFrame:
    """Format the column names for a FeatureGroup

    Module to format correctly the name of the columns of a DataFrame
    to later generate the features names of a Feature Group

    Args:
        data (pandas.DataFrame): dataframe used

    Returns:
        pandas.DataFrame
    """
    data.rename(
        columns=lambda x: x.replace(" ", "_").replace(".", "").lower()[:62],
        inplace=True,
    )
    return data


def _cast_object_to_string(data_frame: pandas.DataFrame) -> pandas.DataFrame:
    """Cast properly pandas object types to strings

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


def prepare_fg_from_dataframe_or_file(
    dataframe_or_path: Union[str, Path, pandas.DataFrame],
    feature_group_name: str,
    role: str = None,
    region: str = None,
    session=None,
    record_id: str = "record_id",
    event_id: str = "data_as_of_date",
    verbose: bool = False,
    **pandas_read_csv_kwargs,
) -> FeatureGroup:
    """Module to prepare a dataframe before creating  Feature Group

    Function to prepare a dataframe for creating a Feature Group from a pandas.DataFrame
    or a path to a file with proper dtypes, feature names and mandatory features (record_id,
    event_id). It needs the sagemaker.Session linked to a role or the role and region used
    to work Feature Stores. If record_id or event_id are not specified it will create ones
    by default with the names

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

    Returns:
        FeatureGroup: FG prepared with all the methods and definitions properly defined
    """

    logger.setLevel(logging.WARNING)
    if verbose:
        logger.setLevel(logging.INFO)

    if isinstance(dataframe_or_path, DataFrame):
        data = dataframe_or_path
    elif isinstance(dataframe_or_path, str):
        pandas_read_csv_kwargs.pop("filepath_or_buffer", None)
        data = read_csv(filepath_or_buffer=dataframe_or_path, **pandas_read_csv_kwargs)
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
        sagemaker_session = _get_session_from_role(role=role, region=region)
    else:
        exc = Exception("Argument Session or role and region must be specified.")
        logger.exception(exc)
        raise exc

    feature_group = FeatureGroup(
        name=feature_group_name, sagemaker_session=sagemaker_session
    )

    feature_group.load_feature_definitions(data_frame=data)

    return feature_group
