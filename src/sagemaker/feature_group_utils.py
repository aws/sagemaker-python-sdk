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
"""

Utilities for working with FeatureGroups and FeatureStores.

"""
import re
import logging
import json
from typing import Union
from pathlib import Path

import pandas
from pandas import DataFrame, Series, read_csv

from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.utils import get_session_from_role

logger = logging.getLogger(__name__)


def get_feature_group_as_dataframe(feature_group_name: str, athena_bucket: str,
                                   query: str = str('SELECT * FROM "sagemaker_featurestore"."#{table}" WHERE '
                                                    + 'is_deleted=False'),
                                   role: str = None, region: str = None, session=None,
                                   event_time_feature_name: str = None, latest_ingestion: bool = True,
                                   logger_level: int = logging.INFO,
                                   **pandas_read_csv_kwargs) -> DataFrame:
    """
    Description:
        Method to run an athena query over a Feature Group in a Feature Store to retrieve its data.
        It needs the sagemaker.Session linked to a role or the role and region used to work Feature Stores.
        Returns a dataframe with the data.

    Args:
        region (str): region of the target feature store
        feature_group_name (str): feature store name
        query (str): query to run. By default, it will take the latest ingest with data that wasn't deleted.
                    If latest_ingestion is False it will take all the data in the feature group that wasn't
                    deleted. It needs to use the keyword "#{table}" to refer to the table. e.g.:
                        'SELECT * FROM "sagemaker_featurestore"."#{table}"'
        athena_bucket (str): S3 bucket for running the query
        role (str): role of the account used to extract data from feature store
        session (str): session of SageMaker used to work with the feature store
        event_time_feature_name (str): eventTimeId feature. Mandatory only if the latest ingestion is True
        latest_ingestion (bool): if True it will get the data only from the latest ingestion. If False it
                                 will take whatever is specified in the query, or if not specify it, it will
                                 get all the data that wasn't deleted.
        logger_level (int): logger level used by lib logging.

    Returns:
        dataset (pandas.DataFrame): dataset with the data retrieved from feature group
    """
    logger.setLevel(logger_level)

    if latest_ingestion:
        if event_time_feature_name is not None:
            query += str(f'AND {event_time_feature_name}=(SELECT MAX({event_time_feature_name}) FROM '
                         + f'"sagemaker_featurestore"."{feature_group_name}")')
    query += ';'

    if session is not None:
        sagemaker_session = session
    elif role is not None and region is not None:
        sagemaker_session = get_session_from_role(role=role, region=region)
    else:
        exc = Exception('Argument Session or role and region must be specified.')
        logger.exception(exc)
        raise exc

    logger.info(f'Feature Group used: {feature_group_name}\n')

    fg = FeatureGroup(name=feature_group_name,
                      sagemaker_session=sagemaker_session)

    sample_query = fg.athena_query()
    query_string = re.sub(r'#\{(table)\}', sample_query.table_name, query)

    logger.info(f"Running query:\n\t{sample_query} \n\n\t-> Save on bucket {athena_bucket}\n")

    sample_query.run(query_string=query_string,
                     output_location=athena_bucket)

    sample_query.wait()

    # run Athena query. The output is loaded to a Pandas dataframe.
    dataset = sample_query.as_dataframe(**pandas_read_csv_kwargs)

    logger.info(f'Data shape retrieve from {feature_group_name}: {dataset.shape}')

    return dataset


def _format_column_names(data: pandas.DataFrame) -> pandas.DataFrame:
    """
    Module to format correctly the name of the columns of a DataFrame to later generate the features names
    of a Feature Group

    Args:
        data (pandas.DataFrame): dataframe used

    Returns:
        pandas.DataFrame
    """
    data.rename(columns=lambda x: x.replace(' ', '_').replace('.', '').lower()[:62], inplace=True)
    return data


def _cast_object_to_string(data_frame: pandas.DataFrame) -> pandas.DataFrame:
    """
    Method to convert 'object' and 'O' column dtypes of a pandas.DataFrame to a valid string type recognized
    by Feature Groups.

    Args:
        data_frame: dataframe used

    Returns:
        pandas.DataFrame
    """
    for label in data_frame.select_dtypes(['object', 'O']).columns.tolist():
        data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame


def get_fg_schema(dataframe_or_path: Union[str, Path, pandas.DataFrame],
                  role: str, region: str,
                  mode: str = 'display', record_id: str = '@index',
                  event_id: str = 'data_as_of_date',
                  saving_file_path: str = '', verbose: bool = False,
                  **pandas_read_csv_kwargs) -> None:
    """
    Method to generate the schema of a Feature Group from a pandas.DataFrame. It has two modes (`mode`):
        - display: the schema is printed on the display
        - make_file: it generates a file with the schema inside. Recommended if it has a lot of features. Then
                     argument `saving_file_path` must be specified.

    Args:
        dataframe_or_path (str, Path, pandas.DataFrame) : pandas.DataFrame or path to the data
        mode (str)               : it changes how the output is displayed or stored, as explained before. By default,
                                    mode='display', and the other mode is `make_file`.
        verbose (bool)           : True for displaying messages, False for silent method.
        record_id (str, '@index'): (Optional) Feature identifier of the rows. If specified each value of that feature
                                    has to be unique. If not specified or record_id='@index', then it will create
                                    a new feature from the index of the pandas.DataFrame.
        event_id (str)           : (Optional) Feature with the time of the creation of data rows. If not specified it
                                    will create one with the current time called `data_as_of_date`
        role (str)               : role used to get the session
        region (str)             : region used to get the session
        saving_file_path (str)   : required if mode='make_file', file path to save the output.

    Returns:
        Save text into a file or displays the feature group schema by teh screen
    """
    MODE = ['display', 'make_file']

    logger.setLevel(logging.WARNING)
    if verbose:
        logger.setLevel(logging.INFO)

    mode = mode.lower()
    if mode not in MODE:
        exc = Exception(f'Invalid value {mode} for parameter mode.\nMode must be in {MODE}')
        logger.exception(exc)
        raise exc

    from sagemaker.feature_store.feature_group import FeatureGroup

    if isinstance(dataframe_or_path, DataFrame):
        data = dataframe_or_path
    elif isinstance(dataframe_or_path, str):
        pandas_read_csv_kwargs.pop('filepath_or_buffer', None)
        data = read_csv(filepath_or_buffer=dataframe_or_path, **pandas_read_csv_kwargs)
    else:
        exc = Exception(str(f'Invalid type {type(dataframe_or_path)} for argument dataframe_or_path.' +
                            f'\nParameter must be of type pandas.DataFrame or string'))
        logger.exception(exc)
        raise exc

    # Formatting cols
    data = _format_column_names(data=data)
    data = _cast_object_to_string(data_frame=data)

    if record_id == '@index':
        record_id = 'index'
        data[record_id] = data.index

    lg_uniq = len(data[record_id].unique())
    lg_id = len(data[record_id])

    if lg_id != lg_uniq:
        exc = Exception(str(f'Record identifier {record_id} have {abs(lg_id - lg_uniq)} duplicated rows.' +
                            f'\nRecord identifier must be unique in each row.'))
        logger.exception(exc)
        raise exc

    session = get_session_from_role(role=role, region=region)
    feature_group = FeatureGroup(
        name='temporalFG', sagemaker_session=session
    )

    if event_id not in data.columns:
        import time
        current_time_sec = int(round(time.time()))

        data[event_id] = Series([current_time_sec] * lg_id, dtype="float64")

    definitions = feature_group.load_feature_definitions(data_frame=data)

    def_list = []
    for ele in definitions:
        def_list.append({'FeatureName': ele.feature_name, 'FeatureType': ele.feature_type.name})

    if mode == MODE[0]:  # display
        logger.info('[')
        for ele in def_list:
            _to_print = json.dumps(ele)
            if ele != def_list[-1]:
                _to_print += ','

            logger.info(f'{_to_print}')
        logger.info(']')
    elif mode == MODE[1]:  # make_file
        if saving_file_path:
            logger.info(f'Saving schema to {saving_file_path}')
            with open(saving_file_path, 'w') as f:
                f.write(json.dumps(def_list))
                f.close()
            logger.info('Finished!.')
        else:
            exc = Exception(str(f'Parameter saving_file_path mandatory if mode {MODE[1]} is specified.'))
            logger.exception(exc)
            raise exc
