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
"""Utilities for working with FeatureGroup and FeatureStores.


"""

import re
import logging

from pandas import DataFrame

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
                    deleted.
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