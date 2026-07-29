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
"""Module for retrieving MLflow tracking server endpoints from SageMaker.

Example:
    .. code:: python

        from sagemaker.train.common_utils.get_mlflow_endpoint import (
            get_mlflow_tracking_server_endpoint
        )
        
        endpoint_url = get_mlflow_tracking_server_endpoint(
            tracking_server_name="my-mlflow-server",
            region="us-west-2"
        )
        print(f"MLflow endpoint: {endpoint_url}")
"""

from typing import Optional

import boto3
from botocore.exceptions import ClientError

from sagemaker.train.common_utils.constants import (
    _ErrorConstants,
    _TrainingJobConstants,
    _ValidationConstants,
)


class MLflowEndpointError(Exception):
    """Raised when unable to retrieve MLflow endpoint."""
    pass


def _get_mlflow_tracking_server_endpoint(
    tracking_server_name: str, 
    region: str = _TrainingJobConstants.DEFAULT_AWS_REGION
) -> str:
    """Get the HTTP endpoint URL for a SageMaker MLflow tracking server.
    
    Args:
        tracking_server_name (str): Name of the MLflow tracking server.
        region (str): AWS region. Defaults to 'us-west-2'.
        
    Returns:
        str: HTTP endpoint URL for the tracking server.
        
    Raises:
        MLflowEndpointError: If unable to retrieve the tracking server endpoint.
        ValueError: If tracking_server_name is empty or invalid.
    """
    if not tracking_server_name or not tracking_server_name.strip():
        raise ValueError(_ValidationConstants.EMPTY_TRACKING_SERVER_NAME_MSG)
    
    if not region or not region.strip():
        raise ValueError(_ValidationConstants.EMPTY_REGION_MSG)
    
    try:
        client = boto3.client('sagemaker', region_name=region.strip())
        
        response = client.describe_mlflow_tracking_server(
            TrackingServerName=tracking_server_name.strip()
        )
        
        tracking_server_url = response.get('TrackingServerUrl')
        if not tracking_server_url:
            raise MLflowEndpointError(
                _ErrorConstants.NO_TRACKING_URL.format(tracking_server_name)
            )
        
        return tracking_server_url
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        if error_code == 'ResourceNotFound':
            raise MLflowEndpointError(
                _ErrorConstants.RESOURCE_NOT_FOUND_ERROR.format(tracking_server_name, region)
            ) from e
        else:
            raise MLflowEndpointError(
                _ErrorConstants.ENDPOINT_RETRIEVAL_ERROR.format(error_message)
            ) from e
