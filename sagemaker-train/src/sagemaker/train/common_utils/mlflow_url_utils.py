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
"""Shared MLflow presigned URL utilities."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_presigned_mlflow_experiment_url(
    mlflow_resource_arn: str,
    mlflow_experiment_name: Optional[str] = None,
) -> Optional[str]:
    """Generate a presigned MLflow URL, optionally deep-linked to an experiment.

    Args:
        mlflow_resource_arn: MLflow tracking server or app ARN.
        mlflow_experiment_name: Optional experiment name for deep-linking.

    Returns:
        Presigned URL with experiment fragment, or base URL, or None on failure.
    """
    try:
        from sagemaker.core.utils.utils import SageMakerClient

        sm_client = SageMakerClient().sagemaker_client
        response = sm_client.create_presigned_mlflow_app_url(Arn=mlflow_resource_arn)
        base_url = response.get("AuthorizedUrl")
        if not base_url:
            return None

        if mlflow_experiment_name:
            try:
                import mlflow
                from mlflow.tracking import MlflowClient

                mlflow.set_tracking_uri(mlflow_resource_arn)
                experiment = MlflowClient(
                    tracking_uri=mlflow_resource_arn
                ).get_experiment_by_name(mlflow_experiment_name)
                if experiment:
                    return f"{base_url}#/experiments/{experiment.experiment_id}"
            except Exception as e:
                logger.debug(f"Failed to resolve MLflow experiment '{mlflow_experiment_name}': {e}")

        return base_url
    except Exception as e:
        logger.debug(f"Failed to generate MLflow experiment URL: {e}")
        return None
