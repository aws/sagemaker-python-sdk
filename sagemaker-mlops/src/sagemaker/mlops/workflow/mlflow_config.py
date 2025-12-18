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
"""MLflow config for SageMaker pipeline."""
from __future__ import absolute_import

from typing import Dict, Any


class MlflowConfig:
    """MLflow configuration for SageMaker pipeline."""

    def __init__(
        self,
        mlflow_resource_arn: str,
        mlflow_experiment_name: str,
    ):
        """Create an MLflow configuration for SageMaker Pipeline.

        Examples:
        Basic MLflow configuration::

            mlflow_config = MlflowConfig(
                mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server",
                mlflow_experiment_name="my-experiment"
            )

            pipeline = Pipeline(
                name="MyPipeline",
                steps=[...],
                mlflow_config=mlflow_config
            )

        Runtime override of experiment name::

            # Override experiment name for a specific execution
            execution = pipeline.start(mlflow_experiment_name="custom-experiment")

        Args:
            mlflow_resource_arn (str): The ARN of the MLflow tracking server resource.
            mlflow_experiment_name (str): The name of the MLflow experiment to be used for tracking.
        """
        self.mlflow_resource_arn = mlflow_resource_arn
        self.mlflow_experiment_name = mlflow_experiment_name

    def to_request(self) -> Dict[str, Any]:
        """Returns: the request structure."""

        return {
            "MlflowResourceArn": self.mlflow_resource_arn,
            "MlflowExperimentName": self.mlflow_experiment_name,
        }
