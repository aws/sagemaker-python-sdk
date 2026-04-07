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
"""Holds the BedrockModelBuilder class."""
from __future__ import absolute_import

import json
import time
import logging
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import TrainingJob, ModelPackage

from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

logger = logging.getLogger(__name__)


def _is_nova_model(container) -> bool:
    """Determine whether a model package container represents a Nova model.

    Checks both recipe_name and hub_content_name for the "nova" substring.

    Args:
        container: A container from ModelPackage.inference_specification.containers.

    Returns:
        True if the container represents a Nova model, False otherwise.
    """
    base_model = getattr(container, "base_model", None)
    if not base_model:
        return False

    recipe_name = getattr(base_model, "recipe_name", None) or ""
    hub_content_name = getattr(base_model, "hub_content_name", None) or ""

    return "nova" in recipe_name.lower() or "nova" in hub_content_name.lower()


class BedrockModelBuilder:
    """Builder class for deploying models to Amazon Bedrock.

    This class provides functionality to deploy SageMaker models to Bedrock
    using either model import jobs or custom model creation, depending on
    the model type (Nova models vs. other models).

    Args:
        model: The model to deploy. Can be a ModelTrainer, TrainingJob, or ModelPackage instance.
    """

    def __init__(self, model: Optional[Union[ModelTrainer, TrainingJob, ModelPackage]]):
        """Initialize BedrockModelBuilder with a model instance.

        Args:
            model: The model to deploy. Can be a ModelTrainer, TrainingJob,
                or ModelPackage instance.
        """
        self.model = model
        self._bedrock_client = None
        self._sagemaker_client = None
        self.boto_session = Session().boto_session
        self.model_package = self._fetch_model_package() if model else None
        self.s3_model_artifacts = self._get_s3_artifacts() if model else None

    def _get_bedrock_client(self):
        """Get or create Bedrock client singleton.

        Returns:
            boto3.client: Bedrock client instance.
        """
        if self._bedrock_client is None:
            self._bedrock_client = self.boto_session.client("bedrock")
        return self._bedrock_client

    def _get_sagemaker_client(self):
        """Get or create SageMaker client singleton.

        Returns:
            boto3.client: SageMaker client instance.
        """
        if self._sagemaker_client is None:
            self._sagemaker_client = self.boto_session.client("sagemaker")
        return self._sagemaker_client

    def _is_nova_model_for_telemetry(self) -> bool:
        """Check if the model is a Nova model for telemetry tracking."""
        try:
            if not self.model_package:
                return False
            container = self.model_package.inference_specification.containers[0]
            return _is_nova_model(container)
        except Exception:
            return False

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="BedrockModelBuilder.deploy")
    def deploy(
        self,
        job_name: Optional[str] = None,
        imported_model_name: Optional[str] = None,
        custom_model_name: Optional[str] = None,
        role_arn: Optional[str] = None,
        job_tags: Optional[list] = None,
        imported_model_tags: Optional[list] = None,
        model_tags: Optional[list] = None,
        client_request_token: Optional[str] = None,
        imported_model_kms_key_id: Optional[str] = None,
        deployment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deploy the model to Bedrock.

        Automatically detects if the model is a Nova model and uses the appropriate
        Bedrock API (create_custom_model for Nova, create_model_import_job for others).
        For Nova models, also creates a custom model deployment for inference.

        Args:
            job_name: Name for the model import job (non-Nova models only).
            imported_model_name: Name for the imported model (non-Nova models only).
            custom_model_name: Name for the custom model (Nova models only).
            role_arn: IAM role ARN with permissions for Bedrock operations.
            job_tags: Tags for the import job (non-Nova models only).
            imported_model_tags: Tags for the imported model (non-Nova models only).
            model_tags: Tags for the custom model (Nova models only).
            client_request_token: Unique token for idempotency (non-Nova models only).
            imported_model_kms_key_id: KMS key ID for encryption (non-Nova models only).
            deployment_name: Name for the deployment (Nova models only). If not provided,
                defaults to custom_model_name suffixed with '-deployment'.

        Returns:
            Response from Bedrock API. For Nova models, returns the
            create_custom_model_deployment response. For others, returns
            the create_model_import_job response.

        Raises:
            ValueError: If model_package is not set or required parameters are missing.
        """
        if not self.model_package:
            raise ValueError(
                "model_package is not set. Provide a valid model during initialization."
            )

        container = self.model_package.inference_specification.containers[0]
        is_nova = _is_nova_model(container)

        if is_nova:
            if not custom_model_name:
                raise ValueError("custom_model_name is required for Nova model deployment.")
            if not role_arn:
                raise ValueError("role_arn is required for Nova model deployment.")

            params = {
                "modelName": custom_model_name,
                "modelSourceConfig": {"s3DataSource": {"s3Uri": self.s3_model_artifacts}},
                "roleArn": role_arn,
            }
            if model_tags:
                params["modelTags"] = model_tags
            params = {k: v for k, v in params.items() if v is not None}

            logger.info("Creating custom model %s for Nova deployment", custom_model_name)
            create_response = self._get_bedrock_client().create_custom_model(**params)

            model_arn = create_response.get("modelArn")
            deploy_name = deployment_name or f"{custom_model_name}-deployment"
            return self.create_deployment(model_arn=model_arn, deployment_name=deploy_name)
        else:
            model_data_source = {"s3DataSource": {"s3Uri": self.s3_model_artifacts}}
            params = {
                "jobName": job_name,
                "importedModelName": imported_model_name,
                "roleArn": role_arn,
                "modelDataSource": model_data_source,
                "jobTags": job_tags,
                "importedModelTags": imported_model_tags,
                "clientRequestToken": client_request_token,
                "importedModelKmsKeyId": imported_model_kms_key_id,
            }
            params = {k: v for k, v in params.items() if v is not None}

            logger.info("Creating model import job for non-Nova deployment")
            return self._get_bedrock_client().create_model_import_job(**params)

    def create_deployment(
        self,
        model_arn: str,
        deployment_name: Optional[str] = None,
        poll_interval: int = 60,
        max_wait: int = 3600,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a deployment for a Nova custom model.

        Polls the model status until it becomes Active before creating the deployment,
        then polls the deployment status until it becomes Active.

        Args:
            model_arn: ARN of the custom model to deploy.
            deployment_name: Name for the deployment.
            poll_interval: Seconds between status checks. Defaults to 60 for model,
                30 for deployment.
            max_wait: Maximum seconds to wait per polling phase. Defaults to 3600.
            **kwargs: Additional parameters for create_custom_model_deployment.

        Returns:
            Response from Bedrock create_custom_model_deployment API.

        Raises:
            RuntimeError: If the model or deployment fails or times out.
            ValueError: If model_arn is not provided.
        """
        if not model_arn:
            raise ValueError("model_arn is required for create_deployment.")

        self._wait_for_model_active(model_arn, poll_interval=poll_interval, max_wait=max_wait)

        params = {
            "modelDeploymentName": deployment_name,
            "modelArn": model_arn,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        params = {k: v for k, v in params.items() if v is not None}

        logger.info("Creating deployment %s for model %s", deployment_name, model_arn)
        response = self._get_bedrock_client().create_custom_model_deployment(**params)

        deployment_arn = response.get("customModelDeploymentArn")
        if deployment_arn:
            self._wait_for_deployment_active(
                deployment_arn, poll_interval=poll_interval, max_wait=max_wait
            )

        return response

    def _wait_for_model_active(
        self, model_arn: str, poll_interval: int = 60, max_wait: int = 3600
    ):
        """Poll Bedrock until the custom model reaches Active status.

        Args:
            model_arn: ARN of the custom model.
            poll_interval: Seconds between status checks.
            max_wait: Maximum seconds to wait.

        Raises:
            RuntimeError: If the model status is Failed or the wait times out.
        """
        elapsed = 0
        status = None
        while elapsed < max_wait:
            resp = self._get_bedrock_client().get_custom_model(modelIdentifier=model_arn)
            status = resp.get("modelStatus")
            logger.info("Custom model status: %s (elapsed %ds)", status, elapsed)
            if status == "Active":
                return
            if status == "Failed":
                raise RuntimeError(
                    f"Custom model {model_arn} failed. Cannot proceed with deployment."
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(
            f"Timed out after {max_wait}s waiting for custom model {model_arn} to become Active. "
            f"Last status: {status}"
        )

    def _wait_for_deployment_active(
        self, deployment_arn: str, poll_interval: int = 30, max_wait: int = 3600
    ):
        """Poll Bedrock until the custom model deployment reaches Active status.

        Args:
            deployment_arn: ARN of the custom model deployment.
            poll_interval: Seconds between status checks. Defaults to 30.
            max_wait: Maximum seconds to wait. Defaults to 3600.

        Raises:
            RuntimeError: If the deployment status is Failed or the wait times out.
        """
        elapsed = 0
        status = None
        while elapsed < max_wait:
            resp = self._get_bedrock_client().get_custom_model_deployment(
                customModelDeploymentIdentifier=deployment_arn
            )
            status = resp.get("status")
            logger.info("Deployment status: %s (elapsed %ds)", status, elapsed)
            if status == "Active":
                return
            if status == "Failed":
                raise RuntimeError(
                    f"Deployment {deployment_arn} failed."
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(
            f"Timed out after {max_wait}s waiting for deployment {deployment_arn} to become Active. "
            f"Last status: {status}"
        )

    def _fetch_model_package(self) -> Optional[ModelPackage]:
        """Fetch the ModelPackage from the provided model.

        Extracts ModelPackage from ModelTrainer, TrainingJob, or returns
        the ModelPackage directly if that's what was provided.

        Returns:
            ModelPackage instance or None if no model was provided.
        """
        if isinstance(self.model, ModelPackage):
            return self.model
        if isinstance(self.model, TrainingJob):
            return ModelPackage.get(self.model.output_model_package_arn)
        if isinstance(self.model, ModelTrainer):
            return ModelPackage.get(
                self.model._latest_training_job.output_model_package_arn
            )
        return None

    def _get_s3_artifacts(self) -> Optional[str]:
        """Extract S3 URI of model artifacts from the model package.

        For Nova models, fetches checkpoint URI from manifest.json in training job output.
        For other models, returns the model data source S3 URI.

        Returns:
            S3 URI string of the model artifacts, or None if not available.
        """
        if not self.model_package:
            return None

        container = self.model_package.inference_specification.containers[0]
        is_nova = _is_nova_model(container)

        if is_nova and isinstance(self.model, TrainingJob):
            return self._get_checkpoint_uri_from_manifest()

        if hasattr(container, "model_data_source") and container.model_data_source:
            data_source = container.model_data_source
            if hasattr(data_source, "s3_data_source") and data_source.s3_data_source:
                return data_source.s3_data_source.s3_uri
        return None

    def _get_checkpoint_uri_from_manifest(self) -> Optional[str]:
        """Get checkpoint URI from manifest.json for Nova models.

        Steps:
        1. Fetch S3 model artifacts from training job
        2. Construct path to manifest.json in the output directory
        3. Read and parse manifest.json
        4. Return checkpoint_s3_bucket value

        Returns:
            Checkpoint URI from manifest.json.

        Raises:
            ValueError: If manifest.json cannot be found or parsed, or if the
                model is not a TrainingJob instance.
        """
        if not isinstance(self.model, TrainingJob):
            raise ValueError("Model must be a TrainingJob instance for Nova models")

        s3_artifacts = self.model.model_artifacts.s3_model_artifacts
        if not s3_artifacts:
            raise ValueError("No S3 model artifacts found in training job")

        logger.info("S3 artifacts path: %s", s3_artifacts)

        # Construct manifest path
        # s3://bucket/path/output/model.tar.gz -> s3://bucket/path/output/output/manifest.json
        parts = s3_artifacts.rstrip("/").rsplit("/", 1)
        manifest_path = parts[0] + "/output/manifest.json"

        logger.info("Manifest path: %s", manifest_path)

        parsed = urlparse(manifest_path)
        bucket = parsed.netloc
        manifest_key = parsed.path.lstrip("/")

        logger.info("Looking for manifest at s3://%s/%s", bucket, manifest_key)

        s3_client = self.boto_session.client("s3")
        try:
            response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
            manifest = json.loads(response["Body"].read().decode("utf-8"))
            logger.info("Manifest content: %s", manifest)

            checkpoint_uri = manifest.get("checkpoint_s3_bucket")
            if not checkpoint_uri:
                raise ValueError(
                    "'checkpoint_s3_bucket' not found in manifest. "
                    "Available keys: %s" % list(manifest.keys())
                )

            logger.info("Checkpoint URI: %s", checkpoint_uri)
            return checkpoint_uri
        except s3_client.exceptions.NoSuchKey:
            raise ValueError(
                "manifest.json not found at s3://%s/%s" % (bucket, manifest_key)
            )
        except json.JSONDecodeError as e:
            raise ValueError("Failed to parse manifest.json: %s" % e)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError("Error reading manifest.json: %s" % e)
