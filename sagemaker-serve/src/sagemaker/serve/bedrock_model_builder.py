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

from typing import Optional, Dict, Any, Union

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import TrainingJob, ModelPackage

from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature


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
            model: The model to deploy. Can be a ModelTrainer, TrainingJob, or ModelPackage instance.
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
    ) -> Dict[str, Any]:
        """Deploy the model to Bedrock.
        
        Automatically detects if the model is a Nova model and uses the appropriate
        Bedrock API (create_custom_model for Nova, create_model_import_job for others).

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

        Returns:
            Response from Bedrock API containing job ARN or model ARN.
            
        Raises:
            ValueError: If required parameters are missing for the detected model type.
        """
        container = self.model_package.inference_specification.containers[0]
        is_nova = (hasattr(container, 'base_model') and container.base_model and
                   hasattr(container.base_model, 'recipe_name') and container.base_model.recipe_name and
                   "nova" in container.base_model.recipe_name.lower()) or \
                  (hasattr(container, 'base_model') and container.base_model and
                   hasattr(container.base_model, 'hub_content_name') and container.base_model.hub_content_name and
                   "nova" in container.base_model.hub_content_name.lower())

        if is_nova:
            params = {
                "modelName": custom_model_name,
                "modelSourceConfig": {"s3DataSource": {"s3Uri": self.s3_model_artifacts}},
                "roleArn": role_arn,
            }
            if model_tags:
                params["modelTags"] = model_tags
            params = {k: v for k, v in params.items() if v is not None}
            return self._get_bedrock_client().create_custom_model(**params)
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
            return self._get_bedrock_client().create_model_import_job(**params)

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
            return ModelPackage.get(self.model._latest_training_job.output_model_package_arn)
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
        is_nova = (hasattr(container, 'base_model') and container.base_model and 
                  hasattr(container.base_model, 'recipe_name') and container.base_model.recipe_name and
                  "nova" in container.base_model.recipe_name.lower()) or \
                  (hasattr(container, 'base_model') and container.base_model and
                   hasattr(container.base_model, 'hub_content_name') and container.base_model.hub_content_name and
                   "nova" in container.base_model.hub_content_name.lower())
        
        if is_nova and isinstance(self.model, TrainingJob):
            return self._get_checkpoint_uri_from_manifest()
        
        if hasattr(container, 'model_data_source') and container.model_data_source:
            if hasattr(container.model_data_source, 's3_data_source') and container.model_data_source.s3_data_source:
                return container.model_data_source.s3_data_source.s3_uri
        return None
    
    def _get_checkpoint_uri_from_manifest(self) -> Optional[str]:
        """Get checkpoint URI from manifest.json for Nova models.
        
        Steps:
        1. Fetch S3 model artifacts from training job
        2. Go one level up in directory
        3. Find manifest.json
        4. Fetch checkpoint_s3_bucket from manifest
        
        Returns:
            Checkpoint URI from manifest.json.
            
        Raises:
            ValueError: If manifest.json cannot be found or parsed.
        """
        import json
        from urllib.parse import urlparse
        import logging
        
        logger = logging.getLogger(__name__)
        
        if not isinstance(self.model, TrainingJob):
            raise ValueError("Model must be a TrainingJob instance for Nova models")
        
        # Step 1: Get S3 model artifacts from training job
        s3_artifacts = self.model.model_artifacts.s3_model_artifacts
        if not s3_artifacts:
            raise ValueError("No S3 model artifacts found in training job")
        
        logger.info(f"S3 artifacts path: {s3_artifacts}")
        
        # Step 2: Construct manifest path (same directory as model artifacts)
        # s3://bucket/path/output/model.tar.gz -> s3://bucket/path/output/output/manifest.json
        parts = s3_artifacts.rstrip('/').rsplit('/', 1)
        manifest_path = parts[0] + '/output/manifest.json'
        
        logger.info(f"Manifest path: {manifest_path}")
        
        # Step 3: Find and read manifest.json
        parsed = urlparse(manifest_path)
        bucket = parsed.netloc
        manifest_key = parsed.path.lstrip('/')
        
        logger.info(f"Looking for manifest at s3://{bucket}/{manifest_key}")
        
        s3_client = self.boto_session.client('s3')
        try:
            response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
            manifest = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Manifest content: {manifest}")
            
            # Step 4: Fetch checkpoint_s3_bucket from manifest
            checkpoint_uri = manifest.get('checkpoint_s3_bucket')
            if not checkpoint_uri:
                raise ValueError(f"'checkpoint_s3_bucket' not found in manifest. Available keys: {list(manifest.keys())}")
            
            logger.info(f"Checkpoint URI: {checkpoint_uri}")
            return checkpoint_uri
        except s3_client.exceptions.NoSuchKey:
            raise ValueError(f"manifest.json not found at s3://{bucket}/{manifest_key}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse manifest.json: {e}")
        except Exception as e:
            raise ValueError(f"Error reading manifest.json: {e}")