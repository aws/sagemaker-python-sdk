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
import os
import time
import logging
from datetime import datetime, timezone

from sagemaker.serve.utils.model_package_utils import is_restricted_model_package
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import TrainingJob, ModelPackage

from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
from sagemaker.train.agent_rft_job import AgentRFTJob
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


_BEDROCK_API_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "bedrock_api_logs")


def _log_bedrock_api_call(api_name: str, params: Dict[str, Any], response: Dict[str, Any]):
    """Log a Bedrock API call to a JSON file in bedrock_api_logs/."""
    log_dir = os.path.normpath(_BEDROCK_API_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{api_name}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    log_entry = {
        "timestamp": timestamp,
        "api": api_name,
        "request": params,
        "response": {k: v for k, v in response.items() if k != "ResponseMetadata"},
    }
    with open(filepath, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)
    logger.info("Bedrock API call logged to %s", filepath)
    print(f"[BedrockModelBuilder] API call logged to: {filepath}")


class BedrockModelBuilder:
    """Builder class for deploying models to Amazon Bedrock.

    This class provides functionality to deploy SageMaker models to Bedrock
    using either model import jobs or custom model creation, depending on
    the model type (Nova models vs. other models).

    Args:
        model: The model to deploy. Can be a ModelTrainer, MultiTurnRLTrainer,
            TrainingJob, or ModelPackage instance.
    """

    def __init__(
        self, model: Optional[Union[ModelTrainer, MultiTurnRLTrainer, AgentRFTJob, TrainingJob, ModelPackage]]
    ):
        """Initialize BedrockModelBuilder with a model instance.

        Args:
            model: The model to deploy. Can be a ModelTrainer, MultiTurnRLTrainer,
                AgentRFTJob, TrainingJob, or ModelPackage instance.
        """
        self.model = model
        self._bedrock_client = None
        self._sagemaker_client = None
        self._imported_model_id = None
        self.boto_session = Session().boto_session
        self.model_package = self._fetch_model_package() if model else None
        self._is_rmp = is_restricted_model_package(self.model_package)
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
        Bedrock API (create_custom_model for Nova, create_model_import_job for OSS).
        For Nova models, creates a custom model deployment and polls until active.
        For OSS models, creates a model import job and polls until complete. Once
        deploy() returns, the model is ready for on-demand inference. For provisioned
        throughput, use the separate create_provisioned_throughput() method.

        Args:
            job_name: Name for the model import job (OSS models only).
            imported_model_name: Name for the imported model (OSS models only).
            custom_model_name: Name for the custom model (Nova models only).
            role_arn: IAM role ARN with permissions for Bedrock operations.
            job_tags: Tags for the import job (OSS models only).
            imported_model_tags: Tags for the imported model (OSS models only).
            model_tags: Tags for the custom model (Nova models only).
            client_request_token: Unique token for idempotency (OSS models only).
            imported_model_kms_key_id: KMS key ID for encryption (OSS models only).
            deployment_name: Name for the deployment (Nova models only). If not provided,
                defaults to custom_model_name suffixed with '-deployment'.

        Returns:
            For Nova models: the create_custom_model_deployment response.
            For OSS models: the completed get_model_import_job response.

        Raises:
            ValueError: If model_package is not set or required parameters are missing.
            RuntimeError: If the import job or deployment fails or times out.
        """
        if not self.model_package:
            raise ValueError(
                "model_package is not set. Provide a valid model during initialization."
            )

        spec = getattr(self.model_package, "inference_specification", None)
        containers = getattr(spec, "containers", None) if spec else None
        container = containers[0] if containers else None
        is_nova = _is_nova_model(container) if container else False

        if self._is_rmp or is_nova:
            if not custom_model_name:
                raise ValueError("custom_model_name is required for Nova model deployment.")
            if not role_arn:
                raise ValueError("role_arn is required for Nova model deployment.")

            if self._is_rmp:
                params = {
                    "modelName": custom_model_name,
                    "customModelDataSource": {
                        "modelPackageArnDataSource": {
                            "modelPackageArn": self.model_package.model_package_arn
                        }
                    },
                    "roleArn": role_arn,
                }
            else:
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
            _log_bedrock_api_call("create_custom_model", params, create_response)

            model_arn = create_response.get("modelArn")
            deploy_name = deployment_name or f"{custom_model_name}-deployment"
            return self.create_deployment(model_arn=model_arn, deployment_name=deploy_name)
        else:
            model_data_source = {"s3DataSource": {"s3Uri": self.s3_model_artifacts}}
            # Auto-generate job_name if not provided
            if not job_name:
                import time
                job_name = f"{imported_model_name or 'import'}-{int(time.time())}"
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

            logger.info("Creating model import job for OSS model deployment")
            print(f"[BedrockModelBuilder] Resolved S3 artifacts path: {self.s3_model_artifacts}")
            print(f"[BedrockModelBuilder] create_model_import_job params: {params}")
            import_response = self._get_bedrock_client().create_model_import_job(**params)
            logger.warning(
                "Bedrock create_model_import_job request: %s, response: %s", params, import_response
            )
            _log_bedrock_api_call("create_model_import_job", params, import_response)

            job_arn = import_response.get("jobArn")
            self._wait_for_import_job_complete(job_arn)

            # Return the completed job details and store imported model ID
            job_details = self._get_bedrock_client().get_model_import_job(
                jobIdentifier=job_arn
            )
            self._imported_model_id = job_details.get("importedModelName")
            return job_details

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
        logger.warning(
            "Bedrock create_custom_model_deployment request: %s, response: %s", params, response
        )
        _log_bedrock_api_call("create_custom_model_deployment", params, response)

        deployment_arn = response.get("customModelDeploymentArn")
        if deployment_arn:
            self._wait_for_deployment_active(
                deployment_arn, poll_interval=poll_interval, max_wait=max_wait
            )

        return response

    def create_provisioned_throughput(
        self,
        model_id: Optional[str] = None,
        provisioned_model_name: str = None,
        model_units: int = 1,
        commitment_duration: Optional[str] = None,
        tags: Optional[list] = None,
        poll_interval: int = 60,
        max_wait: int = 3600,
    ) -> Dict[str, Any]:
        """Create provisioned throughput for an imported model on Bedrock.

        Calls CreateProvisionedModelThroughput and polls until the provisioned
        throughput reaches InService status.

        Args:
            model_id: ARN or name of the model. If not provided, uses the model
                ID from the most recent deploy() call.
            provisioned_model_name: Name for the provisioned throughput resource.
            model_units: Number of model units to provision. Defaults to 1.
            commitment_duration: Commitment duration. Valid values: 'OneMonth',
                'SixMonths'. If not provided, no commitment is set (on-demand).
            tags: Tags for the provisioned throughput resource.
            poll_interval: Seconds between status checks. Defaults to 60.
            max_wait: Maximum seconds to wait. Defaults to 3600.

        Returns:
            Response from Bedrock create_provisioned_model_throughput API.

        Raises:
            RuntimeError: If the provisioned throughput fails or times out.
            ValueError: If model_id cannot be determined or provisioned_model_name
                is not provided.
        """
        resolved_model_id = model_id or self._imported_model_id
        if not resolved_model_id:
            raise ValueError(
                "model_id is required for create_provisioned_throughput. "
                "Either pass it explicitly or call deploy() first."
            )
        if not provisioned_model_name:
            raise ValueError(
                "provisioned_model_name is required for create_provisioned_throughput."
            )

        params = {
            "modelId": resolved_model_id,
            "provisionedModelName": provisioned_model_name,
            "modelUnits": model_units,
        }
        if commitment_duration:
            params["commitmentDuration"] = commitment_duration
        if tags:
            params["tags"] = tags

        logger.info(
            "Creating provisioned throughput '%s' for model %s with %d model units",
            provisioned_model_name,
            resolved_model_id,
            model_units,
        )
        response = self._get_bedrock_client().create_provisioned_model_throughput(**params)

        provisioned_model_arn = response.get("provisionedModelArn")
        if provisioned_model_arn:
            self._wait_for_provisioned_throughput_in_service(
                provisioned_model_arn, poll_interval=poll_interval, max_wait=max_wait
            )

        return response

    def _wait_for_import_job_complete(
        self, job_arn: str, poll_interval: int = 60, max_wait: int = 3600
    ):
        """Poll Bedrock until the model import job reaches Completed status.

        Args:
            job_arn: ARN of the model import job.
            poll_interval: Seconds between status checks. Defaults to 60.
            max_wait: Maximum seconds to wait. Defaults to 3600.

        Raises:
            RuntimeError: If the import job fails or times out.
        """
        elapsed = 0
        status = None
        while elapsed < max_wait:
            resp = self._get_bedrock_client().get_model_import_job(jobIdentifier=job_arn)
            status = resp.get("status")
            logger.info("Import job status: %s (elapsed %ds)", status, elapsed)
            if status == "Completed":
                return
            if status == "Failed":
                failure_reason = resp.get("failureMessage", "Unknown")
                raise RuntimeError(
                    f"Model import job {job_arn} failed. Reason: {failure_reason}"
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(
            f"Timed out after {max_wait}s waiting for import job {job_arn} to complete. "
            f"Last status: {status}"
        )

    def _wait_for_provisioned_throughput_in_service(
        self, provisioned_model_arn: str, poll_interval: int = 60, max_wait: int = 3600
    ):
        """Poll Bedrock until provisioned throughput reaches InService status.

        Args:
            provisioned_model_arn: ARN of the provisioned model throughput.
            poll_interval: Seconds between status checks. Defaults to 60.
            max_wait: Maximum seconds to wait. Defaults to 3600.

        Raises:
            RuntimeError: If the provisioned throughput fails or times out.
        """
        elapsed = 0
        status = None
        while elapsed < max_wait:
            resp = self._get_bedrock_client().get_provisioned_model_throughput(
                provisionedModelId=provisioned_model_arn
            )
            status = resp.get("status")
            logger.info("Provisioned throughput status: %s (elapsed %ds)", status, elapsed)
            if status == "InService":
                return
            if status == "Failed":
                failure_reason = resp.get("failureMessage", "Unknown")
                raise RuntimeError(
                    f"Provisioned throughput {provisioned_model_arn} failed. "
                    f"Reason: {failure_reason}"
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(
            f"Timed out after {max_wait}s waiting for provisioned throughput "
            f"{provisioned_model_arn} to become InService. Last status: {status}"
        )

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

        Extracts ModelPackage from ModelTrainer, MultiTurnRLTrainer, TrainingJob,
        or returns the ModelPackage directly if that's what was provided.

        Returns:
            ModelPackage instance or None if no model was provided.
        """
        if isinstance(self.model, ModelPackage):
            return self.model
        if isinstance(self.model, TrainingJob):
            return ModelPackage.get(self.model.output_model_package_arn)
        if isinstance(self.model, (MultiTurnRLTrainer, AgentRFTJob)):
            arn = self.model.output_model_package_arn
            if not arn:
                job_name = None
                if isinstance(self.model, AgentRFTJob):
                    job_name = self.model.job_name
                elif hasattr(self.model, "_latest_job") and self.model._latest_job:
                    job_name = self.model._latest_job.job_name
                if job_name:
                    from sagemaker.core.resources import Job
                    job = Job.get(
                        job_name=job_name, job_category="AgentRFT"
                    )
                    config = json.loads(job.job_config_document) if job.job_config_document else {}
                    arn = config.get("ServiceOutput", {}).get("OutputModelPackageArn")
            if not arn:
                raise ValueError(
                    "Model has no output_model_package_arn. "
                    "Ensure training has completed successfully."
                )
            return ModelPackage.get(arn)
        if isinstance(self.model, ModelTrainer):
            return ModelPackage.get(
                self.model._latest_training_job.output_model_package_arn
            )
        return None

    def _get_s3_artifacts(self) -> Optional[str]:
        """Extract S3 URI of model artifacts from the model package.

        For Nova models, fetches checkpoint URI from manifest.json in training job output.
        For other models, returns the model data source S3 URI, resolving to the
        hf_merged checkpoint directory if it exists (required for Bedrock import).

        Returns:
            S3 URI string of the model artifacts, or None if not available.
        """
        if not self.model_package:
            return None

        if self._is_rmp:
            return None

        container = self.model_package.inference_specification.containers[0]
        is_nova = _is_nova_model(container)

        if is_nova and isinstance(self.model, TrainingJob):
            return self._get_checkpoint_uri_from_manifest()

        if hasattr(container, "model_data_source") and container.model_data_source:
            data_source = container.model_data_source
            if hasattr(data_source, "s3_data_source") and data_source.s3_data_source:
                s3_uri = data_source.s3_data_source.s3_uri
                if s3_uri:
                    return self._resolve_hf_model_path(s3_uri)
        return None

    def _resolve_hf_model_path(self, s3_uri: str) -> str:
        """Resolve the HuggingFace model directory within model artifacts.

        MTRL training jobs produce checkpoints under checkpoints/:
        - hf_merged/ contains full merged weights (config.json + model shards)
        - hf/ contains LoRA adapter only (adapter_config.json + adapter_model.safetensors)

        The s3_uri from the model package already includes the trailing model/ prefix,
        so this method appends checkpoints/hf_merged/ or checkpoints/hf/ directly.

        This method checks for hf_merged first (preferred for Bedrock import),
        then falls back to hf (LoRA adapter), then the original URI.

        Args:
            s3_uri: Base S3 URI from the model package container (typically ends with model/).

        Returns:
            S3 URI pointing to the resolved model directory.
        """
        s3_uri = s3_uri.rstrip("/") + "/"
        parsed_base = urlparse(s3_uri)
        bucket = parsed_base.netloc
        s3_client = self.boto_session.client("s3")

        print(f"[BedrockModelBuilder] Base s3_uri from model package: {s3_uri}")

        hf_merged_uri = s3_uri + "checkpoints/hf_merged/"
        merged_config_key = urlparse(hf_merged_uri).path.lstrip("/") + "config.json"
        print(f"[BedrockModelBuilder] Probing for hf_merged: s3://{bucket}/{merged_config_key}")
        try:
            s3_client.head_object(Bucket=bucket, Key=merged_config_key)
            logger.info("Found merged HF model at %s", hf_merged_uri)
            print(f"[BedrockModelBuilder] Found hf_merged checkpoint, using: {hf_merged_uri}")
            return hf_merged_uri
        except Exception as e:
            print(f"[BedrockModelBuilder] hf_merged not found: {e}")

        hf_lora_uri = s3_uri + "checkpoints/hf/"
        lora_config_key = urlparse(hf_lora_uri).path.lstrip("/") + "adapter_config.json"
        try:
            s3_client.head_object(Bucket=bucket, Key=lora_config_key)
            logger.info("Found LoRA adapter at %s", hf_lora_uri)
            return hf_lora_uri
        except Exception:
            pass

        logger.info("No hf_merged or hf checkpoint found, using base path: %s", s3_uri)
        return s3_uri.rstrip("/")

    def _get_checkpoint_uri_from_manifest(self) -> Optional[str]:
        """Get checkpoint URI from manifest.json for Nova models.

        Steps:
        1. Build the manifest.json path from the training job output_data_config
        2. Read and parse manifest.json
        3. Return checkpoint_s3_bucket value

        Returns:
            Checkpoint URI from manifest.json.

        Raises:
            ValueError: If manifest.json cannot be found or parsed, or if the
                model is not a TrainingJob instance.
        """
        if not isinstance(self.model, TrainingJob):
            raise ValueError("Model must be a TrainingJob instance for Nova models")

        # Nova serverless training jobs have no model_artifacts; the manifest
        # lives under the job's output_data_config path.
        output_data_config = getattr(self.model, "output_data_config", None)
        s3_output_path = getattr(output_data_config, "s3_output_path", None)
        if not s3_output_path:
            raise ValueError("No S3 output path found in training job output_data_config")

        output_path = s3_output_path.rstrip("/")
        manifest_path = f"{output_path}/{self.model.training_job_name}/output/output/manifest.json"

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
