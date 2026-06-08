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
"""Integration tests for ModelBuilder Nova model customization deployment.

These tests are the Nova counterpart of test_model_customization_deployment.py.
They run against the dedicated Nova test account in us-east-1 (784379639078)
and are marked with ``us_east_1`` so the scheduled GPU integ workflow picks
them up in the us-east-1 job only.
"""
from __future__ import absolute_import

import boto3
import json
import logging
import os
import time
import pytest
import random
from sagemaker.serve import ModelBuilder
from sagemaker.core.resources import TrainingJob

logger = logging.getLogger(__name__)

from sagemaker.core.helper.session_helper import Session

# This test relies on resources in a specific region (Nova test account)
AWS_REGION = "us-east-1"
os.environ.setdefault("AWS_DEFAULT_REGION", AWS_REGION)

# Model package group shared with the Nova SFT/RLVR trainer integ tests.
# Training jobs in those tests register their output here.
MODEL_PACKAGE_GROUP = "sdk-test-finetuned-models"

# Nova base model id (matches the existing Nova trainer/evaluator integ tests).
NOVA_MODEL_ID = "nova-textgeneration-lite-v2"

# Nova deployment instance type (matches test_sft_trainer_nova_workflow setup).
NOVA_INSTANCE_TYPE = "ml.g6.48xlarge"


def _latest_model_package_arn(region=AWS_REGION):
    """Return the ARN of the most recently created Completed model package in
    the Nova model package group, or None if the group has no usable package.

    Mirrors the dynamic lookup used by test_benchmark_evaluation_nova_model so
    these tests stay decoupled from any specific model package version.
    """
    sm_client = boto3.client("sagemaker", region_name=region)
    packages = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )
    summaries = packages.get("ModelPackageSummaryList", [])
    if not summaries:
        # Fall back to any status if no Approved packages exist.
        packages = sm_client.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=10,
        )
        summaries = packages.get("ModelPackageSummaryList", [])

    for summary in summaries:
        if summary.get("ModelPackageStatus") == "Completed":
            return summary["ModelPackageArn"]
    return None


@pytest.fixture(scope="module")
def sagemaker_session():
    """Create a SageMaker session with explicit region."""
    boto_session = boto3.Session(region_name=AWS_REGION)
    return Session(boto_session=boto_session)


@pytest.fixture(scope="module")
def training_job_name():
    """Reusable Nova fine-tuned training job name for testing."""
    return "nova-textgeneration-lite-sft-integ-test-reusable-model-20260531"


@pytest.fixture(scope="module")
def model_package_arn():
    """Latest Completed Nova model package ARN from the shared group.

    Skips the dependent test if no usable model package exists yet (e.g. before
    any Nova SFT/RLVR training job has registered one).
    """
    arn = _latest_model_package_arn()
    if arn is None:
        pytest.skip(
            f"No Completed model package available in {MODEL_PACKAGE_GROUP}. "
            "Run a Nova SFT/RLVR training job first."
        )
    return arn


@pytest.fixture
def endpoint_name():
    """Generate unique endpoint name."""
    return f"e2e-nova-{int(time.time())}-{random.randint(100, 10000)}"


@pytest.fixture(scope="module")
def cleanup_endpoints():
    """Track endpoints to cleanup after tests."""
    endpoints_to_cleanup = []
    yield endpoints_to_cleanup

    for ep_name in endpoints_to_cleanup:
        try:
            from sagemaker.core.resources import Endpoint
            endpoint = Endpoint.get(endpoint_name=ep_name, region=AWS_REGION)
            endpoint.delete()
        except Exception:
            pass


@pytest.mark.us_east_1
class TestModelCustomizationFromTrainingJob:
    """Test Nova model customization deployment from TrainingJob."""

    def test_build_from_training_job(self, training_job_name, sagemaker_session):
        """Test building a Nova model from a training job."""
        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        model_builder = ModelBuilder(
            model=training_job,
            instance_type=NOVA_INSTANCE_TYPE,
            sagemaker_session=sagemaker_session,
        )
        model_builder.accept_eula = True
        model = model_builder.build(
            model_name=f"test-model-{int(time.time())}-{random.randint(100, 10000)}",
            region=AWS_REGION,
        )

        assert model is not None
        assert model.model_arn is not None
        assert model_builder.image_uri is not None
        assert model_builder.instance_type is not None

    def test_deploy_from_training_job(self, training_job_name, endpoint_name, cleanup_endpoints, sagemaker_session):
        """Test deploying a Nova model from a training job and invoking it."""
        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        model_builder = ModelBuilder(
            model=training_job,
            instance_type=NOVA_INSTANCE_TYPE,
            sagemaker_session=sagemaker_session,
        )
        model_builder.accept_eula = True
        model_builder.build(
            model_name=f"test-model-{int(time.time())}-{random.randint(100, 10000)}",
            region=AWS_REGION,
        )

        endpoint = model_builder.deploy(
            endpoint_name=endpoint_name,
        )

        cleanup_endpoints.append(endpoint_name)

        assert endpoint is not None
        assert endpoint.endpoint_arn is not None
        assert endpoint.endpoint_status == "InService"

        # Invoke verification
        time.sleep(10)  # brief buffer for IC readiness

        invoke_response = endpoint.invoke(
            body=json.dumps({
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "What is 7+7?"}]}
                ]
            }),
            content_type="application/json",
            accept="application/json",
        )

        response_body = json.loads(invoke_response.body.read())

        # Validate response structure
        assert response_body is not None, f"Empty response from invoke on {endpoint_name}"
        assert isinstance(response_body, dict)

    def test_fetch_endpoint_names_for_base_model(self, training_job_name, sagemaker_session):
        """Test fetching endpoint names for base model."""
        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        model_builder = ModelBuilder(model=training_job, sagemaker_session=sagemaker_session)
        endpoint_names = model_builder.fetch_endpoint_names_for_base_model()

        assert isinstance(endpoint_names, set)


@pytest.mark.us_east_1
class TestModelCustomizationFromModelPackage:
    """Test Nova model customization deployment from a registered ModelPackage."""

    def test_build_from_model_package(self, model_package_arn, sagemaker_session):
        """Test building a Nova model from a model package."""
        from sagemaker.core.resources import ModelPackage

        model_package = ModelPackage.get(model_package_name=model_package_arn, region=AWS_REGION)
        model_builder = ModelBuilder(
            model=model_package,
            instance_type=NOVA_INSTANCE_TYPE,
            sagemaker_session=sagemaker_session,
        )
        model_builder.accept_eula = True
        model = model_builder.build(region=AWS_REGION)

        assert model is not None
        assert model.model_arn is not None

    def test_deploy_from_model_package(self, model_package_arn, endpoint_name, cleanup_endpoints, sagemaker_session):
        """Test deploying a Nova model from a model package."""
        from sagemaker.core.resources import ModelPackage

        model_package = ModelPackage.get(model_package_name=model_package_arn, region=AWS_REGION)
        model_builder = ModelBuilder(
            model=model_package,
            instance_type=NOVA_INSTANCE_TYPE,
            sagemaker_session=sagemaker_session,
        )
        model_builder.accept_eula = True
        model_builder.build(region=AWS_REGION)
        endpoint = model_builder.deploy(endpoint_name=endpoint_name)

        cleanup_endpoints.append(endpoint_name)

        assert endpoint is not None
        assert endpoint.endpoint_arn is not None


@pytest.mark.us_east_1
class TestInstanceTypeAutoDetection:
    """Test automatic instance type detection for Nova models."""

    def test_instance_type_from_recipe(self, training_job_name, sagemaker_session):
        """Test instance type auto-detection from a Nova recipe."""
        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        model_builder = ModelBuilder(model=training_job, sagemaker_session=sagemaker_session)
        model_builder.accept_eula = True
        model_builder.build(region=AWS_REGION)

        assert model_builder.instance_type is not None
        assert "ml." in model_builder.instance_type


@pytest.mark.us_east_1
class TestModelCustomizationDetection:
    """Test model customization detection logic for Nova models."""

    def test_is_model_customization_training_job(self, training_job_name, sagemaker_session):
        """Test detection from a Nova training job."""
        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        model_builder = ModelBuilder(model=training_job, sagemaker_session=sagemaker_session)

        assert model_builder._is_model_customization() is True

    def test_is_model_customization_model_package(self, model_package_arn, sagemaker_session):
        """Test detection from a Nova model package."""
        from sagemaker.core.resources import ModelPackage

        model_package = ModelPackage.get(model_package_name=model_package_arn, region=AWS_REGION)
        model_builder = ModelBuilder(model=model_package, sagemaker_session=sagemaker_session)

        assert model_builder._is_model_customization() is True

    def test_fetch_model_package_arn(self, training_job_name, sagemaker_session):
        """Test fetching the model package ARN for a Nova training job."""
        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        model_builder = ModelBuilder(model=training_job, sagemaker_session=sagemaker_session)

        arn = model_builder._fetch_model_package_arn()

        assert arn is not None
        assert "model-package" in arn


@pytest.mark.us_east_1
class TestTrainerIntegration:
    """Test ModelBuilder integration with Nova SFTTrainer and RLVRTrainer.

    Nova does not have a DPO recipe in SageMakerPublicHub (only SFT/RLVR/CPT/MTRL),
    so the DPO build test from the open-weights suite is replaced with RLVR here.
    """

    def test_sft_trainer_build(self, training_job_name, sagemaker_session):
        """Test building a model from a Nova SFTTrainer object."""
        from sagemaker.train.sft_trainer import SFTTrainer

        training_job = TrainingJob.get(
            training_job_name=training_job_name, region=AWS_REGION
        )

        trainer = SFTTrainer(
            model=NOVA_MODEL_ID,
            training_dataset="s3://dummy/data.jsonl",
            accept_eula=True,
            model_package_group=MODEL_PACKAGE_GROUP,
            sagemaker_session=sagemaker_session,
        )
        trainer._latest_training_job = training_job

        model_builder = ModelBuilder(model=trainer, sagemaker_session=sagemaker_session)
        model = model_builder.build(region=AWS_REGION)

        assert model is not None
        assert model.model_arn is not None

    def test_rlvr_trainer_build(self, training_job_name, sagemaker_session):
        """Test building a model from a Nova RLVRTrainer object."""
        from sagemaker.train.rlvr_trainer import RLVRTrainer

        training_job = TrainingJob.get(
            training_job_name=training_job_name, region=AWS_REGION
        )

        trainer = RLVRTrainer(
            model=NOVA_MODEL_ID,
            training_dataset="s3://dummy/data.jsonl",
            accept_eula=True,
            model_package_group=MODEL_PACKAGE_GROUP,
            sagemaker_session=sagemaker_session,
        )
        trainer._latest_training_job = training_job

        model_builder = ModelBuilder(model=trainer, sagemaker_session=sagemaker_session)
        model = model_builder.build(region=AWS_REGION)

        assert model is not None
        assert model.model_arn is not None


# -----------------------------------------------------------------------------
# Bedrock deployment tests are intentionally left commented out for Nova.
#
# Bedrock Custom Model Import (CMI) only supports open-weight architectures
# (e.g. Llama, Mistral). Nova is an Amazon proprietary model and cannot be
# imported into Bedrock via CMI, so the Bedrock deployment suite below from the
# open-weights tests has no meaningful Nova equivalent. It is preserved here
# (commented) for parity/reference only.
# -----------------------------------------------------------------------------

# """Integration tests for model customization deployment to Bedrock.
#
# Updated for sagemaker-core integration:
# - Added ModelPackage import for new model handling
# - Enhanced error handling for sagemaker-core compatibility issues
# - Updated model artifacts access to handle both old and new patterns
# - Added fallback logic for different model artifact locations
# - Improved test assertions to work with new object structures
# """
#
# from sagemaker.core.resources import TrainingJob, ModelPackage
# from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder
#
#
# class TestModelCustomizationDeployment:
#     """Test suite for deploying fine-tuned models to Bedrock."""
#
#     @pytest.fixture(scope="class")
#     def setup_config(self, training_job_name):
#         """Setup test configuration."""
#         from sagemaker.core.helper.session_helper import get_execution_role
#         return {
#             "training_job_name": training_job_name,
#             "region": AWS_REGION,
#             "bucket": "models-sdk-testing-pdx",
#             "role_arn": get_execution_role()
#         }
#
#     @pytest.fixture(scope="class")
#     def training_job(self, setup_config):
#         """Get the training job."""
#         return TrainingJob.get(
#             training_job_name=setup_config["training_job_name"],
#             region=setup_config["region"],
#         )
#
#     @pytest.fixture(scope="class")
#     def s3_client(self, setup_config):
#         """Create S3 client."""
#         return boto3.client('s3', region_name=setup_config["region"])
#
#     @pytest.fixture(scope="class")
#     def bedrock_client(self, setup_config):
#         """Create Bedrock client."""
#         client = boto3.client('bedrock', region_name=setup_config["region"])
#         # Cleanup existing import jobs
#         try:
#             jobs = client.list_model_import_jobs()
#             for job in jobs.get('modelImportJobSummaries', []):
#                 if job['jobName'].startswith('test-bedrock-'):
#                     try:
#                         client.stop_model_import_job(jobIdentifier=job['jobArn'])
#                     except Exception:
#                         pass
#         except Exception:
#             pass
#         return client
#
#     @pytest.fixture(scope="class")
#     def bedrock_runtime(self, setup_config):
#         """Create Bedrock runtime client."""
#         return boto3.client('bedrock-runtime', region_name=setup_config["region"])
#
#     @pytest.fixture(scope="class")
#     def deployed_model_arn(self, training_job, bedrock_client, s3_client, setup_config):
#         """Deploy model and return ARN."""
#         self._setup_model_files(training_job, s3_client, setup_config)
#
#         job_name = f"test-bedrock-{random.randint(1000, 9999)}-{int(time.time())}"
#         bedrock_builder = BedrockModelBuilder(model=training_job)
#
#         try:
#             deployment_result = bedrock_builder.deploy(
#                 job_name=job_name,
#                 imported_model_name=job_name,
#                 role_arn=setup_config["role_arn"]
#             )
#
#             job_arn = deployment_result['jobArn']
#
#             # Wait for completion
#             while True:
#                 response = bedrock_client.get_model_import_job(jobIdentifier=job_arn)
#                 status = response['status']
#                 if status in ['Completed', 'Failed']:
#                     break
#                 time.sleep(30)
#
#             model_arn = response['importedModelArn']
#             return model_arn
#
#         except Exception as e:
#             # If there's an issue with the new sagemaker-core integration, provide helpful error info
#             pytest.fail(
#                 f"Deployment failed with error: {str(e)}.")
#
#     def _setup_model_files(self, training_job, s3_client, setup_config):
#         """Setup required model files for Bedrock deployment."""
#         # Get S3 model artifacts path from training job
#         try:
#             # Try to access model artifacts from training job
#             if hasattr(training_job, 'model_artifacts') and hasattr(training_job.model_artifacts, 's3_model_artifacts'):
#                 base_s3_path = training_job.model_artifacts.s3_model_artifacts
#             elif hasattr(training_job, 'output_model_package_arn'):
#                 # If training job has model package ARN, get artifacts from model package
#                 model_package = ModelPackage.get(training_job.output_model_package_arn, region=AWS_REGION)
#                 if hasattr(model_package,
#                            'inference_specification') and model_package.inference_specification.containers:
#                     container = model_package.inference_specification.containers[0]
#                     if hasattr(container, 'model_data_source') and container.model_data_source:
#                         # Access s3_uri from the s3_data_source attribute
#                         if hasattr(container.model_data_source,
#                                    's3_data_source') and container.model_data_source.s3_data_source:
#                             base_s3_path = container.model_data_source.s3_data_source.s3_uri
#                         else:
#                             # Fallback to model_data_url if available
#                             base_s3_path = getattr(container, 'model_data_url', None)
#                     else:
#                         # Fallback to model_data_url if available
#                         base_s3_path = getattr(container, 'model_data_url', None)
#                 else:
#                     raise AttributeError("Cannot find model artifacts in model package")
#             else:
#                 raise AttributeError("Cannot find model artifacts in training job")
#
#             if not base_s3_path:
#                 raise ValueError("Model artifacts S3 path is empty")
#
#         except Exception as e:
#             pytest.fail(
#                 f"Failed to get model artifacts path: {str(e)}. This might be due to sagemaker-core integration changes.")
#
#         bucket = setup_config["bucket"]
#
#         # Create bucket if it doesn't exist
#         try:
#             s3_client.head_bucket(Bucket=bucket)
#         except Exception:
#             try:
#                 s3_client.create_bucket(
#                     Bucket=bucket,
#                     CreateBucketConfiguration={'LocationConstraint': setup_config["region"]}
#                 )
#             except Exception:
#                 pass
#
#         # Copy files from hf_merged to root
#         hf_merged_prefix = base_s3_path.replace(f's3://{bucket}/', '') + 'checkpoints/hf_merged/'
#         root_prefix = base_s3_path.replace(f's3://{bucket}/', '') + '/'
#
#         files_to_copy = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']
#
#         for file in files_to_copy:
#             try:
#                 s3_client.head_object(Bucket=bucket, Key=root_prefix + file)
#             except Exception:
#                 try:
#                     s3_client.copy_object(
#                         Bucket=bucket,
#                         CopySource={'Bucket': bucket, 'Key': hf_merged_prefix + file},
#                         Key=root_prefix + file
#                     )
#                 except Exception as e:
#                     print(f"Warning: Could not copy {file}: {str(e)}")
#
#         # Create added_tokens.json if missing
#         try:
#             s3_client.head_object(Bucket=bucket, Key=root_prefix + 'added_tokens.json')
#         except Exception:
#             try:
#                 s3_client.put_object(
#                     Bucket=bucket,
#                     Key=root_prefix + 'added_tokens.json',
#                     Body=json.dumps({}),
#                     ContentType='application/json'
#                 )
#             except Exception as e:
#                 print(f"Warning: Could not create added_tokens.json: {str(e)}")
#
#     def test_training_job_exists(self, training_job):
#         """Test that the training job exists and is completed."""
#         assert training_job is not None
#         assert training_job.training_job_status == "Completed"
#         # Check for model artifacts in different possible locations due to sagemaker-core changes
#         has_artifacts = (
#                 hasattr(training_job, 'model_artifacts') or
#                 hasattr(training_job, 'output_model_package_arn')
#         )
#         assert has_artifacts, "Training job should have model artifacts or model package ARN"
#
#     def test_bedrock_model_builder_creation(self, training_job):
#         """Test BedrockModelBuilder creation."""
#         try:
#             bedrock_builder = BedrockModelBuilder(model=training_job)
#             assert bedrock_builder is not None
#             assert bedrock_builder.model == training_job
#
#             # Test that the builder can fetch model package if needed
#             if hasattr(bedrock_builder, 'model_package'):
#                 # This tests the new sagemaker-core integration
#                 assert bedrock_builder.model_package is not None or bedrock_builder.model_package is None
#
#         except Exception as e:
#             pytest.fail(
#                 f"BedrockModelBuilder creation failed: {str(e)}. This might be due to sagemaker-core integration issues.")
#
#     @pytest.mark.slow
#     def test_bedrock_job_created(self, deployed_model_arn):
#         """Test that Bedrock import job was created successfully."""
#         assert deployed_model_arn is not None
#
#     @pytest.mark.slow
#     def test_bedrock_model_invoke(self, deployed_model_arn, bedrock_runtime):
#         """Test invoking the imported Bedrock model to ensure it works end-to-end.
#
#         Retries on failure since models can take several minutes
#         to become ready after import.
#         """
#         max_retries = 5
#         base_delay = 10
#
#         for attempt in range(max_retries):
#             try:
#                 response = bedrock_runtime.invoke_model(
#                     modelId=deployed_model_arn,
#                     body=json.dumps({
#                         "prompt": "What is the capital of France?",
#                         "max_gen_len": 100,
#                         "temperature": 0.7,
#                         "top_p": 0.9
#                     })
#                 )
#
#                 result = json.loads(response['body'].read().decode())
#
#                 # Validate response structure
#                 assert "generation" in result, "Response missing 'generation' field"
#                 assert isinstance(result["generation"], str), "'generation' should be a string"
#                 assert len(result["generation"]) > 0, "'generation' should not be empty"
#                 return  # Success
#
#             except Exception as e:
#                 if attempt < max_retries - 1:
#                     logger.info(
#                         f"Invoke failed (attempt {attempt + 1}/{max_retries}): {e}. "
#                         f"Retrying in {base_delay}s..."
#                     )
#                     time.sleep(base_delay)
#                 else:
#                     pytest.fail(
#                         f"Invoke failed after {max_retries} attempts. "
#                         f"Last error: {e}"
#                     )
#
#     def test_zzz_cleanup_deployed_model(self, bedrock_client):
#         """Cleanup deployed model and import jobs (runs last due to zzz prefix)."""
#         if hasattr(self, 'model_arn_for_cleanup'):
#             try:
#                 bedrock_client.delete_imported_model(modelIdentifier=self.model_arn_for_cleanup)
#             except Exception:
#                 pass
#         # Cleanup all test import jobs
#         try:
#             jobs = bedrock_client.list_model_import_jobs()
#             for job in jobs.get('modelImportJobSummaries', []):
#                 if job['jobName'].startswith('test-bedrock-'):
#                     try:
#                         bedrock_client.stop_model_import_job(jobIdentifier=job['jobArn'])
#                     except Exception:
#                         pass
#         except Exception:
#             pass
#
#
# def test_model_customization_workflow(training_job_name):
#     """Standalone test function for pytest discovery.
#
#     Uses explicit region parameter for all SDK calls.
#     """
#     config = {
#         "training_job_name": training_job_name,
#         "region": AWS_REGION,
#         "bucket": "open-models-testing-pdx"
#     }
#
#     try:
#         s3_client = boto3.client('s3', region_name=config["region"])
#         training_job = TrainingJob.get(training_job_name=config["training_job_name"], region=config["region"])
#
#         test_class = TestModelCustomizationDeployment()
#         test_class.test_training_job_exists(training_job)
#         test_class.test_bedrock_model_builder_creation(training_job)
#
#     except Exception as e:
#         print(f"Standalone test failed: {str(e)}")
#         print("This might be due to sagemaker-core integration issues. Please check:")
#         print("1. TrainingJob.get() method compatibility")
#         print("2. Model artifacts access patterns")
#         print("3. BedrockModelBuilder initialization with new sagemaker-core objects")
#         raise
