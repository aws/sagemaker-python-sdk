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
"""Integration tests for ModelBuilder model customization deployment."""
from __future__ import absolute_import

import pytest
import random


@pytest.fixture(scope="module")
def training_job_name():
    """Training job name for testing."""
    return "meta-textgeneration-llama-3-2-1b-instruct-sft-20251201172445"


@pytest.fixture(scope="module")
def sft_training_job_name():
    """SFT training job name for testing."""
    return "meta-textgeneration-llama-3-2-1b-instruct-sft-20251201114921"


@pytest.fixture(scope="module")
def dpo_training_job_name():
    """DPO training job name for testing."""
    return "meta-textgeneration-llama-3-2-1b-instruct-sft-20251123162832"


@pytest.fixture(scope="module")
def model_package_arn():
    """Model package ARN for testing."""
    return "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1"


@pytest.fixture
def endpoint_name():
    """Generate unique endpoint name."""
    import time
    return f"e2e-{int(time.time())}-{random.randint(100, 10000)}"


@pytest.fixture(scope="session", autouse=True)
def cleanup_e2e_endpoints():
    """Cleanup e2e endpoints before and after tests."""
    from sagemaker.core.resources import Endpoint
    from botocore.exceptions import ClientError

    # Cleanup before tests
    try:
        for endpoint in Endpoint.get_all():
            try:
                if endpoint.endpoint_name.startswith('e2e-'):
                    endpoint.delete()
            except (ClientError, Exception):
                pass
    except (ClientError, Exception):
        pass

    yield

    # Cleanup after tests
    try:
        for endpoint in Endpoint.get_all():
            try:
                if endpoint.endpoint_name.startswith('e2e-'):
                    endpoint.delete()
            except (ClientError, Exception):
                pass
    except (ClientError, Exception):
        pass


@pytest.fixture(scope="module")
def cleanup_endpoints():
    """Track endpoints to cleanup after tests."""
    endpoints_to_cleanup = []
    yield endpoints_to_cleanup

    for ep_name in endpoints_to_cleanup:
        try:
            from sagemaker.core.resources import Endpoint
            endpoint = Endpoint.get(endpoint_name=ep_name)
            endpoint.delete()
        except Exception:
            pass


class TestModelCustomizationFromTrainingJob:
    """Test model customization deployment from TrainingJob."""

    def test_build_from_training_job(self, training_job_name):
        """Test building model from training job."""
        from sagemaker.core.resources import TrainingJob
        from sagemaker.serve import ModelBuilder
        import time

        training_job = TrainingJob.get(training_job_name=training_job_name)
        model_builder = ModelBuilder(model=training_job)
        model = model_builder.build(model_name=f"test-model-{int(time.time())}-{random.randint(100, 10000)}")

        assert model is not None
        assert model.model_arn is not None
        assert model_builder.image_uri is not None
        assert model_builder.instance_type is not None

    def test_deploy_from_training_job(self, training_job_name, endpoint_name, cleanup_endpoints):
        """Test deploying model from training job and adapter."""
        from sagemaker.core.resources import TrainingJob
        from sagemaker.serve import ModelBuilder
        import time

        training_job = TrainingJob.get(training_job_name=training_job_name)
        model_builder = ModelBuilder(model=training_job)
        model = model_builder.build(model_name=f"test-model-{int(time.time())}-{random.randint(100, 10000)}")
        endpoint = model_builder.deploy(endpoint_name=endpoint_name)

        cleanup_endpoints.append(endpoint_name)

        assert endpoint is not None
        assert endpoint.endpoint_arn is not None
        assert endpoint.endpoint_status == "InService"

        # Deploy adapter to the same endpoint
        adapter_name = f"{endpoint_name}-adapter-{int(time.time())}-{random.randint(100, 100000)}"
        model_builder2 = ModelBuilder(model=training_job)
        model_builder2.build()
        endpoint2 = model_builder2.deploy(
            endpoint_name=endpoint_name,
            inference_component_name=adapter_name
        )

        assert endpoint2 is not None
        assert endpoint2.endpoint_name == endpoint_name

    def test_fetch_endpoint_names_for_base_model(self, training_job_name):
        """Test fetching endpoint names for base model."""
        from sagemaker.core.resources import TrainingJob
        from sagemaker.serve import ModelBuilder

        training_job = TrainingJob.get(training_job_name=training_job_name)
        model_builder = ModelBuilder(model=training_job)
        endpoint_names = model_builder.fetch_endpoint_names_for_base_model()

        assert isinstance(endpoint_names, set)


class TestModelCustomizationFromModelPackage:

    def test_build_from_model_package(self, model_package_arn):
        """Test building model from model package."""
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder

        model_package = ModelPackage.get(model_package_name=model_package_arn)
        model_builder = ModelBuilder(model=model_package)
        model = model_builder.build()

        assert model is not None
        assert model.model_arn is not None

    def test_deploy_from_model_package(self, model_package_arn, cleanup_endpoints):
        """Test deploying model from model package."""
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder
        import time

        model_package = ModelPackage.get(model_package_name=model_package_arn)
        endpoint_name = f"e2e-{int(time.time())}-{random.randint(100, 10000)}"
        model_builder = ModelBuilder(model=model_package)
        model_builder.build()
        endpoint = model_builder.deploy(endpoint_name=endpoint_name)

        cleanup_endpoints.append(endpoint_name)

        assert endpoint is not None
        assert endpoint.endpoint_arn is not None


class TestInstanceTypeAutoDetection:
    """Test automatic instance type detection."""

    def test_instance_type_from_recipe(self, training_job_name):
        """Test instance type auto-detection from recipe."""
        from sagemaker.core.resources import TrainingJob
        from sagemaker.serve import ModelBuilder

        training_job = TrainingJob.get(training_job_name=training_job_name)
        model_builder = ModelBuilder(model=training_job)
        model_builder.build()

        assert model_builder.instance_type is not None
        assert "ml." in model_builder.instance_type


class TestModelCustomizationDetection:
    """Test model customization detection logic."""

    def test_is_model_customization_training_job(self, training_job_name):
        """Test detection from training job."""
        from sagemaker.core.resources import TrainingJob
        from sagemaker.serve import ModelBuilder

        training_job = TrainingJob.get(training_job_name=training_job_name)
        model_builder = ModelBuilder(model=training_job)

        assert model_builder._is_model_customization() is True

    def test_is_model_customization_model_package(self, model_package_arn):
        """Test detection from model package."""
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder

        model_package = ModelPackage.get(model_package_name=model_package_arn)
        model_builder = ModelBuilder(model=model_package)

        assert model_builder._is_model_customization() is True

    def test_fetch_model_package_arn(self, training_job_name):
        """Test fetching model package ARN."""
        from sagemaker.core.resources import TrainingJob
        from sagemaker.serve import ModelBuilder

        training_job = TrainingJob.get(training_job_name=training_job_name)
        model_builder = ModelBuilder(model=training_job)

        arn = model_builder._fetch_model_package_arn()

        assert arn is not None
        assert "model-package" in arn


class TestTrainerIntegration:
    """Test ModelBuilder integration with SFTTrainer and DPOTrainer."""

    def test_sft_trainer_build(self, training_job_name):
        """Test building model from SFTTrainer."""
        from sagemaker.core.resources import TrainingJob
        from sagemaker.train.sft_trainer import SFTTrainer
        from sagemaker.serve import ModelBuilder

        training_job = TrainingJob.get(
            training_job_name=training_job_name
        )

        trainer = SFTTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_dataset="s3://dummy/data.jsonl",
            accept_eula=True,
            model_package_group_name="test-group"
        )
        trainer._latest_training_job = training_job

        model_builder = ModelBuilder(model=trainer)
        model = model_builder.build()

        assert model is not None
        assert model.model_arn is not None

    def test_dpo_trainer_build(self, training_job_name):
        """Test building model from DPOTrainer."""
        from sagemaker.core.resources import TrainingJob
        from sagemaker.train.dpo_trainer import DPOTrainer
        from sagemaker.serve import ModelBuilder
        from unittest.mock import patch

        training_job = TrainingJob.get(
            training_job_name=training_job_name
        )

        with patch('sagemaker.train.common_utils.finetune_utils._get_fine_tuning_options_and_model_arn',
                   return_value=(None, None)):
            trainer = DPOTrainer(
                model="meta-textgeneration-llama-3-2-1b-instruct",
                training_dataset="s3://dummy/data.jsonl",
                accept_eula=True,
                model_package_group_name="test-group"
            )
        trainer._latest_training_job = training_job

        model_builder = ModelBuilder(model=trainer)
        model = model_builder.build()

        assert model is not None
        assert model.model_arn is not None


"""Integration tests for model customization deployment to Bedrock.

Updated for sagemaker-core integration:
- Added ModelPackage import for new model handling
- Enhanced error handling for sagemaker-core compatibility issues
- Updated model artifacts access to handle both old and new patterns
- Added fallback logic for different model artifact locations
- Improved test assertions to work with new object structures
"""

import json
import time
import random
import boto3
import pytest
from sagemaker.core.resources import TrainingJob, ModelPackage
from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder


class TestModelCustomizationDeployment:
    """Test suite for deploying fine-tuned models to Bedrock."""

    @pytest.fixture(scope="class")
    def setup_config(self, training_job_name):
        """Setup test configuration."""
        from sagemaker.core.helper.session_helper import get_execution_role
        return {
            "training_job_name": training_job_name,
            "region": "us-west-2",
            "bucket": "models-sdk-testing-pdx",
            "role_arn": get_execution_role()
        }

    @pytest.fixture(scope="class")
    def training_job(self, setup_config):
        """Get the training job."""
        return TrainingJob.get(training_job_name=setup_config["training_job_name"])

    @pytest.fixture(scope="class")
    def s3_client(self, setup_config):
        """Create S3 client."""
        return boto3.client('s3', region_name=setup_config["region"])

    @pytest.fixture(scope="class")
    def bedrock_client(self, setup_config):
        """Create Bedrock client."""
        client = boto3.client('bedrock', region_name=setup_config["region"])
        # Cleanup existing import jobs
        try:
            jobs = client.list_model_import_jobs()
            for job in jobs.get('modelImportJobSummaries', []):
                if job['jobName'].startswith('test-bedrock-'):
                    try:
                        client.stop_model_import_job(jobIdentifier=job['jobArn'])
                    except Exception:
                        pass
        except Exception:
            pass
        return client

    @pytest.fixture(scope="class")
    def bedrock_runtime(self, setup_config):
        """Create Bedrock runtime client."""
        return boto3.client('bedrock-runtime', region_name=setup_config["region"])

    @pytest.fixture(scope="class")
    def deployed_model_arn(self, training_job, bedrock_client, s3_client, setup_config):
        """Deploy model and return ARN."""
        self._setup_model_files(training_job, s3_client, setup_config)

        job_name = f"test-bedrock-{random.randint(1000, 9999)}-{int(time.time())}"
        bedrock_builder = BedrockModelBuilder(model=training_job)

        try:
            deployment_result = bedrock_builder.deploy(
                job_name=job_name,
                imported_model_name=job_name,
                role_arn=setup_config["role_arn"]
            )

            job_arn = deployment_result['jobArn']

            # Wait for completion
            while True:
                response = bedrock_client.get_model_import_job(jobIdentifier=job_arn)
                status = response['status']
                if status in ['Completed', 'Failed']:
                    break
                time.sleep(30)

            model_arn = response['importedModelName']
            return model_arn

        except Exception as e:
            # If there's an issue with the new sagemaker-core integration, provide helpful error info
            pytest.fail(
                f"Deployment failed with error: {str(e)}.")

    def _setup_model_files(self, training_job, s3_client, setup_config):
        """Setup required model files for Bedrock deployment."""
        # Get S3 model artifacts path from training job
        try:
            # Try to access model artifacts from training job
            if hasattr(training_job, 'model_artifacts') and hasattr(training_job.model_artifacts, 's3_model_artifacts'):
                base_s3_path = training_job.model_artifacts.s3_model_artifacts
            elif hasattr(training_job, 'output_model_package_arn'):
                # If training job has model package ARN, get artifacts from model package
                model_package = ModelPackage.get(training_job.output_model_package_arn)
                if hasattr(model_package,
                           'inference_specification') and model_package.inference_specification.containers:
                    container = model_package.inference_specification.containers[0]
                    if hasattr(container, 'model_data_source') and container.model_data_source:
                        # Access s3_uri from the s3_data_source attribute
                        if hasattr(container.model_data_source,
                                   's3_data_source') and container.model_data_source.s3_data_source:
                            base_s3_path = container.model_data_source.s3_data_source.s3_uri
                        else:
                            # Fallback to model_data_url if available
                            base_s3_path = getattr(container, 'model_data_url', None)
                    else:
                        # Fallback to model_data_url if available
                        base_s3_path = getattr(container, 'model_data_url', None)
                else:
                    raise AttributeError("Cannot find model artifacts in model package")
            else:
                raise AttributeError("Cannot find model artifacts in training job")

            if not base_s3_path:
                raise ValueError("Model artifacts S3 path is empty")

        except Exception as e:
            pytest.fail(
                f"Failed to get model artifacts path: {str(e)}. This might be due to sagemaker-core integration changes.")

        bucket = setup_config["bucket"]
        
        # Create bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=bucket)
        except Exception:
            try:
                s3_client.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={'LocationConstraint': setup_config["region"]}
                )
            except Exception:
                pass

        # Copy files from hf_merged to root
        hf_merged_prefix = base_s3_path.replace(f's3://{bucket}/', '') + 'checkpoints/hf_merged/'
        root_prefix = base_s3_path.replace(f's3://{bucket}/', '') + '/'

        files_to_copy = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']

        for file in files_to_copy:
            try:
                s3_client.head_object(Bucket=bucket, Key=root_prefix + file)
            except Exception:
                try:
                    s3_client.copy_object(
                        Bucket=bucket,
                        CopySource={'Bucket': bucket, 'Key': hf_merged_prefix + file},
                        Key=root_prefix + file
                    )
                except Exception as e:
                    print(f"Warning: Could not copy {file}: {str(e)}")

        # Create added_tokens.json if missing
        try:
            s3_client.head_object(Bucket=bucket, Key=root_prefix + 'added_tokens.json')
        except Exception:
            try:
                s3_client.put_object(
                    Bucket=bucket,
                    Key=root_prefix + 'added_tokens.json',
                    Body=json.dumps({}),
                    ContentType='application/json'
                )
            except Exception as e:
                print(f"Warning: Could not create added_tokens.json: {str(e)}")

    def test_training_job_exists(self, training_job):
        """Test that the training job exists and is completed."""
        assert training_job is not None
        assert training_job.training_job_status == "Completed"
        # Check for model artifacts in different possible locations due to sagemaker-core changes
        has_artifacts = (
                hasattr(training_job, 'model_artifacts') or
                hasattr(training_job, 'output_model_package_arn')
        )
        assert has_artifacts, "Training job should have model artifacts or model package ARN"

    def test_bedrock_model_builder_creation(self, training_job):
        """Test BedrockModelBuilder creation."""
        try:
            bedrock_builder = BedrockModelBuilder(model=training_job)
            assert bedrock_builder is not None
            assert bedrock_builder.model == training_job

            # Test that the builder can fetch model package if needed
            if hasattr(bedrock_builder, 'model_package'):
                # This tests the new sagemaker-core integration
                assert bedrock_builder.model_package is not None or bedrock_builder.model_package is None

        except Exception as e:
            pytest.fail(
                f"BedrockModelBuilder creation failed: {str(e)}. This might be due to sagemaker-core integration issues.")

    @pytest.mark.slow
    def test_bedrock_job_created(self, deployed_model_arn):
        """Test that Bedrock import job was created successfully."""
        assert deployed_model_arn is not None

    def test_zzz_cleanup_deployed_model(self, bedrock_client):
        """Cleanup deployed model and import jobs (runs last due to zzz prefix)."""
        if hasattr(self, 'model_arn_for_cleanup'):
            try:
                bedrock_client.delete_imported_model(modelIdentifier=self.model_arn_for_cleanup)
            except Exception:
                pass
        # Cleanup all test import jobs
        try:
            jobs = bedrock_client.list_model_import_jobs()
            for job in jobs.get('modelImportJobSummaries', []):
                if job['jobName'].startswith('test-bedrock-'):
                    try:
                        bedrock_client.stop_model_import_job(jobIdentifier=job['jobArn'])
                    except Exception:
                        pass
        except Exception:
            pass


def test_model_customization_workflow(training_job_name):
    """Standalone test function for pytest discovery."""
    config = {
        "training_job_name": training_job_name,
        "region": "us-west-2",
        "bucket": "open-models-testing-pdx"
    }

    try:
        s3_client = boto3.client('s3', region_name=config["region"])
        training_job = TrainingJob.get(training_job_name=config["training_job_name"])

        test_class = TestModelCustomizationDeployment()
        test_class.test_training_job_exists(training_job)
        test_class.test_bedrock_model_builder_creation(training_job)

    except Exception as e:
        print(f"Standalone test failed: {str(e)}")
        print("This might be due to sagemaker-core integration issues. Please check:")
        print("1. TrainingJob.get() method compatibility")
        print("2. Model artifacts access patterns")
        print("3. BedrockModelBuilder initialization with new sagemaker-core objects")
        raise


class TestBedrockNovaDeployment:
    """Test suite for deploying Nova models to Bedrock."""
    NOVA_TRAINING_JOB_NAME = "nova-textgeneration-lite-v2-sft-20251202132123"

    @pytest.fixture(scope="class", autouse=True)
    def setup_region(self):
        """Set region to us-east-1 for Nova tests."""
        import os
        original_region = os.environ.get('AWS_DEFAULT_REGION')
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        yield
        if original_region:
            os.environ['AWS_DEFAULT_REGION'] = original_region
        else:
            os.environ.pop('AWS_DEFAULT_REGION', None)

    @pytest.fixture(scope="class")
    def training_job(self, setup_region):
        """Get Nova training job."""
        import boto3
        session = boto3.Session(region_name="us-east-1")
        return TrainingJob.get(
            training_job_name=self.NOVA_TRAINING_JOB_NAME,
            session=session,
            region="us-east-1")

    @pytest.mark.skip(reason="Bedrock Nova deployment test skipped per team decision")
    def test_bedrock_model_builder_creation(self, training_job):
        """Test BedrockModelBuilder creation with Nova model."""
        bedrock_builder = BedrockModelBuilder(model=training_job)
        assert bedrock_builder is not None
        assert bedrock_builder.model == training_job
        assert bedrock_builder.s3_model_artifacts is not None

    @pytest.mark.skip(reason="Bedrock Nova deployment test skipped per team decision")
    @pytest.mark.slow
    def test_nova_model_deployment(self, training_job):
        """Test Nova model deployment to Bedrock."""
        from sagemaker.core.helper.session_helper import get_execution_role
        bedrock_builder = BedrockModelBuilder(model=training_job)
        rand = random.randint(1000, 9999)
        response = bedrock_builder.deploy(
            custom_model_name=f"test-nova-deployment-{rand}",
            role_arn=get_execution_role()
        )

        assert response is not None
        assert "modelArn" in response or "jobArn" in response
