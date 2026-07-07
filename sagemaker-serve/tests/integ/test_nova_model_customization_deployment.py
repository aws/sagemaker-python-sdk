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
"""Integration tests for Nova model customization deployment.

Covers deploying a fine-tuned Nova model to SageMaker endpoints (via
ModelBuilder) and to Amazon Bedrock custom models (via BedrockModelBuilder).
"""
from __future__ import absolute_import

import boto3
import json
import logging
import os
import re
import time
import pytest
import random
from sagemaker.serve import ModelBuilder
from sagemaker.serve.model_reuse import MODEL_SOURCE_TAG_KEY
from sagemaker.core.resources import TrainingJob

logger = logging.getLogger(__name__)

from sagemaker.core.helper.session_helper import Session

# This test relies on resources in a specific region
AWS_REGION = "us-east-1"
os.environ.setdefault("AWS_DEFAULT_REGION", AWS_REGION)

# Model package group shared with the Nova SFT/RLVR trainer integ tests.
MODEL_PACKAGE_GROUP = "sdk-test-finetuned-models"

NOVA_MODEL_ID = "nova-textgeneration-lite-v2"
NOVA_INSTANCE_TYPE = "ml.g6.48xlarge"


def _deploy_or_skip_on_capacity(model_builder, **deploy_kwargs):
    """Deploy via ModelBuilder, skipping on transient InsufficientInstanceCapacity."""
    try:
        return model_builder.deploy(**deploy_kwargs)
    except Exception as e:
        msg = str(e)
        if "InsufficientInstanceCapacity" in msg or "InsufficientInstance" in msg:
            pytest.skip(f"Skipping due to transient instance capacity shortage: {msg}")
        raise


def _latest_model_package_arn(region=AWS_REGION):
    """Return the ARN of the most recently created Completed model package in
    the Nova model package group, or None if the group has no usable package.
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
    """A completed Nova SFT training job whose output model package still exists.

    The training job's output model package must exist because ModelBuilder.build
    fetches it. Instead of picking a job and hoping its package survived resource
    cleanup, we go the other way: start from a model package that currently exists
    in the group and resolve the training job that produced it (encoded in the
    package's escrow S3 URI). The cleaner always retains the oldest package, so
    this reliably yields a usable job.
    """
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
    packages = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
    ).get("ModelPackageSummaryList", [])

    sft_fallback = None
    for pkg in packages:
        if pkg.get("ModelPackageStatus") != "Completed":
            continue
        detail = sm_client.describe_model_package(ModelPackageName=pkg["ModelPackageArn"])
        containers = detail.get("InferenceSpecification", {}).get("Containers", [])
        if not containers:
            continue
        s3_uri = containers[0].get("ModelDataSource", {}).get("S3DataSource", {}).get("S3Uri", "")
        # Escrow URI looks like s3://.../<training-job-name>/step_N/
        match = re.search(r"/((?:sft|rlvr)-nova-integ-[^/]+)/", s3_uri)
        if not match:
            continue
        job_name = match.group(1)
        try:
            job = sm_client.describe_training_job(TrainingJobName=job_name)
        except sm_client.exceptions.ClientError:
            continue
        if job.get("TrainingJobStatus") != "Completed":
            continue
        if job_name.startswith("sft-nova-integ"):
            return job_name
        sft_fallback = sft_fallback or job_name

    if sft_fallback:
        return sft_fallback

    pytest.skip(
        "No existing Nova model package with a resolvable completed training job "
        "was found. Ensure the scheduled Nova SFT/RLVR workflow has run."
    )


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

        endpoint = _deploy_or_skip_on_capacity(
            model_builder,
            endpoint_name=endpoint_name,
        )

        cleanup_endpoints.append(endpoint_name)

        assert endpoint is not None
        assert endpoint.endpoint_arn is not None
        assert endpoint.endpoint_status == "InService"

        # The endpoint should carry the model-source tag that powers resource reuse.
        sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
        endpoint_tags = sm_client.list_tags(ResourceArn=endpoint.endpoint_arn).get("Tags", [])
        assert MODEL_SOURCE_TAG_KEY in {t["Key"] for t in endpoint_tags}, (
            f"Endpoint {endpoint.endpoint_arn} missing model-source tag for reuse"
        )

        time.sleep(10)  # brief buffer for inference component readiness

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
    """Test Nova model customization deployment via the registered model package.

    A fine-tuned Nova model is built and deployed from its TrainingJob; the
    registered model package is validated along the way.
    """

    def test_build_from_model_package(self, training_job_name, sagemaker_session):
        """Build a Nova model from the training job and validate its model package."""
        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        model_builder = ModelBuilder(
            model=training_job,
            instance_type=NOVA_INSTANCE_TYPE,
            sagemaker_session=sagemaker_session,
        )
        model_builder.accept_eula = True
        model = model_builder.build(region=AWS_REGION)

        assert model is not None
        assert model.model_arn is not None
        assert model_builder._fetch_model_package_arn() is not None

    def test_deploy_from_model_package(self, training_job_name, endpoint_name, cleanup_endpoints, sagemaker_session):
        """Deploy a Nova model via the training-job path and validate the endpoint."""
        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        model_builder = ModelBuilder(
            model=training_job,
            instance_type=NOVA_INSTANCE_TYPE,
            sagemaker_session=sagemaker_session,
        )
        model_builder.accept_eula = True
        model_builder.build(region=AWS_REGION)
        endpoint = _deploy_or_skip_on_capacity(model_builder, endpoint_name=endpoint_name)

        cleanup_endpoints.append(endpoint_name)

        assert endpoint is not None
        assert endpoint.endpoint_arn is not None


@pytest.mark.us_east_1
class TestInstanceTypeAutoDetection:
    """Test instance type handling for Nova models."""

    def test_instance_type_from_recipe(self, training_job_name, sagemaker_session):
        """Nova requires an explicit supported instance type (no auto-detection)."""
        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        model_builder = ModelBuilder(
            model=training_job,
            instance_type=NOVA_INSTANCE_TYPE,
            sagemaker_session=sagemaker_session,
        )
        model_builder.accept_eula = True
        model_builder.build(region=AWS_REGION)

        assert model_builder.instance_type == NOVA_INSTANCE_TYPE
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

    Nova has no DPO recipe, so RLVR is used in place of the open-weights DPO test.
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

        model_builder = ModelBuilder(
            model=trainer,
            instance_type=NOVA_INSTANCE_TYPE,
            sagemaker_session=sagemaker_session,
        )
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

        model_builder = ModelBuilder(
            model=trainer,
            instance_type=NOVA_INSTANCE_TYPE,
            sagemaker_session=sagemaker_session,
        )
        model = model_builder.build(region=AWS_REGION)

        assert model is not None
        assert model.model_arn is not None


@pytest.mark.us_east_1
class TestNovaBedrockDeployment:
    """Test deploying a fine-tuned Nova model to Amazon Bedrock as a custom model."""

    @pytest.fixture(scope="class")
    def role_arn(self):
        """Execution role ARN with Bedrock permissions."""
        from sagemaker.core.helper.session_helper import get_execution_role
        return get_execution_role()

    @pytest.fixture(scope="class")
    def bedrock_client(self):
        """Create a Bedrock control-plane client."""
        return boto3.client("bedrock", region_name=AWS_REGION)

    @pytest.fixture(scope="class")
    def bedrock_runtime(self):
        """Bedrock runtime client with retries for not-yet-ready custom models."""
        from botocore.config import Config
        config = Config(retries={"total_max_attempts": 10, "mode": "standard"})
        return boto3.client("bedrock-runtime", region_name=AWS_REGION, config=config)

    @pytest.fixture(scope="class")
    def deployed_nova_model(self, training_job_name, role_arn, bedrock_client):
        """Deploy a Nova model to Bedrock from its TrainingJob and yield the
        deployment details, cleaning up the custom model and deployment after.
        """
        from sagemaker.core.resources import TrainingJob
        from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder

        unique = f"{int(time.time())}-{random.randint(1000, 9999)}"
        custom_model_name = f"nova-integ-{unique}"
        deployment_name = f"nova-integ-{unique}-deployment"

        training_job = TrainingJob.get(training_job_name=training_job_name, region=AWS_REGION)
        bedrock_builder = BedrockModelBuilder(model=training_job)

        deployment_arn = None
        model_arn = None
        try:
            response = bedrock_builder.deploy(
                custom_model_name=custom_model_name,
                deployment_name=deployment_name,
                role_arn=role_arn,
            )

            assert response is not None
            deployment_arn = response.get("customModelDeploymentArn")
            assert deployment_arn is not None, f"No deployment ARN in response: {response}"

            deployment = bedrock_client.get_custom_model_deployment(
                customModelDeploymentIdentifier=deployment_arn
            )
            model_arn = deployment.get("modelArn")

            yield {
                "deployment_arn": deployment_arn,
                "model_arn": model_arn,
                "custom_model_name": custom_model_name,
            }
        except Exception as e:
            pytest.fail(f"Nova Bedrock deployment failed: {e}")
        finally:
            if deployment_arn:
                try:
                    bedrock_client.delete_custom_model_deployment(
                        customModelDeploymentIdentifier=deployment_arn
                    )
                    logger.info("Deleted custom model deployment: %s", deployment_arn)
                except Exception as e:
                    logger.warning("Failed to delete deployment %s: %s", deployment_arn, e)
            if model_arn:
                try:
                    bedrock_client.delete_custom_model(modelIdentifier=model_arn)
                    logger.info("Deleted custom model: %s", model_arn)
                except Exception as e:
                    logger.warning("Failed to delete custom model %s: %s", model_arn, e)

    def test_nova_bedrock_deployment_active(self, deployed_nova_model, bedrock_client):
        """The Nova custom model deployment should be Active after deploy()."""
        deployment_arn = deployed_nova_model["deployment_arn"]
        deployment = bedrock_client.get_custom_model_deployment(
            customModelDeploymentIdentifier=deployment_arn
        )
        assert deployment.get("status") == "Active"

    def test_nova_bedrock_custom_model_tagged_for_reuse(self, deployed_nova_model, bedrock_client):
        """The Nova custom model should carry the model-source tag that powers reuse."""
        model_arn = deployed_nova_model["model_arn"]
        tags = bedrock_client.list_tags_for_resource(resourceARN=model_arn).get("tags", [])
        assert MODEL_SOURCE_TAG_KEY in {t["key"] for t in tags}, (
            f"Custom model {model_arn} missing model-source tag for reuse"
        )

    @pytest.mark.slow
    def test_nova_bedrock_invoke(self, deployed_nova_model, bedrock_runtime):
        """Invoke the deployed Nova model on Bedrock end-to-end."""
        deployment_arn = deployed_nova_model["deployment_arn"]

        response = bedrock_runtime.invoke_model(
            modelId=deployment_arn,
            body=json.dumps({
                "schemaVersion": "messages-v1",
                "messages": [
                    {"role": "user", "content": [{"text": "What is 7+7?"}]}
                ],
                "inferenceConfig": {"maxTokens": 100, "temperature": 0.0, "topP": 0.9},
            }),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read().decode())

        assert result is not None, "Empty response from Bedrock invoke"
        assert isinstance(result, dict)
        text = result["output"]["message"]["content"][0]["text"]
        assert isinstance(text, str) and len(text) > 0
