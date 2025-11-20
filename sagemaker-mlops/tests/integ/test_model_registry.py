import pytest
import os
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.image_uris import retrieve
import boto3


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def role():
    return get_execution_role()


@pytest.fixture
def model_artifact_path():
    return os.path.join(os.path.dirname(__file__), "data", "model", "registry-test-model.tar.gz")


def test_model_registry(sagemaker_session, role, model_artifact_path):
    region = boto3.Session().region_name
    bucket = sagemaker_session.default_bucket()
    prefix = "test-model-registry"
    model_package_group_name = "test-model-package-group"
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    model_package_arn = None
    
    try:
        # Upload model artifact to S3
        model_s3_key = f"{prefix}/model.tar.gz"
        s3_client = boto3.client('s3')
        s3_client.upload_file(model_artifact_path, bucket, model_s3_key)
        model_url = f"s3://{bucket}/{model_s3_key}"
        
        # Create model for registry
        image_uri = retrieve("xgboost", region, "1.0-1")
        
        model_builder = ModelBuilder(
            image_uri=image_uri,
            s3_model_data_url=model_url,
            role_arn=role,
            sagemaker_session=sagemaker_session,
        )
        
        model = model_builder.build(model_name="test-registry-model")
        assert model is not None
        
        # Register the model
        model_package_arn = model_builder.register(
            model_package_group_name="test-model-package-group",
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge"],
            approval_status="Approved"
        )
        assert model_package_arn is not None
    
    finally:
        # Cleanup model package group
        try:
            response = sagemaker_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name
            )
            for package in response.get('ModelPackageSummaryList', []):
                sagemaker_client.delete_model_package(
                    ModelPackageName=package['ModelPackageArn']
                )
            sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
        except Exception:
            pass
        
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{prefix}/').delete()
