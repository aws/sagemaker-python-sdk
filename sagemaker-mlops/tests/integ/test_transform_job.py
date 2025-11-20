import pytest
import boto3
import os
import time
from sagemaker.core.helper.session_helper import get_execution_role, Session
from sagemaker.core.transformer import Transformer
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.image_uris import retrieve


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def role():
    return get_execution_role()


def test_transform_job(sagemaker_session, role):
    region = boto3.Session().region_name
    bucket = sagemaker_session.default_bucket()
    prefix = "integ-test-transform"
    transform_output_path = f"s3://{bucket}/{prefix}/transform-outputs"
    
    s3_client = boto3.client('s3')
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    try:
        # Upload model and validation data to S3
        model_file = "xgb-churn-prediction-model.tar.gz"
        s3_client.upload_file(
            os.path.join(data_dir, "model", "transform_job", model_file),
            bucket,
            f"{prefix}/{model_file}"
        )
        s3_client.upload_file(
            os.path.join(data_dir, "validation.csv"),
            bucket,
            f"{prefix}/transform_input/validation/validation.csv"
        )
        
        model_url = f"https://{bucket}.s3-{region}.amazonaws.com/{prefix}/{model_file}"
        
        # Build model
        image_uri = retrieve("xgboost", region, "0.90-1")
        model_builder = ModelBuilder(
            image_uri=image_uri,
            s3_model_data_url=model_url,
            role_arn=role,
            sagemaker_session=sagemaker_session,
        )
        model_builder.build(model_name="integ-test-transform-model")

        # Create transformer
        transformer = Transformer(
            model_name="integ-test-transform-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            accept="text/csv",
            assemble_with="Line",
            output_path=transform_output_path,
            sagemaker_session=sagemaker_session,
        )
        
        # Run transform
        data_input = f"s3://{bucket}/{prefix}/transform_input/validation"
        transformer.transform(
            data_input,
            content_type="text/csv",
            split_type="Line",
            input_filter="$[1:]",
            wait=False,
        )
        
        # Poll job status with 10 minute timeout
        job = transformer.latest_transform_job
        timeout = 600
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job.refresh()
            status = job.transform_job_status
            
            if status == "Completed":
                assert status == "Completed"
                break
            elif status in ["Failed", "Stopped"]:
                pytest.fail(f"Transform job {status}")
            
            time.sleep(30)
        else:
            pytest.fail(f"Transform job timed out after {timeout} seconds")
    
    finally:
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{prefix}/').delete()
        
        # Cleanup model
        try:
            sagemaker_session.sagemaker_client.delete_model(ModelName="integ-test-transform-model")
        except Exception:
            pass
