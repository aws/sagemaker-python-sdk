import pytest
import os
import time
import boto3
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import ProcessingInput, ProcessingS3Input, ProcessingOutput, ProcessingS3Output
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core import image_uris


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role()


@pytest.fixture
def abalone_data_path():
    return os.path.join(os.path.dirname(__file__), "data", "abalone.csv")


def test_sklearn_processing_job(sagemaker_session, role, abalone_data_path):
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    prefix = "integ-test-processing-sklearn"
    
    try:
        # Upload abalone data to S3
        input_s3_key = f"{prefix}/input/abalone.csv"
        s3_client = boto3.client('s3')
        s3_client.upload_file(abalone_data_path, bucket, input_s3_key)
        input_data = f"s3://{bucket}/{input_s3_key}"
        
        sklearn_processor = ScriptProcessor(
            image_uri=image_uris.retrieve(
                framework="sklearn",
                region=region,
                version="1.2-1",
                py_version="py3",
                instance_type="ml.m5.xlarge",
            ),
            instance_type="ml.m5.xlarge",
            instance_count=1,
            base_job_name="test-sklearn-preprocess",
            sagemaker_session=sagemaker_session,
            role=role,
        )
        
        processor_args = sklearn_processor.run(
            wait=False,
            inputs=[
                ProcessingInput(
                    input_name="input-1",
                    s3_input=ProcessingS3Input(
                        s3_uri=input_data,
                        local_path="/opt/ml/processing/input",
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                        s3_data_distribution_type="ShardedByS3Key",
                    )
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/{prefix}/train",
                        local_path="/opt/ml/processing/train",
                        s3_upload_mode="EndOfJob"
                    )
                ),
                ProcessingOutput(
                    output_name="validation",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/{prefix}/validation",
                        local_path="/opt/ml/processing/validation",
                        s3_upload_mode="EndOfJob"
                    )
                ),
                ProcessingOutput(
                    output_name="test",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/{prefix}/test",
                        local_path="/opt/ml/processing/test",
                        s3_upload_mode="EndOfJob"
                    )
                ),
            ],
            code=os.path.join(os.path.dirname(__file__), "code", "preprocess.py"),
            arguments=["--input-data", input_data],
        )
        
        # Wait for processing job to complete
        timeout = 600  # 10 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            sklearn_processor.latest_job.refresh()
            status = sklearn_processor.latest_job.processing_job_status
            
            if status == "Completed":
                assert status == "Completed"
                break
            elif status in ["Failed", "Stopped"]:
                pytest.fail(f"Processing job {status}")
            
            time.sleep(30)
        else:
            pytest.fail(f"Processing job timed out after {timeout} seconds")
    
    finally:
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{prefix}/').delete()
