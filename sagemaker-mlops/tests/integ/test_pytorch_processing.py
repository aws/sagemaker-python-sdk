import pytest
import os
import boto3
import time
from time import gmtime, strftime
from sagemaker.core.processing import FrameworkProcessor
from sagemaker.core.shapes import ProcessingOutput, ProcessingS3Output
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.image_uris import get_training_image_uri


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def role():
    return get_execution_role()


def test_pytorch_processing_job(sagemaker_session, role):
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    s3_prefix = "integ-test-pytorch-processing"
    processing_job_name = "{}-{}".format(s3_prefix, strftime("%d-%H-%M-%S", gmtime()))
    output_destination = "s3://{}/{}".format(bucket, s3_prefix)
    
    try:
        image_uri = get_training_image_uri(
            region=region,
            framework="pytorch",
            framework_version="1.13",
            py_version="py39",
            instance_type="ml.m5.xlarge",
        )
        
        pytorch_processor = FrameworkProcessor(
            image_uri=image_uri,
            role=role,
            instance_type="ml.m5.xlarge",
            instance_count=1,
        )
        
        pytorch_processor.run(
            code="preprocessing.py",
            source_dir=os.path.join(os.path.dirname(__file__), "code", "pytorch_processing"),
            requirements="requirements.txt",
            job_name=processing_job_name,
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    s3_output=ProcessingS3Output(
                        s3_uri="{}/train".format(output_destination),
                        local_path="/opt/ml/processing/train",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
                ProcessingOutput(
                    output_name="test",
                    s3_output=ProcessingS3Output(
                        s3_uri="{}/test".format(output_destination),
                        local_path="/opt/ml/processing/test",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
            ],
            wait=False,
        )
        
        # Check job status with 10 minute timeout
        job = pytorch_processor.latest_job
        timeout = 600
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job.refresh()
            status = job.processing_job_status
            
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
        bucket_obj.objects.filter(Prefix=f'{s3_prefix}/').delete()
