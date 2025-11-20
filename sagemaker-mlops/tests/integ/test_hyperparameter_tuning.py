import pytest
import time
import boto3
import os
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, SourceCode, InputData, StoppingCondition
from sagemaker.train.tuner import HyperparameterTuner
from sagemaker.core.parameter import ContinuousParameter, CategoricalParameter


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def role():
    return get_execution_role()


@pytest.fixture
def mnist_data_dir():
    return os.path.join(os.path.dirname(__file__), "data", "mnist")


def test_hyperparameter_tuning_e2e(sagemaker_session, role, mnist_data_dir):
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    prefix = "v3-tunning-integ-test"
    
    try:
        # Upload pre-downloaded MNIST data to S3
        s3_data_uri = sagemaker_session.upload_data(
            path=mnist_data_dir,
            bucket=bucket,
            key_prefix=f"{prefix}/data"
        )
        
        # Configure source code
        source_code = SourceCode(
            source_dir=os.path.join(os.path.dirname(__file__), "code"),
            entry_script="mnist.py"
        )
        
        # Configure compute
        compute = Compute(
            instance_type="ml.m5.xlarge",
            instance_count=1,
            volume_size_in_gb=30
        )
        
        # Configure stopping condition
        stopping_condition = StoppingCondition(
            max_runtime_in_seconds=3600
        )
        
        # Get training image
        training_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:1.10.0-gpu-py38"
        
        # Create ModelTrainer
        model_trainer = ModelTrainer(
            training_image=training_image,
            source_code=source_code,
            compute=compute,
            stopping_condition=stopping_condition,
            hyperparameters={
                "epochs": 1,
                "backend": "gloo"
            },
            sagemaker_session=sagemaker_session,
            role=role,
            base_job_name="test-hpo-pytorch"
        )
        
        # Define hyperparameter ranges
        hyperparameter_ranges = {
            "lr": ContinuousParameter(0.001, 0.1),
            "batch-size": CategoricalParameter([32, 64, 128]),
        }
        
        # Define metric definitions
        metric_definitions = [
            {
                "Name": "average test loss",
                "Regex": "Test set: Average loss: ([0-9\\.]+)"
            }
        ]
        
        # Create HyperparameterTuner
        tuner = HyperparameterTuner(
            model_trainer=model_trainer,
            objective_metric_name="average test loss",
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=metric_definitions,
            max_jobs=2,
            max_parallel_jobs=1,
            strategy="Random",
            objective_type="Minimize",
            early_stopping_type="Auto"
        )
        
        # Prepare input data
        training_data = InputData(
            channel_name="training",
            data_source=s3_data_uri
        )
        
        # Start tuning job
        tuner.tune(
            inputs=[training_data],
            wait=False
        )
        
        tuning_job_name = tuner._current_job_name
        assert tuning_job_name is not None
        
        # Poll for completion
        timeout = 1800  # 30 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = tuner.describe()
            status = response.hyper_parameter_tuning_job_status
            
            if status == "Completed":
                assert status == "Completed"
                break
            elif status in ["Failed", "Stopped"]:
                pytest.fail(f"Tuning job {status}")
            
            time.sleep(60)
        else:
            pytest.fail(f"Tuning job timed out after {timeout} seconds")
    
    finally:
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{prefix}/').delete()
