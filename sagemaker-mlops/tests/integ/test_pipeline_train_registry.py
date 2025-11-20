import pytest
import time
import os
import boto3
import uuid
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import InputData, Compute
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import ProcessingInput, ProcessingS3Input, ProcessingOutput, ProcessingS3Output
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core import image_uris


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def pipeline_session():
    return PipelineSession()


@pytest.fixture
def role():
    return get_execution_role()


def test_pipeline_with_train_and_registry(sagemaker_session, pipeline_session, role):
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    prefix = "integ-test-v3-pipeline"
    base_job_prefix = "train-registry-job"
    
    # Upload abalone data to S3
    s3_client = boto3.client('s3')
    abalone_path = os.path.join(os.path.dirname(__file__), "data", "pipeline", "abalone.csv")
    s3_client.upload_file(abalone_path, bucket, f"{prefix}/input/abalone.csv")
    input_data_s3 = f"s3://{bucket}/{prefix}/input/abalone.csv"
    
    # Parameters
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=input_data_s3,
    )
    
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    
    # Processing step
    sklearn_processor = ScriptProcessor(
        image_uri=image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            py_version="py3",
            instance_type="ml.m5.xlarge",
        ),
        instance_type="ml.m5.xlarge",
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}-sklearn",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    processor_args = sklearn_processor.run(
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
                    s3_uri=f"s3://{sagemaker_session.default_bucket()}/{prefix}/train",
                    local_path="/opt/ml/processing/train",
                    s3_upload_mode="EndOfJob"
                )
            ),
            ProcessingOutput(
                output_name="validation",
                s3_output=ProcessingS3Output(
                    s3_uri=f"s3://{sagemaker_session.default_bucket()}/{prefix}/validation",
                    local_path="/opt/ml/processing/validation",
                    s3_upload_mode="EndOfJob"
                )
            ),
            ProcessingOutput(
                output_name="test",
                s3_output=ProcessingS3Output(
                    s3_uri=f"s3://{sagemaker_session.default_bucket()}/{prefix}/test",
                    local_path="/opt/ml/processing/test",
                    s3_upload_mode="EndOfJob"
                )
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "code", "pipeline", "preprocess.py"),
        arguments=["--input-data", input_data],
    )
    
    step_process = ProcessingStep(
        name="PreprocessData",
        step_args=processor_args,
        cache_config=cache_config,
    )
    
    # Training step
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )
    
    model_trainer = ModelTrainer(
        training_image=image_uri,
        compute=Compute(instance_type="ml.m5.xlarge", instance_count=1),
        base_job_name=f"{base_job_prefix}-xgboost",
        sagemaker_session=pipeline_session,
        role=role,
        hyperparameters={
            "objective": "reg:linear",
            "num_round": 50,
            "max_depth": 5,
        },
        input_data_config=[
            InputData(
                channel_name="train",
                data_source=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
        ],
    )
    
    train_args = model_trainer.train()
    step_train = TrainingStep(
        name="TrainModel",
        step_args=train_args,
        cache_config=cache_config,
    )
    
    # Model step
    model_builder = ModelBuilder(
        s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=image_uri,
        sagemaker_session=pipeline_session,
        role_arn=role,
    )
    
    step_create_model = ModelStep(
        name="CreateModel",
        step_args=model_builder.build()
    )
    
    # Register step
    model_package_group_name = f"integ-test-model-group-{uuid.uuid4().hex[:8]}"
    step_register_model = ModelStep(
        name="RegisterModel",
        step_args=model_builder.register(
            model_package_group_name=model_package_group_name,
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge"],
            approval_status="Approved"
        )
    )
    
    # Pipeline
    pipeline_name = f"integ-test-train-registry-{uuid.uuid4().hex[:8]}"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[processing_instance_count, input_data],
        steps=[step_process, step_train, step_create_model, step_register_model],
        sagemaker_session=pipeline_session,
    )
    
    model_name = None
    try:
        # Upsert and execute pipeline
        pipeline.upsert(role_arn=role)
        execution = pipeline.start()
        
        # Poll execution status with 30 minute timeout
        timeout = 1800
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            execution_desc = execution.describe()
            execution_status = execution_desc["PipelineExecutionStatus"]
            
            if execution_status == "Succeeded":
                # Get model name from execution steps
                steps = sagemaker_session.sagemaker_client.list_pipeline_execution_steps(
                    PipelineExecutionArn=execution_desc["PipelineExecutionArn"]
                )["PipelineExecutionSteps"]
                for step in steps:
                    if step["StepName"] == "CreateModel" and "Metadata" in step:
                        model_name = step["Metadata"].get("Model", {}).get("Arn", "").split("/")[-1]
                        break
                assert execution_status == "Succeeded"
                break
            elif execution_status in ["Failed", "Stopped"]:
                pytest.fail(f"Pipeline execution {execution_status}")
            
            time.sleep(60)
        else:
            pytest.fail(f"Pipeline execution timed out after {timeout} seconds")
    
    finally:
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{prefix}/').delete()
        
        # Cleanup model
        if model_name:
            try:
                sagemaker_session.sagemaker_client.delete_model(ModelName=model_name)
            except Exception:
                pass
        
        # Cleanup model package group
        try:
            sagemaker_session.sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
        except Exception:
            pass
        
        # Cleanup pipeline
        try:
            sagemaker_session.sagemaker_client.delete_pipeline(PipelineName=pipeline_name)
        except Exception:
            pass
