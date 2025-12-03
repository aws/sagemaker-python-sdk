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
"""Integration tests for SFT trainer"""
from __future__ import absolute_import

import os
import time
import pytest
import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType


def test_sft_trainer_lora_complete_workflow(sagemaker_session):
    """Test complete SFT training workflow with LORA."""
    
    sft_trainer = SFTTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group_name="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
        training_dataset="s3://mc-flows-sdk-testing/input_data/sft/",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )
    
    # Create training job
    training_job = sft_trainer.train(wait=False)
    
    # Manual wait loop to avoid resource_config issue
    max_wait_time = 3600  # 1 hour timeout
    poll_interval = 30    # Check every 30 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status
        
        if status in ["Completed", "Failed", "Stopped"]:
            break
            
        time.sleep(poll_interval)
    
    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None


def test_sft_trainer_with_validation_dataset(sagemaker_session):
    """Test SFT trainer with both training and validation datasets."""

    sft_trainer = SFTTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group_name="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
        training_dataset="s3://mc-flows-sdk-testing/input_data/sft/",
        validation_dataset="s3://mc-flows-sdk-testing/input_data/sft/",
        accept_eula=True
    )
    
    training_job = sft_trainer.train(wait=False)
    
    # Manual wait loop
    max_wait_time = 3600
    poll_interval = 30
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status
        
        if status in ["Completed", "Failed", "Stopped"]:
            break
            
        time.sleep(poll_interval)
    
    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')


@pytest.mark.skipif(os.environ.get('AWS_DEFAULT_REGION') != 'us-east-1', reason="Nova models only available in us-east-1")
def test_sft_trainer_nova_workflow(sagemaker_session):
    """Test SFT trainer with Nova model."""
    import os
    os.environ['SAGEMAKER_REGION'] = 'us-east-1'

    # For fine-tuning 
    sft_trainer_nova = SFTTrainer(
        model="nova-textgeneration-lite-v2",
        training_type=TrainingType.LORA, 
        model_package_group_name="sdk-test-finetuned-models",
        mlflow_experiment_name="test-nova-finetuned-models-exp",
        mlflow_run_name="test-nova-finetuned-models-run",
        training_dataset="arn:aws:sagemaker:us-east-1:729646638167:hub-content/sdktest/DataSet/sft-nova-test-dataset/0.0.1",
        s3_output_path="s3://mc-flows-sdk-testing-us-east-1/output/"
    )
    
    # Create training job
    training_job = sft_trainer_nova.train(wait=False)
    
    # Manual wait loop
    max_wait_time = 3600  # 1 hour timeout
    poll_interval = 30    # Check every 30 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status
        
        if status in ["Completed", "Failed", "Stopped"]:
            break
            
        time.sleep(poll_interval)
    
    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None
