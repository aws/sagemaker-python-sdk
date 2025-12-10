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
"""Integration tests for DPO trainer"""
from __future__ import absolute_import

import time
import random
import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.common import TrainingType
import pytest


@pytest.mark.skip(reason="Skipping GPU resource intensive test")
def test_dpo_trainer_lora_complete_workflow(sagemaker_session):
    """Test complete DPO training workflow with LORA."""
    # Create DPOTrainer instance with comprehensive configuration
    trainer = DPOTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group="sdk-test-finetuned-models",
        training_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/dpo-oss-test-data/0.0.1",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )
    
    # Customize hyperparameters for quick training
    trainer.hyperparameters.max_epochs = 1
    
    # Create training job
    training_job = trainer.train(wait=False)
    
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


@pytest.mark.skip(reason="Skipping GPU resource intensive test")
def test_dpo_trainer_with_validation_dataset(sagemaker_session):
    """Test DPO trainer with both training and validation datasets."""
    
    dpo_trainer = DPOTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group="sdk-test-finetuned-models",
        training_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/dpo-oss-test-data/0.0.1",
        validation_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/dpo-oss-test-data/0.0.1",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )
    
    # Customize hyperparameters for quick training
    dpo_trainer.hyperparameters.max_epochs = 1
    
    training_job = dpo_trainer.train(wait=False)
    
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
    assert training_job.output_model_package_arn is not None
