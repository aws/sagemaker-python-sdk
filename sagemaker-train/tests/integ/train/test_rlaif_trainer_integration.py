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
"""Integration tests for RLAIF trainer"""
from __future__ import absolute_import

import time
import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.rlaif_trainer import RLAIFTrainer
from sagemaker.train.common import TrainingType


def test_rlaif_trainer_lora_complete_workflow(sagemaker_session):
    """Test complete RLAIF training workflow with LORA."""
    
    rlaif_trainer = RLAIFTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group_name="sdk-test-finetuned-models",
        reward_model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
        reward_prompt='Builtin.Correctness',
        mlflow_experiment_name="test-rlaif-finetuned-models-exp",
        mlflow_run_name="test-rlaif-finetuned-models-run",
        training_dataset="s3://mc-flows-sdk-testing/input_data/rlvr-rlaif-test-data/train_285.jsonl",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )
    
    # Create training job
    training_job = rlaif_trainer.train(wait=False)
    
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


def test_rlaif_trainer_with_custom_reward_settings(sagemaker_session):
    """Test RLAIF trainer with different reward model and prompt."""
    
    rlaif_trainer = RLAIFTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group_name="sdk-test-finetuned-models",
        reward_model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
        reward_prompt="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/JsonDoc/rlaif-test-prompt/0.0.1",
        mlflow_experiment_name="test-rlaif-finetuned-models-exp",
        mlflow_run_name="test-rlaif-finetuned-models-run",
        training_dataset="s3://mc-flows-sdk-testing/input_data/rlvr-rlaif-test-data/train_285.jsonl",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )
    
    training_job = rlaif_trainer.train(wait=False)
    
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


def test_rlaif_trainer_continued_finetuning(sagemaker_session):
    """Test complete RLAIF training workflow with LORA."""

    rlaif_trainer = RLAIFTrainer(
        model="arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1",
        training_type=TrainingType.LORA,
        model_package_group_name="sdk-test-finetuned-models",
        reward_model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
        reward_prompt='Builtin.Correctness',
        mlflow_experiment_name="test-rlaif-finetuned-models-exp",
        mlflow_run_name="test-rlaif-finetuned-models-run",
        training_dataset="s3://mc-flows-sdk-testing/input_data/rlvr-rlaif-test-data/train_285.jsonl",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )

    # Create training job
    training_job = rlaif_trainer.train(wait=False)

    # Manual wait loop to avoid resource_config issue
    max_wait_time = 3600  # 1 hour timeout
    poll_interval = 30  # Check every 30 seconds
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
