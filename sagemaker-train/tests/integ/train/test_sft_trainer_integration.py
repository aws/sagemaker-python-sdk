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

import time
import random
import pytest
import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType

@pytest.mark.gpu_intensive
def test_sft_trainer_lora_complete_workflow(sagemaker_session, mlflow_resource_arn):
    """Test complete SFT training workflow with LORA, including show_metrics via MLflow."""
    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
    
    sft_trainer = SFTTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
        training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        mlflow_resource_arn=mlflow_resource_arn,
        accept_eula=True,
        base_job_name=f"sft-lora-integ-{unique_id}",
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

    # Verify show_metrics() works via MLflow path for OSS models
    result = sft_trainer.show_metrics()
    # OSS MLflow path renders inline; may return None or a DataFrame
    print(f"show_metrics() returned: {type(result).__name__}")

    # Verify stream_logs() exits without error on a completed job
    sft_trainer.stream_logs(poll=2)


@pytest.mark.gpu_intensive
def test_sft_trainer_with_validation_dataset(sagemaker_session):
    """Test SFT trainer with both training and validation datasets."""
    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"

    sft_trainer = SFTTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
        training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
        validation_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
        accept_eula=True,
        base_job_name=f"sft-val-integ-{unique_id}",
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


@pytest.mark.gpu_intensive
@pytest.mark.us_east_1
def test_sft_trainer_nova_workflow(sagemaker_session_us_east_1):
    """Test SFT trainer with Nova model, including show_metrics() after completion."""
    # sagemaker_session_us_east_1 fixture is defined in conftest.py (us-east-1 region)

    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
    sft_trainer_nova = SFTTrainer(
        model="nova-textgeneration-lite-v2",
        training_type=TrainingType.LORA, 
        model_package_group="sdk-test-finetuned-models",
        mlflow_experiment_name="test-nova-finetuned-models-exp",
        mlflow_run_name="test-nova-finetuned-models-run",
        training_dataset="s3://sagemaker-us-east-1-784379639078/input_data/sft-nova/sft_200_samples.jsonl",
        s3_output_path="s3://sagemaker-us-east-1-784379639078/output/",
        sagemaker_session=sagemaker_session_us_east_1,
        base_job_name=f"sft-nova-integ-{unique_id}",
    )
    
    # Create training job
    training_job = sft_trainer_nova.train(wait=False)
    
    # Manual wait loop
    max_wait_time = 10800  # 3 hour timeout (Nova training takes >1 hour)
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

    # Verify show_metrics() returns valid training metrics
    # Use non-interactive backend so plt.show() doesn't require a display in CI
    import matplotlib
    matplotlib.use("Agg")

    df = sft_trainer_nova.show_metrics()
    assert df is not None, "show_metrics() returned None"
    assert not df.empty, "show_metrics() returned empty DataFrame"
    assert "global_step" in df.columns, (
        f"Expected 'global_step' column, got: {list(df.columns)}"
    )
    assert len(df) > 0

    # Verify metric filter works
    df_filtered = sft_trainer_nova.show_metrics(metrics=["training_loss"])
    assert not df_filtered.empty
    assert set(df_filtered.columns) == {"global_step", "training_loss"}

    # Verify step range filter works
    min_step = int(df["global_step"].min())
    max_step = int(df["global_step"].max())
    if max_step > min_step:
        mid = (min_step + max_step) // 2
        df_range = sft_trainer_nova.show_metrics(starting_step=mid, ending_step=max_step)
        assert not df_range.empty
        assert df_range["global_step"].min() >= mid
        assert df_range["global_step"].max() <= max_step




# @pytest.mark.gpu_intensive
@pytest.mark.gpu_intensive
def test_sft_trainer_lora_with_sequence_length(sagemaker_session):
    """Test SFT training workflow with LORA and sequence_length specified."""
    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"

    sft_trainer = SFTTrainer(
        model="huggingface-vlm-qwen3-5-9b",
        training_type=TrainingType.LORA,
        model_package_group="arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
        training_dataset="s3://mc-flows-sdk-testing/input_data/sft/sample_data_256_final.jsonl",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True,
        sequence_length="16K",
        base_job_name=f"sft-seqlen-integ-{unique_id}",
    )

    training_job = sft_trainer.train(wait=False)

    max_wait_time = 3600
    poll_interval = 30
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status

        if status in ["Completed", "Failed", "Stopped"]:
            break

        time.sleep(poll_interval)

    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None
