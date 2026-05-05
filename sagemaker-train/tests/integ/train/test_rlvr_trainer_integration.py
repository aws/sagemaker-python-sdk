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
"""Integration tests for RLVR trainer"""
from __future__ import absolute_import

import os
import time
import logging
import traceback
import pytest
import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.common import TrainingType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def test_rlvr_trainer_lora_complete_workflow(sagemaker_session):
    """Test complete RLVR training workflow with LORA."""
    logger.info("=== START test_rlvr_trainer_lora_complete_workflow ===")
    logger.info(f"sagemaker_session region: {sagemaker_session.boto_region_name}")

    try:
        rlvr_trainer = RLVRTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="sdk-test-finetuned-models",
            mlflow_experiment_name="test-rlvr-finetuned-models-exp",
            mlflow_run_name="test-rlvr-finetuned-models-run",
            training_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/rlvr-rlaif-oss-test-data/0.0.1",
            s3_output_path="s3://mc-flows-sdk-testing/output/",
            accept_eula=True
        )
        logger.info(f"RLVRTrainer created: model={rlvr_trainer.model}, training_type={rlvr_trainer.training_type}")

        # Create training job
        logger.info("Calling rlvr_trainer.train(wait=False)...")
        training_job = rlvr_trainer.train(wait=False)
        logger.info(f"Training job created: {training_job}")

        # Manual wait loop to avoid resource_config issue
        max_wait_time = 3600  # 1 hour timeout
        poll_interval = 30    # Check every 30 seconds
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            training_job.refresh()
            status = training_job.training_job_status
            elapsed = int(time.time() - start_time)
            logger.info(f"[{elapsed}s] Training job status: {status}")

            if status in ["Completed", "Failed", "Stopped"]:
                break

            time.sleep(poll_interval)

        logger.info(f"Final training job status: {training_job.training_job_status}")
        if training_job.training_job_status == "Failed":
            failure_reason = getattr(training_job, 'failure_reason', 'N/A')
            logger.error(f"Training job FAILED. Failure reason: {failure_reason}")

        # Verify job completed successfully
        assert training_job.training_job_status == "Completed"
        assert hasattr(training_job, 'output_model_package_arn')
        assert training_job.output_model_package_arn is not None
        logger.info(f"output_model_package_arn: {training_job.output_model_package_arn}")
    except Exception as e:
        logger.error(f"test_rlvr_trainer_lora_complete_workflow FAILED: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise
    logger.info("=== END test_rlvr_trainer_lora_complete_workflow - PASSED ===")


def test_rlvr_trainer_with_custom_reward_function(sagemaker_session):
    """Test RLVR trainer with custom reward function."""
    logger.info("=== START test_rlvr_trainer_with_custom_reward_function ===")
    logger.info(f"sagemaker_session region: {sagemaker_session.boto_region_name}")

    try:
        rlvr_trainer = RLVRTrainer(
            model="meta-textgeneration-llama-3-2-1b-instruct",
            training_type=TrainingType.LORA,
            model_package_group="sdk-test-finetuned-models",
            mlflow_experiment_name="test-rlvr-finetuned-models-exp",
            mlflow_run_name="test-rlvr-finetuned-models-run",
            training_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/rlvr-rlaif-oss-test-data/0.0.1",
            s3_output_path="s3://mc-flows-sdk-testing/output/",
            custom_reward_function="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/JsonDoc/rlvr-test-rf/0.0.1",
            accept_eula=True
        )
        logger.info(f"RLVRTrainer created with custom_reward_function ARN")

        logger.info("Calling rlvr_trainer.train(wait=False)...")
        training_job = rlvr_trainer.train(wait=False)
        logger.info(f"Training job created: {training_job}")

        # Manual wait loop
        max_wait_time = 3600
        poll_interval = 30
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            training_job.refresh()
            status = training_job.training_job_status
            elapsed = int(time.time() - start_time)
            logger.info(f"[{elapsed}s] Training job status: {status}")

            if status in ["Completed", "Failed", "Stopped"]:
                break

            time.sleep(poll_interval)

        logger.info(f"Final training job status: {training_job.training_job_status}")
        if training_job.training_job_status == "Failed":
            failure_reason = getattr(training_job, 'failure_reason', 'N/A')
            logger.error(f"Training job FAILED. Failure reason: {failure_reason}")

        # Verify job completed successfully
        assert training_job.training_job_status == "Completed"
        assert hasattr(training_job, 'output_model_package_arn')
        assert training_job.output_model_package_arn is not None
        logger.info(f"output_model_package_arn: {training_job.output_model_package_arn}")
    except Exception as e:
        logger.error(f"test_rlvr_trainer_with_custom_reward_function FAILED: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise
    logger.info("=== END test_rlvr_trainer_with_custom_reward_function - PASSED ===")


def test_rlvr_trainer_nova_workflow(sagemaker_session):
    """Test RLVR training workflow with Nova model."""
    logger.info("=== START test_rlvr_trainer_nova_workflow ===")
    logger.info(f"sagemaker_session region: {sagemaker_session.boto_region_name}")

    try:
        import os
        os.environ['SAGEMAKER_REGION'] = 'us-east-1'
        logger.info("Set SAGEMAKER_REGION=us-east-1")

        # For fine-tuning 
        rlvr_trainer = RLVRTrainer(
            model="nova-textgeneration-lite-v2",
            model_package_group="sdk-test-finetuned-models",
            mlflow_experiment_name="test-nova-rlvr-finetuned-models-exp",
            mlflow_run_name="test-nova-rlvr-finetuned-models-run",
            training_dataset="s3://mc-flows-sdk-testing-us-east-1/input_data/rlvr-nova/grpo-64-sample.jsonl",
            validation_dataset="s3://mc-flows-sdk-testing-us-east-1/input_data/rlvr-nova/grpo-64-sample.jsonl",
            s3_output_path="s3://mc-flows-sdk-testing-us-east-1/output/",
            custom_reward_function="arn:aws:sagemaker:us-east-1:729646638167:hub-content/sdktest/JsonDoc/rlvr-nova-test-rf/0.0.1",
            accept_eula=True
        )
        logger.info(f"RLVRTrainer (Nova) created: model={rlvr_trainer.model}")

        rlvr_trainer.hyperparameters.data_s3_path = 's3://example-bucket'
        rlvr_trainer.hyperparameters.reward_lambda_arn = 'arn:aws:lambda:us-east-1:729646638167:function:rlvr-nova-reward-function'
        logger.info(f"Set hyperparameters: data_s3_path={rlvr_trainer.hyperparameters.data_s3_path}, reward_lambda_arn={rlvr_trainer.hyperparameters.reward_lambda_arn}")

        logger.info("Calling rlvr_trainer.train(wait=False)...")
        training_job = rlvr_trainer.train(wait=False)
        logger.info(f"Training job created: {training_job}")

        # Manual wait loop
        max_wait_time = 3600
        poll_interval = 30
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            training_job.refresh()
            status = training_job.training_job_status
            elapsed = int(time.time() - start_time)
            logger.info(f"[{elapsed}s] Training job status: {status}")

            if status in ["Completed", "Failed", "Stopped"]:
                break

            time.sleep(poll_interval)

        logger.info(f"Final training job status: {training_job.training_job_status}")
        if training_job.training_job_status == "Failed":
            failure_reason = getattr(training_job, 'failure_reason', 'N/A')
            logger.error(f"Training job FAILED. Failure reason: {failure_reason}")

        # Verify job completed successfully
        assert training_job.training_job_status == "Completed"
        assert hasattr(training_job, 'output_model_package_arn')
        assert training_job.output_model_package_arn is not None
        logger.info(f"output_model_package_arn: {training_job.output_model_package_arn}")
    except Exception as e:
        logger.error(f"test_rlvr_trainer_nova_workflow FAILED: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise
    logger.info("=== END test_rlvr_trainer_nova_workflow - PASSED ===")
