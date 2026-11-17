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
"""Integration tests for MTRL Evaluator: Attach to existing job → Evaluate → Wait for completion.

Tests the MTRL evaluation workflow by attaching to existing completed training
jobs (discovered via DescribeJob API) and running evaluations in both
fine-tuned and base model modes, waiting for successful completion.

The job and resources are the account's integ fixtures (see integ_resources).
"""

from __future__ import absolute_import

import pytest
import logging

from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
from sagemaker.train.evaluate import MultiTurnRLEvaluator
from .. import integ_resources

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

_REGION = "us-west-2"

EVAL_TIMEOUT = 14400  # 4 hours


def _get_config():
    """Configuration for the account running the tests, from its integ fixtures."""
    return {
        "env_name": integ_resources.account_id(),
        # Completed MTRL fixture job; its output package lives in a fixture group the CI
        # resource cleaner keeps. Shared with test_mtrl_evaluator_3p_agent.py.
        "existing_job_name": integ_resources.mtrl_job_name(),
        "base_model": "mock-oss-test",
        "agent_core_arn": integ_resources.agent_runtime_arn(),
        "dataset": integ_resources.dataset_uri(integ_resources.MTRL_PROMPTS),
        "s3_output_path": integ_resources.output_uri("mtrl-integ/eval-output"),
        "mlflow_resource_arn": integ_resources.mlflow_app_arn(),
        "model_package_group": integ_resources.arn(
            "sagemaker", f"model-package-group/{integ_resources.mtrl_model_package_group()}"
        ),
    }


@pytest.fixture(scope="module")
def config():
    """Get account-specific test configuration."""
    return _get_config()


@pytest.fixture(scope="module")
def attached_trainer(config):
    """Attach to an existing completed MTRL training job and return a trainer with it set.

    Uses the DescribeJob API (via MultiTurnRLTrainer.attach) to retrieve
    the job and verify it completed successfully with an output model package.
    """
    job = MultiTurnRLTrainer.attach(job_name=config["existing_job_name"])
    logger.info(f"[{config['env_name']}] Attached to job: {job.job_name}")
    logger.info(f"[{config['env_name']}] Status: {job.job_status}")
    logger.info(f"[{config['env_name']}] Output model package: {job.output_model_package_arn}")

    assert job.job_status == "Completed", (
        f"Existing job {config['existing_job_name']} is not Completed "
        f"(status: {job.job_status}). Cannot use for evaluation."
    )
    assert (
        job.output_model_package_arn is not None
    ), f"Existing job {config['existing_job_name']} has no output_model_package_arn."

    trainer = MultiTurnRLTrainer(
        model=config["base_model"],
        agent_env=config["agent_core_arn"],
        training_dataset=config["dataset"],
        output_model_package_group=config["model_package_group"],
        mlflow_app_arn=config["mlflow_resource_arn"],
        s3_output_path=config["s3_output_path"],
        accept_eula=True,
    )
    trainer._latest_job = job
    return trainer


@pytest.mark.gpu_intensive
@pytest.mark.serial
class TestMTRLEvalIntegration:
    """Integration tests for MTRL evaluation: attach → evaluate → wait for success."""

    def test_attach_to_existing_job(self, config):
        """Test that we can attach to an existing completed MTRL job via DescribeJob API."""
        job = MultiTurnRLTrainer.attach(job_name=config["existing_job_name"])

        assert job is not None
        assert job.job_name == config["existing_job_name"]
        assert job.job_status == "Completed"
        assert job.output_model_package_arn is not None

        logger.info(f"[{config['env_name']}] Job name: {job.job_name}")
        logger.info(f"[{config['env_name']}] Job ARN: {job.job_arn}")
        logger.info(f"[{config['env_name']}] Output model package: {job.output_model_package_arn}")

    def test_evaluate_finetuned_model(self, attached_trainer, config):
        """Evaluate a fine-tuned model from attached trainer — submit and wait for completion."""
        evaluator = MultiTurnRLEvaluator(
            model=attached_trainer,
            dataset=config["dataset"],
            s3_output_path=f'{config["s3_output_path"]}finetuned/',
            mlflow_resource_arn=config["mlflow_resource_arn"],
            region=_REGION,
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        assert "pipeline" in execution.arn.lower()
        logger.info(f"[{config['env_name']}] Started finetuned eval: {execution.arn}")

        logger.info(f"[{config['env_name']}] Waiting for finetuned eval to complete...")
        execution.wait(timeout=EVAL_TIMEOUT)

        status = execution.status.overall_status
        logger.info(f"[{config['env_name']}] Finetuned eval completed: {status}")

        assert status == "Succeeded", (
            f"[{config['env_name']}] Finetuned eval failed with status: {status}, "
            f"reason: {execution.status.failure_reason}"
        )

    def test_evaluate_base_model(self, config):
        """Evaluate the base model only — submit and wait for completion."""
        evaluator = MultiTurnRLEvaluator(
            model=config["base_model"],
            dataset=config["dataset"],
            agent_config=config["agent_core_arn"],
            s3_output_path=f'{config["s3_output_path"]}basemodel/',
            mlflow_resource_arn=config["mlflow_resource_arn"],
            region=_REGION,
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        assert "pipeline" in execution.arn.lower()
        logger.info(f"[{config['env_name']}] Started base model eval: {execution.arn}")

        logger.info(f"[{config['env_name']}] Waiting for base model eval to complete...")
        execution.wait(timeout=EVAL_TIMEOUT)

        status = execution.status.overall_status
        logger.info(f"[{config['env_name']}] Base model eval completed: {status}")

        assert status == "Succeeded", (
            f"[{config['env_name']}] Base model eval failed with status: {status}, "
            f"reason: {execution.status.failure_reason}"
        )

    @pytest.mark.skip(
        reason="Comparison template has CreateJob schema validation issue — tracked separately"
    )
    def test_evaluate_comparison(self, attached_trainer, config):
        """Evaluate base + finetuned comparison — submit and wait for completion."""
        evaluator = MultiTurnRLEvaluator(
            model=attached_trainer,
            dataset=config["dataset"],
            s3_output_path=f'{config["s3_output_path"]}comparison/',
            mlflow_resource_arn=config["mlflow_resource_arn"],
            region=_REGION,
            evaluate_base_model=True,
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        assert "pipeline" in execution.arn.lower()
        logger.info(f"[{config['env_name']}] Started comparison eval: {execution.arn}")

        logger.info(f"[{config['env_name']}] Waiting for comparison eval to complete...")
        execution.wait(timeout=EVAL_TIMEOUT)

        status = execution.status.overall_status
        logger.info(f"[{config['env_name']}] Comparison eval completed: {status}")

        assert status == "Succeeded", (
            f"[{config['env_name']}] Comparison eval failed with status: {status}, "
            f"reason: {execution.status.failure_reason}"
        )


@pytest.mark.gpu_intensive
class TestMTRLShowMetrics:
    """Integration tests for show_metrics() on completed MTRL jobs."""

    def test_show_metrics_on_completed_job(self, config):
        """show_metrics() and stream_logs() on a completed MTRL job run without error."""
        job = MultiTurnRLTrainer.attach(job_name=config["existing_job_name"])
        assert job.job_status == "Completed"

        trainer = MultiTurnRLTrainer(
            model=config["base_model"],
            agent_env=config["agent_core_arn"],
            training_dataset=config["dataset"],
            output_model_package_group=config["model_package_group"],
            mlflow_app_arn=config["mlflow_resource_arn"],
            s3_output_path=config["s3_output_path"],
            accept_eula=True,
        )
        trainer._latest_job = job

        result = trainer.show_metrics()
        logger.info(f"[{config['env_name']}] show_metrics() returned: {type(result).__name__}")

        # stream_logs() should exit quickly for a completed job
        trainer.stream_logs(poll=2)
        logger.info(f"[{config['env_name']}] stream_logs() completed without error")
