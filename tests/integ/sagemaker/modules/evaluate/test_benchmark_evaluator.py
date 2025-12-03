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
"""Integration tests for BenchmarkEvaluator"""
from __future__ import absolute_import

import pytest
import logging

from sagemaker.train.evaluate import (
    BenchMarkEvaluator,
    get_benchmarks,
    get_benchmark_properties,
    EvaluationPipelineExecution,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test timeout configuration (in seconds)
EVALUATION_TIMEOUT_SECONDS = 14400  # 4 hours

# Test configuration values from benchmark_demo.ipynb
# TEST_CONFIG = {
#     "model_package_arn": "arn:aws:sagemaker:us-west-2:052150106756:model-package/test-finetuned-models-gamma/28",
#     "dataset_s3_uri": "s3://sagemaker-us-west-2-052150106756/studio-users/d20251107t195443/datasets/2025-11-07T19-55-37-609Z/zc_test.jsonl",
#     "s3_output_path": "s3://mufi-test-serverless-smtj/eval/",
#     "mlflow_tracking_server_arn": "arn:aws:sagemaker:us-west-2:052150106756:mlflow-tracking-server/mmlu-eval-experiment",
#     "model_package_group_arn": "arn:aws:sagemaker:us-west-2:052150106756:model-package-group/example-name-aovqo",
#     "region": "us-west-2",
# }

TEST_CONFIG = {
    "model_package_arn": "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1",
    "dataset_s3_uri": "s3://sagemaker-us-west-2-729646638167/model-customization/eval/zc_test.jsonl",
    "s3_output_path": "s3://sagemaker-us-west-2-729646638167/model-customization/eval/",
    "mlflow_tracking_server_arn": "arn:aws:sagemaker:us-west-2:729646638167:mlflow-app/app-W7FOBBXZANVX",
    "model_package_group_arn": "arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
    "region": "us-west-2",
}

# Base model only evaluation configuration (from commented section in notebook)
BASE_MODEL_ONLY_CONFIG = {
    "base_model_id": "meta-textgeneration-llama-3-2-1b-instruct",
    "dataset_s3_uri": "s3://sagemaker-us-west-2-052150106756/studio-users/d20251107t195443/datasets/2025-11-07T19-55-37-609Z/zc_test.jsonl",
    "s3_output_path": "s3://mufi-test-serverless-smtj/eval/",
    "mlflow_tracking_server_arn": "arn:aws:sagemaker:us-west-2:052150106756:mlflow-tracking-server/mmlu-eval-experiment",
    "region": "us-west-2",
}

# Nova model evaluation configuration (from commented section in notebook)
NOVA_CONFIG = {
    "model_package_arn": "arn:aws:sagemaker:us-east-1:052150106756:model-package/test-nova-finetuned-models/3",
    "dataset_s3_uri": "s3://sagemaker-us-east-1-052150106756/studio-users/d20251107t195443/datasets/2025-11-07T19-55-37-609Z/zc_test.jsonl",
    "s3_output_path": "s3://mufi-test-serverless-iad/eval/",
    "mlflow_tracking_server_arn": "arn:aws:sagemaker:us-east-1:052150106756:mlflow-tracking-server/mlflow-prod-server",
    "model_package_group_arn": "arn:aws:sagemaker:us-east-1:052150106756:model-package-group/test-nova-finetuned-models",
    "region": "us-east-1",
}


class TestBenchmarkEvaluatorIntegration:
    """Integration tests for BenchmarkEvaluator with fine-tuned model package"""

    def test_get_benchmarks_and_properties(self):
        """Test getting available benchmarks and their properties"""
        # Get available benchmarks
        Benchmark = get_benchmarks()
        
        # Verify it's an enum
        assert hasattr(Benchmark, "__members__")
        
        # Verify GEN_QA is available
        assert hasattr(Benchmark, "GEN_QA")
        
        # Get properties for GEN_QA benchmark
        properties = get_benchmark_properties(benchmark=Benchmark.GEN_QA)
        
        # Verify properties structure
        assert isinstance(properties, dict)
        assert "modality" in properties
        assert "description" in properties
        assert "metrics" in properties
        assert "strategy" in properties
        
        logger.info(f"GEN_QA properties: {properties}")

    def test_benchmark_evaluation_full_flow(self):
        """
        Test complete benchmark evaluation flow with fine-tuned model package.
        
        This test mirrors the flow from benchmark_demo.ipynb and covers:
        1. Creating BenchMarkEvaluator with GEN_QA benchmark
        2. Accessing hyperparameters
        3. Starting evaluation
        4. Monitoring execution
        5. Waiting for completion
        6. Viewing results
        7. Retrieving execution by ARN
        8. Listing all evaluations
        
        Test configuration values are taken directly from the notebook example.
        """
        # Get benchmarks
        Benchmark = get_benchmarks()
        
        # Step 1: Create BenchmarkEvaluator
        logger.info("Creating BenchmarkEvaluator with GEN_QA benchmark")
        
        # Create evaluator (matching notebook configuration)
        evaluator = BenchMarkEvaluator(
            benchmark=Benchmark.GEN_QA,
            model=TEST_CONFIG["model_package_arn"],
            s3_output_path=TEST_CONFIG["s3_output_path"],
            mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
            dataset=TEST_CONFIG["dataset_s3_uri"],
            model_package_group=TEST_CONFIG["model_package_group_arn"],
            base_eval_name="integ-test-gen-qa-eval",
        )
        
        # Verify evaluator was created
        assert evaluator is not None
        assert evaluator.benchmark == Benchmark.GEN_QA
        assert evaluator.model == TEST_CONFIG["model_package_arn"]
        assert evaluator.dataset == TEST_CONFIG["dataset_s3_uri"]
        
        logger.info(f"Created evaluator: {evaluator.base_eval_name}")
        
        # Step 2: Access hyperparameters
        logger.info("Accessing hyperparameters")
        hyperparams = evaluator.hyperparameters.to_dict()
        
        # Verify hyperparameters structure
        assert isinstance(hyperparams, dict)
        assert "max_new_tokens" in hyperparams
        assert "temperature" in hyperparams
        
        logger.info(f"Hyperparameters: {hyperparams}")
        
        # Step 3: Start evaluation
        logger.info("Starting evaluation execution")
        execution = evaluator.evaluate()
        
        # Verify execution was created
        assert execution is not None
        assert execution.arn is not None
        assert execution.name is not None
        assert execution.eval_type is not None
        
        logger.info(f"Pipeline Execution ARN: {execution.arn}")
        logger.info(f"Initial Status: {execution.status.overall_status}")
        
        # Step 4: Monitor execution
        logger.info("Refreshing execution status")
        execution.refresh()
        
        # Verify status was updated
        assert execution.status.overall_status is not None
        
        # Log step details if available
        if execution.status.step_details:
            logger.info("Step Details:")
            for step in execution.status.step_details:
                logger.info(f"  {step.name}: {step.status}")
        
        # Step 5: Wait for completion
        logger.info(f"Waiting for evaluation to complete (timeout: {EVALUATION_TIMEOUT_SECONDS}s / {EVALUATION_TIMEOUT_SECONDS//3600}h)")
        
        try:
            execution.wait(target_status="Succeeded", poll=30, timeout=EVALUATION_TIMEOUT_SECONDS)
            logger.info(f"Final Status: {execution.status.overall_status}")
            
            # Verify completion
            assert execution.status.overall_status == "Succeeded"
            
            # Step 6: View results
            logger.info("Displaying results")
            execution.show_results()
            
            # Verify S3 output path is set
            assert execution.s3_output_path is not None
            logger.info(f"Results stored at: {execution.s3_output_path}")
            
        except Exception as e:
            logger.error(f"Evaluation failed or timed out: {e}")
            logger.error(f"Final status: {execution.status.overall_status}")
            if execution.status.failure_reason:
                logger.error(f"Failure reason: {execution.status.failure_reason}")
            
            # Log step failures
            if execution.status.step_details:
                for step in execution.status.step_details:
                    if "failed" in step.status.lower():
                        logger.error(f"Failed step: {step.name}")
                        if step.failure_reason:
                            logger.error(f"  Reason: {step.failure_reason}")
            
            # Re-raise to fail the test
            raise
        
        # Step 7: Retrieve execution by ARN
        logger.info("Retrieving execution by ARN")
        retrieved_execution = EvaluationPipelineExecution.get(
            arn=execution.arn,
            region=TEST_CONFIG["region"]
        )
        
        # Verify retrieved execution matches
        assert retrieved_execution.arn == execution.arn
        
        logger.info(f"Retrieved execution status: {retrieved_execution.status.overall_status}")
        
        # Step 8: List all benchmark evaluations
        logger.info("Listing all benchmark evaluations")
        all_executions_iter = BenchMarkEvaluator.get_all(region=TEST_CONFIG["region"])
        all_executions = list(all_executions_iter)
        
        if all_executions:
            # Verify our execution is in the list
            execution_arns = [exec.arn for exec in all_executions]
            assert execution.arn in execution_arns
        
        logger.info("Integration test completed successfully")

    def test_benchmark_evaluator_validation(self):
        """Test BenchmarkEvaluator validation of inputs"""
        Benchmark = get_benchmarks()
        
        # Test invalid benchmark type
        with pytest.raises(ValueError):
            BenchMarkEvaluator(
                benchmark="invalid_benchmark",
                model=TEST_CONFIG["model_package_arn"],
                s3_output_path=TEST_CONFIG["s3_output_path"],
                mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
                dataset="s3://bucket/dataset.jsonl",
            )
        
        # Test invalid MLflow ARN format
        with pytest.raises(ValueError, match="Invalid MLFlow resource ARN"):
            BenchMarkEvaluator(
                benchmark=Benchmark.GEN_QA,
                model=TEST_CONFIG["model_package_arn"],
                s3_output_path=TEST_CONFIG["s3_output_path"],
                mlflow_resource_arn="invalid-arn",
                dataset="s3://bucket/dataset.jsonl",
            )
        
        logger.info("Validation tests passed")

    def test_benchmark_subtasks_validation(self):
        """Test benchmark subtask validation"""
        Benchmark = get_benchmarks()
        
        # Test valid subtask for MMLU (has subtask support)
        evaluator = BenchMarkEvaluator(
            benchmark=Benchmark.MMLU,
            model=TEST_CONFIG["model_package_arn"],
            s3_output_path=TEST_CONFIG["s3_output_path"],
            mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
            dataset="s3://bucket/dataset.jsonl",
            subtasks=["abstract_algebra", "anatomy"],
            model_package_group="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test",
        )
        assert evaluator.subtasks == ["abstract_algebra", "anatomy"]
        
        # Test invalid subtask for benchmark without subtask support
        with pytest.raises(ValueError, match="Subtask is not supported"):
            BenchMarkEvaluator(
                benchmark=Benchmark.GEN_QA,
                model=TEST_CONFIG["model_package_arn"],
                s3_output_path=TEST_CONFIG["s3_output_path"],
                mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
                dataset="s3://bucket/dataset.jsonl",
                subtasks=["invalid"],
                model_package_group="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test",
            )
        
        logger.info("Subtask validation tests passed")

    @pytest.mark.skip(reason="Base model only evaluation - to be enabled when needed")
    def test_benchmark_evaluation_base_model_only(self):
        """
        Test benchmark evaluation with base model only (no fine-tuned model).
        
        This test uses a JumpStart model ID directly instead of a model package ARN.
        Configuration from commented section in benchmark_demo.ipynb.
        
        Note: This test is currently skipped. Remove the @pytest.mark.skip decorator
        when you want to enable it.
        """
        # Get benchmarks
        Benchmark = get_benchmarks()
        
        logger.info("Creating BenchmarkEvaluator with base model only (JumpStart model ID)")
        
        # Create evaluator with JumpStart model ID (no model package)
        evaluator = BenchMarkEvaluator(
            benchmark=Benchmark.GEN_QA,
            model=BASE_MODEL_ONLY_CONFIG["base_model_id"],
            s3_output_path=BASE_MODEL_ONLY_CONFIG["s3_output_path"],
            mlflow_resource_arn=BASE_MODEL_ONLY_CONFIG["mlflow_tracking_server_arn"],
            dataset=BASE_MODEL_ONLY_CONFIG["dataset_s3_uri"],
            base_eval_name="integ-test-base-model-only",
            # Note: model_package_group not needed for JumpStart models
        )
        
        # Verify evaluator was created
        assert evaluator is not None
        assert evaluator.benchmark == Benchmark.GEN_QA
        assert evaluator.model == BASE_MODEL_ONLY_CONFIG["base_model_id"]
        
        logger.info(f"Created evaluator: {evaluator.base_eval_name}")
        
        # Start evaluation
        logger.info("Starting evaluation execution")
        execution = evaluator.evaluate()
        
        # Verify execution was created
        assert execution is not None
        assert execution.arn is not None
        assert execution.name is not None
        
        logger.info(f"Pipeline Execution ARN: {execution.arn}")
        logger.info(f"Initial Status: {execution.status.overall_status}")
        
        # Wait for completion
        logger.info(f"Waiting for evaluation to complete (timeout: {EVALUATION_TIMEOUT_SECONDS}s / {EVALUATION_TIMEOUT_SECONDS//3600}h)")
        execution.wait(target_status="Succeeded", poll=30, timeout=EVALUATION_TIMEOUT_SECONDS)
        
        # Verify completion
        assert execution.status.overall_status == "Succeeded"
        logger.info("Base model only evaluation completed successfully")

    @pytest.mark.skip(reason="Nova model evaluation - to be enabled when needed")
    def test_benchmark_evaluation_nova_model(self):
        """
        Test benchmark evaluation with Nova model.
        
        This test uses a Nova fine-tuned model package in us-east-1 region.
        Configuration from commented section in benchmark_demo.ipynb.
        
        Note: This test is currently skipped. Remove the @pytest.mark.skip decorator
        when you want to enable it.
        """
        # Get benchmarks
        Benchmark = get_benchmarks()
        
        logger.info("Creating BenchmarkEvaluator with Nova model")
        
        # Create evaluator with Nova model package
        evaluator = BenchMarkEvaluator(
            benchmark=Benchmark.GEN_QA,
            model=NOVA_CONFIG["model_package_arn"],
            s3_output_path=NOVA_CONFIG["s3_output_path"],
            mlflow_resource_arn=NOVA_CONFIG["mlflow_tracking_server_arn"],
            dataset=NOVA_CONFIG["dataset_s3_uri"],
            model_package_group=NOVA_CONFIG["model_package_group_arn"],
            base_eval_name="integ-test-nova-eval",
            region=NOVA_CONFIG["region"],
        )
        
        # Verify evaluator was created
        assert evaluator is not None
        assert evaluator.benchmark == Benchmark.GEN_QA
        assert evaluator.model == NOVA_CONFIG["model_package_arn"]
        assert evaluator.region == NOVA_CONFIG["region"]
        
        logger.info(f"Created evaluator: {evaluator.base_eval_name}")
        
        # Access hyperparameters (Nova models may have different hyperparameters)
        logger.info("Accessing hyperparameters")
        hyperparams = evaluator.hyperparameters.to_dict()
        
        # Verify hyperparameters structure
        assert isinstance(hyperparams, dict)
        logger.info(f"Hyperparameters: {hyperparams}")
        
        # Start evaluation
        logger.info("Starting evaluation execution")
        execution = evaluator.evaluate()
        
        # Verify execution was created
        assert execution is not None
        assert execution.arn is not None
        assert execution.name is not None
        
        logger.info(f"Pipeline Execution ARN: {execution.arn}")
        logger.info(f"Initial Status: {execution.status.overall_status}")
        
        # Monitor execution
        execution.refresh()
        logger.info(f"Status after refresh: {execution.status.overall_status}")
        
        # Wait for completion
        logger.info("Waiting for evaluation to complete (timeout: 1 hour)")
        execution.wait(target_status="Succeeded", poll=30, timeout=3600)
        
        # Verify completion
        assert execution.status.overall_status == "Succeeded"
        
        # View results
        logger.info("Displaying results")
        execution.show_results()
        
        logger.info("Nova model evaluation completed successfully")
