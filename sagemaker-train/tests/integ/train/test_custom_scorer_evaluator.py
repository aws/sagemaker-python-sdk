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
"""Integration tests for CustomScorerEvaluator"""
from __future__ import absolute_import

import pytest
import logging

from sagemaker.train.evaluate import (
    CustomScorerEvaluator,
    get_builtin_metrics,
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

# Test configuration values from custom_scorer_demo.ipynb
# TEST_CONFIG = {
#     "evaluator_arn": "arn:aws:sagemaker:us-west-2:052150106756:hub-content/F3LMYANDKWPZCROJVCKMJ7TOML6QMZBZRRQOVTUL45VUK7PJ4SXA/JsonDoc/eval-lambda-test/0.0.1",
#     "dataset_s3_uri": "s3://sagemaker-us-west-2-052150106756/studio-users/d20251107t195443/datasets/2025-11-07T19-55-37-609Z/zc_test.jsonl",
#     "model_package_arn": "arn:aws:sagemaker:us-west-2:052150106756:model-package/test-finetuned-models-gamma/28",
#     "s3_output_path": "s3://mufi-test-serverless-smtj/eval/",
#     "mlflow_tracking_server_arn": "arn:aws:sagemaker:us-west-2:052150106756:mlflow-tracking-server/mmlu-eval-experiment",
#     "evaluate_base_model": False,
#     "region": "us-west-2",
# }

TEST_CONFIG = {
    "evaluator_arn": "arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/JsonDoc/eval-lambda-test/0.0.1",
    "model_package_arn": "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1",
    "dataset_s3_uri": "s3://sagemaker-us-west-2-729646638167/model-customization/eval/zc_test.jsonl",
    "s3_output_path": "s3://sagemaker-us-west-2-729646638167/model-customization/eval/",
    "mlflow_tracking_server_arn": "arn:aws:sagemaker:us-west-2:729646638167:mlflow-app/app-W7FOBBXZANVX",
    "model_package_group_arn": "arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
    "evaluate_base_model": False,
    "region": "us-west-2",
}


@pytest.mark.skip(reason="Temporarily skipped - moved from tests/integ/sagemaker/modules/evaluate/")
class TestCustomScorerEvaluatorIntegration:
    """Integration tests for CustomScorerEvaluator with custom evaluator"""

    def test_get_builtin_metrics(self):
        """Test getting available built-in metrics"""
        # Get available built-in metrics
        BuiltInMetric = get_builtin_metrics()
        
        # Verify it's an enum
        assert hasattr(BuiltInMetric, "__members__")
        
        # Verify PRIME_MATH is available
        assert hasattr(BuiltInMetric, "PRIME_MATH")
        
        # Verify PRIME_CODE is available
        assert hasattr(BuiltInMetric, "PRIME_CODE")
        
        logger.info(f"Built-in metrics: {list(BuiltInMetric.__members__.keys())}")

    def test_custom_scorer_evaluation_full_flow(self):
        """
        Test complete custom scorer evaluation flow with custom evaluator ARN.
        
        This test mirrors the flow from custom_scorer_demo.ipynb and covers:
        1. Creating CustomScorerEvaluator with custom evaluator ARN
        2. Accessing hyperparameters
        3. Starting evaluation
        4. Monitoring execution
        5. Waiting for completion
        6. Viewing results
        7. Retrieving execution by ARN
        8. Listing all evaluations
        
        Test configuration values are taken directly from the notebook example.
        """
        # Step 1: Create CustomScorerEvaluator
        logger.info("Creating CustomScorerEvaluator with custom evaluator ARN")
        
        # Create evaluator (matching notebook configuration)
        evaluator = CustomScorerEvaluator(
            evaluator=TEST_CONFIG["evaluator_arn"],
            dataset=TEST_CONFIG["dataset_s3_uri"],
            model=TEST_CONFIG["model_package_arn"],
            s3_output_path=TEST_CONFIG["s3_output_path"],
            # mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
            evaluate_base_model=TEST_CONFIG["evaluate_base_model"],
        )
        
        # Verify evaluator was created
        assert evaluator is not None
        assert evaluator.evaluator == TEST_CONFIG["evaluator_arn"]
        assert evaluator.model == TEST_CONFIG["model_package_arn"]
        assert evaluator.dataset == TEST_CONFIG["dataset_s3_uri"]
        assert evaluator.evaluate_base_model == TEST_CONFIG["evaluate_base_model"]
        
        logger.info(f"Created evaluator with custom evaluator ARN")
        
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
        
        # Step 8: List all custom scorer evaluations
        logger.info("Listing all custom scorer evaluations")
        all_executions_iter = CustomScorerEvaluator.get_all(region=TEST_CONFIG["region"])
        all_executions = list(all_executions_iter)
        
        # Verify our execution is in the list
        execution_arns = [exec.arn for exec in all_executions]
        if execution_arns:
            assert execution.arn in execution_arns
        
        logger.info("Integration test completed successfully")

    def test_custom_scorer_evaluator_validation(self):
        """Test CustomScorerEvaluator validation of inputs"""
        # Test invalid evaluator type
        with pytest.raises(ValueError, match="Invalid evaluator type"):
            CustomScorerEvaluator(
                evaluator=123,  # Invalid type (not string, enum, or object)
                model=TEST_CONFIG["model_package_arn"],
                s3_output_path=TEST_CONFIG["s3_output_path"],
                # mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
                dataset=TEST_CONFIG["dataset_s3_uri"],
            )
        
        # Test invalid MLflow ARN format
        with pytest.raises(ValueError, match="Invalid MLFlow resource ARN"):
            CustomScorerEvaluator(
                evaluator=TEST_CONFIG["evaluator_arn"],
                model=TEST_CONFIG["model_package_arn"],
                s3_output_path=TEST_CONFIG["s3_output_path"],
                mlflow_resource_arn="invalid-arn",
                dataset=TEST_CONFIG["dataset_s3_uri"],
            )
        
        logger.info("Validation tests passed")

    @pytest.mark.skip(reason="Built-in metric evaluation - to be enabled when needed")
    def test_custom_scorer_with_builtin_metric(self):
        """
        Test custom scorer evaluation with built-in metric.
        
        This test uses a built-in metric (PRIME_MATH) instead of a custom evaluator ARN.
        Configuration adapted from commented section in custom_scorer_demo.ipynb.
        
        Note: This test is currently skipped. Remove the @pytest.mark.skip decorator
        when you want to enable it.
        """
        # Get built-in metrics
        BuiltInMetric = get_builtin_metrics()
        
        logger.info("Creating CustomScorerEvaluator with built-in metric")
        
        # Create evaluator with built-in metric
        evaluator = CustomScorerEvaluator(
            evaluator=BuiltInMetric.PRIME_MATH,  # Built-in metric enum
            dataset=TEST_CONFIG["dataset_s3_uri"],
            model=TEST_CONFIG["model_package_arn"],
            s3_output_path=TEST_CONFIG["s3_output_path"],
            # mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
            evaluate_base_model=False,
        )
        
        # Verify evaluator was created
        assert evaluator is not None
        assert evaluator.evaluator == BuiltInMetric.PRIME_MATH
        
        logger.info(f"Created evaluator with built-in metric: {BuiltInMetric.PRIME_MATH}")
        
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
        logger.info("Built-in metric evaluation completed successfully")

    @pytest.mark.skip(reason="Base model only evaluation - not working yet per notebook")
    def test_custom_scorer_base_model_only(self):
        """
        Test custom scorer evaluation with base model only (no fine-tuned model).
        
        Note: Per the notebook, "Evaluation with Base Model Only is yet to be 
        implemented/tested - Not Working currently". This test is skipped until
        that functionality is available.
        """
        logger.info("Base model only evaluation - not yet implemented")
        pass
