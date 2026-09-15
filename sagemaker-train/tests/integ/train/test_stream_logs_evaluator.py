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
"""Integration tests for evaluator stream_logs()"""
from __future__ import annotations

import logging
import time

import boto3
import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.train.evaluate.benchmark_evaluator import BenchMarkEvaluator, get_benchmarks
from sagemaker.train.evaluate.custom_scorer_evaluator import CustomScorerEvaluator, get_builtin_metrics
from sagemaker.train.evaluate.llm_as_judge_evaluator import LLMAsJudgeEvaluator
from sagemaker.train.evaluate.execution import (
    EvaluationPipelineExecution,
    PipelineExecutionStatus,
    StepDetail,
)

logger = logging.getLogger(__name__)

REGION = "us-west-2"

S3_OUTPUT = "s3://sagemaker-us-west-2-729646638167/model-customization/eval/"
MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1"
DATASET_S3 = "s3://sagemaker-us-west-2-729646638167/model-customization/eval/zc_test.jsonl"

BENCHMARK_EXECUTION_ARN = "arn:aws:sagemaker:us-west-2:729646638167:pipeline/SagemakerEvaluation-BenchmarkEvaluation-499b3c7e-e456-4297-9dc0-cc5737137c9c/execution/p1gtwhjm9dzt"
BENCHMARK_STEP_ARN = "arn:aws:sagemaker:us-west-2:729646638167:training-job/pipelines-p1gtwhjm9dzt-EvaluateCustomModel-XEdt5h2gQC"

CUSTOM_SCORER_EXECUTION_ARN = "arn:aws:sagemaker:us-west-2:729646638167:pipeline/SagemakerEvaluation-CustomScorerEvaluation-2d0fde36-af0f-49d7-8b8e-a5e11352dc1f/execution/yca2ij65mlhr"
CUSTOM_SCORER_STEP_ARN = "arn:aws:sagemaker:us-west-2:729646638167:training-job/pipelines-yca2ij65mlhr-EvaluateCustomModel-MlMUskwbNB"

LLMAJ_EXECUTION_ARN = "arn:aws:sagemaker:us-west-2:729646638167:pipeline/SagemakerEvaluation-LLMAJEvaluation-ac7a1fe7-fe8a-445c-8aa5-702b3d6b7771/execution/hmk0lcu6ufzc"
LLMAJ_STEP_ARN = "arn:aws:sagemaker:us-west-2:729646638167:training-job/pipelines-hmk0lcu6ufzc-EvaluateCustomModelM-6UaY2bgNL5"



@pytest.fixture(scope="module")
def sagemaker_session():
    boto_session = boto3.Session(region_name=REGION)
    return Session(boto_session=boto_session)


def _make_execution(execution_arn: str, step_name: str, step_arn: str):
    """Construct a completed execution from known pipeline step details."""
    return EvaluationPipelineExecution(
        arn=execution_arn,
        name=execution_arn.split("/execution/")[1],
        status=PipelineExecutionStatus(
            overall_status="Succeeded",
            step_details=[
                StepDetail(
                    name=step_name,
                    status="Succeeded",
                    display_name=step_name,
                    job_arn=step_arn,
                )
            ],
        ),
    )


class TestEvaluatorStreamLogsFromCompletedJobs:
    """Verify evaluator.stream_logs() works for each evaluator type.

    Each test uses the actual evaluator class that produced the pipeline
    execution, with real step ARNs that have CloudWatch logs available.
    """

    def test_benchmark_evaluator_stream_logs(self, sagemaker_session):
        """BenchMarkEvaluator.stream_logs() on a completed benchmark pipeline."""
        Benchmark = get_benchmarks()
        evaluator = BenchMarkEvaluator(
            benchmark=Benchmark.MMLU,
            model=MODEL_PACKAGE_ARN,
            s3_output_path=S3_OUTPUT,
            sagemaker_session=sagemaker_session,
        )
        evaluator._latest_execution = _make_execution(
            BENCHMARK_EXECUTION_ARN, "EvaluateCustomModel", BENCHMARK_STEP_ARN
        )

        start = time.time()
        evaluator.stream_logs(poll=2)
        elapsed = time.time() - start

        assert elapsed < 30, (
            f"stream_logs() took {elapsed:.1f}s — should exit quickly for completed job"
        )
        print(f"✓ BenchMarkEvaluator.stream_logs() completed in {elapsed:.1f}s")

    def test_custom_scorer_evaluator_stream_logs(self, sagemaker_session):
        """CustomScorerEvaluator.stream_logs() on a completed custom scorer pipeline."""
        BuiltInMetric = get_builtin_metrics()
        evaluator = CustomScorerEvaluator(
            evaluator=BuiltInMetric.PRIME_MATH,
            dataset=DATASET_S3,
            model=MODEL_PACKAGE_ARN,
            s3_output_path=S3_OUTPUT,
            sagemaker_session=sagemaker_session,
        )
        evaluator._latest_execution = _make_execution(
            CUSTOM_SCORER_EXECUTION_ARN, "EvaluateCustomModel", CUSTOM_SCORER_STEP_ARN
        )

        start = time.time()
        evaluator.stream_logs(poll=2)
        elapsed = time.time() - start

        assert elapsed < 30, (
            f"stream_logs() took {elapsed:.1f}s — should exit quickly for completed job"
        )
        print(f"✓ CustomScorerEvaluator.stream_logs() completed in {elapsed:.1f}s")

    def test_llm_as_judge_evaluator_stream_logs(self, sagemaker_session):
        """LLMAsJudgeEvaluator.stream_logs() on a completed LLMAJ pipeline."""
        evaluator = LLMAsJudgeEvaluator(
            model=MODEL_PACKAGE_ARN,
            evaluator_model="amazon.nova-pro-v1:0",
            dataset=DATASET_S3,
            builtin_metrics=["Completeness"],
            s3_output_path=S3_OUTPUT,
            sagemaker_session=sagemaker_session,
            evaluate_base_model=False,
        )
        evaluator._latest_execution = _make_execution(
            LLMAJ_EXECUTION_ARN, "EvaluateCustomModelMetrics", LLMAJ_STEP_ARN
        )

        start = time.time()
        evaluator.stream_logs(poll=2)
        elapsed = time.time() - start

        assert elapsed < 30, (
            f"stream_logs() took {elapsed:.1f}s — should exit quickly for completed job"
        )
        print(f"✓ LLMAsJudgeEvaluator.stream_logs() completed in {elapsed:.1f}s")
