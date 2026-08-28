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
"""Integration tests for LLMAsJudgeEvaluator ``evaluator_model`` validation.

These tests exercise ONLY the two-step judge-model validation against live AWS —
they do NOT submit an evaluation job:

* Step 1 (construction): the model is checked for membership in the
  service-maintained supported-judge-models list read from
  ``s3://jumpstart-cache-prod-<region>/fmhMetadata/supported-llmaj-judge-models.json``.
* Step 2 (``_check_evaluator_model_lifecycle``): the model's live Bedrock
  lifecycle is checked via ``bedrock:GetFoundationModel`` to confirm it is still
  in service (not unavailable in-region / past end of life).

Each test hits real AWS (S3, JumpStart hub for base-model resolution, and Bedrock)
but makes only read-only calls; none of them start a SageMaker pipeline or a
Bedrock evaluation job.
"""
from __future__ import absolute_import

import logging

import pytest
from pydantic import ValidationError

from sagemaker.train.evaluate import LLMAsJudgeEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

REGION = "us-west-2"

# A real, resolvable public JumpStart base model — keeps construction's model
# resolution cheap and account-independent (no fine-tuned model package needed).
BASE_MODEL = "meta-textgeneration-llama-3-2-1b-instruct"

# Format-only fields (not existence-checked at construction): a well-formed S3 URI
# and a well-formed MLflow ARN are enough to construct the evaluator.
DATASET_S3_URI = "s3://sagemaker-us-west-2-729646638167/model-customization/eval/gen_qa.jsonl"
S3_OUTPUT_PATH = "s3://sagemaker-us-west-2-729646638167/model-customization/eval/"
MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:729646638167:mlflow-app/app-TTAUWUNMUHH6"

# An in-service judge model in us-west-2 (present in the supported list AND ACTIVE
# in Bedrock) — used for the positive path.
ACTIVE_JUDGE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

# A judge model that is still advertised in the supported-judge-models list but is
# no longer available in us-west-2 (Bedrock GetFoundationModel returns
# ResourceNotFoundException). This is exactly the stale-list / end-of-life case the
# feature guards against: it passes step 1 (membership) but must fail step 2.
RETIRED_JUDGE_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# A model id that is not a supported judge at all — must fail step 1 at construction.
UNSUPPORTED_JUDGE_MODEL = "not-a-real-judge-model-v1:0"


def _build_evaluator(evaluator_model):
    """Construct an evaluator (runs step-1 validation) without submitting a job."""
    return LLMAsJudgeEvaluator(
        model=BASE_MODEL,
        evaluator_model=evaluator_model,
        dataset=DATASET_S3_URI,
        s3_output_path=S3_OUTPUT_PATH,
        mlflow_resource_arn=MLFLOW_ARN,
        region=REGION,
    )


class TestLLMAsJudgeEvaluatorModelValidation:
    """Live validation-only tests for ``evaluator_model`` (no evaluation job)."""

    def test_supported_active_model_passes_both_validation_steps(self):
        """Positive: an in-service judge model passes step 1 and step 2.

        Step 1 succeeds at construction (the model is in the live supported-models
        list); step 2 succeeds because Bedrock reports the model as in service.
        """
        evaluator = _build_evaluator(ACTIVE_JUDGE_MODEL)
        assert evaluator.evaluator_model == ACTIVE_JUDGE_MODEL

        # Step 2: real bedrock:GetFoundationModel lifecycle check — must not raise.
        evaluator._check_evaluator_model_lifecycle(REGION)
        logger.info("Active judge model %s passed both validation steps", ACTIVE_JUDGE_MODEL)

    def test_unsupported_model_fails_construction(self):
        """Negative (step 1): a non-judge model is rejected at construction.

        The real supported-judge-models list is fetched and the model is absent, so
        construction fails fast — long before any evaluation job could be started.
        """
        with pytest.raises(ValidationError) as exc_info:
            _build_evaluator(UNSUPPORTED_JUDGE_MODEL)

        message = str(exc_info.value)
        assert "is not a supported LLM-as-Judge model" in message
        assert UNSUPPORTED_JUDGE_MODEL in message
        logger.info("Unsupported model correctly rejected at construction")

    def test_retired_model_fails_lifecycle_check(self):
        """Negative (step 2): a listed-but-retired judge model fails the live check.

        The model is still in the supported-models list (step 1 passes at
        construction), but Bedrock no longer offers it in this region, so the
        lifecycle check fails fast instead of letting a doomed job spin up.

        Assumes the test identity holds ``bedrock:GetFoundationModel``; without it
        the check degrades to a warning by design and this assertion would not hold.
        """
        evaluator = _build_evaluator(RETIRED_JUDGE_MODEL)
        assert evaluator.evaluator_model == RETIRED_JUDGE_MODEL  # step 1 passed

        with pytest.raises(ValueError, match="not available in region"):
            evaluator._check_evaluator_model_lifecycle(REGION)
        logger.info("Retired model %s correctly failed the lifecycle check", RETIRED_JUDGE_MODEL)
