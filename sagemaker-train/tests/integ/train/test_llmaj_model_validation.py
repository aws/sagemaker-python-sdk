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
"""Integration tests for LLMAsJudgeEvaluator model validation.

These tests exercise ONLY input validation against live AWS — they do NOT submit
an evaluation job. Two areas are covered:

1. ``evaluator_model`` (the judge) two-step validation:
   * Step 1 (construction): membership in the service-maintained
     supported-judge-models list read from
     ``s3://jumpstart-cache-prod-<region>/fmhMetadata/supported-llmaj-judge-models.json``.
   * Step 2 (``_check_evaluator_model_lifecycle``): the model's live Bedrock
     lifecycle via ``bedrock:GetFoundationModel`` (still in service vs unavailable
     in-region / past end of life).

2. ``model`` (the model under evaluation) resolution against the JumpStart hub at
   construction time.

Each test hits real AWS (S3, JumpStart hub for base-model resolution, and Bedrock)
but makes only read-only calls; none of them start a SageMaker pipeline or a
Bedrock evaluation job.
"""
from __future__ import absolute_import

import json
import logging
import time
import uuid
from contextlib import contextmanager

import boto3
import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

from sagemaker.core.helper.session_helper import Session
from sagemaker.train.evaluate import LLMAsJudgeEvaluator

_LIFECYCLE_LOGGER = "sagemaker.train.evaluate.llm_as_judge_evaluator"

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

# A model id that does not exist in the JumpStart hub — used for the negative
# base-model-resolution test.
INVALID_BASE_MODEL = "not-a-real-jumpstart-model-xyz-123"


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

    def test_retired_model_lifecycle_enforced_or_degrades(self, caplog):
        """Step 2 on a listed-but-retired judge model, under the ambient identity.

        The model is still in the supported-models list (step 1 passes at
        construction), but Bedrock no longer offers it in this region. The outcome
        depends on the runner's own permissions, and BOTH are correct:

        * identity WITH ``bedrock:GetFoundationModel`` → the check fails fast
          (``ValueError``), stopping a doomed job before it starts;
        * identity WITHOUT it → the check can't verify, so it degrades to a warning
          and continues (never blocks).

        This test tolerates both so it passes regardless of the runner's baseline
        permissions. The deterministic per-permission behavior is pinned down in
        :class:`TestEvaluatorModelLifecycleBedrockPermission`, which provisions its
        own roles.
        """
        evaluator = _build_evaluator(RETIRED_JUDGE_MODEL)
        assert evaluator.evaluator_model == RETIRED_JUDGE_MODEL  # step 1 passed

        with caplog.at_level(logging.WARNING, logger=_LIFECYCLE_LOGGER):
            try:
                evaluator._check_evaluator_model_lifecycle(REGION)
                enforced = False
            except ValueError as e:
                enforced = True
                assert "not available in region" in str(e)

        if enforced:
            logger.info("Retired model enforced (identity can verify lifecycle)")
        else:
            # Degraded rather than blocked — a warning must explain why.
            assert any(
                "still in service" in record.getMessage()
                for record in caplog.records
            ), "expected a lifecycle warning when the check degrades"
            logger.info("Retired model degraded gracefully (identity cannot verify)")


class TestJumpStartBaseModelValidation:
    """Live validation of the ``model`` argument against the JumpStart hub."""

    def test_valid_jumpstart_model_resolves(self):
        """Positive: a real JumpStart model id resolves against the hub at construction."""
        evaluator = LLMAsJudgeEvaluator(
            model=BASE_MODEL,
            evaluator_model=ACTIVE_JUDGE_MODEL,
            dataset=DATASET_S3_URI,
            s3_output_path=S3_OUTPUT_PATH,
            mlflow_resource_arn=MLFLOW_ARN,
            region=REGION,
        )
        assert evaluator.model == BASE_MODEL
        # Resolution populated the base-model identity from the hub.
        assert evaluator._base_model_name
        assert evaluator._base_model_arn and "hub-content" in evaluator._base_model_arn
        logger.info("JumpStart model %s resolved to %s", BASE_MODEL, evaluator._base_model_arn)

    def test_nonexistent_jumpstart_model_fails_construction(self):
        """Negative: a model id absent from the JumpStart hub fails fast at construction."""
        with pytest.raises(ValidationError) as exc_info:
            LLMAsJudgeEvaluator(
                model=INVALID_BASE_MODEL,
                evaluator_model=ACTIVE_JUDGE_MODEL,
                dataset=DATASET_S3_URI,
                s3_output_path=S3_OUTPUT_PATH,
                mlflow_resource_arn=MLFLOW_ARN,
                region=REGION,
            )

        message = str(exc_info.value)
        assert "Failed to resolve" in message
        assert INVALID_BASE_MODEL in message
        logger.info("Nonexistent JumpStart model correctly rejected at construction")


# IAM policy statements used to provision the two throwaway roles below.
_BEDROCK_GETMODEL_ALLOW = [
    {"Effect": "Allow", "Action": ["bedrock:GetFoundationModel"], "Resource": "*"}
]
# No bedrock grant — ``bedrock:GetFoundationModel`` is implicitly denied.
_NO_BEDROCK_ALLOW = [
    {"Effect": "Allow", "Action": ["sts:GetCallerIdentity"], "Resource": "*"}
]


@contextmanager
def _assumed_role_session(permission_statements, label):
    """Create a throwaway IAM role with ``permission_statements``, assume it, and
    yield a SageMaker session backed by its temporary credentials.

    Mirrors the self-provisioning scoped-role pattern in
    ``sagemaker-serve/tests/integ/test_private_hub_artifact_resolution.py``: the
    role is created, used, and deleted on exit. ``pytest.skip`` is raised (cleanly)
    when the runner cannot create or assume roles (e.g. missing ``iam:CreateRole``
    / ``sts:AssumeRole``). Provisioning the exact permission set makes the caller's
    behavior deterministic regardless of the runner's baseline identity.
    """
    iam = boto3.client("iam")
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    role_name = f"sdk-llmaj-{label}-{uuid.uuid4().hex[:8]}"

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    permission_policy = {"Version": "2012-10-17", "Statement": permission_statements}

    try:
        role_arn = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"SDK integ test role ({label}) — auto-deleted",
        )["Role"]["Arn"]
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="llmaj-test-policy",
            PolicyDocument=json.dumps(permission_policy),
        )
    except ClientError as e:
        pytest.skip(f"Cannot create IAM role (likely missing permissions): {e}")

    def _cleanup():
        try:
            iam.delete_role_policy(RoleName=role_name, PolicyName="llmaj-test-policy")
            iam.delete_role(RoleName=role_name)
        except Exception as e:  # noqa: BLE001 - best-effort teardown
            logger.warning("Role cleanup failed for %s: %s", role_name, e)

    # IAM role creation + assume-role are eventually consistent; wait then retry.
    time.sleep(15)
    credentials = None
    last_err = None
    for _ in range(6):
        try:
            credentials = sts.assume_role(
                RoleArn=role_arn, RoleSessionName=f"llmaj-{label}"
            )["Credentials"]
            break
        except ClientError as e:
            last_err = e
            time.sleep(5)
    if credentials is None:
        _cleanup()
        pytest.skip(f"Cannot assume test role: {last_err}")

    boto_session = boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=REGION,
    )
    try:
        yield Session(boto_session=boto_session)
    finally:
        _cleanup()


class TestEvaluatorModelLifecycleBedrockPermission:
    """Live tests for how step 2 behaves w.r.t. the caller's Bedrock permission.

    No evaluation job is submitted. Each test provisions its OWN throwaway role
    (with / without ``bedrock:GetFoundationModel``) and runs the lifecycle call
    under it, so the outcome is deterministic regardless of the runner's baseline
    permissions. Both use the same retired judge model to contrast the paths.
    """

    def test_with_bedrock_permission_lifecycle_check_enforces(self):
        """Positive: WITH the permission, the lifecycle check runs and enforces.

        Under a role that grants ``bedrock:GetFoundationModel``, the retired model
        is looked up, found unavailable, and the check fails fast — proving the
        Bedrock lookup actually executed (a permission-less degrade would not raise).
        """
        evaluator = _build_evaluator(RETIRED_JUDGE_MODEL)
        with _assumed_role_session(_BEDROCK_GETMODEL_ALLOW, "with-bedrock") as session:
            evaluator.sagemaker_session = session
            with pytest.raises(ValueError, match="not available in region"):
                evaluator._check_evaluator_model_lifecycle(REGION)
        logger.info("Permitted identity enforced the lifecycle check")

    def test_without_bedrock_permission_lifecycle_check_degrades(self, caplog):
        """Negative: WITHOUT the permission, the check degrades — never blocks.

        The evaluator is built under the default identity (so step 1 and base-model
        resolution succeed), then the lifecycle call is run under a role lacking
        ``bedrock:GetFoundationModel``. Bedrock returns AccessDenied, so the SDK
        warns and continues instead of raising — even for a retired model that
        would otherwise fail fast under a permitted identity.
        """
        evaluator = _build_evaluator(RETIRED_JUDGE_MODEL)
        with _assumed_role_session(_NO_BEDROCK_ALLOW, "no-bedrock") as session:
            evaluator.sagemaker_session = session
            with caplog.at_level(logging.WARNING, logger=_LIFECYCLE_LOGGER):
                # Must NOT raise despite the model being retired — we can't verify.
                evaluator._check_evaluator_model_lifecycle(REGION)

        assert any(
            "bedrock:GetFoundationModel" in record.getMessage()
            for record in caplog.records
        ), "expected a warning naming the missing bedrock:GetFoundationModel permission"
        logger.info("Unpermitted identity degraded gracefully (no block)")
